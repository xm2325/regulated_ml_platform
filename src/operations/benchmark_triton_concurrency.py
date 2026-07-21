from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

_LABEL_PATTERN = re.compile(r'(\w+)="((?:\\.|[^"])*)"')
_METRIC_PATTERN = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$")


def _dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def parse_prometheus(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    snapshot: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_PATTERN.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        labels = tuple(sorted((item.group(1), item.group(2)) for item in _LABEL_PATTERN.finditer(raw_labels or "")))
        try:
            snapshot[(name, labels)] = float(raw_value)
        except ValueError:
            continue
    return snapshot


def _metric_total(
    snapshot: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    metric_name: str,
    model_name: str,
) -> float:
    total = 0.0
    for (name, label_items), value in snapshot.items():
        if name != metric_name:
            continue
        labels = dict(label_items)
        if labels.get("model") == model_name:
            total += value
    return total


def _metric_delta(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    metric_name: str,
    model_name: str,
) -> float:
    return max(_metric_total(after, metric_name, model_name) - _metric_total(before, metric_name, model_name), 0.0)


def _payload(matrix: np.ndarray) -> dict[str, Any]:
    return {
        "inputs": [
            {
                "name": "FEATURES",
                "shape": list(matrix.shape),
                "datatype": "FP32",
                "data": matrix.tolist(),
            }
        ],
        "outputs": [{"name": "SUPPORT_PROBABILITY"}],
    }


def _infer(client: httpx.Client, endpoint: str, payload: dict[str, Any], expected: np.ndarray) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        response = client.post(endpoint, json=payload)
        response.raise_for_status()
        body = response.json()
        outputs = {item["name"]: item for item in body.get("outputs", [])}
        output = outputs.get("SUPPORT_PROBABILITY")
        if output is None:
            raise ValueError(f"Triton response did not contain SUPPORT_PROBABILITY: {body}")
        observed = np.asarray(output["data"], dtype=np.float32).reshape(-1)
        if observed.shape != expected.shape:
            raise ValueError(f"Unexpected Triton output shape {observed.shape}; expected {expected.shape}")
        max_error = float(np.max(np.abs(expected.astype(np.float64) - observed.astype(np.float64))))
        return {
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "max_absolute_probability_error": max_error,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - benchmark records request failures rather than dropping the scenario
        return {
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "max_absolute_probability_error": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _burst_infer(
    client: httpx.Client,
    endpoint: str,
    barrier: threading.Barrier,
    payload: dict[str, Any],
    expected: np.ndarray,
) -> dict[str, Any]:
    barrier.wait()
    return _infer(client, endpoint, payload, expected)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile)) if values else 0.0


def benchmark_concurrency(
    triton_url: str,
    metrics_url: str,
    model_path: Path,
    sample_path: Path,
    triton_root: Path,
    output_path: Path,
    concurrency_levels: list[int],
    rounds_per_level: int = 12,
    request_batch_size: int = 1,
    tolerance: float = 5e-5,
) -> dict[str, Any]:
    if not concurrency_levels or any(level < 1 for level in concurrency_levels):
        raise ValueError("concurrency_levels must contain positive integers")
    if rounds_per_level < 1 or request_batch_size < 1:
        raise ValueError("rounds_per_level and request_batch_size must be positive")

    model = joblib.load(model_path)
    preprocessor = joblib.load(triton_root / "preprocessor.joblib")
    frame = pd.read_csv(sample_path)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if frame.empty:
        raise ValueError("sample data is empty")

    max_requests = max(concurrency_levels) * rounds_per_level
    required_rows = max_requests * request_batch_size
    row_indices = np.arange(required_rows) % len(frame)
    workload = frame.iloc[row_indices][feature_columns].reset_index(drop=True)
    transformed = _dense_float32(preprocessor.transform(workload))
    native = np.asarray(model.predict_proba(workload)[:, 1], dtype=np.float64)

    payloads: list[dict[str, Any]] = []
    expected_outputs: list[np.ndarray] = []
    for request_index in range(max_requests):
        start = request_index * request_batch_size
        end = start + request_batch_size
        payloads.append(_payload(transformed[start:end]))
        expected_outputs.append(native[start:end])

    endpoint = triton_url.rstrip("/") + "/v2/models/support_ensemble/infer"
    metrics_endpoint = metrics_url.rstrip("/")
    max_concurrency = max(concurrency_levels)
    limits = httpx.Limits(max_connections=max_concurrency * 2, max_keepalive_connections=max_concurrency * 2)
    scenarios: list[dict[str, Any]] = []

    with httpx.Client(timeout=30.0, limits=limits) as client:
        health = client.get(triton_url.rstrip("/") + "/v2/health/ready")
        health.raise_for_status()

        for warmup_index in range(min(4, len(payloads))):
            warmup = _infer(client, endpoint, payloads[warmup_index], expected_outputs[warmup_index])
            if warmup["error"] is not None:
                raise RuntimeError(f"Triton warmup failed: {warmup['error']}")

        request_offset = 0
        for concurrency in concurrency_levels:
            before_response = client.get(metrics_endpoint)
            before_response.raise_for_status()
            before = parse_prometheus(before_response.text)
            results: list[dict[str, Any]] = []
            scenario_start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                for round_index in range(rounds_per_level):
                    barrier = threading.Barrier(concurrency + 1)
                    futures = []
                    for slot in range(concurrency):
                        index = (request_offset + round_index * concurrency + slot) % len(payloads)
                        futures.append(
                            executor.submit(
                                _burst_infer,
                                client,
                                endpoint,
                                barrier,
                                payloads[index],
                                expected_outputs[index],
                            )
                        )
                    barrier.wait()
                    results.extend(future.result() for future in futures)

            elapsed_seconds = max(time.perf_counter() - scenario_start, 1e-12)
            request_offset += concurrency * rounds_per_level
            after_response = client.get(metrics_endpoint)
            after_response.raise_for_status()
            after = parse_prometheus(after_response.text)

            successful = [item for item in results if item["error"] is None]
            failures = [item for item in results if item["error"] is not None]
            latencies = [float(item["latency_ms"]) for item in successful]
            probability_errors = [float(item["max_absolute_probability_error"]) for item in successful]

            base_inferences = _metric_delta(before, after, "nv_inference_count", "support_base")
            base_exec = _metric_delta(before, after, "nv_inference_exec_count", "support_base")
            calibrator_inferences = _metric_delta(before, after, "nv_inference_count", "support_calibrator")
            calibrator_exec = _metric_delta(before, after, "nv_inference_exec_count", "support_calibrator")
            base_queue_us = _metric_delta(before, after, "nv_inference_queue_duration_us", "support_base")
            base_average_batch = base_inferences / base_exec if base_exec > 0 else 0.0
            calibrator_average_batch = calibrator_inferences / calibrator_exec if calibrator_exec > 0 else 0.0
            completed_rows = len(successful) * request_batch_size
            max_probability_error = max(probability_errors, default=0.0)
            http_error_rate = len(failures) / max(len(results), 1)

            scenarios.append(
                {
                    "concurrency": concurrency,
                    "request_batch_size": request_batch_size,
                    "rounds": rounds_per_level,
                    "requests": len(results),
                    "successful_requests": len(successful),
                    "http_failures": len(failures),
                    "http_error_rate": http_error_rate,
                    "elapsed_seconds": elapsed_seconds,
                    "requests_per_second": len(successful) / elapsed_seconds,
                    "rows_per_second": completed_rows / elapsed_seconds,
                    "p50_latency_ms": _percentile(latencies, 50),
                    "p95_latency_ms": _percentile(latencies, 95),
                    "p99_latency_ms": _percentile(latencies, 99),
                    "max_absolute_probability_error": max_probability_error,
                    "parity_status": "PASS" if max_probability_error <= tolerance and not failures else "FAIL",
                    "support_base_average_batch_size": base_average_batch,
                    "support_calibrator_average_batch_size": calibrator_average_batch,
                    "support_base_batching_gain": base_average_batch / request_batch_size if request_batch_size else 0.0,
                    "support_base_queue_us_per_inference": base_queue_us / base_inferences if base_inferences > 0 else 0.0,
                    "metric_deltas": {
                        "support_base_inference_count": base_inferences,
                        "support_base_execution_count": base_exec,
                        "support_calibrator_inference_count": calibrator_inferences,
                        "support_calibrator_execution_count": calibrator_exec,
                    },
                    "sample_errors": [item["error"] for item in failures[:5]],
                }
            )

    status = "PASS" if all(item["parity_status"] == "PASS" for item in scenarios) else "FAIL"
    report = {
        "status": status,
        "benchmark_type": "real_triton_concurrent_http",
        "runtime_evidence": "real_triton_server",
        "model": "support_ensemble",
        "request_batch_size": request_batch_size,
        "concurrency_levels": concurrency_levels,
        "rounds_per_level": rounds_per_level,
        "tolerance": tolerance,
        "scenarios": scenarios,
        "boundary": (
            "This is a short hosted-runner concurrency experiment. It proves the exercised server behaviour and metric deltas, "
            "not production capacity under a representative traffic distribution."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-url", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8002/metrics")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--sample", default="data/processed/features.csv")
    parser.add_argument("--triton-root", default="models/triton")
    parser.add_argument("--output", default="reports/triton_concurrency_benchmark.json")
    parser.add_argument("--concurrency-levels", default="1,4,8,16,32")
    parser.add_argument("--rounds-per-level", type=int, default=12)
    parser.add_argument("--request-batch-size", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    concurrency_levels = [int(value.strip()) for value in args.concurrency_levels.split(",") if value.strip()]
    benchmark_concurrency(
        args.triton_url,
        args.metrics_url,
        Path(args.model),
        Path(args.sample),
        Path(args.triton_root),
        Path(args.output),
        concurrency_levels,
        args.rounds_per_level,
        args.request_batch_size,
        args.tolerance,
    )


if __name__ == "__main__":
    main()
