from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile)) if values else 0.0


def _infer(client: httpx.Client, endpoint: str, matrix: np.ndarray) -> np.ndarray:
    payload = {
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
    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    body = response.json()
    outputs = {item["name"]: item for item in body.get("outputs", [])}
    output = outputs.get("SUPPORT_PROBABILITY")
    if output is None:
        raise ValueError(f"Triton response did not contain SUPPORT_PROBABILITY: {body}")
    values = np.asarray(output["data"], dtype=np.float32)
    return values.reshape(matrix.shape[0], -1)[:, 0]


def benchmark(
    triton_url: str,
    model_path: Path,
    sample_path: Path,
    triton_root: Path,
    output_path: Path,
    batch_sizes: list[int],
    repeats: int = 25,
    warmup: int = 5,
    tolerance: float = 5e-5,
) -> dict[str, Any]:
    model = joblib.load(model_path)
    preprocessor = joblib.load(triton_root / "preprocessor.joblib")
    frame = pd.read_csv(sample_path)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    endpoint = triton_url.rstrip("/") + "/v2/models/support_ensemble/infer"
    results: list[dict[str, Any]] = []

    with httpx.Client(timeout=30.0) as client:
        health = client.get(triton_url.rstrip("/") + "/v2/health/ready")
        health.raise_for_status()
        for batch_size in batch_sizes:
            batch = frame.iloc[np.arange(batch_size) % len(frame)][feature_columns].reset_index(drop=True)
            transformed = _dense_float32(preprocessor.transform(batch))
            native = np.asarray(model.predict_proba(batch)[:, 1], dtype=np.float64)
            for _ in range(warmup):
                _infer(client, endpoint, transformed)
            timings: list[float] = []
            latest = np.empty(batch_size, dtype=np.float32)
            for _ in range(repeats):
                start = time.perf_counter()
                latest = _infer(client, endpoint, transformed)
                timings.append((time.perf_counter() - start) * 1000.0)
            max_error = float(np.max(np.abs(native - latest.astype(np.float64))))
            p50_ms = median(timings)
            results.append(
                {
                    "batch_size": batch_size,
                    "p50_ms": float(p50_ms),
                    "p95_ms": _percentile(timings, 95),
                    "rows_per_second_at_p50": float(batch_size / max(p50_ms / 1000.0, 1e-12)),
                    "max_absolute_probability_error": max_error,
                    "parity_status": "PASS" if max_error <= tolerance else "FAIL",
                }
            )

    status = "PASS" if all(item["parity_status"] == "PASS" for item in results) else "FAIL"
    report = {
        "status": status,
        "benchmark_type": "real_triton_http",
        "runtime_evidence": "real_triton_server",
        "triton_url": triton_url,
        "model": "support_ensemble",
        "tolerance": tolerance,
        "repeats": repeats,
        "warmup": warmup,
        "results": results,
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
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--sample", default="data/processed/features.csv")
    parser.add_argument("--triton-root", default="models/triton")
    parser.add_argument("--output", default="reports/triton_http_benchmark.json")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()]
    benchmark(
        args.triton_url,
        Path(args.model),
        Path(args.sample),
        Path(args.triton_root),
        Path(args.output),
        batch_sizes,
        args.repeats,
        args.warmup,
        args.tolerance,
    )


if __name__ == "__main__":
    main()
