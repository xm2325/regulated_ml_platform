#!/usr/bin/env python3
"""Run same-input CPU and Triton HTTP qualification for the synthetic MLP.

CPU mode requires PyTorch and creates immutable NumPy fixtures. GPU and parity
modes require the Triton SDK's ``tritonclient`` package. The formal Slurm path
runs CPU and GPU for at least 300 seconds and keeps this module FP32-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from accelerator_workload import MAX_BATCH_SIZE, build_model, make_input, percentile, workload_contract

SCHEMA_VERSION = "regulated-ml-platform.roihu-http-qualification/v1"


def _bounded_int(raw: str, *, minimum: int, maximum: int, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise argparse.ArgumentTypeError(f"{label} must be between {minimum} and {maximum}")
    return value


def _batch_size(raw: str) -> int:
    return _bounded_int(raw, minimum=1, maximum=MAX_BATCH_SIZE, label="batch size")


def _cpu_threads(raw: str) -> int:
    return _bounded_int(raw, minimum=1, maximum=72, label="CPU threads")


def _concurrency(raw: str) -> int:
    return _bounded_int(raw, minimum=1, maximum=32, label="concurrency")


def _minimum_samples(raw: str) -> int:
    return _bounded_int(raw, minimum=1, maximum=10_000_000, label="minimum samples")


def _parity_rows(raw: str) -> int:
    return _bounded_int(raw, minimum=1, maximum=100_000, label="parity rows")


def _duration(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("duration must be a number") from exc
    if not 0.1 <= value <= 3600:
        raise argparse.ArgumentTypeError("duration must be between 0.1 and 3600 seconds")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu", "parity"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=_duration, default=300.0)
    parser.add_argument("--minimum-samples", type=_minimum_samples, default=1000)
    parser.add_argument("--batch-size", type=_batch_size, default=64)
    parser.add_argument("--parity-rows", type=_parity_rows, default=1024)
    parser.add_argument("--cpu-threads", type=_cpu_threads, default=72)
    parser.add_argument("--url", default="127.0.0.1:8000")
    parser.add_argument("--model-name", default="accelerator_qualification")
    parser.add_argument("--concurrency", type=_concurrency, default=4)
    parser.add_argument("--warmup", type=lambda raw: _bounded_int(raw, minimum=1, maximum=100, label="warmup"), default=5)
    parser.add_argument("--maximum-http-error-rate", type=float, default=0.001)
    parser.add_argument("--maximum-absolute-probability-error", type=float, default=5e-5)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-short-development-run",
        action="store_true",
        help="allow less than 300 seconds/1000 samples, but mark output DEVELOPMENT_ONLY",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise RuntimeError("no successful timed samples were observed")
    return {
        "p50": percentile(latencies_ms, 0.50),
        "p95": percentile(latencies_ms, 0.95),
        "p99": percentile(latencies_ms, 0.99),
        "minimum": min(latencies_ms),
        "maximum": max(latencies_ms),
    }


def _qualification_status(args: argparse.Namespace, duration: float, sample_count: int, extra_checks: bool = True) -> str:
    formal = duration >= 300.0 and sample_count >= 1000 and extra_checks
    if formal:
        return "PASS"
    if args.allow_short_development_run:
        return "DEVELOPMENT_ONLY"
    return "FAIL"


def _run_cpu(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    args.fixture_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.manual_seed(1729)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    model = build_model(torch)
    benchmark_input = make_input(torch, args.batch_size)
    with torch.inference_mode():
        for _ in range(args.warmup):
            torch.sigmoid(model(benchmark_input))

    latencies_ms: list[float] = []
    started = time.perf_counter()
    while time.perf_counter() - started < args.duration_seconds or len(latencies_ms) < args.minimum_samples:
        request_started = time.perf_counter_ns()
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(benchmark_input))
        probabilities.numpy()
        latencies_ms.append((time.perf_counter_ns() - request_started) / 1_000_000.0)
    duration = time.perf_counter() - started

    parity_batches = []
    remaining = args.parity_rows
    with torch.inference_mode():
        while remaining:
            rows = min(remaining, MAX_BATCH_SIZE)
            parity_batches.append(make_input(torch, rows))
            remaining -= rows
        parity_inputs = torch.cat(parity_batches, dim=0)
        cpu_probabilities = torch.sigmoid(model(parity_inputs))

    benchmark_input_path = args.fixture_dir / "benchmark_input.npy"
    parity_input_path = args.fixture_dir / "parity_inputs.npy"
    cpu_probability_path = args.fixture_dir / "cpu_parity_probabilities.npy"
    np.save(benchmark_input_path, benchmark_input.numpy().astype(np.float32, copy=False), allow_pickle=False)
    np.save(parity_input_path, parity_inputs.numpy().astype(np.float32, copy=False), allow_pickle=False)
    np.save(cpu_probability_path, cpu_probabilities.numpy().astype(np.float32, copy=False), allow_pickle=False)
    for path in (benchmark_input_path, parity_input_path, cpu_probability_path):
        os.chmod(path, 0o600)

    workload = workload_contract(model)
    fixture_manifest = {
        "schema_version": SCHEMA_VERSION,
        "workload": workload,
        "precision": "fp32",
        "decision_threshold": args.decision_threshold,
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in (benchmark_input_path, parity_input_path, cpu_probability_path)
        },
    }
    _atomic_json(args.fixture_dir / "fixture_manifest.json", fixture_manifest)
    status = _qualification_status(args, duration, len(latencies_ms))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": "cpu_fp32",
        "workload": workload,
        "duration_seconds": duration,
        "sample_count": len(latencies_ms),
        "request_batch_size": args.batch_size,
        "row_count": len(latencies_ms) * args.batch_size,
        "throughput_requests_per_second": len(latencies_ms) / duration,
        "throughput_rows_per_second": len(latencies_ms) * args.batch_size / duration,
        "latency_ms": _latency_summary(latencies_ms),
        "timing_scope": "model forward plus sigmoid; model and fixed input resident on NVIDIA Grace CPU memory",
        "cpu_threads": args.cpu_threads,
        "warmup_iterations": args.warmup,
        "fixture_manifest_sha256": _sha256(args.fixture_dir / "fixture_manifest.json"),
        "claim_boundary": {"synthetic_accelerator_qualification_only": True, "production_capacity_claim_allowed": False},
    }


def _triton_client(url: str) -> Any:
    import tritonclient.http as httpclient

    try:
        return httpclient.InferenceServerClient(url=url, verbose=False, concurrency=1, connection_timeout=60.0, network_timeout=60.0)
    except TypeError:
        return httpclient.InferenceServerClient(url=url, verbose=False, concurrency=1)


def _infer_setup(data: Any) -> tuple[Any, Any]:
    import tritonclient.http as httpclient

    infer_input = httpclient.InferInput("INPUT__0", list(data.shape), "FP32")
    infer_input.set_data_from_numpy(data, binary_data=True)
    requested_output = httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True)
    return infer_input, requested_output


def _sigmoid_numpy(values: Any) -> Any:
    import numpy as np

    values = values.astype(np.float32, copy=False)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _run_gpu(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    benchmark_input = np.load(args.fixture_dir / "benchmark_input.npy", allow_pickle=False)
    fixture_manifest = json.loads((args.fixture_dir / "fixture_manifest.json").read_text(encoding="utf-8"))
    if benchmark_input.dtype != np.float32 or benchmark_input.shape != (args.batch_size, 1024):
        raise ValueError("benchmark fixture does not match the fixed FP32 input contract")

    clients: list[tuple[Any, Any, Any]] = []
    for _ in range(args.concurrency):
        client = _triton_client(args.url)
        infer_input, requested_output = _infer_setup(benchmark_input)
        for _ in range(args.warmup):
            result = client.infer(args.model_name, [infer_input], outputs=[requested_output])
            _sigmoid_numpy(result.as_numpy("OUTPUT__0"))
        clients.append((client, infer_input, requested_output))

    start_event = threading.Event()
    deadline_holder: dict[str, float] = {}

    def worker(resources: tuple[Any, Any, Any]) -> tuple[list[float], int, list[str]]:
        client, infer_input, requested_output = resources
        latencies: list[float] = []
        errors = 0
        error_types: list[str] = []
        start_event.wait()
        deadline = deadline_holder["deadline"]
        while time.perf_counter() < deadline:
            request_started = time.perf_counter_ns()
            try:
                result = client.infer(args.model_name, [infer_input], outputs=[requested_output])
                _sigmoid_numpy(result.as_numpy("OUTPUT__0"))
                latencies.append((time.perf_counter_ns() - request_started) / 1_000_000.0)
            except Exception as exc:  # noqa: BLE001 - HTTP errors are governed evidence
                errors += 1
                if len(error_types) < 10:
                    error_types.append(type(exc).__name__)
        return latencies, errors, error_types

    started = time.perf_counter()
    deadline_holder["deadline"] = started + args.duration_seconds
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, resources) for resources in clients]
        start_event.set()
        observations = [future.result() for future in futures]
    duration = time.perf_counter() - started
    latencies_ms = [latency for observed, _, _ in observations for latency in observed]
    http_errors = sum(errors for _, errors, _ in observations)
    error_types = [name for _, _, names in observations for name in names][:10]
    http_requests = len(latencies_ms) + http_errors
    error_rate = http_errors / http_requests if http_requests else 1.0
    status = _qualification_status(
        args,
        duration,
        len(latencies_ms),
        extra_checks=http_requests > 0 and error_rate <= args.maximum_http_error_rate,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": "triton_http_fp32",
        "workload": fixture_manifest["workload"],
        "duration_seconds": duration,
        "sample_count": len(latencies_ms),
        "request_batch_size": args.batch_size,
        "row_count": len(latencies_ms) * args.batch_size,
        "throughput_requests_per_second": len(latencies_ms) / duration,
        "throughput_rows_per_second": len(latencies_ms) * args.batch_size / duration,
        "latency_ms": _latency_summary(latencies_ms),
        "http_requests": http_requests,
        "http_errors": http_errors,
        "http_error_rate": error_rate,
        "observed_error_types": error_types,
        "concurrency": args.concurrency,
        "warmup_iterations_per_worker": args.warmup,
        "timing_scope": "binary Triton HTTP request/response plus client sigmoid; fixed input resident in client memory",
        "fixture_manifest_sha256": _sha256(args.fixture_dir / "fixture_manifest.json"),
        "claim_boundary": {"synthetic_accelerator_qualification_only": True, "production_capacity_claim_allowed": False},
    }


def _run_parity(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    parity_inputs = np.load(args.fixture_dir / "parity_inputs.npy", allow_pickle=False)
    cpu_probabilities = np.load(args.fixture_dir / "cpu_parity_probabilities.npy", allow_pickle=False)
    fixture_manifest = json.loads((args.fixture_dir / "fixture_manifest.json").read_text(encoding="utf-8"))
    if parity_inputs.dtype != np.float32 or cpu_probabilities.dtype != np.float32:
        raise ValueError("parity fixtures must be FP32")
    if len(parity_inputs) != len(cpu_probabilities) or len(parity_inputs) < args.parity_rows:
        raise ValueError("parity fixtures do not contain the required rows")

    client = _triton_client(args.url)
    gpu_outputs = []
    http_errors = 0
    for offset in range(0, args.parity_rows, MAX_BATCH_SIZE):
        batch = np.ascontiguousarray(parity_inputs[offset : offset + MAX_BATCH_SIZE])
        infer_input, requested_output = _infer_setup(batch)
        try:
            result = client.infer(args.model_name, [infer_input], outputs=[requested_output])
            gpu_outputs.append(_sigmoid_numpy(result.as_numpy("OUTPUT__0")))
        except Exception:  # noqa: BLE001 - aggregate without leaking endpoint details
            http_errors += 1
    if http_errors or not gpu_outputs:
        raise RuntimeError("one or more Triton parity requests failed")
    gpu_probabilities = np.concatenate(gpu_outputs, axis=0)
    cpu_probabilities = cpu_probabilities[: args.parity_rows]
    absolute_error = np.abs(gpu_probabilities - cpu_probabilities)
    cpu_decisions = cpu_probabilities >= args.decision_threshold
    gpu_decisions = gpu_probabilities >= args.decision_threshold
    decision_mismatches = int(np.count_nonzero(cpu_decisions != gpu_decisions))
    max_error = float(absolute_error.max())
    gpu_probability_path = args.fixture_dir / "gpu_parity_probabilities.npy"
    np.save(gpu_probability_path, gpu_probabilities.astype(np.float32, copy=False), allow_pickle=False)
    os.chmod(gpu_probability_path, 0o600)
    formal = args.parity_rows >= 1000 and max_error <= args.maximum_absolute_probability_error and decision_mismatches == 0
    status = "PASS" if formal else ("DEVELOPMENT_ONLY" if args.allow_short_development_run else "FAIL")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": "cpu_fp32_vs_triton_http_fp32",
        "workload": fixture_manifest["workload"],
        "sample_count": args.parity_rows,
        "output_value_count": int(gpu_probabilities.size),
        "max_absolute_probability_error": max_error,
        "mean_absolute_probability_error": float(absolute_error.mean()),
        "decision_threshold": args.decision_threshold,
        "policy_decision_mismatches": decision_mismatches,
        "http_errors": http_errors,
        "fixtures": {
            "parity_inputs_sha256": _sha256(args.fixture_dir / "parity_inputs.npy"),
            "cpu_probabilities_sha256": _sha256(args.fixture_dir / "cpu_parity_probabilities.npy"),
            "gpu_probabilities_sha256": _sha256(gpu_probability_path),
        },
        "claim_boundary": {"synthetic_accelerator_qualification_only": True, "production_capacity_claim_allowed": False},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.maximum_http_error_rate <= 1.0:
        raise ValueError("maximum HTTP error rate must be between zero and one")
    if not 0.0 <= args.maximum_absolute_probability_error < float("inf"):
        raise ValueError("maximum probability error must be finite and non-negative")
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError("decision threshold must be strictly between zero and one")
    if not args.allow_short_development_run and (args.duration_seconds < 300.0 or args.minimum_samples < 1000 or args.parity_rows < 1000):
        raise ValueError("formal qualification requires >=300 seconds, >=1000 timed samples, and >=1000 parity rows")
    if args.mode == "cpu":
        return _run_cpu(args)
    if args.mode == "gpu":
        return _run_gpu(args)
    return _run_parity(args)


def main() -> int:
    os.umask(0o077)
    args = build_parser().parse_args()
    try:
        payload = run(args)
        _atomic_json(args.output, payload)
        print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
        return 0 if payload["status"] in {"PASS", "DEVELOPMENT_ONLY"} else 1
    except Exception as exc:  # noqa: BLE001 - emit fail-closed evidence
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ERROR",
            "mode": args.mode,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "claim_boundary": {"production_capacity_claim_allowed": False},
        }
        try:
            _atomic_json(args.output, payload)
        except Exception:  # noqa: BLE001
            pass
        print(f"qualification client failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
