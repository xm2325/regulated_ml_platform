#!/usr/bin/env python3
"""Benchmark the deterministic accelerator-qualification MLP on Roihu GH200.

The result is synthetic runtime evidence.  It is not production capacity,
customer-data validation, or evidence about the platform's champion model.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from accelerator_workload import (
    DEFAULT_BATCH_SIZES,
    MAX_BATCH_SIZE,
    build_model,
    make_input,
    parse_batch_sizes,
    percentile,
    sha256_file,
    validate_source_commit,
    verify_source_archive,
    workload_contract,
)

SCHEMA_VERSION = "regulated-ml-platform.roihu-pytorch-benchmark/v1"


def _bounded_int(raw: str, *, minimum: int, maximum: int, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise argparse.ArgumentTypeError(f"{label} must be between {minimum} and {maximum}")
    return value


def _positive_repetitions(raw: str) -> int:
    return _bounded_int(raw, minimum=5, maximum=1000, label="repetitions")


def _bounded_warmup(raw: str) -> int:
    return _bounded_int(raw, minimum=1, maximum=100, label="warmup")


def _positive_threads(raw: str) -> int:
    return _bounded_int(raw, minimum=1, maximum=72, label="CPU threads")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence path")
    parser.add_argument("--source-root", type=Path, required=True, help="extracted, digest-verified source root")
    parser.add_argument("--source-archive", type=Path, required=True, help="immutable source archive")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--batch-sizes", default=",".join(str(value) for value in DEFAULT_BATCH_SIZES))
    parser.add_argument("--warmup", type=_bounded_warmup, default=10)
    parser.add_argument("--repetitions", type=_positive_repetitions, default=50)
    parser.add_argument(
        "--cpu-threads",
        type=_positive_threads,
        default=min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))),
    )
    parser.add_argument("--fp32-atol", type=float, default=1e-4)
    parser.add_argument("--fp32-rtol", type=float, default=1e-4)
    parser.add_argument("--bf16-atol", type=float, default=2e-2)
    parser.add_argument("--bf16-rtol", type=float, default=2e-2)
    parser.add_argument("--expected-gpu-name", default="GH200")
    parser.add_argument("--expected-compute-capability", default="9.0")
    parser.add_argument(
        "--allow-non-slurm",
        action="store_true",
        help="local development only; evidence cannot pass the Roihu qualification gate",
    )
    parser.add_argument(
        "--allow-cpu-only",
        action="store_true",
        help="local development only; evidence cannot pass the GPU qualification gate",
    )
    return parser


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


def _source_provenance(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    script_path = Path(__file__).resolve(strict=True)
    try:
        script_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError("executed benchmark is not inside the verified source root") from exc
    archive = verify_source_archive(args.source_archive, args.expected_source_sha256)
    commit = validate_source_commit(args.source_git_commit)
    return {
        "archive": archive,
        "git_commit": commit,
        "git_commit_attestation": "operator_supplied_and_archive_root_validated_by_sbatch_wrapper",
        "source_root_name": source_root.name,
        "executed_files": {
            "benchmark_pytorch_inference.py": sha256_file(script_path),
            "accelerator_workload.py": sha256_file(source_root / "hpc/roihu/accelerator_workload.py"),
        },
    }


def _slurm_provenance() -> dict[str, Any]:
    names = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "account": "SLURM_JOB_ACCOUNT",
        "partition": "SLURM_JOB_PARTITION",
        "cluster_name": "SLURM_CLUSTER_NAME",
        "node_list": "SLURM_JOB_NODELIST",
        "nodes": "SLURM_JOB_NUM_NODES",
        "tasks": "SLURM_NTASKS",
        "cpus_per_task": "SLURM_CPUS_PER_TASK",
        "gres": "SLURM_JOB_GPUS",
    }
    return {key: os.environ.get(environment_name) for key, environment_name in names.items()}


def _latency_summary(latencies_ms: list[float], batch_size: int) -> dict[str, Any]:
    elapsed_seconds = sum(latencies_ms) / 1000.0
    return {
        "observations": len(latencies_ms),
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "minimum": min(latencies_ms),
            "maximum": max(latencies_ms),
        },
        "steady_state_throughput_items_per_second": (batch_size * len(latencies_ms)) / elapsed_seconds,
    }


def _measure_cpu(torch: Any, model: Any, inputs: Any, *, warmup: int, repetitions: int) -> tuple[dict[str, Any], Any]:
    latencies: list[float] = []
    output = None
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(inputs)
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            output = model(inputs)
            elapsed = time.perf_counter_ns() - started
            latencies.append(elapsed / 1_000_000.0)
    if output is None:
        raise RuntimeError("CPU benchmark produced no output")
    return _latency_summary(latencies, int(inputs.shape[0])), output.detach().cpu().float()


def _measure_cuda(
    torch: Any,
    model: Any,
    inputs: Any,
    *,
    warmup: int,
    repetitions: int,
) -> tuple[dict[str, Any], Any]:
    latencies: list[float] = []
    output = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(inputs)
        torch.cuda.synchronize()
        for _ in range(repetitions):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            output = model(inputs)
            finished.record()
            finished.synchronize()
            latencies.append(float(started.elapsed_time(finished)))
    if output is None:
        raise RuntimeError("CUDA benchmark produced no output")
    free_after, _ = torch.cuda.mem_get_info()
    result = _latency_summary(latencies, int(inputs.shape[0]))
    result["gpu_memory_bytes"] = {
        "device_total": int(total_memory),
        "free_before": int(free_before),
        "free_after": int(free_after),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }
    return result, output.detach().cpu().float()


def _parity(torch: Any, reference: Any, candidate: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    absolute = (candidate - reference).abs()
    denominator = reference.abs().clamp_min(1e-8)
    relative = absolute / denominator
    passed = bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol))
    return {
        "status": "PASS" if passed else "FAIL",
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "maximum_absolute_error": float(absolute.max().item()),
        "maximum_relative_error": float(relative.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
    }


def _runtime_provenance(torch: Any, cpu_threads: int) -> dict[str, Any]:
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "python": platform.python_version(),
        "platform_machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_runtime": cuda_version,
        "cudnn": cudnn_version,
        "cpu_threads": cpu_threads,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if any(
        tolerance < 0 or not float(tolerance) < float("inf")
        for tolerance in (args.fp32_atol, args.fp32_rtol, args.bf16_atol, args.bf16_rtol)
    ):
        raise ValueError("parity tolerances must be finite and non-negative")
    batch_sizes = parse_batch_sizes(args.batch_sizes, maximum=MAX_BATCH_SIZE)
    source = _source_provenance(args)
    slurm = _slurm_provenance()
    is_slurm = bool(slurm["job_id"])
    if not is_slurm and not args.allow_non_slurm:
        raise RuntimeError("SLURM_JOB_ID is required; use --allow-non-slurm only for local development")

    torch.manual_seed(1729)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    model_cpu = build_model(torch)
    workload = workload_contract(model_cpu)

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available and not args.allow_cpu_only:
        raise RuntimeError("CUDA is required; use --allow-cpu-only only for local development")

    device: Any = None
    model_cuda_fp32: Any = None
    model_cuda_bf16: Any = None
    gpu: dict[str, Any] | None = None
    if cuda_available:
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(device)
        if args.expected_gpu_name.lower() not in gpu_name.lower():
            raise RuntimeError(f"allocated GPU does not match expected accelerator family {args.expected_gpu_name!r}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("allocated CUDA device does not report BF16 support")
        properties = torch.cuda.get_device_properties(device)
        compute_capability = f"{properties.major}.{properties.minor}"
        if compute_capability != args.expected_compute_capability:
            raise RuntimeError(
                f"allocated GPU compute capability {compute_capability} does not match expected {args.expected_compute_capability}"
            )
        gpu = {
            "name": gpu_name,
            "device_index": 0,
            "compute_capability": compute_capability,
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
            "bf16_supported": True,
        }
        model_cuda_fp32 = copy.deepcopy(model_cpu).to(device=device, dtype=torch.float32).eval()
        model_cuda_bf16 = copy.deepcopy(model_cpu).to(device=device, dtype=torch.bfloat16).eval()

    results: list[dict[str, Any]] = []
    all_parity_passed = True
    for batch_size in batch_sizes:
        inputs_cpu = make_input(torch, batch_size)
        cpu_metrics, cpu_output = _measure_cpu(
            torch,
            model_cpu,
            inputs_cpu,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        row: dict[str, Any] = {"batch_size": batch_size, "cpu_fp32": cpu_metrics}
        if cuda_available:
            inputs_cuda_fp32 = inputs_cpu.to(device=device, dtype=torch.float32)
            cuda_fp32, output_fp32 = _measure_cuda(
                torch,
                model_cuda_fp32,
                inputs_cuda_fp32,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            fp32_parity = _parity(torch, cpu_output, output_fp32, atol=args.fp32_atol, rtol=args.fp32_rtol)

            inputs_cuda_bf16 = inputs_cpu.to(device=device, dtype=torch.bfloat16)
            cuda_bf16, output_bf16 = _measure_cuda(
                torch,
                model_cuda_bf16,
                inputs_cuda_bf16,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            bf16_parity = _parity(torch, cpu_output, output_bf16, atol=args.bf16_atol, rtol=args.bf16_rtol)
            all_parity_passed = all_parity_passed and fp32_parity["status"] == "PASS" and bf16_parity["status"] == "PASS"
            row.update(
                {
                    "cuda_fp32": cuda_fp32,
                    "cuda_bf16": cuda_bf16,
                    "parity_vs_cpu_fp32": {"cuda_fp32": fp32_parity, "cuda_bf16": bf16_parity},
                    "throughput_speedup_vs_cpu_fp32": {
                        "cuda_fp32": cuda_fp32["steady_state_throughput_items_per_second"]
                        / cpu_metrics["steady_state_throughput_items_per_second"],
                        "cuda_bf16": cuda_bf16["steady_state_throughput_items_per_second"]
                        / cpu_metrics["steady_state_throughput_items_per_second"],
                    },
                }
            )
        results.append(row)

    qualification_passed = is_slurm and cuda_available and all_parity_passed
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if qualification_passed else "DEVELOPMENT_ONLY",
        "evidence_tier": "SMOKE_ONLY",
        "workload": workload,
        "configuration": {
            "batch_sizes": batch_sizes,
            "warmup_iterations": args.warmup,
            "measured_repetitions": args.repetitions,
            "cpu_threads": args.cpu_threads,
            "precision_paths": ["cpu_fp32"] + (["cuda_fp32", "cuda_bf16"] if cuda_available else []),
            "timing_scope": "steady-state forward pass with model and input resident on the measured device",
        },
        "provenance": {
            "source": source,
            "slurm": slurm,
            "runtime": _runtime_provenance(torch, args.cpu_threads),
        },
        "accelerator": gpu,
        "results": results,
        "checks": {
            "source_archive_digest_verified": source["archive"]["digest_status"] == "PASS",
            "slurm_job_observed": is_slurm,
            "cuda_observed": cuda_available,
            "expected_gpu_family_observed": cuda_available,
            "expected_compute_capability_observed": bool(gpu and gpu["compute_capability"] == args.expected_compute_capability),
            "bf16_supported": bool(gpu and gpu["bf16_supported"]),
            "all_precision_parity_checks_passed": all_parity_passed if cuda_available else False,
        },
        "claim_boundary": {
            "synthetic_accelerator_qualification_only": True,
            "production_model_claim_allowed": False,
            "champion_model_claim_allowed": False,
            "production_capacity_claim_allowed": False,
            "customer_data_used": False,
            "promotion_decision_allowed": False,
            "gpu_eligibility_decision_allowed": False,
            "note": "Repeat under a reviewed production workload and policy before any deployment or capacity decision.",
        },
    }


def main() -> int:
    os.umask(0o077)
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = run(args)
        _atomic_json(args.output, payload)
        print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
        return 0 if payload["status"] in {"PASS", "DEVELOPMENT_ONLY"} else 1
    except Exception as exc:  # noqa: BLE001 - evidence must survive unexpected runtime failures
        failure = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ERROR",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "claim_boundary": {
                "synthetic_accelerator_qualification_only": True,
                "production_model_claim_allowed": False,
                "champion_model_claim_allowed": False,
                "production_capacity_claim_allowed": False,
                "promotion_decision_allowed": False,
            },
        }
        try:
            _atomic_json(args.output, failure)
        except Exception as write_exc:  # noqa: BLE001
            print(f"unable to write failure evidence: {type(write_exc).__name__}", file=sys.stderr)
        print(f"benchmark failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
