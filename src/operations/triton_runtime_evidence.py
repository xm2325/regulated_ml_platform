from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_BATCH_SIZES = {8, 32, 64}


def _as_int(value: Any) -> int:
    return int(value)


def _preferred_batches(config: dict[str, Any]) -> set[int]:
    dynamic = config.get("dynamic_batching") or {}
    values = dynamic.get("preferred_batch_size") or []
    return {_as_int(value) for value in values}


def validate_runtime_evidence(
    base_config: dict[str, Any],
    calibrator_config: dict[str, Any],
    benchmark: dict[str, Any],
    metrics_text: str,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "triton_http_benchmark_pass": benchmark.get("status") == "PASS",
        "real_triton_runtime": benchmark.get("runtime_evidence") == "real_triton_server",
        "base_batch_capacity": _as_int(base_config.get("max_batch_size", 0)) >= 128,
        "calibrator_batch_capacity": _as_int(calibrator_config.get("max_batch_size", 0)) >= 128,
        "base_dynamic_batching_loaded": EXPECTED_BATCH_SIZES.issubset(_preferred_batches(base_config)),
        "calibrator_dynamic_batching_loaded": EXPECTED_BATCH_SIZES.issubset(_preferred_batches(calibrator_config)),
        "triton_inference_metrics_present": "nv_inference_" in metrics_text,
    }
    results = benchmark.get("results") or []
    required_runtime_batches = {1, 8, 32, 64, 128}
    observed_batches = {int(item.get("batch_size", 0)) for item in results}
    checks["required_batch_sizes_executed"] = required_runtime_batches.issubset(observed_batches)
    checks["all_runtime_probability_parity_pass"] = bool(results) and all(
        item.get("parity_status") == "PASS" for item in results
    )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failures else "FAIL",
        "runtime_evidence": "real_triton_cpu_hosted_ci",
        "accelerator": "cpu",
        "gpu_runtime_claim": False,
        "checks": checks,
        "failures": failures,
        "benchmark_results": results,
        "boundary": (
            "This evidence proves the validated tree-ensemble serving path on a real Triton server using CPU. "
            "It does not prove CUDA, TensorRT, A100 throughput, GPU memory behaviour, or GPU autoscaling."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--calibrator-config", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", default="reports/triton_cpu_runtime/runtime_evidence.json")
    args = parser.parse_args()

    report = validate_runtime_evidence(
        json.loads(Path(args.base_config).read_text(encoding="utf-8")),
        json.loads(Path(args.calibrator_config).read_text(encoding="utf-8")),
        json.loads(Path(args.benchmark).read_text(encoding="utf-8")),
        Path(args.metrics).read_text(encoding="utf-8"),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
