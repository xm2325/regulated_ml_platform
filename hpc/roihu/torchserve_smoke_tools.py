from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "regulated-ml-platform.roihu-torchserve-smoke/v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(output: Path) -> None:
    import torch

    torch.manual_seed(20260723)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 16),
    ).eval()
    traced = torch.jit.trace(model, torch.ones((1, 256), dtype=torch.float32))
    torch.jit.save(traced, str(output))


def _telemetry(path: Path) -> dict[str, Any]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key.strip(): value.strip() for key, value in row.items() if key})
    if len(rows) < 10:
        raise ValueError("at least ten GPU telemetry samples are required")
    return {
        "samples": len(rows),
        "uuid": rows[0]["uuid"],
        "name": rows[0]["name"],
        "peak_utilization_percent": max(float(row["utilization.gpu"]) for row in rows),
        "peak_memory_used_mib": max(float(row["memory.used"]) for row in rows),
        "mean_power_draw_watts": sum(float(row["power.draw"]) for row in rows) / len(rows),
    }


def summarize(args: argparse.Namespace) -> int:
    response = json.loads(args.response.read_text(encoding="utf-8"))
    if isinstance(response, list) and len(response) == 1:
        response = response[0]
    telemetry = _telemetry(args.telemetry)
    checks = {
        "torchserve_response_received": isinstance(response, dict),
        "cuda_handler_confirmed": isinstance(response, dict) and response.get("device") == "cuda",
        "declared_batch_observed": isinstance(response, dict) and response.get("batch") == args.batch,
        "expected_output_shape": (isinstance(response, dict) and response.get("output_shape") == [args.batch, 16]),
        "gh200_confirmed": "GH200" in telemetry["name"],
        "gpu_memory_allocated": telemetry["peak_memory_used_mib"] > 0,
    }
    payload = {
        "schema_version": SCHEMA,
        "status": "SMOKE_PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source": {
            "git_commit": args.source_commit,
            "source_archive_sha256": args.source_archive_sha256,
            "torchserve_wheel_sha256": args.torchserve_wheel_sha256,
            "model_archiver_wheel_sha256": args.model_archiver_wheel_sha256,
        },
        "slurm": {
            "job_id": args.slurm_job_id,
            "partition": args.partition,
            "completed_state_attested_by_job": False,
        },
        "runtime": {
            "torchserve_version": args.torchserve_version,
            "pytorch_version": args.pytorch_version,
            "cuda_version": args.cuda_version,
            "java_version": args.java_version,
        },
        "request": response,
        "gpu": telemetry,
        "artifacts": {
            "model_sha256": _sha256(args.model),
            "response_sha256": _sha256(args.response),
            "telemetry_sha256": _sha256(args.telemetry),
        },
        "claim_boundary": {
            "classification": "SMOKE_ONLY",
            "performance_claim_allowed": False,
            "production_readiness_claim_allowed": False,
            "recommended_runtime_claim_allowed": False,
            "note": (
                "This confirms one synthetic loopback request path through an archived "
                "TorchServe release on one Roihu GH200. It is not a benchmark, security "
                "approval, production recommendation, or comparison with Triton."
            ),
        },
    }
    _atomic_json(args.output, payload)
    return 0 if payload["status"] == "SMOKE_PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and summarize a bounded TorchServe smoke")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-model")
    build.add_argument("--output", type=Path, required=True)

    report = subparsers.add_parser("summarize")
    report.add_argument("--response", type=Path, required=True)
    report.add_argument("--telemetry", type=Path, required=True)
    report.add_argument("--model", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--batch", type=int, default=256)
    report.add_argument("--source-commit", required=True)
    report.add_argument("--source-archive-sha256", required=True)
    report.add_argument("--torchserve-wheel-sha256", required=True)
    report.add_argument("--model-archiver-wheel-sha256", required=True)
    report.add_argument("--slurm-job-id", required=True)
    report.add_argument("--partition", required=True)
    report.add_argument("--torchserve-version", required=True)
    report.add_argument("--pytorch-version", required=True)
    report.add_argument("--cuda-version", required=True)
    report.add_argument("--java-version", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "build-model":
        build_model(args.output)
        return 0
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
