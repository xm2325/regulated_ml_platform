from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "regulated-ml-platform.roihu-triton-rightsizing/v1"
AGGREGATE_SCHEMA = "regulated-ml-platform.roihu-triton-rightsizing-aggregate/v1"
HEX_40 = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_perf(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Perf Analyzer emitted no data rows: {path}")
    row = max(rows, key=lambda item: float(item["Inferences/Second"]))
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "concurrency": int(row["Concurrency"]),
        "inferences_per_second": float(row["Inferences/Second"]),
        "p50_latency_ms": float(row["p50 latency"]) / 1000.0,
        "p95_latency_ms": float(row["p95 latency"]) / 1000.0,
        "p99_latency_ms": float(row["p99 latency"]) / 1000.0,
    }


def _read_telemetry(path: Path) -> list[dict[str, Any]]:
    by_index: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cleaned = {key.strip(): value.strip() for key, value in row.items() if key}
            index = int(cleaned["index"])
            by_index[index].append(
                {
                    "uuid": cleaned["uuid"],
                    "name": cleaned["name"],
                    "utilization_gpu_percent": float(cleaned["utilization.gpu"]),
                    "memory_used_mib": float(cleaned["memory.used"]),
                    "memory_total_mib": float(cleaned["memory.total"]),
                    "power_draw_watts": float(cleaned["power.draw"]),
                }
            )
    output = []
    for index, samples in sorted(by_index.items()):
        totals = {sample["memory_total_mib"] for sample in samples}
        uuids = {sample["uuid"] for sample in samples}
        names = {sample["name"] for sample in samples}
        if len(totals) != 1 or len(uuids) != 1 or len(names) != 1:
            raise ValueError(f"GPU identity changed within telemetry for index {index}")
        output.append(
            {
                "index": index,
                "uuid": next(iter(uuids)),
                "name": next(iter(names)),
                "samples": len(samples),
                "mean_utilization_percent": sum(float(sample["utilization_gpu_percent"]) for sample in samples) / len(samples),
                "peak_utilization_percent": max(float(sample["utilization_gpu_percent"]) for sample in samples),
                "peak_memory_used_mib": max(float(sample["memory_used_mib"]) for sample in samples),
                "memory_total_mib": next(iter(totals)),
                "mean_power_draw_watts": sum(float(sample["power_draw_watts"]) for sample in samples) / len(samples),
                "peak_power_draw_watts": max(float(sample["power_draw_watts"]) for sample in samples),
            }
        )
    return output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def summarize_run(args: argparse.Namespace) -> int:
    root = args.run_dir.resolve(strict=True)
    if not HEX_40.fullmatch(args.source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA-1")
    perf_paths = sorted(root.glob("perf-server-*.csv"))
    perf = [_read_perf(path) for path in perf_paths]
    telemetry_path = root / "nvidia-smi.csv"
    telemetry = _read_telemetry(telemetry_path)
    checks = {
        "expected_server_results": len(perf) == args.gpu_count,
        "expected_gpu_telemetry": len(telemetry) == args.gpu_count,
        "telemetry_sample_floor": bool(telemetry) and all(item["samples"] >= args.minimum_telemetry_samples for item in telemetry),
        "positive_throughput": bool(perf) and all(item["inferences_per_second"] > 0.0 for item in perf),
        "positive_latency": bool(perf) and all(item["p95_latency_ms"] > 0.0 and item["p99_latency_ms"] > 0.0 for item in perf),
        "gh200_only": bool(telemetry) and all("GH200" in item["name"] for item in telemetry),
    }
    total_throughput = sum(item["inferences_per_second"] for item in perf)
    payload = {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source": {
            "git_commit": args.source_commit,
            "source_archive_sha256": args.source_archive_sha256,
            "parent_formal_job_id": args.parent_formal_job_id,
            "parent_formal_decision": "GPU_REJECTED",
            "model_plan_sha256": args.model_plan_sha256,
        },
        "slurm": {
            "job_id": args.slurm_job_id,
            "partition": args.partition,
            "gpu_count": args.gpu_count,
            "completed_state_attested_by_job": False,
        },
        "measurement": {
            "batch_size": args.batch_size,
            "concurrency_per_server": args.concurrency,
            "server_count": len(perf),
            "total_inferences_per_second": total_throughput,
            "mean_server_inferences_per_second": total_throughput / len(perf) if perf else 0.0,
            "worst_p95_latency_ms": max((item["p95_latency_ms"] for item in perf), default=math.nan),
            "worst_p99_latency_ms": max((item["p99_latency_ms"] for item in perf), default=math.nan),
            "servers": perf,
            "gpus": telemetry,
        },
        "profiler": {
            "enabled": args.profiler_enabled,
            "nsys_report_present": (root / "nsys-profile.nsys-rep").is_file(),
            "cuda_kernel_summary_present": (root / "nsys-cuda-kernel-summary.csv").is_file(),
            "cuda_api_summary_present": (root / "nsys-cuda-api-summary.csv").is_file(),
        },
        "claim_boundary": {
            "classification": "SMOKE_ONLY",
            "semantic_parity_reestablished": False,
            "gpu_eligibility_decision_allowed": False,
            "production_capacity_claim_allowed": False,
            "kubernetes_gpu_fleet_claim_allowed": False,
            "note": (
                "This job measures short single-node GH200 scheduling and serving behaviour. "
                "It reuses a plan from the named formal job and does not replace that job's "
                "GPU_REJECTED decision."
            ),
        },
        "artifacts": {
            "telemetry": {
                "file": telemetry_path.name,
                "sha256": _sha256(telemetry_path),
            }
        },
    }
    _atomic_json(args.output, payload)
    return 0 if payload["status"] == "PASS" else 2


def aggregate_runs(args: argparse.Namespace) -> int:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    if not summaries or any(item.get("schema_version") != SCHEMA for item in summaries):
        raise ValueError("all inputs must be v1 Roihu right-sizing summaries")
    by_count = {int(item["slurm"]["gpu_count"]): item for item in summaries}
    if set(by_count) != {1, 2, 4}:
        raise ValueError("aggregate evidence requires exactly 1, 2, and 4 GPU summaries")
    baseline = float(by_count[1]["measurement"]["total_inferences_per_second"])
    points = []
    for count in (1, 2, 4):
        item = by_count[count]
        throughput = float(item["measurement"]["total_inferences_per_second"])
        speedup = throughput / baseline
        points.append(
            {
                "gpu_count": count,
                "slurm_job_id": item["slurm"]["job_id"],
                "total_inferences_per_second": throughput,
                "speedup_vs_one_gpu": speedup,
                "scaling_efficiency": speedup / count,
                "worst_p95_latency_ms": item["measurement"]["worst_p95_latency_ms"],
                "mean_gpu_utilization_percent": sum(gpu["mean_utilization_percent"] for gpu in item["measurement"]["gpus"]) / count,
            }
        )
    checks = {
        "all_runs_passed": all(item.get("status") == "PASS" for item in summaries),
        "source_commit_consistent": len({item["source"]["git_commit"] for item in summaries}) == 1,
        "model_plan_consistent": len({item["source"]["model_plan_sha256"] for item in summaries}) == 1,
    }
    payload = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "points": points,
        "best_observed_gpu_count_by_total_throughput": max(points, key=lambda item: item["total_inferences_per_second"])["gpu_count"],
        "claim_boundary": {
            "classification": "SMOKE_ONLY",
            "automatic_rightsizing_decision_allowed": False,
            "production_capacity_claim_allowed": False,
            "note": (
                "Observed scaling is a short same-node comparison. It is not a cost model, "
                "production SLO, or authorization to allocate additional GPUs."
            ),
        },
    }
    _atomic_json(args.output, payload)
    return 0 if payload["status"] == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize governed Roihu Triton right-sizing evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpu-count", type=int, choices=(1, 2, 4), required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--source-archive-sha256", required=True)
    run.add_argument("--model-plan-sha256", required=True)
    run.add_argument("--parent-formal-job-id", required=True)
    run.add_argument("--slurm-job-id", required=True)
    run.add_argument("--partition", choices=("gputest", "gpumedium"), required=True)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--minimum-telemetry-samples", type=int, default=10)
    run.add_argument("--profiler-enabled", action="store_true")
    run.set_defaults(func=summarize_run)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--summaries", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(func=aggregate_runs)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
