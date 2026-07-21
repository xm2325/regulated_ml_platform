#!/usr/bin/env python3
"""Finalize a completed Roihu job into the formal GPU evidence gate schema.

Run this only after the qualification Slurm job has left the queue. The finalizer
queries ``sacct`` for the completed state, derives governed benchmark fields from
raw files, writes SHA256SUMS, and invokes the bundled fail-closed validator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "regulated-ml-platform.roihu-gpu-evidence/v1"
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
RELEASE_PATHS = {
    "source_archive": "release/source.tar.gz",
    "model": "release/model.plan",
    "sif_image": "release/triton-server.sif",
}
RAW_PATHS = {
    "slurm_job": "raw/slurm-job.json",
    "nvidia_smi": "raw/nvidia-smi.csv",
    "software_versions": "raw/software-versions.json",
    "cpu_benchmark": "raw/cpu-benchmark.json",
    "gpu_benchmark": "raw/gpu-benchmark.json",
    "parity": "raw/parity.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sacct-file", type=Path, help="offline test input; otherwise query sacct")
    parser.add_argument("--validator", type=Path)
    parser.add_argument("--policy", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sacct_text(job_id: str, offline: Path | None) -> str:
    if offline:
        return offline.read_text(encoding="utf-8")
    command = [
        "sacct",
        "--jobs",
        job_id,
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode,Partition,NodeList,Cluster",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout


def _parse_sacct(text: str, job_id: str) -> dict[str, str]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        if len(fields) != 6:
            raise ValueError("unexpected sacct field count")
        rows.append(fields)
    matches = [fields for fields in rows if fields[0] == job_id]
    if len(matches) != 1:
        raise ValueError("sacct must contain exactly one allocation row for the qualification job")
    _, state, exit_code, partition, node_list, cluster = matches[0]
    state = state.split()[0].rstrip("+")
    if state != "COMPLETED" or exit_code != "0:0":
        raise ValueError("qualification job is not recorded as COMPLETED with exit code 0:0")
    if cluster.casefold() != "roihu":
        raise ValueError("sacct cluster is not Roihu")
    return {
        "cluster": "roihu",
        "partition": partition,
        "job_id": job_id,
        "node_list": node_list,
        "state": state,
        "exit_code": exit_code,
    }


def _numeric_cell(row: dict[str, str], prefix: str) -> float:
    matches = [value for key, value in row.items() if key and key.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"nvidia-smi telemetry is missing the {prefix} column")
    value = matches[0].strip().split()[0]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nvidia-smi {prefix} value is not finite")
    return number


def _telemetry(path: Path) -> dict[str, float | int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("nvidia-smi telemetry contains no samples")
    utilizations = [_numeric_cell(row, "utilization.gpu") for row in rows]
    used_memory = [_numeric_cell(row, "memory.used") for row in rows]
    total_memory = [_numeric_cell(row, "memory.total") for row in rows]
    if any(not 0 <= value <= 100 for value in utilizations):
        raise ValueError("nvidia-smi utilization must be between zero and 100 percent")
    if max(total_memory) - min(total_memory) > 1:
        raise ValueError("nvidia-smi total memory changed during the qualification window")
    return {
        "sample_count": len(rows),
        "sustained_gpu_utilization_fraction": sum(utilizations) / len(utilizations) / 100.0,
        "peak_gpu_memory_used_mib": max(used_memory),
        "gpu_memory_total_mib": max(total_memory),
    }


def _required_number(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _build_benchmark(
    source_commit: str, cpu: dict[str, Any], gpu: dict[str, Any], parity: dict[str, Any], telemetry: dict[str, Any]
) -> dict[str, Any]:
    for name, record in (("CPU", cpu), ("GPU", gpu), ("parity", parity)):
        if record.get("status") != "PASS":
            raise ValueError(f"{name} raw evidence did not pass")
    cpu_throughput = _required_number(cpu, "throughput_requests_per_second")
    gpu_throughput = _required_number(gpu, "throughput_requests_per_second")
    cpu_p95 = _required_number(cpu.get("latency_ms", {}), "p95")
    gpu_p95 = _required_number(gpu.get("latency_ms", {}), "p95")
    total_memory = _required_number(telemetry, "gpu_memory_total_mib")
    peak_memory = _required_number(telemetry, "peak_gpu_memory_used_mib")
    headroom = (total_memory - peak_memory) / total_memory
    http_requests = int(gpu.get("http_requests"))
    http_errors = int(gpu.get("http_errors"))
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "source_commit": source_commit,
        "runtime_evidence": "real_gpu",
        "cpu": {
            "duration_seconds": _required_number(cpu, "duration_seconds"),
            "sample_count": int(cpu.get("sample_count")),
            "throughput_requests_per_second": cpu_throughput,
            "p95_latency_ms": cpu_p95,
        },
        "gpu": {
            "duration_seconds": _required_number(gpu, "duration_seconds"),
            "sample_count": int(gpu.get("sample_count")),
            "throughput_requests_per_second": gpu_throughput,
            "p95_latency_ms": gpu_p95,
            "http_requests": http_requests,
            "http_errors": http_errors,
            "http_error_rate": http_errors / http_requests if http_requests else 1.0,
            "telemetry_sample_count": int(telemetry["sample_count"]),
            "sustained_gpu_utilization_fraction": telemetry["sustained_gpu_utilization_fraction"],
            "peak_gpu_memory_used_mib": peak_memory,
            "gpu_memory_total_mib": total_memory,
            "gpu_memory_headroom_fraction": headroom,
        },
        "comparison": {
            "throughput_speedup_ratio": gpu_throughput / cpu_throughput,
            "gpu_p95_to_cpu_p95_ratio": gpu_p95 / cpu_p95,
        },
        "parity": {
            "sample_count": int(parity.get("sample_count")),
            "max_absolute_probability_error": _required_number(parity, "max_absolute_probability_error"),
            "policy_decision_mismatches": int(parity.get("policy_decision_mismatches")),
        },
    }


def run(args: argparse.Namespace) -> int:
    os.umask(0o077)
    root = args.evidence_root.resolve(strict=True)
    if not args.job_id.isdigit():
        raise ValueError("job ID must contain only digits")
    source_commit = args.source_commit.strip().lower()
    if not FULL_COMMIT.fullmatch(source_commit):
        raise ValueError("source commit must be a lowercase full 40-character Git commit")

    all_paths = {**RELEASE_PATHS, **RAW_PATHS}
    preexisting_paths = {name: relative for name, relative in all_paths.items() if name != "slurm_job"}
    resolved = {name: (root / relative).resolve(strict=True) for name, relative in preexisting_paths.items()}
    for path in resolved.values():
        path.relative_to(root)
        if not path.is_file():
            raise ValueError(f"required evidence file is missing: {path.name}")

    sacct_text = _sacct_text(args.job_id, args.sacct_file)
    slurm = _parse_sacct(sacct_text, args.job_id)
    _atomic_text(root / "raw/sacct.txt", sacct_text)
    _atomic_json(root / RAW_PATHS["slurm_job"], slurm)
    resolved["slurm_job"] = (root / RAW_PATHS["slurm_job"]).resolve(strict=True)

    cpu = _load_json(resolved["cpu_benchmark"])
    gpu = _load_json(resolved["gpu_benchmark"])
    parity = _load_json(resolved["parity"])
    software_versions = _load_json(resolved["software_versions"])
    telemetry = _telemetry(resolved["nvidia_smi"])
    benchmark = _build_benchmark(source_commit, cpu, gpu, parity, telemetry)
    _atomic_json(root / "benchmark.json", benchmark)

    hardware = software_versions.get("hardware")
    software = software_versions.get("software")
    if not isinstance(hardware, dict) or not isinstance(software, dict):
        raise ValueError("software-versions.json must contain hardware and software objects")
    digests = {name: _sha256(path) for name, path in resolved.items()}
    manifest = {
        "schema_version": EVIDENCE_SCHEMA,
        "source_commit": source_commit,
        "runtime_evidence": "real_gpu",
        "qualified_workload": {
            "kind": "neural_accelerator_qualification",
            "model_family": "neural_network",
            "profile": "candidate_gpu_profile",
            "is_current_tree_champion": False,
        },
        "slurm": slurm,
        "hardware": hardware,
        "software": software,
        "artifacts": {name: {"path": RELEASE_PATHS[name], "sha256": f"sha256:{digests[name]}"} for name in RELEASE_PATHS},
        "raw_artifacts": [{"name": name, "path": RAW_PATHS[name], "sha256": digests[name]} for name in RAW_PATHS],
    }
    _atomic_json(root / "manifest.json", manifest)
    checksum_lines = [f"{digests[name]}  {all_paths[name]}" for name in sorted(all_paths)]
    _atomic_text(root / "SHA256SUMS", "\n".join(checksum_lines) + "\n")

    validator = (args.validator or root / "validator/roihu_gpu_evidence.py").resolve(strict=True)
    policy = (args.policy or root / "validator/roihu_gpu_evidence_policy.yaml").resolve(strict=True)
    command = [
        sys.executable,
        str(validator),
        "--manifest",
        str(root / "manifest.json"),
        "--benchmark",
        str(root / "benchmark.json"),
        "--policy",
        str(policy),
        "--checksums",
        str(root / "SHA256SUMS"),
        "--artifact-root",
        str(root),
        "--output",
        str(root / "decision.json"),
    ]
    result = subprocess.run(command, check=False, timeout=120)
    return result.returncode


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - finalization must fail closed
        root = args.evidence_root.resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        failure = {
            "schema_version": "regulated-ml-platform.roihu-gpu-decision/v1",
            "decision": "GPU_REJECTED",
            "status": "FAIL",
            "reasons": [f"finalization: {type(exc).__name__}: {exc}"],
            "gpu_profile_enabled": False,
        }
        try:
            _atomic_json(root / "decision.json", failure)
        except Exception:  # noqa: BLE001
            pass
        print(f"qualification finalization failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
