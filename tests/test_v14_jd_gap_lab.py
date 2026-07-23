from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROIHU = ROOT / "hpc" / "roihu"


def _load():
    path = ROIHU / "summarize_triton_rightsizing.py"
    spec = importlib.util.spec_from_file_location("rightsizing", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_perf(path: Path, throughput: float, p95_us: int = 1500) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Concurrency",
                "Inferences/Second",
                "p50 latency",
                "p95 latency",
                "p99 latency",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Concurrency": 4,
                "Inferences/Second": throughput,
                "p50 latency": 1000,
                "p95 latency": p95_us,
                "p99 latency": p95_us + 200,
            }
        )


def _write_telemetry(path: Path, gpu_count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "index",
                "uuid",
                "name",
                "utilization.gpu",
                "memory.used",
                "memory.total",
                "power.draw",
            ],
        )
        writer.writeheader()
        for sample in range(12):
            for index in range(gpu_count):
                writer.writerow(
                    {
                        "timestamp": f"2026-07-23 12:00:{sample:02d}",
                        "index": index,
                        "uuid": f"GPU-{index}",
                        "name": "NVIDIA GH200 120GB",
                        "utilization.gpu": 40 + index,
                        "memory.used": 1000 + index,
                        "memory.total": 97871,
                        "power.draw": 250 + index,
                    }
                )


def _args(module, root: Path, gpu_count: int):
    return module.build_parser().parse_args(
        [
            "run",
            "--run-dir",
            str(root),
            "--output",
            str(root / "summary.json"),
            "--gpu-count",
            str(gpu_count),
            "--source-commit",
            "a" * 40,
            "--source-archive-sha256",
            "b" * 64,
            "--model-plan-sha256",
            "c" * 64,
            "--parent-formal-job-id",
            "304890",
            "--slurm-job-id",
            f"900{gpu_count}",
            "--partition",
            "gputest",
        ]
    )


def test_rightsizing_run_summary_preserves_claim_boundary(tmp_path: Path):
    module = _load()
    _write_perf(tmp_path / "perf-server-0.csv", 1000.0)
    _write_telemetry(tmp_path / "nvidia-smi.csv", 1)

    assert module.summarize_run(_args(module, tmp_path, 1)) == 0
    report = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["measurement"]["total_inferences_per_second"] == 1000.0
    assert report["source"]["parent_formal_decision"] == "GPU_REJECTED"
    assert report["claim_boundary"]["classification"] == "SMOKE_ONLY"
    assert report["claim_boundary"]["gpu_eligibility_decision_allowed"] is False
    assert report["claim_boundary"]["production_capacity_claim_allowed"] is False


def test_rightsizing_run_fails_when_expected_server_is_missing(tmp_path: Path):
    module = _load()
    _write_perf(tmp_path / "perf-server-0.csv", 1000.0)
    _write_telemetry(tmp_path / "nvidia-smi.csv", 2)

    assert module.summarize_run(_args(module, tmp_path, 2)) == 2
    report = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["checks"]["expected_server_results"] is False


def test_rightsizing_aggregate_calculates_speedup_without_authorizing_capacity(tmp_path: Path):
    module = _load()
    paths = []
    for gpu_count, throughput in ((1, 1000.0), (2, 1800.0), (4, 3000.0)):
        root = tmp_path / str(gpu_count)
        root.mkdir()
        for index in range(gpu_count):
            _write_perf(root / f"perf-server-{index}.csv", throughput / gpu_count)
        _write_telemetry(root / "nvidia-smi.csv", gpu_count)
        assert module.summarize_run(_args(module, root, gpu_count)) == 0
        paths.append(root / "summary.json")

    args = module.build_parser().parse_args(
        [
            "aggregate",
            "--summaries",
            *(str(path) for path in paths),
            "--output",
            str(tmp_path / "aggregate.json"),
        ]
    )
    assert module.aggregate_runs(args) == 0
    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["points"][1]["speedup_vs_one_gpu"] == 1.8
    assert aggregate["points"][2]["scaling_efficiency"] == 0.75
    assert aggregate["claim_boundary"]["automatic_rightsizing_decision_allowed"] is False
    assert aggregate["claim_boundary"]["production_capacity_claim_allowed"] is False


def test_roihu_profile_script_is_bounded_source_bound_and_offline():
    script = (ROIHU / "triton_profile_rightsizing.sbatch").read_text(encoding="utf-8")
    for contract in (
        "#SBATCH --partition=gputest",
        "#SBATCH --time=00:15:00",
        "GPU_COUNT must be 1, 2, or 4",
        "request exactly 72 Grace CPU cores per GH200",
        "source archive marker mismatch",
        "--argos=no",
        "/usr/local/cuda/bin/nsys profile",
        "--duration=45",
        "--kill=sigterm",
        "APPTAINERENV_TMPDIR=/work/nsys-tmp",
        "env -u LD_PRELOAD",
        "--concurrency-range=4",
        "--measurement-interval=3000",
        "-lms 200",
        "SMOKE_ONLY",
        "summarize_triton_rightsizing.py",
    ):
        assert contract in script
    assert "apptainer pull" not in script
    assert "pip install" not in script
    assert "--http-address=127.0.0.1" in script
    assert "--grpc-address=127.0.0.1" in script
    assert "--metrics-address=127.0.0.1" in script
