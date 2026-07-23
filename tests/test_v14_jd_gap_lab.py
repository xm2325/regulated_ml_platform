from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
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
        "-name 'nsys-cuda-gpu-kern-sum*.csv'",
        "-name 'nsys-cuda-api-sum*.csv'",
        "unexpected number of CUDA kernel summary files",
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


def test_torchserve_summary_requires_cuda_and_preserves_archived_runtime_boundary(
    tmp_path: Path,
):
    path = ROIHU / "torchserve_smoke_tools.py"
    spec = importlib.util.spec_from_file_location("torchserve_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "batch": 256,
                "device": "cuda",
                "output_shape": [256, 16],
                "output_checksum": 1.0,
            }
        ),
        encoding="utf-8",
    )
    telemetry = tmp_path / "nvidia-smi.csv"
    _write_telemetry(telemetry, 1)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    output = tmp_path / "summary.json"
    args = Namespace(
        response=response,
        telemetry=telemetry,
        model=model,
        output=output,
        batch=256,
        source_commit="a" * 40,
        source_archive_sha256="b" * 64,
        torchserve_wheel_sha256="c" * 64,
        model_archiver_wheel_sha256="d" * 64,
        slurm_job_id="9001",
        partition="gputest",
        torchserve_version="0.12.0",
        pytorch_version="2.10.0+cu130",
        cuda_version="13.0",
        java_version="25.0.1",
    )

    assert module.summarize(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "SMOKE_PASS"
    assert report["claim_boundary"]["performance_claim_allowed"] is False
    assert report["claim_boundary"]["production_readiness_claim_allowed"] is False
    assert report["claim_boundary"]["recommended_runtime_claim_allowed"] is False


def test_torchserve_sbatch_is_offline_loopback_and_claim_bounded():
    script = (ROIHU / "torchserve_gpu_smoke.sbatch").read_text(encoding="utf-8")
    for contract in (
        "#SBATCH --partition=gputest",
        "#SBATCH --time=00:10:00",
        "--no-index --no-deps",
        "JAVA_HOME=",
        "disable_token_authorization=true",
        "inference_address=http://127.0.0.1:8080",
        "metrics_address=http://127.0.0.1:8082",
        "source archive marker mismatch",
        "SMOKE_ONLY",
        "recommended_runtime_claim_allowed",
    ):
        assert contract in script or contract in (ROIHU / "torchserve_smoke_tools.py").read_text(encoding="utf-8")
    assert "pip download" not in script
    assert "apptainer pull" not in script
