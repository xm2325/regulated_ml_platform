from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from src.operations.roihu_gpu_evidence import cli, parse_checksums, validate_roihu_gpu_evidence

SOURCE_SHA = "a" * 40
EVIDENCE_SCHEMA = "regulated-ml-platform.roihu-gpu-evidence/v1"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy() -> dict:
    return yaml.safe_load(Path("config/roihu_gpu_evidence_policy.yaml").read_text(encoding="utf-8"))


def _evidence(tmp_path: Path) -> tuple[dict, dict, dict[str, str]]:
    release_files = {
        "source_archive": "release/source.tar.gz",
        "model": "release/model.plan",
        "sif_image": "release/triton.sif",
    }
    raw_files = {
        "slurm_job": "raw/sacct.json",
        "nvidia_smi": "raw/nvidia-smi.csv",
        "software_versions": "raw/software-versions.json",
        "cpu_benchmark": "raw/cpu-benchmark.json",
        "gpu_benchmark": "raw/gpu-benchmark.json",
        "parity": "raw/parity.json",
    }
    for name, relative in {**release_files, **raw_files}.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"immutable test evidence: {name}\n".encode())

    checksums = {relative: _digest(tmp_path / relative) for relative in [*release_files.values(), *raw_files.values()]}
    manifest = {
        "schema_version": EVIDENCE_SCHEMA,
        "source_commit": SOURCE_SHA,
        # This claim is intentionally not consulted by the validator.
        "runtime_evidence": "real_gpu",
        "qualified_workload": {
            "kind": "neural_accelerator_qualification",
            "model_family": "transformer",
            "profile": "candidate_gpu_profile",
            "is_current_tree_champion": False,
        },
        "slurm": {
            "cluster": "roihu",
            "partition": "gpumedium",
            "job_id": "123456",
            "node_list": "r-gpu001",
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
        "hardware": {
            "gpu_product_name": "NVIDIA GH200 96GB HBM3",
            "compute_capability": "9.0",
            "cpu_architecture": "aarch64",
            "gpu_count": 1,
            "gpu_memory_total_mib": 98304,
            "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "driver_version": "575.51.03",
        },
        "software": {
            "cuda_version": "12.9.1",
            "tensorrt_version": "10.9.0",
            "triton_server_version": "2.59.0",
            "triton_container_release": "25.03",
        },
        "artifacts": {name: {"path": relative, "sha256": f"sha256:{checksums[relative]}"} for name, relative in release_files.items()},
        "raw_artifacts": [{"name": name, "path": relative, "sha256": checksums[relative]} for name, relative in raw_files.items()],
    }
    memory_headroom = (98304 - 70000) / 98304
    benchmark = {
        "schema_version": EVIDENCE_SCHEMA,
        "source_commit": SOURCE_SHA,
        "runtime_evidence": "real_gpu",
        "cpu": {
            "duration_seconds": 600,
            "sample_count": 2000,
            "throughput_requests_per_second": 1000.0,
            "p95_latency_ms": 10.0,
        },
        "gpu": {
            "duration_seconds": 600,
            "sample_count": 2000,
            "throughput_requests_per_second": 2000.0,
            "p95_latency_ms": 7.5,
            "http_requests": 10000,
            "http_errors": 1,
            "http_error_rate": 0.0001,
            "telemetry_sample_count": 2000,
            "sustained_gpu_utilization_fraction": 0.65,
            "peak_gpu_memory_used_mib": 70000,
            "gpu_memory_total_mib": 98304,
            "gpu_memory_headroom_fraction": memory_headroom,
        },
        "comparison": {
            "throughput_speedup_ratio": 2.0,
            "gpu_p95_to_cpu_p95_ratio": 0.75,
        },
        "parity": {
            "sample_count": 2000,
            "max_absolute_probability_error": 0.000001,
            "policy_decision_mismatches": 0,
        },
    }
    return manifest, benchmark, checksums


def test_complete_roihu_gh200_evidence_is_gpu_eligible(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_ELIGIBLE"
    assert report["status"] == "PASS"
    assert report["gpu_profile_enabled"] is True
    assert report["reasons"] == []
    assert report["artifacts"]["verified_file_count"] == 9
    assert len(report["artifacts"]["verified_sha256"]) == 9
    assert report["artifacts"]["verified_sha256"]["artifact:sif_image"] == checksums["release/triton.sif"]
    assert report["runtime_evidence_string_trusted"] is False
    assert report["accelerator_product"] == "nvidia-gh200"
    assert report["qualified_workload"] == {
        "verified": True,
        "kind": "neural_accelerator_qualification",
        "model_family": "transformer",
        "profile": "candidate_gpu_profile",
        "applies_to_current_tree_champion": False,
    }
    assert report["deployment_effects"] == {
        "automatic_promotion_authorized": False,
        "current_model_accelerator": "unchanged",
        "current_model_profile": "unchanged",
    }


def test_tampered_sif_bytes_are_rejected(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    (tmp_path / manifest["artifacts"]["sif_image"]["path"]).write_bytes(b"tampered image")

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["gpu_profile_enabled"] is False
    assert report["checks"]["artifact_file_sha256_verified"] is False


def test_missing_raw_parity_artifact_is_rejected(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    manifest["raw_artifacts"] = [item for item in manifest["raw_artifacts"] if item["name"] != "parity"]

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["checks"]["required_raw_artifacts_present"] is False


def test_checksum_file_tampering_is_rejected(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    checksums[manifest["artifacts"]["model"]["path"]] = "f" * 64

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["checks"]["checksum_declarations_match_manifest"] is False
    assert report["checks"]["artifact_file_sha256_verified"] is False


def test_lone_runtime_evidence_claim_never_unlocks_gpu(tmp_path: Path):
    manifest = {"runtime_evidence": "real_gpu"}
    benchmark = {"runtime_evidence": "real_gpu"}

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), {}, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["runtime_evidence_string_trusted"] is False
    assert report["accelerator_product"] is None
    assert report["qualified_workload"]["verified"] is False
    assert report["checks"]["real_gh200_product_metadata"] is False
    assert report["checks"]["artifact_file_sha256_verified"] is False


def test_incomplete_source_sha_and_nonzero_slurm_exit_are_rejected(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    manifest["source_commit"] = "a" * 12
    benchmark["source_commit"] = "a" * 12
    manifest["slurm"]["exit_code"] = "1:0"

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["checks"]["manifest_full_source_sha"] is False
    assert report["checks"]["benchmark_full_source_sha"] is False
    assert report["checks"]["slurm_exit_code_zero"] is False


def test_short_gputest_partition_cannot_issue_qualification_decision(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    manifest["slurm"]["partition"] = "gputest"

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["checks"]["slurm_gpu_partition"] is False


def test_reported_speedup_must_match_raw_measurements(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    benchmark["comparison"]["throughput_speedup_ratio"] = 9.0

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["checks"]["throughput_speedup_consistent"] is False
    assert report["derived"]["throughput_speedup_ratio"] == 2.0


def test_cli_reads_four_inputs_and_writes_machine_decision(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    benchmark_path = tmp_path / "benchmark.json"
    policy_path = tmp_path / "policy.yaml"
    checksums_path = tmp_path / "SHA256SUMS"
    output_path = tmp_path / "decision.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(_policy()), encoding="utf-8")
    checksums_path.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(checksums.items())),
        encoding="utf-8",
    )

    exit_code = cli(
        [
            "--manifest",
            str(manifest_path),
            "--benchmark",
            str(benchmark_path),
            "--policy",
            str(policy_path),
            "--checksums",
            str(checksums_path),
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["decision"] == "GPU_ELIGIBLE"


def test_checksum_parser_rejects_duplicate_and_unsafe_paths():
    digest = "a" * 64
    assert parse_checksums(f"{digest}  raw/evidence.json\n") == {"raw/evidence.json": digest}

    for invalid in (
        f"{digest}  ../outside.json\n",
        f"{digest}  raw/evidence.json\n{digest}  raw/evidence.json\n",
    ):
        try:
            parse_checksums(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe or duplicate checksum path was accepted")


def test_policy_threshold_failure_is_fail_closed(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    policy = copy.deepcopy(_policy())
    del policy["benchmark"]["minimum_duration_seconds"]

    report = validate_roihu_gpu_evidence(manifest, benchmark, policy, checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["checks"]["policy_minimum_duration_seconds"] is False


def test_policy_cannot_remove_mandatory_sif_or_raw_gpu_evidence(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    policy = copy.deepcopy(_policy())
    policy["artifacts"]["required_artifacts"] = ["source_archive", "model"]
    policy["artifacts"]["required_raw_artifacts"] = ["slurm_job", "nvidia_smi"]

    report = validate_roihu_gpu_evidence(manifest, benchmark, policy, checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["checks"]["policy_required_artifacts"] is False
    assert report["checks"]["policy_required_raw_artifacts"] is False


def test_qualification_cannot_change_current_tree_champion_profile(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    manifest["qualified_workload"]["is_current_tree_champion"] = True

    report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert report["decision"] == "GPU_REJECTED"
    assert report["accelerator_product"] is None
    assert report["qualified_workload"]["verified"] is False
    assert report["checks"]["current_tree_champion_excluded"] is False
    assert report["deployment_effects"]["automatic_promotion_authorized"] is False
    assert report["deployment_effects"]["current_model_accelerator"] == "unchanged"
    assert report["deployment_effects"]["current_model_profile"] == "unchanged"


def test_triton_server_and_container_release_versions_are_independent(tmp_path: Path):
    manifest, benchmark, checksums = _evidence(tmp_path)
    manifest["software"]["triton_server_version"] = "2.41.0"

    server_report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert server_report["decision"] == "GPU_REJECTED"
    assert server_report["checks"]["triton_server_version_validated"] is False
    assert server_report["checks"]["triton_container_release_validated"] is True

    manifest["software"]["triton_server_version"] = "2.59.0"
    manifest["software"]["triton_container_release"] = "23.12"
    release_report = validate_roihu_gpu_evidence(manifest, benchmark, _policy(), checksums, tmp_path)

    assert release_report["decision"] == "GPU_REJECTED"
    assert release_report["checks"]["triton_server_version_validated"] is True
    assert release_report["checks"]["triton_container_release_validated"] is False
