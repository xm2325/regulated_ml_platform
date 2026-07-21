from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

FULL_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{16,}$")
VERSION_RE = re.compile(r"^(\d+(?:\.\d+){1,3})(?:[-+][0-9A-Za-z.-]+)?$")
GNU_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})\s+[ *](.+)$")
MANDATORY_RELEASE_ARTIFACTS = frozenset({"source_archive", "model", "sif_image"})
MANDATORY_RAW_ARTIFACTS = frozenset({"slurm_job", "nvidia_smi", "software_versions", "cpu_benchmark", "gpu_benchmark", "parity"})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _sha256(value: Any) -> str | None:
    match = SHA256_RE.fullmatch(value) if isinstance(value, str) else None
    return match.group(1) if match else None


def _version(value: Any) -> tuple[int, ...] | None:
    match = VERSION_RE.fullmatch(value) if isinstance(value, str) else None
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _version_at_least(observed: Any, minimum: Any) -> bool:
    observed_version = _version(observed)
    minimum_version = _version(minimum)
    if observed_version is None or minimum_version is None:
        return False
    width = max(len(observed_version), len(minimum_version))
    return observed_version + (0,) * (width - len(observed_version)) >= minimum_version + (0,) * (width - len(minimum_version))


def _canonical_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(text: str) -> dict[str, str]:
    """Parse a JSON/YAML checksum mapping or a GNU sha256sum file.

    Accepted structured forms are ``{"path": "digest"}``,
    ``{"checksums": {...}}``, or ``{"files": [{"path": ..., "sha256": ...}]}``.
    Duplicate, unsafe, or malformed entries are rejected rather than overwritten.
    """

    stripped = text.strip()
    if not stripped:
        raise ValueError("checksum input is empty")

    parsed: Any = None
    if stripped[0] in "{[":
        parsed = json.loads(stripped)
    elif ":" in stripped and not GNU_CHECKSUM_RE.fullmatch(stripped.splitlines()[0].strip()):
        parsed = yaml.safe_load(stripped)

    entries: list[tuple[Any, Any]] = []
    if parsed is not None:
        if isinstance(parsed, Mapping) and isinstance(parsed.get("checksums"), Mapping):
            entries = list(parsed["checksums"].items())
        elif isinstance(parsed, Mapping) and isinstance(parsed.get("files"), Sequence):
            entries = [(_mapping(item).get("path"), _mapping(item).get("sha256")) for item in parsed["files"]]
        elif isinstance(parsed, Mapping):
            entries = list(parsed.items())
        else:
            raise ValueError("structured checksum input must be a mapping")
    else:
        for line_number, raw_line in enumerate(stripped.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = GNU_CHECKSUM_RE.fullmatch(line)
            if not match:
                raise ValueError(f"invalid GNU checksum line {line_number}")
            entries.append((match.group(2), match.group(1)))

    checksums: dict[str, str] = {}
    for raw_path, raw_digest in entries:
        path = _canonical_relative_path(raw_path)
        digest = _sha256(raw_digest)
        if path is None or digest is None:
            raise ValueError("checksum entries require canonical relative paths and SHA-256 digests")
        if path in checksums:
            raise ValueError(f"duplicate checksum path: {path}")
        checksums[path] = digest
    if not checksums:
        raise ValueError("checksum input contains no entries")
    return checksums


class _Evaluation:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.reasons: list[str] = []

    def require(self, name: str, passed: bool, failure: str) -> bool:
        passed = bool(passed)
        self.checks[name] = passed
        if not passed:
            self.reasons.append(f"{name}: {failure}")
        return passed


def _required_policy_number(
    evaluation: _Evaluation,
    section: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = _finite_number(section.get(key))
    valid = value is not None
    if valid and minimum is not None:
        valid = value >= minimum
    if valid and maximum is not None:
        valid = value <= maximum
    evaluation.require(
        f"policy_{key}",
        valid,
        f"policy must define a finite {key}" + (f" >= {minimum}" if minimum is not None else ""),
    )
    return value if valid else None


def _validate_artifacts(
    evaluation: _Evaluation,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    checksums: Mapping[str, str],
    artifact_root: pathlib.Path,
) -> dict[str, Any]:
    artifact_policy = _mapping(policy.get("artifacts"))
    artifacts = _mapping(manifest.get("artifacts"))
    required_artifacts = [str(value) for value in _sequence(artifact_policy.get("required_artifacts"))]
    required_raw = [str(value) for value in _sequence(artifact_policy.get("required_raw_artifacts"))]
    evaluation.require(
        "policy_required_artifacts",
        MANDATORY_RELEASE_ARTIFACTS.issubset(required_artifacts),
        "policy must require the source archive, model, and SIF image",
    )
    evaluation.require(
        "policy_required_raw_artifacts",
        MANDATORY_RAW_ARTIFACTS.issubset(required_raw),
        "policy must require Slurm, nvidia-smi, software, CPU/GPU benchmark, and parity records",
    )

    records: dict[str, tuple[str, str]] = {}
    artifact_names_valid = True
    for name in required_artifacts:
        item = _mapping(artifacts.get(name))
        path = _canonical_relative_path(item.get("path"))
        digest = _sha256(item.get("sha256"))
        if path is None or digest is None:
            artifact_names_valid = False
            continue
        records[name] = (path, digest)
    evaluation.require(
        "release_artifact_digests_present",
        artifact_names_valid and len(records) == len(required_artifacts),
        "source archive, model, and SIF records must have canonical paths and SHA-256 digests",
    )

    raw_records: dict[str, tuple[str, str]] = {}
    raw_items = _sequence(manifest.get("raw_artifacts"))
    raw_valid = bool(raw_items)
    for raw_item in raw_items:
        item = _mapping(raw_item)
        name = item.get("name")
        path = _canonical_relative_path(item.get("path"))
        digest = _sha256(item.get("sha256"))
        if not isinstance(name, str) or not name or name in raw_records or path is None or digest is None:
            raw_valid = False
            continue
        raw_records[name] = (path, digest)
    evaluation.require(
        "raw_artifact_manifest_valid",
        raw_valid,
        "raw artifact records must be non-empty, unique, and include canonical path plus SHA-256",
    )
    evaluation.require(
        "required_raw_artifacts_present",
        all(name in raw_records for name in required_raw),
        "one or more policy-required raw evidence artifacts are missing",
    )

    all_records = {f"artifact:{name}": record for name, record in records.items()}
    all_records.update({f"raw:{name}": record for name, record in raw_records.items()})
    paths = [record[0] for record in all_records.values()]
    evaluation.require(
        "artifact_paths_unique",
        len(paths) == len(set(paths)),
        "each release and raw artifact must have a distinct path",
    )

    declarations_match = bool(all_records) and all(checksums.get(path) == digest for path, digest in all_records.values())
    evaluation.require(
        "checksum_declarations_match_manifest",
        declarations_match,
        "checksum file must contain the manifest digest for every release and raw artifact",
    )

    root = artifact_root.resolve()
    files_exist = bool(all_records)
    files_match = bool(all_records)
    observed_hashes: dict[str, str] = {}
    for key, (relative, expected) in all_records.items():
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            files_exist = False
            files_match = False
            continue
        if not candidate.is_file():
            files_exist = False
            files_match = False
            continue
        observed = _hash_file(candidate)
        observed_hashes[key] = observed
        if observed != expected or checksums.get(relative) != observed:
            files_match = False
    evaluation.require(
        "artifact_files_present",
        files_exist,
        "every declared release and raw artifact must exist below the evidence root",
    )
    evaluation.require(
        "artifact_file_sha256_verified",
        files_match,
        "one or more artifact bytes do not match both manifest and checksum-file SHA-256",
    )
    return {
        "release_artifact_count": len(records),
        "raw_artifact_count": len(raw_records),
        "verified_file_count": len(observed_hashes),
        "verified_sha256": dict(sorted(observed_hashes.items())),
    }


def validate_roihu_gpu_evidence(
    manifest: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    policy: Mapping[str, Any],
    checksums: Mapping[str, str],
    artifact_root: pathlib.Path | str,
) -> dict[str, Any]:
    """Return a fail-closed Roihu GPU eligibility decision.

    ``runtime_evidence`` is deliberately ignored. Eligibility is derived from
    independently cross-checked provenance, scheduler, hardware, runtime,
    benchmark, telemetry, and artifact-integrity fields.
    """

    evaluation = _Evaluation()
    policy_schema = policy.get("schema_version")
    evidence_schema = policy.get("required_evidence_schema_version")
    evaluation.require(
        "policy_schema_version",
        policy_schema == "regulated-ml-platform.roihu-gpu-policy/v1",
        "unsupported or missing Roihu GPU policy schema",
    )
    evaluation.require(
        "evidence_schema_policy_configured",
        isinstance(evidence_schema, str) and bool(evidence_schema),
        "policy must configure the required evidence schema",
    )
    evaluation.require(
        "manifest_schema_version",
        isinstance(evidence_schema, str) and manifest.get("schema_version") == evidence_schema,
        "manifest schema does not match policy",
    )
    evaluation.require(
        "benchmark_schema_version",
        isinstance(evidence_schema, str) and benchmark.get("schema_version") == evidence_schema,
        "benchmark schema does not match policy",
    )

    manifest_sha = manifest.get("source_commit")
    benchmark_sha = benchmark.get("source_commit")
    evaluation.require(
        "manifest_full_source_sha",
        isinstance(manifest_sha, str) and FULL_GIT_SHA_RE.fullmatch(manifest_sha) is not None,
        "manifest source_commit must be a lowercase full 40- or 64-hex Git SHA",
    )
    evaluation.require(
        "benchmark_full_source_sha",
        isinstance(benchmark_sha, str) and FULL_GIT_SHA_RE.fullmatch(benchmark_sha) is not None,
        "benchmark source_commit must be a lowercase full 40- or 64-hex Git SHA",
    )
    evaluation.require(
        "source_sha_consistent",
        isinstance(manifest_sha, str) and manifest_sha == benchmark_sha,
        "manifest and benchmark source commits differ",
    )

    qualification_policy = _mapping(policy.get("qualification"))
    qualified_workload = _mapping(manifest.get("qualified_workload"))
    required_workload_kind = qualification_policy.get("required_kind")
    required_workload_profile = qualification_policy.get("required_profile")
    allowed_model_families = {str(value) for value in _sequence(qualification_policy.get("allowed_model_families"))}
    configured_accelerator_product = qualification_policy.get("accelerator_product")
    evaluation.require(
        "qualification_policy_configured",
        required_workload_kind == "neural_accelerator_qualification"
        and required_workload_profile == "candidate_gpu_profile"
        and bool(allowed_model_families)
        and configured_accelerator_product == "nvidia-gh200",
        "policy must scope evidence to a neural candidate profile on nvidia-gh200",
    )
    evaluation.require(
        "neural_accelerator_qualification_workload",
        qualified_workload.get("kind") == required_workload_kind,
        "manifest is not a neural accelerator qualification workload",
    )
    evaluation.require(
        "candidate_gpu_profile_scope",
        qualified_workload.get("profile") == required_workload_profile,
        "qualification must target an isolated candidate GPU profile",
    )
    evaluation.require(
        "gpu_candidate_model_family",
        isinstance(qualified_workload.get("model_family"), str) and qualified_workload.get("model_family") in allowed_model_families,
        "model family is not approved for neural accelerator qualification",
    )
    evaluation.require(
        "current_tree_champion_excluded",
        qualified_workload.get("is_current_tree_champion") is False,
        "GPU qualification must not target or unlock the current tree champion",
    )

    slurm_policy = _mapping(policy.get("slurm"))
    slurm = _mapping(manifest.get("slurm"))
    allowed_partitions = {str(value) for value in _sequence(slurm_policy.get("allowed_partitions"))}
    expected_cluster = slurm_policy.get("expected_cluster")
    evaluation.require(
        "slurm_policy_configured",
        isinstance(expected_cluster, str) and bool(expected_cluster) and bool(allowed_partitions),
        "policy must define expected cluster and allowed GPU partitions",
    )
    evaluation.require(
        "slurm_roihu_cluster",
        isinstance(expected_cluster, str) and slurm.get("cluster") == expected_cluster,
        "job was not recorded on the configured Roihu cluster",
    )
    evaluation.require(
        "slurm_gpu_partition",
        isinstance(slurm.get("partition"), str) and slurm.get("partition") in allowed_partitions,
        "job partition is not an approved full-GH200 partition",
    )
    evaluation.require(
        "slurm_job_identity",
        isinstance(slurm.get("job_id"), str)
        and bool(slurm.get("job_id"))
        and isinstance(slurm.get("node_list"), str)
        and bool(slurm.get("node_list")),
        "job_id and node_list are required",
    )
    evaluation.require(
        "slurm_completed",
        slurm.get("state") == "COMPLETED",
        "Slurm state must be COMPLETED",
    )
    evaluation.require(
        "slurm_exit_code_zero",
        slurm.get("exit_code") == "0:0",
        "Slurm exit code must be 0:0",
    )

    hardware_policy = _mapping(policy.get("hardware"))
    hardware = _mapping(manifest.get("hardware"))
    allowed_product_tokens = [str(value).casefold() for value in _sequence(hardware_policy.get("required_product_tokens"))]
    product = hardware.get("gpu_product_name")
    product_folded = product.casefold() if isinstance(product, str) else ""
    evaluation.require(
        "gpu_product_policy_configured",
        bool(allowed_product_tokens),
        "policy must name required GH200 product tokens",
    )
    evaluation.require(
        "real_gh200_product_metadata",
        bool(product_folded) and all(token in product_folded for token in allowed_product_tokens),
        "nvidia-smi product metadata does not identify an NVIDIA GH200",
    )
    evaluation.require(
        "gh200_compute_capability",
        str(hardware.get("compute_capability")) == str(hardware_policy.get("required_compute_capability")),
        "GPU compute capability does not match GH200 Hopper policy",
    )
    evaluation.require(
        "grace_arm_host",
        hardware.get("cpu_architecture") == hardware_policy.get("required_cpu_architecture"),
        "GPU job host is not the configured NVIDIA Grace ARM architecture",
    )
    minimum_gpu_count = _non_negative_int(hardware_policy.get("minimum_gpu_count"))
    observed_gpu_count = _non_negative_int(hardware.get("gpu_count"))
    evaluation.require(
        "gpu_count",
        minimum_gpu_count is not None and observed_gpu_count is not None and observed_gpu_count >= minimum_gpu_count >= 1,
        "GPU count is missing or below policy",
    )
    minimum_gpu_memory = _finite_number(hardware_policy.get("minimum_gpu_memory_total_mib"))
    hardware_memory = _finite_number(hardware.get("gpu_memory_total_mib"))
    evaluation.require(
        "full_gh200_memory_metadata",
        minimum_gpu_memory is not None and hardware_memory is not None and hardware_memory >= minimum_gpu_memory > 0,
        "GPU memory metadata is missing or indicates less than a full 96 GiB GH200",
    )
    evaluation.require(
        "gpu_uuid_metadata",
        isinstance(hardware.get("gpu_uuid"), str) and GPU_UUID_RE.fullmatch(hardware["gpu_uuid"]) is not None,
        "a valid nvidia-smi GPU UUID is required",
    )
    evaluation.require(
        "driver_version_metadata",
        _version(hardware.get("driver_version")) is not None,
        "a parseable NVIDIA driver version is required",
    )

    software_policy = _mapping(policy.get("software"))
    software = _mapping(manifest.get("software"))
    for key, label in (
        ("cuda_version", "CUDA"),
        ("tensorrt_version", "TensorRT"),
        ("triton_server_version", "Triton Server"),
        ("triton_container_release", "Triton container release"),
    ):
        minimum = software_policy.get(f"minimum_{key}")
        evaluation.require(
            f"{key}_policy_configured",
            _version(minimum) is not None,
            f"policy must define a parseable minimum {label} version",
        )
        evaluation.require(
            f"{key}_validated",
            _version_at_least(software.get(key), minimum),
            f"observed {label} version is missing, malformed, or below policy",
        )

    benchmark_policy = _mapping(policy.get("benchmark"))
    minimum_duration = _required_policy_number(evaluation, benchmark_policy, "minimum_duration_seconds", minimum=1)
    minimum_cpu_samples = _required_policy_number(evaluation, benchmark_policy, "minimum_cpu_samples", minimum=1)
    minimum_gpu_samples = _required_policy_number(evaluation, benchmark_policy, "minimum_gpu_samples", minimum=1)
    minimum_parity_samples = _required_policy_number(evaluation, benchmark_policy, "minimum_parity_samples", minimum=1)
    maximum_probability_error = _required_policy_number(evaluation, benchmark_policy, "maximum_absolute_probability_error", minimum=0)
    maximum_decision_mismatches = _required_policy_number(evaluation, benchmark_policy, "maximum_policy_decision_mismatches", minimum=0)
    maximum_http_error_rate = _required_policy_number(evaluation, benchmark_policy, "maximum_http_error_rate", minimum=0, maximum=1)
    minimum_speedup = _required_policy_number(evaluation, benchmark_policy, "minimum_throughput_speedup_ratio", minimum=1)
    maximum_p95_ratio = _required_policy_number(evaluation, benchmark_policy, "maximum_gpu_p95_to_cpu_p95_ratio", minimum=0)
    minimum_utilization = _required_policy_number(
        evaluation, benchmark_policy, "minimum_sustained_gpu_utilization_fraction", minimum=0, maximum=1
    )
    maximum_utilization = _required_policy_number(
        evaluation, benchmark_policy, "maximum_sustained_gpu_utilization_fraction", minimum=0, maximum=1
    )
    minimum_headroom = _required_policy_number(evaluation, benchmark_policy, "minimum_gpu_memory_headroom_fraction", minimum=0, maximum=1)
    evaluation.require(
        "policy_utilization_range",
        minimum_utilization is not None and maximum_utilization is not None and minimum_utilization < maximum_utilization,
        "minimum sustained utilization must be below maximum sustained utilization",
    )

    cpu = _mapping(benchmark.get("cpu"))
    gpu = _mapping(benchmark.get("gpu"))
    cpu_duration = _finite_number(cpu.get("duration_seconds"))
    gpu_duration = _finite_number(gpu.get("duration_seconds"))
    evaluation.require(
        "minimum_cpu_benchmark_duration",
        minimum_duration is not None and cpu_duration is not None and cpu_duration >= minimum_duration,
        "CPU benchmark duration is missing or too short",
    )
    evaluation.require(
        "minimum_gpu_benchmark_duration",
        minimum_duration is not None and gpu_duration is not None and gpu_duration >= minimum_duration,
        "GPU benchmark duration is missing or too short",
    )
    cpu_samples = _non_negative_int(cpu.get("sample_count"))
    gpu_samples = _non_negative_int(gpu.get("sample_count"))
    evaluation.require(
        "minimum_cpu_sample_count",
        minimum_cpu_samples is not None and cpu_samples is not None and cpu_samples >= minimum_cpu_samples,
        "CPU timed sample count is missing or below policy",
    )
    evaluation.require(
        "minimum_gpu_sample_count",
        minimum_gpu_samples is not None and gpu_samples is not None and gpu_samples >= minimum_gpu_samples,
        "GPU timed sample count is missing or below policy",
    )

    parity = _mapping(benchmark.get("parity"))
    parity_samples = _non_negative_int(parity.get("sample_count"))
    probability_error = _finite_number(parity.get("max_absolute_probability_error"))
    decision_mismatches = _non_negative_int(parity.get("policy_decision_mismatches"))
    evaluation.require(
        "minimum_parity_sample_count",
        minimum_parity_samples is not None and parity_samples is not None and parity_samples >= minimum_parity_samples,
        "CPU-to-GPU parity sample count is missing or below policy",
    )
    evaluation.require(
        "probability_parity",
        maximum_probability_error is not None and probability_error is not None and probability_error <= maximum_probability_error,
        "CPU-to-GPU maximum absolute probability error exceeds policy",
    )
    evaluation.require(
        "policy_decision_parity",
        maximum_decision_mismatches is not None and decision_mismatches is not None and decision_mismatches <= maximum_decision_mismatches,
        "CPU-to-GPU policy decisions differ",
    )

    http_requests = _non_negative_int(gpu.get("http_requests"))
    http_errors = _non_negative_int(gpu.get("http_errors"))
    reported_error_rate = _finite_number(gpu.get("http_error_rate"))
    derived_error_rate = http_errors / http_requests if http_requests and http_errors is not None else None
    evaluation.require(
        "http_request_accounting",
        http_requests is not None
        and http_errors is not None
        and http_requests > 0
        and http_errors <= http_requests
        and gpu_samples is not None
        and http_requests >= gpu_samples,
        "HTTP request/error counts are invalid or do not cover the GPU samples",
    )
    evaluation.require(
        "http_error_rate_consistent",
        derived_error_rate is not None
        and reported_error_rate is not None
        and math.isclose(reported_error_rate, derived_error_rate, rel_tol=0.0, abs_tol=1e-12),
        "reported HTTP error rate does not equal errors divided by requests",
    )
    evaluation.require(
        "http_error_rate_slo",
        maximum_http_error_rate is not None and derived_error_rate is not None and derived_error_rate <= maximum_http_error_rate,
        "GPU HTTP error rate exceeds policy",
    )

    cpu_throughput = _finite_number(cpu.get("throughput_requests_per_second"))
    gpu_throughput = _finite_number(gpu.get("throughput_requests_per_second"))
    cpu_p95 = _finite_number(cpu.get("p95_latency_ms"))
    gpu_p95 = _finite_number(gpu.get("p95_latency_ms"))
    comparison = _mapping(benchmark.get("comparison"))
    reported_speedup = _finite_number(comparison.get("throughput_speedup_ratio"))
    reported_p95_ratio = _finite_number(comparison.get("gpu_p95_to_cpu_p95_ratio"))
    derived_speedup = gpu_throughput / cpu_throughput if cpu_throughput and gpu_throughput is not None else None
    derived_p95_ratio = gpu_p95 / cpu_p95 if cpu_p95 and gpu_p95 is not None else None
    evaluation.require(
        "positive_performance_measurements",
        all(value is not None and value > 0 for value in (cpu_throughput, gpu_throughput, cpu_p95, gpu_p95)),
        "CPU and GPU throughput and p95 measurements must be finite and positive",
    )
    evaluation.require(
        "throughput_speedup_consistent",
        derived_speedup is not None
        and reported_speedup is not None
        and math.isclose(reported_speedup, derived_speedup, rel_tol=1e-6, abs_tol=1e-9),
        "reported throughput speedup does not match raw CPU/GPU throughput",
    )
    evaluation.require(
        "throughput_speedup_slo",
        minimum_speedup is not None and derived_speedup is not None and derived_speedup >= minimum_speedup,
        "GPU throughput speedup is below policy",
    )
    evaluation.require(
        "p95_ratio_consistent",
        derived_p95_ratio is not None
        and reported_p95_ratio is not None
        and math.isclose(reported_p95_ratio, derived_p95_ratio, rel_tol=1e-6, abs_tol=1e-9),
        "reported p95 ratio does not match raw CPU/GPU latency",
    )
    evaluation.require(
        "p95_ratio_slo",
        maximum_p95_ratio is not None and derived_p95_ratio is not None and derived_p95_ratio <= maximum_p95_ratio,
        "GPU-to-CPU p95 ratio exceeds policy",
    )

    telemetry_samples = _non_negative_int(gpu.get("telemetry_sample_count"))
    utilization = _finite_number(gpu.get("sustained_gpu_utilization_fraction"))
    evaluation.require(
        "minimum_gpu_telemetry_samples",
        minimum_gpu_samples is not None and telemetry_samples is not None and telemetry_samples >= minimum_gpu_samples,
        "GPU utilization telemetry sample count is below policy",
    )
    evaluation.require(
        "sustained_gpu_utilization_range",
        minimum_utilization is not None
        and maximum_utilization is not None
        and utilization is not None
        and minimum_utilization <= utilization <= maximum_utilization,
        "sustained GPU utilization is outside the governed efficiency/headroom range",
    )
    gpu_memory_used = _finite_number(gpu.get("peak_gpu_memory_used_mib"))
    gpu_memory_total = _finite_number(gpu.get("gpu_memory_total_mib"))
    reported_headroom = _finite_number(gpu.get("gpu_memory_headroom_fraction"))
    derived_headroom = (
        (gpu_memory_total - gpu_memory_used) / gpu_memory_total
        if gpu_memory_total is not None
        and gpu_memory_total > 0
        and gpu_memory_used is not None
        and 0 <= gpu_memory_used <= gpu_memory_total
        else None
    )
    evaluation.require(
        "gpu_memory_matches_hardware",
        gpu_memory_total is not None
        and hardware_memory is not None
        and math.isclose(gpu_memory_total, hardware_memory, rel_tol=0.0, abs_tol=1.0),
        "benchmark GPU memory total does not match nvidia-smi hardware metadata",
    )
    evaluation.require(
        "gpu_memory_headroom_consistent",
        derived_headroom is not None
        and reported_headroom is not None
        and math.isclose(reported_headroom, derived_headroom, rel_tol=1e-6, abs_tol=1e-9),
        "reported GPU memory headroom does not match used/total memory",
    )
    evaluation.require(
        "gpu_memory_headroom_slo",
        minimum_headroom is not None and derived_headroom is not None and derived_headroom >= minimum_headroom,
        "peak GPU memory leaves insufficient policy headroom",
    )

    artifact_summary = _validate_artifacts(
        evaluation,
        manifest,
        policy,
        checksums,
        pathlib.Path(artifact_root),
    )

    decision = "GPU_ELIGIBLE" if not evaluation.reasons else "GPU_REJECTED"
    qualification_scope_valid = all(
        evaluation.checks.get(name, False)
        for name in (
            "qualification_policy_configured",
            "neural_accelerator_qualification_workload",
            "candidate_gpu_profile_scope",
            "gpu_candidate_model_family",
            "current_tree_champion_excluded",
        )
    )
    return {
        "schema_version": "regulated-ml-platform.roihu-gpu-decision/v1",
        "decision": decision,
        "status": "PASS" if decision == "GPU_ELIGIBLE" else "FAIL",
        "gpu_profile_enabled": decision == "GPU_ELIGIBLE",
        "source_commit": manifest_sha if isinstance(manifest_sha, str) else None,
        "accelerator_product": "nvidia-gh200" if decision == "GPU_ELIGIBLE" else None,
        "qualified_workload": {
            "verified": decision == "GPU_ELIGIBLE" and qualification_scope_valid,
            "kind": qualified_workload.get("kind") if qualification_scope_valid else None,
            "model_family": qualified_workload.get("model_family") if qualification_scope_valid else None,
            "profile": qualified_workload.get("profile") if qualification_scope_valid else None,
            "applies_to_current_tree_champion": False if qualification_scope_valid else None,
        },
        "deployment_effects": {
            "automatic_promotion_authorized": False,
            "current_model_accelerator": "unchanged",
            "current_model_profile": "unchanged",
        },
        "checks": evaluation.checks,
        "reasons": evaluation.reasons,
        "derived": {
            "http_error_rate": derived_error_rate,
            "throughput_speedup_ratio": derived_speedup,
            "gpu_p95_to_cpu_p95_ratio": derived_p95_ratio,
            "gpu_memory_headroom_fraction": derived_headroom,
        },
        "artifacts": artifact_summary,
        "runtime_evidence_string_trusted": False,
        "claim_boundary": (
            "GPU eligibility applies only to the exact source commit, immutable artifacts, Roihu GH200 job, "
            "software versions, and benchmark inputs validated here; it is not production approval."
        ),
    }


def _load_mapping(path: pathlib.Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON/YAML mapping")
    return value


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate fail-closed CSC Roihu GH200 evidence")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--checksums", required=True)
    parser.add_argument("--artifact-root", help="root for manifest artifact paths; defaults to manifest directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    output = pathlib.Path(args.output)
    try:
        manifest_path = pathlib.Path(args.manifest)
        manifest = _load_mapping(manifest_path)
        benchmark = _load_mapping(pathlib.Path(args.benchmark))
        policy = _load_mapping(pathlib.Path(args.policy))
        checksums = parse_checksums(pathlib.Path(args.checksums).read_text(encoding="utf-8"))
        artifact_root = pathlib.Path(args.artifact_root) if args.artifact_root else manifest_path.parent
        report = validate_roihu_gpu_evidence(manifest, benchmark, policy, checksums, artifact_root)
    except Exception as exc:  # fail closed while still producing a machine-readable decision
        report = {
            "schema_version": "regulated-ml-platform.roihu-gpu-decision/v1",
            "decision": "GPU_REJECTED",
            "status": "FAIL",
            "gpu_profile_enabled": False,
            "accelerator_product": None,
            "qualified_workload": {
                "verified": False,
                "kind": None,
                "model_family": None,
                "profile": None,
                "applies_to_current_tree_champion": None,
            },
            "deployment_effects": {
                "automatic_promotion_authorized": False,
                "current_model_accelerator": "unchanged",
                "current_model_profile": "unchanged",
            },
            "checks": {"inputs_loaded": False},
            "reasons": [f"inputs_loaded: {type(exc).__name__}: {exc}"],
            "runtime_evidence_string_trusted": False,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["decision"] == "GPU_ELIGIBLE" else 2


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
