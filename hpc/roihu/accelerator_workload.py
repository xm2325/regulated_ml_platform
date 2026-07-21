"""Deterministic synthetic workload used only for accelerator qualification.

This module deliberately does not import PyTorch at module import time.  The Roihu
batch scripts load CSC's pinned ``python-pytorch/2.10`` module before invoking the
benchmark or exporter.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

WORKLOAD_NAME = "regulated_ml_platform_accelerator_qualification_mlp"
WORKLOAD_CLASSIFICATION = "synthetic_accelerator_qualification_only"
INPUT_NAME = "INPUT__0"
OUTPUT_NAME = "OUTPUT__0"
INPUT_FEATURES = 1024
HIDDEN_FEATURES = (2048, 2048)
OUTPUT_FEATURES = 512
MAX_BATCH_SIZE = 256
DEFAULT_BATCH_SIZES = (1, 16, 64, 256)
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def validate_sha256(value: str, *, label: str) -> str:
    """Return a normalized SHA-256 or fail closed."""

    normalized = value.strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{label} must be exactly 64 hexadecimal characters")
    return normalized


def validate_source_commit(value: str) -> str:
    """Return a normalized full Git commit or fail closed."""

    normalized = value.strip().lower()
    if not SOURCE_COMMIT_PATTERN.fullmatch(normalized):
        raise ValueError("source commit must be a full 40-character hexadecimal Git commit")
    return normalized


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading the whole artifact into memory."""

    if not path.is_file():
        raise ValueError(f"required artifact is not a regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_archive(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify and describe the immutable source archive."""

    expected = validate_sha256(expected_sha256, label="expected source SHA-256")
    resolved = path.expanduser().resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError("source archive SHA-256 mismatch")
    return {
        "path": str(resolved),
        "name": resolved.name,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "size_bytes": resolved.stat().st_size,
        "digest_status": "PASS",
    }


def parse_batch_sizes(raw: str, *, maximum: int = MAX_BATCH_SIZE) -> list[int]:
    """Parse a small, bounded, duplicate-free batch-size set."""

    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("batch sizes must be comma-separated integers") from exc
    if not values:
        raise ValueError("at least one batch size is required")
    if len(values) > 8:
        raise ValueError("at most eight batch sizes are allowed")
    if len(values) != len(set(values)):
        raise ValueError("batch sizes must not contain duplicates")
    if any(value < 1 or value > maximum for value in values):
        raise ValueError(f"batch sizes must be between 1 and {maximum}")
    return values


def percentile(values: list[float], quantile: float) -> float:
    """Calculate a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("cannot calculate a percentile from no observations")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_model(torch: Any) -> Any:
    """Build the fixed MLP and initialize every parameter analytically.

    Analytical initialization avoids relying on a PyTorch RNG implementation, so
    the benchmark and ONNX exporter reconstruct the exact same source weights.
    """

    model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES[0]),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(HIDDEN_FEATURES[0], HIDDEN_FEATURES[1]),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(HIDDEN_FEATURES[1], OUTPUT_FEATURES),
    )
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            positions = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            phase = (parameter_index + 1) * 0.173
            scale = 0.018 if parameter.ndim > 1 else 0.004
            parameter.copy_(torch.sin(positions * 0.00037 + phase) * scale)
    return model.eval()


def make_input(torch: Any, batch_size: int) -> Any:
    """Construct deterministic, bounded synthetic feature rows."""

    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
    positions = torch.arange(batch_size * INPUT_FEATURES, dtype=torch.float32).reshape(batch_size, INPUT_FEATURES)
    return torch.sin(positions * 0.0017) + 0.25 * torch.cos(positions * 0.00031)


def workload_fingerprint(model: Any) -> str:
    """Hash tensor names, shapes, dtypes, and bytes in a stable order."""

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def workload_contract(model: Any) -> dict[str, Any]:
    """Return the claim-bounded workload identity embedded in every artifact."""

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "name": WORKLOAD_NAME,
        "classification": WORKLOAD_CLASSIFICATION,
        "production_model": False,
        "champion_model": False,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input_features": INPUT_FEATURES,
        "hidden_features": list(HIDDEN_FEATURES),
        "output_features": OUTPUT_FEATURES,
        "parameter_count": parameter_count,
        "maximum_batch_size": MAX_BATCH_SIZE,
        "weights_sha256": workload_fingerprint(model),
    }
