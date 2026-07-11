from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_release_consistency import (
    ConsistencyError,
    check_generated_artifacts,
    check_source_consistency,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _artifact_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    version = "0.8.0"
    source_sha = "a" * 40
    model_dir = tmp_path / "models"
    evidence_dir = tmp_path / "evidence"
    site_dir = tmp_path / "site"

    metadata = {
        "model_version": version,
        "git_commit": source_sha[:7],
        "created_at": "2026-07-11T12:00:00+00:00",
        "policy_version": "policy-v1",
        "feature_schema_version": "features-v1",
    }
    _write_json(model_dir / "metadata.json", metadata)
    _write_json(
        model_dir / "model_contract.json",
        {
            "model_version": version,
            "policy_version": "policy-v1",
            "feature_schema_version": "features-v1",
        },
    )
    _write_json(
        evidence_dir / "reports/model_metrics.json",
        {
            "model_version": version,
            "trained_at": metadata["created_at"],
            "best_model": "champion",
            "models": {"champion": {"auc": 0.8}},
            "metadata": metadata,
        },
    )
    _write_json(evidence_dir / "reports/promotion_gate.json", {"status": "PASS"})
    _write_json(
        evidence_dir / "reports/sbom.cdx.json",
        {"metadata": {"component": {"version": version}}},
    )
    _write_json(
        evidence_dir / "reports/release_approval_pack.json",
        {
            "model_version": version,
            "overall_status": "PASS",
            "control_statuses": {"promotion": "PASS"},
        },
    )
    _write_json(site_dir / "data/evidence.json", {"model_version": version, "release_status": "PASS"})

    for relative in (
        "reports/model_evaluation.md",
        "reports/calibration_report.md",
        "reports/fairness_report.md",
        "reports/deployment_validation.md",
        "docs/reproducibility_manifest.md",
    ):
        path = evidence_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated evidence\n", encoding="utf-8")
    _write_json(evidence_dir / "reports/reproducibility_manifest.json", {"status": "PASS"})
    for relative in ("docs/model_contract.md", "docs/release_approval_pack.md"):
        path = evidence_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"generated model version {version}\n", encoding="utf-8")
    (site_dir / "index.html").write_text(f"<p>Version {version}</p>", encoding="utf-8")

    return model_dir, evidence_dir, site_dir, source_sha


def test_source_release_versions_are_consistent() -> None:
    assert check_source_consistency(ROOT) == "0.8.0"


def test_generated_evidence_matches_version_and_source_commit(tmp_path: Path) -> None:
    model_dir, evidence_dir, site_dir, source_sha = _artifact_fixture(tmp_path)

    check_generated_artifacts(
        version="0.8.0",
        source_sha=source_sha,
        model_dir=model_dir,
        evidence_dir=evidence_dir,
        site_dir=site_dir,
    )


def test_generated_evidence_rejects_another_commit(tmp_path: Path) -> None:
    model_dir, evidence_dir, site_dir, _ = _artifact_fixture(tmp_path)

    with pytest.raises(ConsistencyError, match="not the release source"):
        check_generated_artifacts(
            version="0.8.0",
            source_sha="b" * 40,
            model_dir=model_dir,
            evidence_dir=evidence_dir,
            site_dir=site_dir,
        )


def test_generated_evidence_rejects_stale_sbom_version(tmp_path: Path) -> None:
    model_dir, evidence_dir, site_dir, source_sha = _artifact_fixture(tmp_path)
    _write_json(
        evidence_dir / "reports/sbom.cdx.json",
        {"metadata": {"component": {"version": "0.6.0"}}},
    )

    with pytest.raises(ConsistencyError, match="SBOM application version"):
        check_generated_artifacts(
            version="0.8.0",
            source_sha=source_sha,
            model_dir=model_dir,
            evidence_dir=evidence_dir,
            site_dir=site_dir,
        )
