from __future__ import annotations

from pathlib import Path

import yaml

from src.operations.gitops_promotion import build_release_identity, evaluate_promotion

POLICY = yaml.safe_load(Path("config/environment_promotion.yaml").read_text(encoding="utf-8"))
METADATA = {
    "model_version": "0.6.0",
    "policy_version": "targeted-support-policy-v3",
    "feature_schema_version": "financial_customer_features_v4",
}
DIGEST = "sha256:" + "a" * 64


def _checks(names: list[str]) -> dict[str, str]:
    return {name: "PASS" for name in names}


def test_dev_to_preprod_accepts_same_immutable_release() -> None:
    identity = build_release_identity(DIGEST, "abc1234", METADATA)
    required = POLICY["transitions"]["dev->preprod"]["required_checks"]
    report = evaluate_promotion("dev", "preprod", identity, _checks(required), POLICY, source_release_identity=identity)
    assert report["status"] == "READY"


def test_mutable_or_non_digest_image_is_blocked() -> None:
    identity = build_release_identity("ghcr.io/example/app:latest", "abc1234", METADATA)
    required = POLICY["transitions"]["dev->preprod"]["required_checks"]
    report = evaluate_promotion("dev", "preprod", identity, _checks(required), POLICY)
    assert report["status"] == "BLOCKED"
    assert any("immutable" in reason for reason in report["reasons"])


def test_environment_rebuild_is_blocked() -> None:
    source = build_release_identity(DIGEST, "abc1234", METADATA)
    rebuilt = build_release_identity("sha256:" + "b" * 64, "abc1234", METADATA)
    required = POLICY["transitions"]["dev->preprod"]["required_checks"]
    report = evaluate_promotion("dev", "preprod", rebuilt, _checks(required), POLICY, source_release_identity=source)
    assert report["status"] == "BLOCKED"
    assert any("image_digest" in reason for reason in report["reasons"])


def test_prod_requires_manual_approval_even_when_all_checks_pass() -> None:
    identity = build_release_identity(DIGEST, "abc1234", METADATA)
    required = POLICY["transitions"]["preprod->prod"]["required_checks"]
    report = evaluate_promotion(
        "preprod",
        "prod",
        identity,
        _checks(required),
        POLICY,
        approval_status="pending",
        source_release_identity=identity,
    )
    assert report["status"] == "BLOCKED"
    assert "manual production approval is required" in report["reasons"]


def test_prod_can_be_ready_only_after_explicit_approval() -> None:
    identity = build_release_identity(DIGEST, "abc1234", METADATA)
    required = POLICY["transitions"]["preprod->prod"]["required_checks"]
    report = evaluate_promotion(
        "preprod",
        "prod",
        identity,
        _checks(required),
        POLICY,
        approval_status="approved",
        source_release_identity=identity,
    )
    assert report["status"] == "READY"
