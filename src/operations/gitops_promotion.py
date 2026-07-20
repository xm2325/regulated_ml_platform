from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def build_release_identity(
    image_digest: str,
    git_commit: str,
    model_metadata: dict[str, Any],
) -> dict[str, str]:
    identity = {
        "image_digest": image_digest,
        "git_commit": git_commit,
        "model_release_version": str(model_metadata["model_version"]),
        "policy_version": str(model_metadata["policy_version"]),
        "feature_schema_version": str(model_metadata["feature_schema_version"]),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["release_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    return identity


def evaluate_promotion(
    source_environment: str,
    target_environment: str,
    release_identity: dict[str, str],
    checks: dict[str, str],
    policy: dict[str, Any],
    approval_status: str = "pending",
    source_release_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    transition_name = f"{source_environment}->{target_environment}"
    transition = policy.get("transitions", {}).get(transition_name)
    reasons: list[str] = []

    if transition is None:
        reasons.append(f"transition {transition_name} is not allowed")
        return {"status": "BLOCKED", "transition": transition_name, "reasons": reasons}

    if policy["controls"].get("require_immutable_image_digest", True) and not IMAGE_DIGEST_RE.fullmatch(
        release_identity.get("image_digest", "")
    ):
        reasons.append("image reference is not an immutable sha256 digest")

    if source_release_identity is not None and policy["controls"].get(
        "require_same_release_identity_between_environments", True
    ):
        for field in [
            "image_digest",
            "git_commit",
            "model_release_version",
            "policy_version",
            "feature_schema_version",
            "release_id",
        ]:
            if source_release_identity.get(field) != release_identity.get(field):
                reasons.append(f"release identity changed between environments: {field}")

    missing_or_failed_checks = [
        name for name in transition.get("required_checks", []) if checks.get(name) != "PASS"
    ]
    if missing_or_failed_checks:
        reasons.append("required checks are not PASS: " + ", ".join(missing_or_failed_checks))

    if transition.get("manual_approval_required", False) and approval_status != "approved":
        reasons.append("manual production approval is required")

    if target_environment == "prod" and policy["controls"].get("production_auto_promotion_allowed", False):
        reasons.append("invalid policy: production auto-promotion must remain disabled")

    return {
        "status": "READY" if not reasons else "BLOCKED",
        "transition": transition_name,
        "release_identity": release_identity,
        "required_checks": transition.get("required_checks", []),
        "checks": checks,
        "approval_status": approval_status,
        "manual_approval_required": bool(transition.get("manual_approval_required", False)),
        "reasons": reasons or ["same immutable release identity and all environment gates passed"],
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    reasons = "\n".join(f"- {reason}" for reason in report["reasons"])
    identity = report.get("release_identity", {})
    output.write_text(
        "# Environment promotion decision\n\n"
        f"**Transition:** `{report['transition']}`  \n"
        f"**Status:** `{report['status']}`\n\n"
        "## Immutable release identity\n\n"
        f"- release ID: `{identity.get('release_id', 'n/a')}`\n"
        f"- image digest: `{identity.get('image_digest', 'n/a')}`\n"
        f"- git commit: `{identity.get('git_commit', 'n/a')}`\n"
        f"- model release: `{identity.get('model_release_version', 'n/a')}`\n"
        f"- policy: `{identity.get('policy_version', 'n/a')}`\n"
        f"- feature schema: `{identity.get('feature_schema_version', 'n/a')}`\n\n"
        "## Decision reasons\n\n"
        f"{reasons}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--model-metadata", default="models/metadata.json")
    parser.add_argument("--checks", required=True)
    parser.add_argument("--policy", default="config/environment_promotion.yaml")
    parser.add_argument("--approval-status", default="pending", choices=["pending", "approved", "rejected"])
    parser.add_argument("--source-release-identity")
    parser.add_argument("--output-json", default="reports/environment_promotion.json")
    parser.add_argument("--output-md", default="reports/environment_promotion.md")
    args = parser.parse_args()

    metadata = json.loads(Path(args.model_metadata).read_text(encoding="utf-8"))
    checks = json.loads(Path(args.checks).read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
    source_identity = None
    if args.source_release_identity:
        source_identity = json.loads(Path(args.source_release_identity).read_text(encoding="utf-8"))
    identity = build_release_identity(args.image_digest, args.git_commit, metadata)
    report = evaluate_promotion(
        args.source,
        args.target,
        identity,
        checks,
        policy,
        args.approval_status,
        source_identity,
    )
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report, indent=2))
    if report["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
