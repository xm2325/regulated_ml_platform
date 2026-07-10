from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_pack(reports_dir: Path) -> dict[str, Any]:
    metrics = _load(reports_dir / "model_metrics.json")
    gate = _load(reports_dir / "promotion_gate.json")
    quality = _load(reports_dir / "data_quality_report.json")
    privacy = _load(reports_dir / "privacy_report.json")
    drift = _load(reports_dir / "drift_summary.json")
    load = _load(reports_dir / "load_test_summary.json")
    deployment = _load(reports_dir / "deployment_validation.json")
    incident = _load(reports_dir / "incident_drill_report.json")
    controls = {"promotion": gate.get("status", "MISSING"), "data_quality": quality.get("status", "MISSING"), "privacy": privacy.get("status", "MISSING"), "drift": "PASS" if drift.get("overall_status") == "ok" else "REVIEW", "load_slo": "PASS" if load.get("slo_pass") else "REVIEW", "deployment": deployment.get("status", "MISSING"), "incident": incident.get("status", "MISSING")}
    overall = "PASS" if all(value == "PASS" for value in controls.values()) else "REVIEW"
    best = metrics.get("best_model", "unknown")
    result = metrics.get("models", {}).get(best, {})
    return {"overall_status": overall, "release_decision": "eligible_for_controlled_demo_release" if overall == "PASS" else "hold_for_review", "model_version": metrics.get("model_version", "unknown"), "policy_version": metrics.get("metadata", {}).get("policy_version", "unknown"), "feature_schema_version": metrics.get("metadata", {}).get("feature_schema_version", "unknown"), "best_model": best, "control_statuses": controls, "key_metrics": {"auc": result.get("auc"), "brier": result.get("brier"), "expected_calibration_error": result.get("expected_calibration_error"), "precision_at_policy_threshold": result.get("precision_at_policy_threshold"), "load_test_p95_ms": load.get("latency_ms_p95")}, "evaluation_design": {"model_selection": metrics.get("metadata", {}).get("model_selection_split"), "calibration": metrics.get("metadata", {}).get("calibration_split"), "threshold_selection": metrics.get("metadata", {}).get("threshold_selection_split"), "final_evaluation": metrics.get("metadata", {}).get("final_evaluation_split")}}


def write_markdown(pack: dict[str, Any], output: Path) -> None:
    lines = ["# Release approval pack", "", f"## Release decision: **{pack['overall_status']}**", "", f"Recommendation: `{pack['release_decision']}`  ", f"Model: `{pack['best_model']} / {pack['model_version']}`  ", f"Policy: `{pack['policy_version']}`  ", f"Feature schema: `{pack['feature_schema_version']}`", "", "## Control status", "", "| Control | Status |", "|---|---|"]
    lines.extend(f"| {name} | {status} |" for name, status in pack["control_statuses"].items())
    lines.extend(["", "## Key evidence", "", "| Metric | Value |", "|---|---:|"])
    lines.extend(f"| {name} | {value:.4f} |" if isinstance(value, float) else f"| {name} | {value} |" for name, value in pack["key_metrics"].items())
    lines.extend(["", "## Boundary", "", "This synthetic release still requires independent validation, security review, data-protection review, model-risk approval, and accountable business sign-off before any real use."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output-json", default="reports/release_approval_pack.json")
    parser.add_argument("--output-md", default="docs/release_approval_pack.md")
    args = parser.parse_args()
    pack = build_pack(Path(args.reports_dir))
    Path(args.output_json).write_text(json.dumps(pack, indent=2), encoding="utf-8")
    write_markdown(pack, Path(args.output_md))
    print(json.dumps(pack, indent=2))


if __name__ == "__main__":
    main()
