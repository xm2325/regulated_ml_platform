from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_incident_drill(reports_dir: Path) -> dict[str, Any]:
    load = _load_json(reports_dir / "load_test_summary.json")
    drift = _load_json(reports_dir / "drift_summary.json")
    gate = _load_json(reports_dir / "promotion_gate.json")
    privacy = _load_json(reports_dir / "privacy_report.json")
    scenarios = [
        {"scenario": "latency_slo_breach", "trigger": "p95 latency exceeds 250 ms for 15 minutes", "evidence_in_demo": f"p95 latency = {load.get('latency_ms_p95', 'unknown')} ms", "first_action": "freeze release and scale model service replicas", "rollback_condition": "p95 remains above target after scaling or error rate rises"},
        {"scenario": "prediction_drift", "trigger": "drift report status is not ok", "evidence_in_demo": f"drift status = {drift.get('status', drift.get('overall_status', 'unknown'))}", "first_action": "route new model to shadow mode and request model-risk review", "rollback_condition": "high-impact segment drift is confirmed"},
        {"scenario": "promotion_gate_review", "trigger": "promotion gate returns REVIEW", "evidence_in_demo": f"promotion status = {gate.get('status', 'unknown')}", "first_action": "block production promotion and open approval ticket", "rollback_condition": "current production model is already exposed to the failing change"},
        {"scenario": "privacy_guard_failure", "trigger": "privacy report contains blocked direct identifiers", "evidence_in_demo": f"privacy status = {privacy.get('status', 'unknown')}", "first_action": "stop data export and remove blocked fields", "rollback_condition": "blocked fields reached a served endpoint or report artifact"},
    ]
    pass_count = sum("unknown" not in scenario["evidence_in_demo"] for scenario in scenarios)
    return {"created_at": datetime.now(timezone.utc).isoformat(), "status": "PASS" if pass_count >= 3 else "REVIEW", "scenarios_checked": len(scenarios), "scenarios_with_demo_evidence": pass_count, "scenarios": scenarios}


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = ["# Incident drill report", "", f"Created at: `{report['created_at']}`", f"Status: `{report['status']}`", f"Scenarios checked: `{report['scenarios_checked']}`", ""]
    for scenario in report["scenarios"]:
        lines.extend([f"## {scenario['scenario']}", "", f"Trigger: {scenario['trigger']}", f"Evidence in demo: {scenario['evidence_in_demo']}", f"First action: {scenario['first_action']}", f"Rollback condition: {scenario['rollback_condition']}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output-json", default="reports/incident_drill_report.json")
    parser.add_argument("--output-md", default="reports/incident_drill_report.md")
    args = parser.parse_args()
    report = build_incident_drill(Path(args.reports_dir))
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
