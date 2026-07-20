from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def validate_alert_rules(rules_path: Path, repo_root: Path) -> dict[str, Any]:
    document = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    alerts: list[dict[str, Any]] = []
    failures: list[str] = []
    names: set[str] = set()

    for group in document.get("groups", []):
        for rule in group.get("rules", []):
            name = str(rule.get("alert", ""))
            if not name:
                failures.append("rule without alert name")
                continue
            if name in names:
                failures.append(f"duplicate alert name: {name}")
            names.add(name)
            labels = rule.get("labels", {})
            annotations = rule.get("annotations", {})
            runbook = str(annotations.get("runbook_url", ""))
            if labels.get("severity") not in {"page", "ticket"}:
                failures.append(f"{name}: severity must be page or ticket")
            if not rule.get("for"):
                failures.append(f"{name}: missing sustained-duration 'for' field")
            if not annotations.get("summary"):
                failures.append(f"{name}: missing summary annotation")
            if not runbook.startswith("repo://"):
                failures.append(f"{name}: runbook_url must use repo:// for reviewable repository evidence")
            else:
                runbook_path = runbook.removeprefix("repo://").split("#", 1)[0]
                if not (repo_root / runbook_path).is_file():
                    failures.append(f"{name}: runbook path does not exist: {runbook_path}")
            expression = str(rule.get("expr", "")).strip()
            if not expression:
                failures.append(f"{name}: missing PromQL expression")
            alerts.append(
                {
                    "name": name,
                    "severity": labels.get("severity"),
                    "component": labels.get("component"),
                    "for": rule.get("for"),
                    "runbook_url": runbook,
                }
            )

    if not alerts:
        failures.append("no alert rules found")
    return {
        "status": "PASS" if not failures else "FAIL",
        "alert_count": len(alerts),
        "alerts": alerts,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="observability/prometheus/regulated-ai-alerts.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="reports/alerting_validation.json")
    args = parser.parse_args()
    report = validate_alert_rules(Path(args.rules), Path(args.repo_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
