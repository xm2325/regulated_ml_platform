from pathlib import Path

from src.operations.validate_alerting import validate_alert_rules


def test_all_alerts_have_actionable_runbooks() -> None:
    report = validate_alert_rules(Path("observability/prometheus/regulated-ai-alerts.yaml"), Path("."))
    assert report["status"] == "PASS"
    assert report["alert_count"] >= 7
    assert not report["failures"]
    assert all(alert["severity"] in {"page", "ticket"} for alert in report["alerts"])
    assert all(alert["runbook_url"].startswith("repo://") for alert in report["alerts"])
