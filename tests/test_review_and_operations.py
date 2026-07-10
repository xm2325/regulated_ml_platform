from pathlib import Path

from fastapi.testclient import TestClient

from src.governance.change_approval_pack import build_pack
from src.operations.load_test import run_load_test
from src.serving.app import app

client = TestClient(app)
REQUEST = {"customer_id": "C_REVIEW", "age": 82, "annual_income": 30000, "cash_balance": 120000, "investment_balance": 20000, "debt_balance": 25000, "risk_score": 0.55, "recent_activity_count": 2, "account_type": "savings", "employment_status": "retired"}


def test_review_route_endpoint():
    response = client.post("/review-route", json=REQUEST)
    assert response.status_code == 200
    assert response.json()["review_route"] in {"auto_serve", "manual_review"}


def test_predict_includes_audit_and_review_route():
    body = client.post("/predict", json=REQUEST).json()
    assert body["audit_event_id"].startswith("audit_")
    assert body["review_route"] in {"auto_serve", "manual_review"}


def test_load_test_runs_small_sample():
    summary = run_load_test(n_requests=5, seed=101)
    assert summary["ok"] == 5
    assert "latency_ms_p95" in summary


def test_release_approval_pack_handles_existing_reports(tmp_path: Path):
    (tmp_path / "model_metrics.json").write_text('{"best_model":"m","models":{"m":{"auc":0.8,"brier":0.1}}}', encoding="utf-8")
    (tmp_path / "promotion_gate.json").write_text('{"status":"PASS"}', encoding="utf-8")
    pack = build_pack(tmp_path)
    assert pack["control_statuses"]["promotion"] == "PASS"
    assert pack["key_metrics"]["auc"] == 0.8
