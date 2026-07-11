from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)
REQUEST = {"customer_id": "C_TEST", "age": 45, "annual_income": 52000, "cash_balance": 24000, "investment_balance": 15000, "debt_balance": 3500, "risk_score": 0.45, "recent_activity_count": 5, "account_type": "isa", "employment_status": "employed"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "policy_threshold" in response.json()


def test_version_endpoint():
    assert client.get("/version").json()["service_version"] == "0.8.0"


def test_predict_endpoint():
    response = client.post("/predict", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "C_TEST"
    assert 0 <= body["support_probability"] <= 1
    assert body["recommended_action"] in {"no_support", "cash_buffer_warning", "investment_support", "risk_review"}
    assert body["decision_id"].startswith("decision_")
    assert body["policy_version"] == "targeted-support-policy-v3"
    assert body["feature_schema_version"] == "financial_customer_features_v4"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_request_count" in response.text


def test_shadow_predict_endpoint():
    response = client.post("/shadow-predict", json=REQUEST)
    assert response.status_code == 200
    assert response.json()["status"] in {"candidate_available", "candidate_unavailable"}
