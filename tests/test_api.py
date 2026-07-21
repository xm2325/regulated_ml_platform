from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)
REQUEST = {"customer_id": "C_TEST", "age": 45, "annual_income": 52000, "cash_balance": 24000, "investment_balance": 15000, "debt_balance": 3500, "risk_score": 0.45, "recent_activity_count": 5, "account_type": "isa", "employment_status": "employed"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "policy_threshold" in body
    assert body["model_source"] in {"local", "registry"}
    assert "runtime_state" in body
    assert "canary_state" in body


def test_version_endpoint():
    body = client.get("/version").json()
    assert body["platform_version"] == "1.3.0"
    assert body["service_version"] == "1.3.0"
    assert body["model_release_version"] == "0.6.0"
    assert body["model_source"] in {"local", "registry"}
    assert body["canary_enabled"] is False
    assert body["canary_state"] == "disabled"


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
    assert body["model_source"] == "local"
    assert body["runtime_state"] == "ready_local"
    assert body["registry_model_version"] is None
    assert body["served_model_role"] == "champion"
    assert body["canary_state"] == "disabled"
    assert body["canary_assignment"] == "champion"


def test_runtime_model_endpoint():
    response = client.get("/runtime/model")
    assert response.status_code == 200
    body = response.json()
    assert body["requested_source"] == "local"
    assert body["active_source"] == "local"
    assert body["service_version"] == "1.3.0"
    assert body["model_release_version"] == "0.6.0"
    assert "last_reload_error" not in body


def test_canary_status_endpoint_is_disabled_by_default():
    response = client.get("/canary/status")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["state"] == "disabled"
    assert body["traffic_percent"] >= 0


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_request_count" in response.text
    assert "regulated_ai_registry_model_active" in response.text
    assert "regulated_ai_model_reload_successes" in response.text
    assert "regulated_ai_canary_state" in response.text
    assert "regulated_ai_canary_action_disagreement_rate" in response.text


def test_shadow_predict_endpoint():
    response = client.post("/shadow-predict", json=REQUEST)
    assert response.status_code == 200
    assert response.json()["status"] in {"candidate_available", "candidate_unavailable"}
