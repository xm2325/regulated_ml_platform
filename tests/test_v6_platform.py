from pathlib import Path

from fastapi.testclient import TestClient

from src.core.telemetry import pseudonymous_customer_key
from src.data.make_dataset import generate_customers
from src.features.build_features import build_features
from src.governance.model_contract import build_contract
from src.governance.reproducibility_manifest import build_manifest
from src.models.train import train
from src.operations.validate_deployment import validate
from src.serving.app import app
from src.serving.policy import TargetedSupportPolicy
from src.serving.schemas import PredictionRequest

client = TestClient(app)
REQUEST = {"customer_id": "C_V6", "request_id": "request-v6-0001", "age": 45, "annual_income": 52000, "cash_balance": 24000, "investment_balance": 15000, "debt_balance": 3500, "risk_score": 0.45, "recent_activity_count": 5, "account_type": "isa", "employment_status": "employed"}


def test_decision_contract_endpoint():
    body = client.get("/decision-contract").json()
    assert body["contract_version"] == "decision-contract-v3"
    assert "audit_fields" in body
    assert "served_model_role" in body["audit_fields"]
    assert "comparison_challenger_registry_version" in body["audit_fields"]


def test_request_id_header_is_returned():
    response = client.post("/predict", json=REQUEST, headers={"X-Request-ID": "trace-12345678"})
    assert response.headers["X-Request-ID"] == "trace-12345678"
    assert response.headers["X-Service-Version"] == "1.2.0"


def test_unknown_request_field_is_rejected():
    assert client.post("/predict", json={**REQUEST, "email": "person@example.com"}).status_code == 422


def test_policy_hard_debt_gate():
    request = PredictionRequest(**{**REQUEST, "annual_income": 10000, "debt_balance": 9000})
    decision = TargetedSupportPolicy().decide(request, probability=0.2, threshold=0.5)
    assert decision.action == "risk_review"
    assert decision.hard_safety_gate_triggered is True


def test_customer_key_is_one_way_and_stable():
    first = pseudonymous_customer_key("customer-123")
    assert first == pseudonymous_customer_key("customer-123")
    assert "customer-123" not in first


def test_model_contract_uses_versioned_layers(tmp_path: Path):
    features = build_features(generate_customers(n=400, seed=55))
    input_path = tmp_path / "features.csv"
    features.to_csv(input_path, index=False)
    train(input_path, tmp_path / "models", tmp_path / "reports", random_state=55)
    import json
    metadata = json.loads((tmp_path / "models/metadata.json").read_text())
    contract = build_contract(metadata)
    assert contract["model_version"] == "0.6.0"
    assert contract["policy_version"] == "targeted-support-policy-v3"
    assert "financial advice" in contract["prohibited_uses"]


def test_reproducibility_manifest_hashes_artifacts(tmp_path: Path):
    (tmp_path / "requirements-runtime.lock").write_text("x", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    assert any(item["sha256"] for item in manifest["files"])


def test_kubernetes_deployment_controls_pass():
    report = validate(Path("k8s"))
    assert report["status"] == "PASS"
    assert report["checks"]["run_as_non_root"] is True
