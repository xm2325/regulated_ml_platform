import json
from pathlib import Path

from src.data.make_dataset import generate_customers
from src.features.build_features import build_features
from src.governance.explainability_report import build_explainability_report
from src.governance.privacy_guard import inspect_columns
from src.models.champion_challenger import build_champion_challenger_report
from src.models.train import train
from src.operations.export_openapi import app
from src.operations.incident_drill import build_incident_drill


def test_privacy_guard_passes_generated_data():
    report = inspect_columns(generate_customers(n=50, seed=21).drop(columns=["support_needed", "true_support_probability"]))
    assert report["status"] == "PASS"


def test_privacy_guard_finds_direct_identifier():
    raw = generate_customers(n=10, seed=22).drop(columns=["support_needed", "true_support_probability"])
    raw["email"] = [f"u{i}@example.com" for i in range(len(raw))]
    assert "email" in inspect_columns(raw)["blocked_direct_identifier_hits"]


def test_champion_challenger_report_from_training(tmp_path: Path):
    features = build_features(generate_customers(n=500, seed=23))
    input_path = tmp_path / "features.csv"
    features.to_csv(input_path, index=False)
    report = build_champion_challenger_report(train(input_path, tmp_path / "models", tmp_path / "reports", random_state=23))
    assert report["status"] in {"PROMOTE_CHAMPION", "KEEP_CHAMPION", "REVIEW_REQUIRED"}
    assert len(report["all_candidates"]) >= 2


def test_explainability_report_runs(tmp_path: Path):
    features = build_features(generate_customers(n=500, seed=24))
    input_path = tmp_path / "features.csv"
    features.to_csv(input_path, index=False)
    train(input_path, tmp_path / "models", tmp_path / "reports", random_state=24)
    report = build_explainability_report(tmp_path / "models/model.joblib", input_path, 100)
    assert report["top_features"]


def test_incident_drill_reads_reports(tmp_path: Path):
    for name, content in {"load_test_summary.json": {"latency_ms_p95": 90}, "drift_summary.json": {"status": "ok"}, "promotion_gate.json": {"status": "PASS"}, "privacy_report.json": {"status": "PASS"}}.items():
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")
    assert build_incident_drill(tmp_path)["status"] == "PASS"


def test_openapi_contains_shadow_endpoint():
    assert "/shadow-predict" in app.openapi()["paths"]
