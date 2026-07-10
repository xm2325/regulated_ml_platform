from pathlib import Path

from src.data.make_dataset import generate_customers
from src.features.build_features import build_features
from src.governance.promotion_gate import evaluate_gate
from src.models.train import train
from src.monitoring.data_quality import check_data_quality
from src.serving.batch_score import score_frame
from src.serving.predictor import ModelPredictor


def test_data_quality_passes_for_generated_data():
    frame = generate_customers(n=50, seed=11).drop(columns=["support_needed", "true_support_probability"])
    report = check_data_quality(frame)
    assert report["status"] == "PASS"
    assert report["rows"] == 50


def test_data_quality_fails_for_bad_category():
    frame = generate_customers(n=10, seed=12).drop(columns=["support_needed", "true_support_probability"])
    frame.loc[0, "account_type"] = "bad_type"
    report = check_data_quality(frame)
    assert report["status"] == "FAIL"
    assert report["category_violations"]["account_type"] == 1


def test_promotion_gate_returns_status(tmp_path: Path):
    features = build_features(generate_customers(n=500, seed=13))
    input_path = tmp_path / "features.csv"
    features.to_csv(input_path, index=False)
    metrics = train(input_path, tmp_path / "models", tmp_path / "reports", random_state=13)
    gate = evaluate_gate(metrics)
    assert gate["status"] in {"PASS", "REVIEW"}
    assert "auc" in gate["checks"]


def test_batch_scoring(tmp_path: Path):
    raw = generate_customers(n=500, seed=14)
    features = build_features(raw)
    input_path = tmp_path / "features.csv"
    features.to_csv(input_path, index=False)
    train(input_path, tmp_path / "models", tmp_path / "reports", random_state=14)
    sample = raw.drop(columns=["support_needed", "true_support_probability"]).head(5)
    predictor = ModelPredictor(model_path=str(tmp_path / "models" / "model.joblib"), metadata_path=str(tmp_path / "models" / "metadata.json"))
    scored = score_frame(sample, predictor=predictor)
    assert len(scored) == 5
    assert {"customer_id", "support_probability", "recommended_action", "model_version"}.issubset(scored.columns)
