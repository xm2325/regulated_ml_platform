from pathlib import Path

from src.data.make_dataset import generate_customers
from src.features.build_features import build_features
from src.models.train import train


def test_training_writes_artifacts(tmp_path: Path):
    features = build_features(generate_customers(n=400, seed=5))
    input_path = tmp_path / "features.csv"
    features.to_csv(input_path, index=False)
    metrics = train(input_path, tmp_path / "models", tmp_path / "reports", random_state=5)
    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "models" / "metadata.json").exists()
    assert (tmp_path / "reports" / "model_metrics.json").exists()
    best = metrics["best_model"]
    assert 0 <= metrics["models"][best]["brier"] <= 1
    assert 0 <= metrics["models"][best]["auc"] <= 1
    assert metrics["metadata"]["model_selection_split"] == "model_selection"
    assert metrics["metadata"]["calibration_split"] == "calibration"
    assert metrics["metadata"]["threshold_selection_split"] == "policy_validation"
    assert metrics["metadata"]["final_evaluation_split"] == "out_of_time_test"
    assert sum(metrics["split_counts"].values()) == 400
    assert metrics["split_ranges"]["train"]["end_date"] <= metrics["split_ranges"]["out_of_time_test"]["start_date"]
    assert metrics["calibration_comparison"]["method"] == "platt_scaling"
    assert "auc" in metrics["confidence_intervals"]
