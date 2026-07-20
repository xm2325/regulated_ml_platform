from __future__ import annotations

from pathlib import Path

import yaml

from src.operations.continuous_ops import decide_continuous_action

POLICY = yaml.safe_load(Path("config/continuous_ops.yaml").read_text(encoding="utf-8"))


def _clean_drift() -> dict[str, object]:
    return {
        "overall_status": "ok",
        "numeric": [{"feature": "age", "psi": 0.02, "status": "ok"}],
        "categorical": [{"feature": "account_type", "total_variation_distance": 0.01, "status": "ok"}],
    }


def test_clean_monitoring_does_not_retrain() -> None:
    report = decide_continuous_action(_clean_drift(), {"status": "PASS"}, POLICY)
    assert report["decision"] == "NO_TRAINING"
    assert report["controls"]["auto_promote"] is False


def test_material_drift_trains_challenger_only() -> None:
    drift = _clean_drift()
    drift["overall_status"] = "review"
    drift["numeric"] = [{"feature": "age", "psi": 0.31, "status": "review"}]
    report = decide_continuous_action(drift, {"status": "PASS"}, POLICY)
    assert report["decision"] == "TRAIN_CANDIDATE"
    assert report["controls"]["training_destination_alias"] == "challenger"
    assert report["controls"]["require_canary_before_champion"] is True
    assert report["controls"]["auto_promote"] is False


def test_bad_data_blocks_retraining_even_when_drift_is_high() -> None:
    drift = _clean_drift()
    drift["overall_status"] = "review"
    drift["numeric"] = [{"feature": "annual_income", "psi": 0.60, "status": "review"}]
    report = decide_continuous_action(drift, {"status": "FAIL"}, POLICY)
    assert report["decision"] == "BLOCKED_DATA_QUALITY"


def test_live_performance_failure_requires_investigation_not_blind_retraining() -> None:
    report = decide_continuous_action(
        _clean_drift(),
        {"status": "PASS"},
        POLICY,
        {"auc": 0.68, "brier": 0.28, "expected_calibration_error": 0.14, "selection_rate_gap": 0.21},
    )
    assert report["decision"] == "INVESTIGATE_MODEL"
    assert len(report["reasons"]) == 4
    assert report["controls"]["auto_promote"] is False
