from pathlib import Path

import numpy as np

from src.data.make_dataset import generate_customers
from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TIME_COLUMN, build_features
from src.models.calibration import PlattCalibratedClassifier
from src.models.train import make_models, temporal_split


def test_generated_data_has_ordered_observation_dates():
    raw = generate_customers(n=500, seed=77)
    assert TIME_COLUMN in raw.columns
    assert raw[TIME_COLUMN].is_monotonic_increasing


def test_temporal_split_is_strictly_ordered():
    splits = temporal_split(build_features(generate_customers(n=500, seed=78)))
    order = ["train", "model_selection", "calibration", "policy_validation", "out_of_time_test"]
    for left, right in zip(order[:-1], order[1:], strict=True):
        assert splits[left][TIME_COLUMN].max() <= splits[right][TIME_COLUMN].min()


def test_platt_calibrated_model_serializable(tmp_path: Path):
    splits = temporal_split(build_features(generate_customers(n=500, seed=79)))
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = make_models(79)["logistic_regression"]
    model.fit(splits["train"][columns], splits["train"]["support_needed"])
    calibrated = PlattCalibratedClassifier.fit_calibrator(model, splits["calibration"][columns], splits["calibration"]["support_needed"].to_numpy())
    probability = calibrated.predict_proba(splits["out_of_time_test"][columns])[:, 1]
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
