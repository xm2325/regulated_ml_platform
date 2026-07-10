from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.config import settings
from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, TIME_COLUMN
from src.models.calibration import PlattCalibratedClassifier
from src.models.evaluation import (
    POLICY_MIN_COVERAGE,
    POLICY_MIN_PRECISION,
    bootstrap_intervals,
    calibration_bins,
    choose_policy_threshold,
    compute_metrics,
    fairness_summary,
    log_with_mlflow_or_fallback,
    split_ranges,
    temporal_split,
    threshold_table,
    write_calibration_report,
    write_fairness_report,
    write_model_evaluation,
)

MODEL_VERSION = settings.service_version


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=18, random_state=random_state, class_weight="balanced_subsample", n_jobs=-1)),
            ]
        ),
    }


def train(input_path: Path, model_dir: Path, reports_dir: Path, random_state: int = 42) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path)
    splits = temporal_split(frame)
    split_info = split_ranges(splits)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    x_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET].astype(int).to_numpy()
    x_selection = splits["model_selection"][feature_columns]
    y_selection = splits["model_selection"][TARGET].astype(int).to_numpy()
    x_calibration = splits["calibration"][feature_columns]
    y_calibration = splits["calibration"][TARGET].astype(int).to_numpy()
    x_policy = splits["policy_validation"][feature_columns]
    y_policy = splits["policy_validation"][TARGET].astype(int).to_numpy()
    x_test = splits["out_of_time_test"][feature_columns]
    y_test = splits["out_of_time_test"][TARGET].astype(int).to_numpy()

    trained_models = make_models(random_state=random_state)
    selection_metrics: dict[str, dict[str, float]] = {}
    test_raw_probabilities: dict[str, np.ndarray] = {}
    for name, model in trained_models.items():
        model.fit(x_train, y_train)
        selection_probability = model.predict_proba(x_selection)[:, 1]
        selection_metrics[name] = compute_metrics(y_selection, selection_probability)
        test_raw_probabilities[name] = model.predict_proba(x_test)[:, 1]

    best_name = max(selection_metrics, key=lambda name: (selection_metrics[name]["auc"], -selection_metrics[name]["brier"]))
    calibrated_model = PlattCalibratedClassifier.fit_calibrator(trained_models[best_name], x_calibration, y_calibration)
    policy_probability = calibrated_model.predict_proba(x_policy)[:, 1]
    policy_thresholds = threshold_table(y_policy, policy_probability)
    policy_threshold = choose_policy_threshold(policy_thresholds)

    calibrated_test_probability = calibrated_model.predict_proba(x_test)[:, 1]
    raw_champion_test_probability = test_raw_probabilities[best_name]
    model_metrics = {
        name: compute_metrics(y_test, probability, threshold=policy_threshold if name == best_name else 0.5)
        for name, probability in test_raw_probabilities.items()
    }
    model_metrics[best_name] = compute_metrics(y_test, calibrated_test_probability, threshold=policy_threshold)

    model_path = model_dir / "model.joblib"
    joblib.dump(calibrated_model, model_path)
    candidate_dir = model_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for old_candidate in candidate_dir.glob("*.joblib"):
        old_candidate.unlink()
    for candidate_name, candidate_model in trained_models.items():
        if candidate_name != best_name:
            joblib.dump(candidate_model, candidate_dir / f"{candidate_name}.joblib")

    metadata = {
        "model_name": best_name,
        "model_variant": "platt_calibrated",
        "model_version": MODEL_VERSION,
        "policy_version": settings.policy_version,
        "feature_schema_version": settings.feature_schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "time_column": TIME_COLUMN,
        "threshold": policy_threshold,
        "model_selection_split": "model_selection",
        "calibration_split": "calibration",
        "threshold_selection_split": "policy_validation",
        "final_evaluation_split": "out_of_time_test",
        "policy_min_precision": POLICY_MIN_PRECISION,
        "policy_min_coverage": POLICY_MIN_COVERAGE,
        "split_counts": {name: summary["n"] for name, summary in split_info.items()},
        "split_ranges": split_info,
        "calibration": {
            "method": "platt_scaling",
            "intercept": calibrated_model.calibration_intercept,
            "slope": calibrated_model.calibration_slope,
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fairness = fairness_summary(splits["out_of_time_test"], y_test, calibrated_test_probability, policy_threshold)
    calibrated_bins = calibration_bins(y_test, calibrated_test_probability)
    intervals = bootstrap_intervals(y_test, calibrated_test_probability, policy_threshold, random_state=random_state)
    raw_metrics = compute_metrics(y_test, raw_champion_test_probability, threshold=policy_threshold)
    calibrated_metrics = model_metrics[best_name]
    pd.DataFrame(
        {
            "customer_id": splits["out_of_time_test"]["customer_id"].to_numpy(),
            "observation_date": splits["out_of_time_test"][TIME_COLUMN].to_numpy(),
            "y_true": y_test,
            "raw_probability": raw_champion_test_probability,
            "calibrated_probability": calibrated_test_probability,
            "predicted_support": (calibrated_test_probability >= policy_threshold).astype(int),
        }
    ).to_csv(reports_dir / "test_predictions.csv", index=False)

    metrics_blob = {
        "trained_at": metadata["created_at"],
        "model_version": MODEL_VERSION,
        "best_model": best_name,
        "policy_threshold": policy_threshold,
        "split_counts": metadata["split_counts"],
        "split_ranges": split_info,
        "selection_metrics": selection_metrics,
        "models": model_metrics,
        "threshold_table": policy_thresholds,
        "calibration_bins": calibrated_bins,
        "calibration_comparison": {
            "method": "platt_scaling",
            "intercept": calibrated_model.calibration_intercept,
            "slope": calibrated_model.calibration_slope,
            "raw": {key: raw_metrics[key] for key in ["auc", "brier", "expected_calibration_error"]},
            "calibrated": {key: calibrated_metrics[key] for key in ["auc", "brier", "expected_calibration_error"]},
        },
        "confidence_intervals": intervals,
        "fairness_summary": fairness,
        "metadata": metadata,
    }
    (reports_dir / "model_metrics.json").write_text(json.dumps(metrics_blob, indent=2), encoding="utf-8")
    (reports_dir / "threshold_policy.json").write_text(
        json.dumps(
            {
                "policy_threshold": policy_threshold,
                "selection_split": "policy_validation",
                "policy_min_precision": POLICY_MIN_PRECISION,
                "policy_min_coverage": POLICY_MIN_COVERAGE,
                "threshold_table": policy_thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_model_evaluation(metrics_blob, reports_dir / "model_evaluation.md")
    write_fairness_report(fairness, reports_dir / "fairness_report.md")
    write_calibration_report(metrics_blob, reports_dir / "calibration_report.md")
    log_with_mlflow_or_fallback(best_name, calibrated_metrics, metadata, model_path, reports_dir)
    return metrics_blob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/features.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    metrics = train(Path(args.input), Path(args.model_dir), Path(args.reports_dir), args.random_state)
    best = metrics["best_model"]
    print(f"Selected and calibrated model: {best}")
    print(json.dumps(metrics["models"][best], indent=2))
    print(f"Frozen policy-validation threshold: {metrics['policy_threshold']:.2f}")


if __name__ == "__main__":
    main()
