from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score

from src.features.build_features import TARGET, TIME_COLUMN

POLICY_MIN_PRECISION = 0.80
POLICY_MIN_COVERAGE = 0.10
MIN_GROUP_SIZE = 30


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
        if not mask.any():
            continue
        observed = float(y_true[mask].mean())
        predicted = float(prob[mask].mean())
        ece += float(mask.mean()) * abs(observed - predicted)
    return float(ece)


def calibration_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:], strict=True), start=1):
        mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
        if not mask.any():
            rows.append({"bin": idx, "left": float(left), "right": float(right), "n": 0})
            continue
        observed = float(y_true[mask].mean())
        predicted = float(prob[mask].mean())
        rows.append({"bin": idx, "left": float(left), "right": float(right), "n": int(mask.sum()), "mean_probability": predicted, "observed_rate": observed, "absolute_error": abs(observed - predicted)})
    return rows


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    high_conf_mask = (prob >= 0.75) | (prob <= 0.25)
    high_conf_positive = prob >= 0.75
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "expected_calibration_error": expected_calibration_error(y_true, prob),
        "precision_at_policy_threshold": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_policy_threshold": float(recall_score(y_true, pred, zero_division=0)),
        "policy_support_rate": float(pred.mean()),
        "precision_at_0_5": float(precision_score(y_true, (prob >= 0.5).astype(int), zero_division=0)),
        "high_confidence_rate": float(high_conf_mask.mean()),
        "positive_high_confidence_rate": float(high_conf_positive.mean()),
        "precision_at_high_confidence": float(precision_score(y_true[high_conf_positive], np.ones(high_conf_positive.sum()), zero_division=0)) if high_conf_positive.any() else 0.0,
    }


def threshold_table(y_true: np.ndarray, prob: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold in np.round(np.arange(0.30, 0.91, 0.05), 2):
        pred = (prob >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        coverage = pred.mean()
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append({"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "support_rate": float(coverage), "f1": float(f1)})
    return rows


def choose_policy_threshold(rows: list[dict[str, float]]) -> float:
    candidates = [row for row in rows if row["precision"] >= POLICY_MIN_PRECISION and row["support_rate"] >= POLICY_MIN_COVERAGE]
    if candidates:
        return float(max(candidates, key=lambda row: (row["f1"], row["precision"], row["support_rate"]))["threshold"])
    return float(max(rows, key=lambda row: (row["precision"], row["support_rate"]))["threshold"])


def bootstrap_intervals(y_true: np.ndarray, prob: np.ndarray, threshold: float, n_bootstrap: int = 250, random_state: int = 42) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {"auc": [], "brier": [], "precision_at_policy_threshold": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample_y, sample_prob = y_true[idx], prob[idx]
        if np.unique(sample_y).size < 2:
            continue
        values["auc"].append(float(roc_auc_score(sample_y, sample_prob)))
        values["brier"].append(float(brier_score_loss(sample_y, sample_prob)))
        values["precision_at_policy_threshold"].append(float(precision_score(sample_y, (sample_prob >= threshold).astype(int), zero_division=0)))
    return {name: {"lower_95": float(np.quantile(samples, 0.025)), "median": float(np.quantile(samples, 0.5)), "upper_95": float(np.quantile(samples, 0.975)), "bootstrap_samples": len(samples)} for name, samples in values.items()}


def temporal_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = frame.assign(_parsed_date=pd.to_datetime(frame[TIME_COLUMN])).sort_values(["_parsed_date", "customer_id"]).drop(columns="_parsed_date")
    n = len(ordered)
    boundaries = [0, int(0.60 * n), int(0.70 * n), int(0.80 * n), int(0.90 * n), n]
    names = ["train", "model_selection", "calibration", "policy_validation", "out_of_time_test"]
    splits = {name: ordered.iloc[left:right].copy() for name, left, right in zip(names, boundaries[:-1], boundaries[1:], strict=True)}
    for name, split in splits.items():
        if len(split) < 20 or split[TARGET].nunique() < 2:
            raise ValueError(f"Temporal split {name} is too small or contains one target class")
    return splits


def split_ranges(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    return {name: {"n": int(len(split)), "start_date": str(split[TIME_COLUMN].min()), "end_date": str(split[TIME_COLUMN].max()), "positive_rate": float(split[TARGET].mean())} for name, split in splits.items()}


def _gap(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None and np.isfinite(value)]
    return float(max(clean) - min(clean)) if len(clean) >= 2 else None


def fairness_summary(frame: pd.DataFrame, y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    work = frame.copy()
    work["y_true"], work["prob"], work["pred"] = y_true, prob, (prob >= threshold).astype(int)
    work["age_band"] = pd.cut(work["age"], bins=[17, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "61+"])
    work["income_band"] = pd.qcut(work["annual_income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    output: dict[str, Any] = {"threshold": threshold, "minimum_group_size": MIN_GROUP_SIZE, "groups": {}, "gaps": {}}
    for group_col in ["age_band", "income_band", "employment_status"]:
        rows = []
        for group, subset in work.groupby(group_col, observed=False):
            n = len(subset)
            has_both_classes = subset["y_true"].nunique() == 2
            tp = int(((subset["pred"] == 1) & (subset["y_true"] == 1)).sum())
            fp = int(((subset["pred"] == 1) & (subset["y_true"] == 0)).sum())
            tn = int(((subset["pred"] == 0) & (subset["y_true"] == 0)).sum())
            fn = int(((subset["pred"] == 0) & (subset["y_true"] == 1)).sum())
            rows.append({
                "group": str(group), "n": int(n),
                "evidence_status": "sufficient" if n >= MIN_GROUP_SIZE and has_both_classes else "insufficient_evidence",
                "mean_probability": float(subset["prob"].mean()), "observed_positive_rate": float(subset["y_true"].mean()),
                "predicted_support_rate": float(subset["pred"].mean()),
                "precision_at_policy_threshold": float(precision_score(subset["y_true"], subset["pred"], zero_division=0)),
                "recall_at_policy_threshold": float(recall_score(subset["y_true"], subset["pred"], zero_division=0)),
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
                "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
                "auc": float(roc_auc_score(subset["y_true"], subset["prob"])) if n >= MIN_GROUP_SIZE and has_both_classes else None,
                "brier": float(brier_score_loss(subset["y_true"], subset["prob"])),
                "expected_calibration_error": expected_calibration_error(subset["y_true"].to_numpy(), subset["prob"].to_numpy()),
            })
        output["groups"][group_col] = rows
        sufficient = [row for row in rows if row["evidence_status"] == "sufficient"]
        output["gaps"][group_col] = {
            "predicted_support_rate_gap": _gap([row["predicted_support_rate"] for row in sufficient]),
            "mean_probability_gap": _gap([row["mean_probability"] for row in sufficient]),
            "precision_gap": _gap([row["precision_at_policy_threshold"] for row in sufficient]),
            "false_positive_rate_gap": _gap([row["false_positive_rate"] for row in sufficient]),
            "expected_calibration_error_gap": _gap([row["expected_calibration_error"] for row in sufficient]),
            "auc_gap": _gap([row["auc"] for row in sufficient]),
        }
    return output


def write_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    output = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---" for _ in columns]) + "|"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else "n/a" if value is None else str(value))
        output.append("| " + " | ".join(cells) + " |")
    return output


def write_model_evaluation(metrics: dict[str, Any], output_path: Path) -> None:
    best = metrics["best_model"]
    best_metrics = metrics["models"][best]
    lines = ["# Model evaluation report", "", f"**Release conclusion:** the calibrated `{best}` is evaluated on a later, untouched time window.", "", f"Model version: `{metrics['model_version']}`", "", f"Frozen policy threshold: `{metrics['policy_threshold']:.2f}`", "", "## Out-of-time test result", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in best_metrics.items() if isinstance(value, float))
    lines.extend(["", "## Temporal split", "", "| Split | Rows | Start | End | Positive rate |", "|---|---:|---|---|---:|"])
    for name, summary in metrics["split_ranges"].items():
        lines.append(f"| {name} | {summary['n']} | {summary['start_date']} | {summary['end_date']} | {summary['positive_rate']:.4f} |")
    lines.extend(["", "## Calibration effect", "", "| Metric | Raw champion | Calibrated champion |", "|---|---:|---:|"])
    for key in ["brier", "expected_calibration_error", "auc"]:
        lines.append(f"| {key} | {metrics['calibration_comparison']['raw'][key]:.4f} | {metrics['calibration_comparison']['calibrated'][key]:.4f} |")
    lines.extend(["", "## Bootstrap uncertainty", "", "| Metric | 95% lower | Median | 95% upper |", "|---|---:|---:|---:|"])
    for name, interval in metrics["confidence_intervals"].items():
        lines.append(f"| {name} | {interval['lower_95']:.4f} | {interval['median']:.4f} | {interval['upper_95']:.4f} |")
    lines.extend(["", "## Leakage control", "", "The five windows are ordered by observation date. Candidate selection, probability calibration, policy-threshold selection, and final evaluation use separate windows.", "", "## Policy-validation threshold search", ""])
    lines.extend(write_markdown_table(metrics["threshold_table"], ["threshold", "precision", "recall", "support_rate", "f1"]))
    lines.extend(["", "## Boundary", "", "The data and target are synthetic. This report tests the engineering path and does not support real financial advice."])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fairness_report(fairness: dict[str, Any], output_path: Path) -> None:
    lines = ["# Segment behaviour report", "", f"Frozen policy threshold: `{fairness['threshold']:.2f}`", "", f"Minimum group size for comparative conclusions: `{fairness['minimum_group_size']}`", "", "Groups below the evidence threshold are reported but excluded from gap-based release checks.", ""]
    columns = ["group", "n", "evidence_status", "observed_positive_rate", "predicted_support_rate", "precision_at_policy_threshold", "recall_at_policy_threshold", "false_positive_rate", "expected_calibration_error", "auc", "brier"]
    for group_col, rows in fairness["groups"].items():
        lines.extend([f"## {group_col}", ""])
        lines.extend(write_markdown_table(rows, columns))
        lines.append("")
        gaps = fairness["gaps"][group_col]
        lines.append("Gaps among groups with sufficient evidence: " + ", ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}=n/a" for key, value in gaps.items()))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_calibration_report(metrics: dict[str, Any], output_path: Path) -> None:
    comparison = metrics["calibration_comparison"]
    lines = ["# Calibration report", "", "Platt scaling is fitted on a dedicated calibration window after the base model is selected.", "", f"Calibration intercept: `{comparison['intercept']:.4f}`", "", f"Calibration slope: `{comparison['slope']:.4f}`", "", "| Metric | Raw champion | Calibrated champion |", "|---|---:|---:|"]
    for key in ["brier", "expected_calibration_error", "auc"]:
        lines.append(f"| {key} | {comparison['raw'][key]:.4f} | {comparison['calibrated'][key]:.4f} |")
    lines.extend(["", "## Calibrated out-of-time bins", ""])
    lines.extend(write_markdown_table(metrics["calibration_bins"], ["bin", "left", "right", "n", "mean_probability", "observed_rate", "absolute_error"]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_with_mlflow_or_fallback(model_name: str, metrics: dict[str, float], params: dict[str, Any], artifact_path: Path, reports_dir: Path) -> None:
    fallback = {"tracking_mode": "local_json_fallback", "reason": "MLflow is optional at service runtime. When installed during training, metrics and the model artifact are logged.", "model_name": model_name, "params": params, "metrics": metrics, "artifact_path": str(artifact_path)}
    try:
        import mlflow
        mlflow.set_experiment("regulated-targeted-support")
        with mlflow.start_run(run_name=f"{model_name}-{params.get('model_version', 'unknown')}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact_path))
        fallback["tracking_mode"] = "mlflow"
    except Exception as exc:
        fallback["mlflow_error"] = str(exc)
    (reports_dir / "local_mlflow_fallback.json").write_text(json.dumps(fallback, indent=2), encoding="utf-8")
