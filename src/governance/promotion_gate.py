from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_THRESHOLDS = {"auc_min": 0.75, "brier_max": 0.20, "ece_max": 0.08, "high_conf_precision_min": 0.80, "policy_precision_min": 0.80, "employment_support_rate_gap_max": 0.35, "age_support_rate_gap_max": 0.35}


def load_thresholds(path: str | Path | None = "config/promotion_gate.yaml") -> dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if path and Path(path).exists():
        thresholds.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return thresholds


def _within_limit_or_unavailable(value: float | None, limit: float) -> bool:
    return value is None or value <= limit


def evaluate_gate(metrics: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    limits = thresholds or load_thresholds()
    best = metrics["best_model"]
    model_metrics = metrics["models"][best]
    fairness = metrics["fairness_summary"]
    checks = {
        "auc": model_metrics["auc"] >= float(limits["auc_min"]),
        "brier": model_metrics["brier"] <= float(limits["brier_max"]),
        "expected_calibration_error": model_metrics["expected_calibration_error"] <= float(limits["ece_max"]),
        "precision_at_high_confidence": model_metrics["precision_at_high_confidence"] >= float(limits["high_conf_precision_min"]),
        "precision_at_policy_threshold": model_metrics["precision_at_policy_threshold"] >= float(limits["policy_precision_min"]),
        "employment_support_rate_gap": _within_limit_or_unavailable(fairness["gaps"]["employment_status"]["predicted_support_rate_gap"], float(limits["employment_support_rate_gap_max"])),
        "age_support_rate_gap": _within_limit_or_unavailable(fairness["gaps"]["age_band"]["predicted_support_rate_gap"], float(limits["age_support_rate_gap_max"])),
        "model_selected_on_dedicated_window": metrics["metadata"].get("model_selection_split") == "model_selection",
        "calibration_fitted_on_dedicated_window": metrics["metadata"].get("calibration_split") == "calibration",
        "threshold_selected_on_policy_window": metrics["metadata"].get("threshold_selection_split") == "policy_validation",
        "final_metrics_from_out_of_time_test": metrics["metadata"].get("final_evaluation_split") == "out_of_time_test",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"status": "PASS" if not failed else "REVIEW", "release_recommendation": "eligible_for_controlled_release" if not failed else "hold_for_model_risk_review", "failed_checks": failed, "checks": checks, "thresholds": limits, "model_version": metrics["model_version"], "best_model": best, "policy_threshold": metrics["policy_threshold"], "evaluation_design": {"model_selection": metrics["metadata"].get("model_selection_split"), "calibration": metrics["metadata"].get("calibration_split"), "threshold_selection": metrics["metadata"].get("threshold_selection_split"), "final_evaluation": metrics["metadata"].get("final_evaluation_split")}}


def write_markdown(gate: dict[str, Any], output: Path) -> None:
    lines = ["# Model promotion gate", "", f"## Release conclusion: **{gate['status']}**", "", f"Recommendation: `{gate['release_recommendation']}`", "", f"Model version: `{gate['model_version']}`", "", f"Selected model: `{gate['best_model']}`", "", f"Frozen policy threshold: `{gate['policy_threshold']:.2f}`", "", "## Evidence checks", "", "| Check | Result |", "|---|---|"]
    lines.extend(f"| {name} | {'PASS' if passed else 'REVIEW'} |" for name, passed in gate["checks"].items())
    lines.extend(["", "## Evaluation design", "", f"Model selection: `{gate['evaluation_design']['model_selection']}`", "", f"Calibration: `{gate['evaluation_design']['calibration']}`", "", f"Threshold selection: `{gate['evaluation_design']['threshold_selection']}`", "", f"Final evaluation: `{gate['evaluation_design']['final_evaluation']}`"])
    if gate["failed_checks"]:
        lines.extend(["", "Checks requiring review: " + ", ".join(gate["failed_checks"])])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/model_metrics.json")
    parser.add_argument("--config", default="config/promotion_gate.yaml")
    parser.add_argument("--output-json", default="reports/promotion_gate.json")
    parser.add_argument("--output-md", default="reports/promotion_gate.md")
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    gate = evaluate_gate(metrics, load_thresholds(args.config))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_markdown(gate, Path(args.output_md))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
