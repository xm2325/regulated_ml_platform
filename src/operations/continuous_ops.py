from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _max_numeric_psi(drift: dict[str, Any]) -> float:
    return max((float(row.get("psi", 0.0)) for row in drift.get("numeric", [])), default=0.0)


def _max_categorical_distance(drift: dict[str, Any]) -> float:
    return max(
        (float(row.get("total_variation_distance", 0.0)) for row in drift.get("categorical", [])),
        default=0.0,
    )


def _performance_failures(observed: dict[str, Any] | None, policy: dict[str, Any]) -> list[str]:
    if not observed:
        return []
    failures: list[str] = []
    limits = policy["performance"]
    if "auc" in observed and float(observed["auc"]) < float(limits["min_auc"]):
        failures.append("observed AUC is below the operational floor")
    if "brier" in observed and float(observed["brier"]) > float(limits["max_brier"]):
        failures.append("observed Brier score is above the operational ceiling")
    if (
        "expected_calibration_error" in observed
        and float(observed["expected_calibration_error"]) > float(limits["max_expected_calibration_error"])
    ):
        failures.append("observed expected calibration error is above the operational ceiling")
    return failures


def _fairness_failures(observed: dict[str, Any] | None, policy: dict[str, Any]) -> list[str]:
    if not observed or "selection_rate_gap" not in observed:
        return []
    if float(observed["selection_rate_gap"]) > float(policy["fairness"]["max_selection_rate_gap"]):
        return ["observed selection-rate gap is above the operational fairness ceiling"]
    return []


def decide_continuous_action(
    drift: dict[str, Any],
    data_quality: dict[str, Any],
    policy: dict[str, Any],
    observed_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controls = policy["controls"]
    max_psi = _max_numeric_psi(drift)
    max_distance = _max_categorical_distance(drift)
    data_quality_pass = data_quality.get("status") == "PASS"
    performance_failures = _performance_failures(observed_metrics, policy)
    fairness_failures = _fairness_failures(observed_metrics, policy)
    reasons: list[str] = []

    if controls.get("data_quality_must_pass", True) and not data_quality_pass:
        decision = "BLOCKED_DATA_QUALITY"
        reasons.append("data quality failed; retraining on suspect data is blocked")
    elif performance_failures or fairness_failures:
        decision = "INVESTIGATE_MODEL"
        reasons.extend(performance_failures)
        reasons.extend(fairness_failures)
    elif (
        drift.get("overall_status") == "review"
        or max_psi >= float(policy["drift"]["numeric_psi_train_threshold"])
        or max_distance >= float(policy["drift"]["categorical_distance_train_threshold"])
    ):
        decision = "TRAIN_CANDIDATE"
        reasons.append("material drift detected; create a new candidate for offline validation")
    else:
        decision = "NO_TRAINING"
        reasons.append("no configured continuous-training trigger is active")

    return {
        "policy_version": policy.get("version", "unknown"),
        "decision": decision,
        "reasons": reasons,
        "signals": {
            "data_quality_status": data_quality.get("status"),
            "drift_status": drift.get("overall_status"),
            "max_numeric_psi": max_psi,
            "max_categorical_distance": max_distance,
            "observed_metrics": observed_metrics or {},
        },
        "controls": {
            "training_destination_alias": controls["retraining_destination_alias"],
            "auto_promote": bool(controls["auto_promote_after_retraining"]),
            "require_offline_promotion_gate": bool(controls["require_offline_promotion_gate"]),
            "require_canary_before_champion": bool(controls["require_canary_before_champion"]),
        },
        "next_steps": {
            "BLOCKED_DATA_QUALITY": ["quarantine the affected data window", "repair data contract violations", "rerun monitoring before training"],
            "INVESTIGATE_MODEL": ["open model-performance incident", "segment the failure", "decide whether retraining is justified"],
            "TRAIN_CANDIDATE": ["run controlled training pipeline", "register output as challenger only", "run offline gate before any canary traffic"],
            "NO_TRAINING": ["continue monitoring", "do not create a redundant model release"],
        }[decision],
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    reasons = "\n".join(f"- {item}" for item in report["reasons"])
    steps = "\n".join(f"- {item}" for item in report["next_steps"])
    output.write_text(
        "# Continuous training and monitoring decision\n\n"
        f"**Decision:** `{report['decision']}`\n\n"
        "## Why\n\n"
        f"{reasons}\n\n"
        "## Safety controls\n\n"
        f"- training destination: `{report['controls']['training_destination_alias']}`\n"
        f"- automatic promotion: `{str(report['controls']['auto_promote']).lower()}`\n"
        f"- offline promotion gate required: `{str(report['controls']['require_offline_promotion_gate']).lower()}`\n"
        f"- canary required before champion: `{str(report['controls']['require_canary_before_champion']).lower()}`\n\n"
        "## Next steps\n\n"
        f"{steps}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", default="reports/drift_summary.json")
    parser.add_argument("--data-quality", default="reports/data_quality_report.json")
    parser.add_argument("--observed-metrics")
    parser.add_argument("--config", default="config/continuous_ops.yaml")
    parser.add_argument("--output-json", default="reports/continuous_ops_decision.json")
    parser.add_argument("--output-md", default="reports/continuous_ops_decision.md")
    args = parser.parse_args()

    policy = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = decide_continuous_action(
        _load_json(args.drift) or {},
        _load_json(args.data_quality) or {},
        policy,
        _load_json(args.observed_metrics),
    )
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
