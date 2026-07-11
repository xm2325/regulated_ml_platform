"""Fail-closed decision gate for progressive model delivery.

The gate consumes a bounded canary observation window and emits a deterministic
PASS or ROLLBACK decision.  It intentionally has no Prometheus dependency: a
workflow can export the same metrics used by Argo Rollouts and retain the JSON
decision as release evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": "1.0",
    "thresholds": {
        "minimum_requests": 1_000,
        "availability_min": 0.995,
        "error_rate_max": 0.01,
        "p95_latency_ms_max": 250.0,
        "drift_psi_max": 0.20,
        "fairness_gap_max": 0.10,
    },
    "baseline_regression": {
        "error_rate_increase_max": 0.005,
        "p95_latency_increase_pct_max": 20.0,
    },
    "decision": {"pass": "PASS", "fail": "ROLLBACK", "fail_closed": True},
}

_METRIC_PATHS = (
    "sample_count",
    "canary.availability",
    "canary.error_rate",
    "canary.p95_latency_ms",
    "canary.drift_psi",
    "canary.fairness_gap",
    "baseline.error_rate",
    "baseline.p95_latency_ms",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_policy(path: str | Path | None = "config/canary_gate.yaml") -> dict[str, Any]:
    """Load policy overrides while retaining safe defaults."""

    if path is None:
        return copy.deepcopy(DEFAULT_POLICY)
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Canary gate policy not found: {policy_path}")
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Canary gate policy must be a YAML mapping")
    return _deep_merge(DEFAULT_POLICY, raw)


def _lookup(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def _finite_number(payload: dict[str, Any], dotted_path: str) -> float:
    value = _lookup(payload, dotted_path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{dotted_path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{dotted_path} must be finite")
    return numeric


def _check(observed: float, comparator: str, threshold: float) -> dict[str, Any]:
    if comparator == ">=":
        passed = observed >= threshold
    elif comparator == "<=":
        passed = observed <= threshold
    else:  # pragma: no cover - only internal constants call this helper
        raise ValueError(f"Unsupported comparator: {comparator}")
    return {
        "passed": passed,
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
    }


def _invalid_result(metrics: dict[str, Any], policy: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    fail_decision = str(policy["decision"]["fail"])
    return {
        "schema_version": "1.0",
        "decision": fail_decision,
        "status": fail_decision,
        "release_recommendation": "abort_canary_and_restore_stable",
        "release_id": metrics.get("release_id", "unknown"),
        "observed_at": metrics.get("observed_at"),
        "failed_checks": ["input_valid"],
        "checks": {
            "input_valid": {
                "passed": False,
                "observed": errors,
                "comparator": "valid_metric_contract",
                "threshold": True,
            }
        },
        "input_errors": errors,
        "fail_closed": bool(policy["decision"].get("fail_closed", True)),
        "policy_schema_version": str(policy["schema_version"]),
    }


def evaluate_gate(metrics: dict[str, Any], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one canary window and return audit-friendly decision evidence.

    Missing, non-numeric, or non-finite metrics fail closed to ROLLBACK rather
    than accidentally promoting an unobserved release.
    """

    limits = _deep_merge(DEFAULT_POLICY, policy or {})
    errors: list[str] = []
    values: dict[str, float] = {}
    for path in _METRIC_PATHS:
        try:
            values[path] = _finite_number(metrics, path)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))

    if errors:
        return _invalid_result(metrics, limits, errors)

    if values["sample_count"] < 0:
        return _invalid_result(metrics, limits, ["sample_count must be non-negative"])
    for rate_path in ("canary.availability", "canary.error_rate", "canary.fairness_gap", "baseline.error_rate"):
        if not 0.0 <= values[rate_path] <= 1.0:
            errors.append(f"{rate_path} must be between 0 and 1")
    if values["canary.drift_psi"] < 0:
        errors.append("canary.drift_psi must be non-negative")
    if values["canary.p95_latency_ms"] < 0:
        errors.append("canary.p95_latency_ms must be non-negative")
    if values["baseline.p95_latency_ms"] <= 0:
        errors.append("baseline.p95_latency_ms must be greater than zero")
    if errors:
        return _invalid_result(metrics, limits, errors)

    thresholds = limits["thresholds"]
    regression = limits["baseline_regression"]
    latency_regression_pct = (
        (values["canary.p95_latency_ms"] / values["baseline.p95_latency_ms"]) - 1.0
    ) * 100.0
    error_rate_increase = values["canary.error_rate"] - values["baseline.error_rate"]

    checks = {
        "minimum_requests": _check(values["sample_count"], ">=", float(thresholds["minimum_requests"])),
        "availability": _check(values["canary.availability"], ">=", float(thresholds["availability_min"])),
        "error_rate": _check(values["canary.error_rate"], "<=", float(thresholds["error_rate_max"])),
        "p95_latency_ms": _check(values["canary.p95_latency_ms"], "<=", float(thresholds["p95_latency_ms_max"])),
        "drift_psi": _check(values["canary.drift_psi"], "<=", float(thresholds["drift_psi_max"])),
        "fairness_gap": _check(values["canary.fairness_gap"], "<=", float(thresholds["fairness_gap_max"])),
        "error_rate_regression": _check(
            error_rate_increase,
            "<=",
            float(regression["error_rate_increase_max"]),
        ),
        "p95_latency_regression_pct": _check(
            latency_regression_pct,
            "<=",
            float(regression["p95_latency_increase_pct_max"]),
        ),
    }
    failed = [name for name, result in checks.items() if not result["passed"]]
    decision = str(limits["decision"]["pass"] if not failed else limits["decision"]["fail"])
    return {
        "schema_version": "1.0",
        "decision": decision,
        "status": decision,
        "release_recommendation": (
            "continue_progressive_promotion" if not failed else "abort_canary_and_restore_stable"
        ),
        "release_id": metrics.get("release_id", "unknown"),
        "observed_at": metrics.get("observed_at"),
        "window_seconds": metrics.get("window_seconds"),
        "failed_checks": failed,
        "checks": checks,
        "fail_closed": bool(limits["decision"].get("fail_closed", True)),
        "policy_schema_version": str(limits["schema_version"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a canary release gate")
    parser.add_argument("--metrics", required=True, help="JSON canary metrics export")
    parser.add_argument("--config", default="config/canary_gate.yaml", help="YAML gate policy")
    parser.add_argument("--output", help="Optional JSON evidence output")
    args = parser.parse_args(argv)

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    result = evaluate_gate(metrics, load_policy(args.config))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
