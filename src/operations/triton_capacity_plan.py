from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _load_perf_analyzer_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            normalized: dict[str, Any] = dict(row)
            for key in ["Concurrency", "Inferences/Second", "p50 latency", "p95 latency", "p99 latency", "Server Queue"]:
                value = row.get(key)
                if value not in {None, ""}:
                    try:
                        normalized[key] = float(value)
                    except ValueError:
                        pass
            rows.append(normalized)
    return rows


def build_capacity_plan(
    benchmark: dict[str, Any],
    policy: dict[str, Any],
    perf_analyzer_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    service = policy.get("service_objectives", {})
    batching = policy.get("batching", {})
    capacity = policy.get("capacity", {})
    claim_boundary = policy.get("claim_boundary", {})

    p95_limit = float(service.get("p95_latency_ms", 20.0))
    p99_limit = float(service.get("p99_latency_ms", 35.0))
    max_http_error_rate = float(service.get("max_http_error_rate", 0.0))
    max_probability_error = float(service.get("max_absolute_probability_error", 5e-5))
    min_batch_gain = float(batching.get("minimum_batching_gain_at_concurrency", 1.25))
    min_effective_batch = float(batching.get("minimum_effective_average_batch_size", 1.5))
    safety_fraction = float(capacity.get("safety_headroom_fraction", 0.70))
    target_rows_per_second = float(capacity.get("reference_target_rows_per_second", 10000.0))
    max_reference_replicas = int(capacity.get("max_reference_replicas", 16))
    minimum_passing = int(capacity.get("minimum_slo_passing_scenarios", 2))

    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_headroom_fraction must be in (0, 1]")

    evaluated: list[dict[str, Any]] = []
    for scenario in benchmark.get("scenarios", []):
        checks = {
            "parity": scenario.get("parity_status") == "PASS",
            "p95_latency": float(scenario.get("p95_latency_ms", math.inf)) <= p95_limit,
            "p99_latency": float(scenario.get("p99_latency_ms", math.inf)) <= p99_limit,
            "http_error_rate": float(scenario.get("http_error_rate", math.inf)) <= max_http_error_rate,
            "probability_error": float(scenario.get("max_absolute_probability_error", math.inf)) <= max_probability_error,
        }
        item = dict(scenario)
        item["slo_checks"] = checks
        item["slo_pass"] = all(checks.values())
        evaluated.append(item)

    passing = [item for item in evaluated if item["slo_pass"]]
    best = max(passing, key=lambda item: float(item.get("rows_per_second", 0.0)), default=None)
    concurrent = [item for item in evaluated if int(item.get("concurrency", 1)) > 1]
    batching_observed = any(float(item.get("support_base_batching_gain", 0.0)) >= min_batch_gain for item in concurrent)
    max_average_batch = max((float(item.get("support_base_average_batch_size", 0.0)) for item in evaluated), default=0.0)
    effective_batching = batching_observed and max_average_batch >= min_effective_batch

    if best is None:
        measured_capacity = 0.0
        safe_capacity = 0.0
        recommended_replicas = max_reference_replicas
        decision = "NO_SLO_PASSING_CAPACITY_POINT"
        reason = "No measured concurrency scenario passed all configured latency, error, and parity objectives."
    else:
        measured_capacity = float(best.get("rows_per_second", 0.0))
        safe_capacity = measured_capacity * safety_fraction
        recommended_replicas = max(1, math.ceil(target_rows_per_second / max(safe_capacity, 1e-12)))
        if len(passing) < minimum_passing:
            decision = "INSUFFICIENT_CAPACITY_EVIDENCE"
            reason = f"Only {len(passing)} SLO-passing scenario(s) were observed; policy requires at least {minimum_passing}."
        elif not effective_batching:
            decision = "FIX_BATCHING_BEFORE_REPLICA_SCALING"
            reason = "Concurrency was exercised but scheduler batching gain did not reach the configured evidence threshold."
        elif recommended_replicas > max_reference_replicas:
            decision = "REFERENCE_TARGET_EXCEEDS_REPLICA_BOUNDARY"
            reason = "The reference throughput target exceeds the configured maximum replica boundary after safety headroom."
        elif recommended_replicas == 1:
            decision = "ONE_REPLICA_SUFFICIENT_FOR_REFERENCE_TARGET"
            reason = "One measured CPU instance covers the reference target after applying the configured safety headroom."
        else:
            decision = "SCALE_REPLICAS_FOR_REFERENCE_TARGET"
            reason = "Multiple replicas are required for the reference target after applying measured single-instance headroom."

    perf_rows = perf_analyzer_rows or []
    perf_summary = {
        "available": bool(perf_rows),
        "rows": len(perf_rows),
        "max_reported_inferences_per_second": max(
            (float(item.get("Inferences/Second", 0.0)) for item in perf_rows if isinstance(item.get("Inferences/Second"), (int, float))),
            default=0.0,
        ),
        "source": "NVIDIA Triton Perf Analyzer" if perf_rows else None,
    }

    checks = {
        "benchmark_pass": benchmark.get("status") == "PASS",
        "minimum_slo_passing_scenarios": len(passing) >= minimum_passing,
        "dynamic_batching_observed": effective_batching,
        "capacity_point_available": best is not None,
        "production_capacity_claim_disabled": claim_boundary.get("production_capacity_claim_allowed") is False,
        "gpu_capacity_claim_disabled": claim_boundary.get("gpu_capacity_claim_allowed") is False,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    return {
        "status": status,
        "platform_version": policy.get("platform_version"),
        "decision": decision,
        "reason": reason,
        "checks": checks,
        "service_objectives": {
            "p95_latency_ms": p95_limit,
            "p99_latency_ms": p99_limit,
            "max_http_error_rate": max_http_error_rate,
            "max_absolute_probability_error": max_probability_error,
        },
        "batching_evidence": {
            "observed": effective_batching,
            "minimum_required_gain": min_batch_gain,
            "maximum_observed_average_batch_size": max_average_batch,
        },
        "capacity_evidence": {
            "best_slo_passing_concurrency": int(best["concurrency"]) if best else None,
            "measured_rows_per_second": measured_capacity,
            "safety_headroom_fraction": safety_fraction,
            "safe_reference_rows_per_second_per_replica": safe_capacity,
            "reference_target_rows_per_second": target_rows_per_second,
            "recommended_reference_replicas": recommended_replicas,
            "max_reference_replicas": max_reference_replicas,
            "slo_passing_scenarios": len(passing),
            "total_scenarios": len(evaluated),
        },
        "perf_analyzer": perf_summary,
        "scenarios": evaluated,
        "claim_boundary": {
            "production_capacity_claim_allowed": False,
            "gpu_capacity_claim_allowed": False,
            "statement": (
                "Replica guidance is a reference calculation from short hosted-CI measurements. It is not a production capacity "
                "commitment and must be re-measured with representative traffic, resource limits, failure modes, and SLOs."
            ),
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    capacity = report["capacity_evidence"]
    batching = report["batching_evidence"]
    lines = [
        "# Triton capacity evidence",
        "",
        f"Status: **{report['status']}**",
        f"Decision: **{report['decision']}**",
        "",
        report["reason"],
        "",
        "## Measured reference envelope",
        "",
        f"- Best SLO-passing concurrency: `{capacity['best_slo_passing_concurrency']}`",
        f"- Measured rows/s at that point: `{capacity['measured_rows_per_second']:.2f}`",
        f"- Safety headroom fraction: `{capacity['safety_headroom_fraction']:.2f}`",
        f"- Safe reference rows/s per replica: `{capacity['safe_reference_rows_per_second_per_replica']:.2f}`",
        f"- Reference target rows/s: `{capacity['reference_target_rows_per_second']:.2f}`",
        f"- Recommended reference replicas: `{capacity['recommended_reference_replicas']}`",
        "",
        "## Dynamic batching evidence",
        "",
        f"- Observed: `{batching['observed']}`",
        f"- Maximum observed average batch size: `{batching['maximum_observed_average_batch_size']:.3f}`",
        f"- Minimum configured batching gain: `{batching['minimum_required_gain']:.3f}`",
        "",
        "## Boundary",
        "",
        report["claim_boundary"]["statement"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--policy", default="config/triton_capacity_policy.yaml")
    parser.add_argument("--perf-analyzer-csv")
    parser.add_argument("--output-json", default="reports/triton_capacity_plan.json")
    parser.add_argument("--output-md", default="reports/triton_capacity_plan.md")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")) or {}
    perf_rows = _load_perf_analyzer_csv(Path(args.perf_analyzer_csv) if args.perf_analyzer_csv else None)
    report = build_capacity_plan(benchmark, policy, perf_rows)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, Path(args.output_md))
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
