from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

PERF_ANALYZER_LATENCY_TO_MS = 0.001


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_perf_row(row: dict[str, Any]) -> dict[str, Any]:
    concurrency = _optional_float(row.get("concurrency", row.get("Concurrency")))
    throughput = _optional_float(row.get("inferences_per_second", row.get("Inferences/Second")))

    def latency_ms(normalized_key: str, csv_key: str) -> float | None:
        normalized = _optional_float(row.get(normalized_key))
        if normalized is not None:
            return normalized
        raw = _optional_float(row.get(csv_key))
        return raw * PERF_ANALYZER_LATENCY_TO_MS if raw is not None else None

    return {
        "concurrency": int(concurrency) if concurrency is not None else None,
        "inferences_per_second": throughput,
        "p50_latency_ms": latency_ms("p50_latency_ms", "p50 latency"),
        "p95_latency_ms": latency_ms("p95_latency_ms", "p95 latency"),
        "p99_latency_ms": latency_ms("p99_latency_ms", "p99 latency"),
        "server_queue_ms": latency_ms("server_queue_ms", "Server Queue"),
    }


def _load_perf_analyzer_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [_normalize_perf_row(dict(row)) for row in csv.DictReader(handle)]


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

    semantic_scenarios: list[dict[str, Any]] = []
    for scenario in benchmark.get("scenarios", []):
        checks = {
            "parity": scenario.get("parity_status") == "PASS",
            "p95_latency": float(scenario.get("p95_latency_ms", math.inf)) <= p95_limit,
            "p99_latency": float(scenario.get("p99_latency_ms", math.inf)) <= p99_limit,
            "http_error_rate": float(scenario.get("http_error_rate", math.inf)) <= max_http_error_rate,
            "probability_error": float(scenario.get("max_absolute_probability_error", math.inf)) <= max_probability_error,
        }
        item = dict(scenario)
        item["end_to_end_checks"] = checks
        item["end_to_end_slo_pass"] = all(checks.values())
        semantic_scenarios.append(item)

    concurrent = [item for item in semantic_scenarios if int(item.get("concurrency", 1)) > 1]
    batching_observed = any(float(item.get("support_base_batching_gain", 0.0)) >= min_batch_gain for item in concurrent)
    max_average_batch = max(
        (float(item.get("support_base_average_batch_size", 0.0)) for item in semantic_scenarios),
        default=0.0,
    )
    effective_batching = batching_observed and max_average_batch >= min_effective_batch
    semantic_parity_pass = bool(semantic_scenarios) and all(
        item.get("parity_status") == "PASS" and float(item.get("max_absolute_probability_error", math.inf)) <= max_probability_error
        for item in semantic_scenarios
    )
    semantic_http_pass = bool(semantic_scenarios) and all(
        float(item.get("http_error_rate", math.inf)) <= max_http_error_rate for item in semantic_scenarios
    )
    semantic_end_to_end_passing = [item for item in semantic_scenarios if item["end_to_end_slo_pass"]]
    semantic_best = max(
        semantic_end_to_end_passing,
        key=lambda item: float(item.get("rows_per_second", 0.0)),
        default=None,
    )

    normalized_perf_rows = [_normalize_perf_row(item) for item in (perf_analyzer_rows or [])]
    evaluated_perf_rows: list[dict[str, Any]] = []
    for row in normalized_perf_rows:
        p95 = row.get("p95_latency_ms")
        p99 = row.get("p99_latency_ms")
        throughput = row.get("inferences_per_second")
        checks = {
            "concurrency_present": row.get("concurrency") is not None,
            "throughput_positive": throughput is not None and float(throughput) > 0,
            "p95_latency": p95 is not None and float(p95) <= p95_limit,
            "p99_latency": p99 is not None and float(p99) <= p99_limit,
        }
        item = dict(row)
        item["slo_checks"] = checks
        item["slo_pass"] = all(checks.values())
        evaluated_perf_rows.append(item)

    perf_passing = [item for item in evaluated_perf_rows if item["slo_pass"]]
    perf_best = max(
        perf_passing,
        key=lambda item: float(item.get("inferences_per_second") or 0.0),
        default=None,
    )

    measured_capacity = float(perf_best.get("inferences_per_second") or 0.0) if perf_best else 0.0
    safe_capacity = measured_capacity * safety_fraction
    recommended_replicas = (
        max(1, math.ceil(target_rows_per_second / max(safe_capacity, 1e-12)))
        if perf_best
        else max_reference_replicas
    )

    if benchmark.get("status") != "PASS" or not semantic_parity_pass or not semantic_http_pass:
        decision = "SEMANTIC_RUNTIME_EVIDENCE_FAILED"
        reason = "Concurrent HTTP evidence did not preserve request correctness and probability semantics."
    elif not effective_batching:
        decision = "FIX_BATCHING_BEFORE_REPLICA_SCALING"
        reason = "Concurrency was exercised but runtime scheduler batching gain did not reach the configured evidence threshold."
    elif not evaluated_perf_rows:
        decision = "PERF_ANALYZER_CAPACITY_EVIDENCE_REQUIRED"
        reason = "Server capacity is not estimated from the Python semantic client when Triton Perf Analyzer evidence is absent."
    elif len(perf_passing) < minimum_passing:
        decision = "INSUFFICIENT_PERF_ANALYZER_SLO_EVIDENCE"
        reason = f"Only {len(perf_passing)} Perf Analyzer point(s) passed the configured SLO; policy requires at least {minimum_passing}."
    elif perf_best is None:
        decision = "NO_PERF_ANALYZER_SLO_CAPACITY_POINT"
        reason = "No Perf Analyzer concurrency point passed the configured p95 and p99 latency objectives."
    elif recommended_replicas > max_reference_replicas:
        decision = "REFERENCE_TARGET_EXCEEDS_REPLICA_BOUNDARY"
        reason = "The reference throughput target exceeds the configured maximum replica boundary after safety headroom."
    elif recommended_replicas == 1:
        decision = "ONE_REPLICA_SUFFICIENT_FOR_REFERENCE_TARGET"
        reason = "One measured Triton CPU instance covers the reference target after applying configured safety headroom."
    else:
        decision = "SCALE_REPLICAS_FOR_REFERENCE_TARGET"
        reason = "Multiple replicas are required for the reference target after applying Perf Analyzer capacity and safety headroom."

    checks = {
        "benchmark_pass": benchmark.get("status") == "PASS",
        "semantic_probability_parity": semantic_parity_pass,
        "semantic_http_correctness": semantic_http_pass,
        "dynamic_batching_observed": effective_batching,
        "perf_analyzer_available": bool(evaluated_perf_rows),
        "minimum_perf_analyzer_slo_points": len(perf_passing) >= minimum_passing,
        "capacity_point_available": perf_best is not None,
        "production_capacity_claim_disabled": claim_boundary.get("production_capacity_claim_allowed") is False,
        "gpu_capacity_claim_disabled": claim_boundary.get("gpu_capacity_claim_allowed") is False,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    max_perf_throughput = max(
        (float(item.get("inferences_per_second") or 0.0) for item in evaluated_perf_rows),
        default=0.0,
    )
    max_semantic_throughput = max(
        (float(item.get("rows_per_second", 0.0)) for item in semantic_scenarios),
        default=0.0,
    )

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
        "evidence_roles": {
            "semantic_and_batching_gate": "custom_concurrent_http_with_native_probability_parity",
            "server_capacity_source": "nvidia_triton_perf_analyzer",
            "why_separate": (
                "The custom Python client includes JSON serialization, Python scheduling, and client-stack overhead. "
                "Perf Analyzer is used for Triton server capacity while the custom client protects semantics and proves batching."
            ),
        },
        "batching_evidence": {
            "observed": effective_batching,
            "minimum_required_gain": min_batch_gain,
            "maximum_observed_average_batch_size": max_average_batch,
        },
        "semantic_client_observation": {
            "maximum_observed_rows_per_second": max_semantic_throughput,
            "best_end_to_end_slo_concurrency": int(semantic_best["concurrency"]) if semantic_best else None,
            "best_end_to_end_slo_rows_per_second": float(semantic_best.get("rows_per_second", 0.0)) if semantic_best else 0.0,
            "end_to_end_slo_passing_scenarios": len(semantic_end_to_end_passing),
            "total_scenarios": len(semantic_scenarios),
            "capacity_source": False,
        },
        "capacity_evidence": {
            "source": "nvidia_triton_perf_analyzer",
            "best_slo_passing_concurrency": int(perf_best["concurrency"]) if perf_best else None,
            "measured_inferences_per_second": measured_capacity,
            "best_point_p95_latency_ms": float(perf_best["p95_latency_ms"]) if perf_best else None,
            "best_point_p99_latency_ms": float(perf_best["p99_latency_ms"]) if perf_best else None,
            "safety_headroom_fraction": safety_fraction,
            "safe_reference_rows_per_second_per_replica": safe_capacity,
            "reference_target_rows_per_second": target_rows_per_second,
            "recommended_reference_replicas": recommended_replicas,
            "max_reference_replicas": max_reference_replicas,
            "slo_passing_points": len(perf_passing),
            "total_points": len(evaluated_perf_rows),
        },
        "perf_analyzer": {
            "available": bool(evaluated_perf_rows),
            "rows": len(evaluated_perf_rows),
            "max_reported_inferences_per_second": max_perf_throughput,
            "latency_unit_in_csv": "microseconds",
            "normalized_latency_unit": "milliseconds",
            "source": "NVIDIA Triton Perf Analyzer" if evaluated_perf_rows else None,
            "points": evaluated_perf_rows,
        },
        "scenarios": semantic_scenarios,
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
    semantic = report["semantic_client_observation"]
    lines = [
        "# Triton capacity evidence",
        "",
        f"Status: **{report['status']}**",
        f"Decision: **{report['decision']}**",
        "",
        report["reason"],
        "",
        "## Evidence separation",
        "",
        "- Semantic/batching gate: custom concurrent HTTP + native calibrated-model parity",
        "- Capacity source: NVIDIA Triton Perf Analyzer",
        "",
        "## Server capacity reference",
        "",
        f"- Best Perf Analyzer SLO-passing concurrency: `{capacity['best_slo_passing_concurrency']}`",
        f"- Measured inferences/s: `{capacity['measured_inferences_per_second']:.2f}`",
        f"- p95/p99 at selected point: `{capacity['best_point_p95_latency_ms']:.3f}` / `{capacity['best_point_p99_latency_ms']:.3f}` ms",
        f"- Safety headroom fraction: `{capacity['safety_headroom_fraction']:.2f}`",
        f"- Safe reference rows/s per replica: `{capacity['safe_reference_rows_per_second_per_replica']:.2f}`",
        f"- Reference target rows/s: `{capacity['reference_target_rows_per_second']:.2f}`",
        f"- Recommended reference replicas: `{capacity['recommended_reference_replicas']}`",
        "",
        "## Semantic client observation",
        "",
        f"- Best end-to-end SLO concurrency: `{semantic['best_end_to_end_slo_concurrency']}`",
        f"- Best end-to-end SLO rows/s: `{semantic['best_end_to_end_slo_rows_per_second']:.2f}`",
        "- This client measurement is not used as the Triton server capacity source.",
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
