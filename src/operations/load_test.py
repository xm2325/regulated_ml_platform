from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.data.make_dataset import generate_customers
from src.serving.app import app


def request_from_row(row: Any) -> dict[str, Any]:
    return {"customer_id": str(row["customer_id"]), "age": int(row["age"]), "annual_income": float(row["annual_income"]), "cash_balance": float(row["cash_balance"]), "investment_balance": float(row["investment_balance"]), "debt_balance": float(row["debt_balance"]), "risk_score": float(row["risk_score"]), "recent_activity_count": int(row["recent_activity_count"]), "account_type": str(row["account_type"]), "employment_status": str(row["employment_status"])}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def run_load_test(n_requests: int = 200, seed: int = 42) -> dict[str, Any]:
    previous_logging = os.environ.get("DISABLE_STRUCTURED_LOGGING")
    os.environ["DISABLE_STRUCTURED_LOGGING"] = "1"
    client = TestClient(app)
    rows = generate_customers(n=n_requests, seed=seed).drop(columns=["support_needed", "true_support_probability"])
    latencies_ms: list[float] = []
    status_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    for _, row in rows.iterrows():
        start = time.perf_counter()
        response = client.post("/predict", json=request_from_row(row))
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        key = str(response.status_code)
        status_counts[key] = status_counts.get(key, 0) + 1
        if response.status_code == 200:
            route = response.json().get("review_route", "unknown")
            review_counts[route] = review_counts.get(route, 0) + 1
    ok = status_counts.get("200", 0)
    errors = n_requests - ok
    summary = {"requests": n_requests, "ok": ok, "errors": errors, "error_rate": errors / max(n_requests, 1), "latency_ms_mean": float(statistics.mean(latencies_ms)), "latency_ms_p50": percentile(latencies_ms, 50), "latency_ms_p95": percentile(latencies_ms, 95), "latency_ms_p99": percentile(latencies_ms, 99), "status_counts": status_counts, "review_route_counts": review_counts, "slo_pass": errors == 0 and percentile(latencies_ms, 95) < 250.0, "slo_latency_ms_p95_target": 250.0}
    if previous_logging is None:
        os.environ.pop("DISABLE_STRUCTURED_LOGGING", None)
    else:
        os.environ["DISABLE_STRUCTURED_LOGGING"] = previous_logging
    return summary


def write_report(summary: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# API load test report", "", f"Requests: `{summary['requests']}`", f"Errors: `{summary['errors']}`", f"Error rate: `{summary['error_rate']:.4f}`", f"Mean latency: `{summary['latency_ms_mean']:.2f} ms`", f"p95 latency: `{summary['latency_ms_p95']:.2f} ms`", f"p99 latency: `{summary['latency_ms_p99']:.2f} ms`", f"SLO pass: `{summary['slo_pass']}`", "", "## Review route counts", "", "| Route | Count |", "|---|---:|"]
    lines.extend(f"| {route} | {count} |" for route, count in sorted(summary["review_route_counts"].items()))
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="reports/load_test_summary.json")
    parser.add_argument("--output-md", default="reports/load_test_report.md")
    args = parser.parse_args()
    summary = run_load_test(args.requests, args.seed)
    write_report(summary, Path(args.output_json), Path(args.output_md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
