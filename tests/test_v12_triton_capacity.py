from src.operations.benchmark_triton_concurrency import _metric_delta, _metric_total, parse_prometheus
from src.operations.triton_capacity_plan import _normalize_perf_row, build_capacity_plan


def _policy():
    return {
        "platform_version": "1.2.0",
        "service_objectives": {
            "p95_latency_ms": 20.0,
            "p99_latency_ms": 35.0,
            "max_http_error_rate": 0.0,
            "max_absolute_probability_error": 5e-5,
        },
        "batching": {
            "minimum_effective_average_batch_size": 1.5,
            "minimum_batching_gain_at_concurrency": 1.25,
        },
        "capacity": {
            "safety_headroom_fraction": 0.70,
            "reference_target_rows_per_second": 10000.0,
            "max_reference_replicas": 16,
            "minimum_slo_passing_scenarios": 2,
        },
        "claim_boundary": {
            "production_capacity_claim_allowed": False,
            "gpu_capacity_claim_allowed": False,
        },
    }


def _scenario(concurrency, rows_per_second, average_batch_size, p95=10.0, p99=15.0):
    return {
        "concurrency": concurrency,
        "request_batch_size": 1,
        "rows_per_second": rows_per_second,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "http_error_rate": 0.0,
        "max_absolute_probability_error": 1e-7,
        "parity_status": "PASS",
        "support_base_average_batch_size": average_batch_size,
        "support_base_batching_gain": average_batch_size,
    }


def _perf(concurrency, throughput, p95_usec=9000.0, p99_usec=12000.0):
    return {
        "Concurrency": concurrency,
        "Inferences/Second": throughput,
        "p50 latency": 5000.0,
        "p95 latency": p95_usec,
        "p99 latency": p99_usec,
        "Server Queue": 10.0,
    }


def test_prometheus_parser_aggregates_model_metrics_and_deltas():
    before = parse_prometheus(
        'nv_inference_count{model="support_base",version="1"} 100\n'
        'nv_inference_exec_count{model="support_base",version="1"} 100\n'
    )
    after = parse_prometheus(
        'nv_inference_count{model="support_base",version="1"} 132\n'
        'nv_inference_exec_count{model="support_base",version="1"} 104\n'
    )
    assert _metric_total(after, "nv_inference_count", "support_base") == 132
    assert _metric_delta(before, after, "nv_inference_count", "support_base") == 32
    assert _metric_delta(before, after, "nv_inference_exec_count", "support_base") == 4


def test_perf_analyzer_csv_latency_is_normalized_from_microseconds_to_ms():
    row = _normalize_perf_row(_perf(8, 3000.0, p95_usec=9000.0, p99_usec=12000.0))
    assert row["concurrency"] == 8
    assert row["inferences_per_second"] == 3000.0
    assert row["p95_latency_ms"] == 9.0
    assert row["p99_latency_ms"] == 12.0


def test_capacity_plan_uses_perf_analyzer_for_capacity_and_custom_client_for_semantics():
    benchmark = {
        "status": "PASS",
        "scenarios": [
            _scenario(1, 500.0, 1.0),
            _scenario(4, 1600.0, 2.4),
            _scenario(8, 3000.0, 4.5),
        ],
    }
    perf_rows = [
        _perf(4, 2200.0, p95_usec=6000.0, p99_usec=9000.0),
        _perf(8, 3000.0, p95_usec=9000.0, p99_usec=12000.0),
    ]
    report = build_capacity_plan(benchmark, _policy(), perf_rows)
    assert report["status"] == "PASS"
    assert report["decision"] == "SCALE_REPLICAS_FOR_REFERENCE_TARGET"
    assert report["batching_evidence"]["observed"] is True
    assert report["capacity_evidence"]["source"] == "nvidia_triton_perf_analyzer"
    assert report["capacity_evidence"]["best_slo_passing_concurrency"] == 8
    assert report["capacity_evidence"]["safe_reference_rows_per_second_per_replica"] == 2100.0
    assert report["capacity_evidence"]["recommended_reference_replicas"] == 5
    assert report["semantic_client_observation"]["capacity_source"] is False
    assert report["claim_boundary"]["production_capacity_claim_allowed"] is False


def test_capacity_plan_blocks_replica_recommendation_when_batching_is_not_effective():
    benchmark = {
        "status": "PASS",
        "scenarios": [
            _scenario(1, 500.0, 1.0),
            _scenario(4, 1200.0, 1.05),
            _scenario(8, 1800.0, 1.10),
        ],
    }
    perf_rows = [_perf(4, 2500.0), _perf(8, 3500.0)]
    report = build_capacity_plan(benchmark, _policy(), perf_rows)
    assert report["status"] == "FAIL"
    assert report["decision"] == "FIX_BATCHING_BEFORE_REPLICA_SCALING"
    assert report["checks"]["dynamic_batching_observed"] is False


def test_capacity_plan_requires_perf_analyzer_capacity_evidence():
    benchmark = {
        "status": "PASS",
        "scenarios": [
            _scenario(1, 500.0, 1.0),
            _scenario(4, 1200.0, 2.0),
        ],
    }
    report = build_capacity_plan(benchmark, _policy())
    assert report["status"] == "FAIL"
    assert report["decision"] == "PERF_ANALYZER_CAPACITY_EVIDENCE_REQUIRED"
    assert report["capacity_evidence"]["best_slo_passing_concurrency"] is None


def test_capacity_plan_fails_when_perf_analyzer_points_miss_latency_slo():
    benchmark = {
        "status": "PASS",
        "scenarios": [
            _scenario(1, 500.0, 1.0),
            _scenario(4, 900.0, 2.0),
        ],
    }
    perf_rows = [
        _perf(4, 2500.0, p95_usec=25000.0, p99_usec=40000.0),
        _perf(8, 3500.0, p95_usec=30000.0, p99_usec=45000.0),
    ]
    report = build_capacity_plan(benchmark, _policy(), perf_rows)
    assert report["status"] == "FAIL"
    assert report["decision"] == "INSUFFICIENT_PERF_ANALYZER_SLO_EVIDENCE"
    assert report["capacity_evidence"]["best_slo_passing_concurrency"] is None
