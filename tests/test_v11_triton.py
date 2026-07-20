from src.operations.accelerator_policy import evaluate_accelerator_policy
from src.operations.gpu_autoscaling import decide_gpu_scaling
from src.serving.triton_export import _base_config, _ensemble_config


ACCELERATOR_POLICY = {
    "gpu_candidate_families": ["neural_network", "transformer"],
    "minimum_gpu_speedup_ratio": 1.5,
    "maximum_gpu_p95_latency_ratio": 1.1,
    "minimum_sustained_gpu_utilization": 0.35,
    "maximum_sustained_gpu_utilization": 0.90,
    "require_real_gpu_runtime_evidence": True,
    "require_probability_parity_pass": True,
    "require_policy_decision_parity_pass": True,
}

AUTOSCALING_POLICY = {
    "min_replicas": 1,
    "max_replicas": 8,
    "scale_out": {"gpu_utilization_at_or_above": 0.85, "queue_time_ms_at_or_above": 8.0},
    "scale_in": {
        "gpu_utilization_at_or_below": 0.30,
        "queue_time_ms_at_or_below": 1.0,
        "minimum_average_batch_size": 2.0,
    },
    "cooldown_seconds": {"scale_out": 120, "scale_in": 600},
    "require_accelerator_decision": "GPU_ELIGIBLE",
}


def test_current_tree_model_is_cpu_only_without_fake_gpu_claim():
    report = evaluate_accelerator_policy({"model_family": "tree_ensemble"}, ACCELERATOR_POLICY)
    assert report["decision"] == "CPU_ONLY"
    assert report["gpu_profile_enabled"] is False


def test_neural_model_requires_real_gpu_benchmark():
    report = evaluate_accelerator_policy({"model_family": "neural_network"}, ACCELERATOR_POLICY)
    assert report["decision"] == "GPU_BENCHMARK_REQUIRED"


def test_neural_model_can_become_gpu_eligible_with_strong_real_evidence():
    benchmark = {
        "runtime_evidence": "real_gpu",
        "probability_parity_status": "PASS",
        "policy_decision_parity_status": "PASS",
        "throughput_speedup_ratio": 2.1,
        "gpu_p95_to_cpu_p95_ratio": 0.72,
        "sustained_gpu_utilization": 0.68,
    }
    report = evaluate_accelerator_policy({"model_family": "neural_network"}, ACCELERATOR_POLICY, benchmark)
    assert report["decision"] == "GPU_ELIGIBLE"
    assert report["gpu_profile_enabled"] is True


def test_gpu_claim_is_rejected_when_evidence_is_not_real_gpu():
    benchmark = {
        "runtime_evidence": "contract_only",
        "probability_parity_status": "PASS",
        "policy_decision_parity_status": "PASS",
        "throughput_speedup_ratio": 3.0,
        "gpu_p95_to_cpu_p95_ratio": 0.5,
        "sustained_gpu_utilization": 0.7,
    }
    report = evaluate_accelerator_policy({"model_family": "neural_network"}, ACCELERATOR_POLICY, benchmark)
    assert report["decision"] == "GPU_REJECTED"


def test_gpu_autoscaling_is_disabled_without_gpu_eligibility():
    report = decide_gpu_scaling(
        {"decision": "CPU_ONLY"},
        {"replicas": 2, "gpu_utilization": 0.95, "queue_time_ms_per_request": 20.0, "average_batch_size": 32},
        AUTOSCALING_POLICY,
    )
    assert report["decision"] == "GPU_PROFILE_DISABLED"
    assert report["desired_replicas"] == 2


def test_gpu_autoscaling_scales_out_on_queue_pressure():
    report = decide_gpu_scaling(
        {"decision": "GPU_ELIGIBLE"},
        {"replicas": 2, "gpu_utilization": 0.72, "queue_time_ms_per_request": 12.0, "average_batch_size": 32},
        AUTOSCALING_POLICY,
    )
    assert report["decision"] == "SCALE_OUT"
    assert report["desired_replicas"] == 3


def test_gpu_autoscaling_does_not_scale_in_when_batching_is_weak():
    report = decide_gpu_scaling(
        {"decision": "GPU_ELIGIBLE"},
        {"replicas": 3, "gpu_utilization": 0.15, "queue_time_ms_per_request": 0.2, "average_batch_size": 1.0},
        AUTOSCALING_POLICY,
    )
    assert report["decision"] == "HOLD"
    assert "batching" in report["reason"]


def test_triton_base_config_has_bounded_dynamic_batching():
    config = _base_config(19, "probabilities", 128)
    assert "max_batch_size: 128" in config
    assert "preferred_batch_size: [ 8, 32, 64 ]" in config
    assert "max_queue_delay_microseconds: 500" in config
    assert "KIND_CPU" in config


def test_triton_ensemble_preserves_calibration_stage():
    config = _ensemble_config(19, "probabilities", 128)
    assert 'model_name: "support_base"' in config
    assert 'model_name: "support_calibrator"' in config
    assert "RAW_PROBABILITIES" in config
    assert "SUPPORT_PROBABILITY" in config
