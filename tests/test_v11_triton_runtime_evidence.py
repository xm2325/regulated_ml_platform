from src.operations.triton_runtime_evidence import validate_runtime_evidence


def _config(preferred=(8, 32, 64)):
    return {
        "max_batch_size": "128",
        "dynamic_batching": {"preferred_batch_size": [str(value) for value in preferred]},
    }


def _benchmark():
    return {
        "status": "PASS",
        "runtime_evidence": "real_triton_server",
        "results": [
            {"batch_size": size, "parity_status": "PASS", "max_absolute_probability_error": 1e-7}
            for size in [1, 8, 32, 64, 128]
        ],
    }


def test_real_triton_cpu_runtime_evidence_passes_with_loaded_batching_contract():
    report = validate_runtime_evidence(
        _config(),
        _config(),
        _benchmark(),
        'nv_inference_request_success{model="support_ensemble"} 5\n',
    )
    assert report["status"] == "PASS"
    assert report["runtime_evidence"] == "real_triton_cpu_hosted_ci"
    assert report["gpu_runtime_claim"] is False
    assert report["checks"]["required_batch_sizes_executed"] is True


def test_runtime_evidence_fails_when_dynamic_batching_was_not_loaded():
    report = validate_runtime_evidence(
        _config(preferred=(8, 32)),
        _config(),
        _benchmark(),
        'nv_inference_request_success{model="support_ensemble"} 5\n',
    )
    assert report["status"] == "FAIL"
    assert "base_dynamic_batching_loaded" in report["failures"]
