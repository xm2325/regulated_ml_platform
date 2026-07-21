from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_repository(root: Path) -> dict[str, Any]:
    import onnx

    contract_path = root / "contract.json"
    repository = root / "model_repository"
    base_model_path = repository / "support_base" / "1" / "model.onnx"
    calibrator_model_path = repository / "support_calibrator" / "1" / "model.onnx"
    base_config = repository / "support_base" / "config.pbtxt"
    calibrator_config = repository / "support_calibrator" / "config.pbtxt"
    ensemble_config = repository / "support_ensemble" / "config.pbtxt"
    ensemble_version_dir = repository / "support_ensemble" / "1"
    ensemble_version_marker = ensemble_version_dir / "version.txt"
    preprocessor = root / "preprocessor.joblib"
    required = [
        contract_path,
        base_model_path,
        calibrator_model_path,
        base_config,
        calibrator_config,
        ensemble_config,
        ensemble_version_marker,
        preprocessor,
    ]
    failures: list[str] = []
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {path}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    base_model = onnx.load(base_model_path)
    calibrator_model = onnx.load(calibrator_model_path)
    onnx.checker.check_model(base_model)
    onnx.checker.check_model(calibrator_model)
    base_ir_version = int(base_model.ir_version)
    calibrator_ir_version = int(calibrator_model.ir_version)
    runtime_ir_limit = int(contract.get("triton_onnx_max_ir_version", 0))

    base_text = base_config.read_text(encoding="utf-8")
    calibrator_text = calibrator_config.read_text(encoding="utf-8")
    ensemble_text = ensemble_config.read_text(encoding="utf-8")
    checks = {
        "contract_version": contract.get("contract_version") == "triton-serving-contract-v1",
        "cpu_default": contract.get("accelerator_default") == "cpu",
        "gpu_not_implicitly_enabled": contract.get("gpu_profile_allowed_without_benchmark") is False,
        "base_dynamic_batching": "dynamic_batching" in base_text and "preferred_batch_size" in base_text,
        "calibrator_dynamic_batching": "dynamic_batching" in calibrator_text and "preferred_batch_size" in calibrator_text,
        "ensemble_version_directory": ensemble_version_dir.is_dir(),
        "ensemble_maps_base": 'model_name: "support_base"' in ensemble_text,
        "ensemble_maps_calibrator": 'model_name: "support_calibrator"' in ensemble_text,
        "ensemble_output": "SUPPORT_PROBABILITY" in ensemble_text,
        "base_hash_matches": contract["artifacts"]["support_base_onnx_sha256"] == _sha256(base_model_path),
        "calibrator_hash_matches": contract["artifacts"]["support_calibrator_onnx_sha256"] == _sha256(calibrator_model_path),
        "ensemble_marker_hash_matches": contract["artifacts"]["support_ensemble_version_marker_sha256"]
        == _sha256(ensemble_version_marker),
        "preprocessor_hash_matches": contract["artifacts"]["preprocessor_sha256"] == _sha256(preprocessor),
        "triton_ir_limit_declared": runtime_ir_limit > 0,
        "base_ir_within_triton_limit": runtime_ir_limit > 0 and base_ir_version <= runtime_ir_limit,
        "calibrator_ir_within_triton_limit": runtime_ir_limit > 0 and calibrator_ir_version <= runtime_ir_limit,
        "base_ir_matches_contract": contract.get("support_base_onnx_ir_version") == base_ir_version,
        "calibrator_ir_matches_contract": contract.get("support_calibrator_onnx_ir_version") == calibrator_ir_version,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "model_family": contract.get("model_family"),
        "max_batch_size": contract.get("max_batch_size"),
        "preferred_batch_sizes": contract.get("preferred_batch_sizes"),
        "onnx_ir_versions": {
            "runtime_limit": runtime_ir_limit,
            "support_base": base_ir_version,
            "support_calibrator": calibrator_ir_version,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="models/triton")
    parser.add_argument("--output", default="reports/triton_repository_validation.json")
    args = parser.parse_args()
    report = validate_repository(Path(args.root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
