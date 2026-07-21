from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

TRITON_ONNX_MAX_IR_VERSION = 10
ONNX_TARGET_OPSET = 18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D transformed feature matrix, got shape={array.shape}")
    return array


def _model_family(estimator: Any) -> str:
    name = estimator.__class__.__name__.lower()
    if "randomforest" in name or "gradientboost" in name or "xgb" in name or "tree" in name:
        return "tree_ensemble"
    if "logistic" in name or "linear" in name:
        return "linear"
    if "mlp" in name or "neural" in name:
        return "neural_network"
    return name


def _probability_output_name(model: Any) -> str:
    outputs = list(model.graph.output)
    for output in outputs:
        if "prob" in output.name.lower():
            return output.name
    if len(outputs) < 2:
        raise ValueError("Classifier ONNX export did not expose a probability output")
    return outputs[-1].name


def _require_triton_ir_compatibility(model: Any, model_name: str) -> int:
    ir_version = int(model.ir_version)
    if ir_version > TRITON_ONNX_MAX_IR_VERSION:
        raise ValueError(
            f"{model_name} ONNX IR version {ir_version} exceeds the validated Triton runtime limit "
            f"{TRITON_ONNX_MAX_IR_VERSION}"
        )
    return ir_version


def _build_calibrator_onnx(slope: float, intercept: float, output_path: Path) -> int:
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("RAW_PROBABILITIES", TensorProto.FLOAT, [None, 2])
    output_info = helper.make_tensor_value_info("SUPPORT_PROBABILITY", TensorProto.FLOAT, [None, 1])
    initializers = [
        helper.make_tensor("CLASS_ONE", TensorProto.INT64, [1], [1]),
        helper.make_tensor("CLIP_MIN", TensorProto.FLOAT, [], [1e-6]),
        helper.make_tensor("CLIP_MAX", TensorProto.FLOAT, [], [1.0 - 1e-6]),
        helper.make_tensor("ONE", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("SLOPE", TensorProto.FLOAT, [], [float(slope)]),
        helper.make_tensor("INTERCEPT", TensorProto.FLOAT, [], [float(intercept)]),
    ]
    nodes = [
        helper.make_node("Gather", ["RAW_PROBABILITIES", "CLASS_ONE"], ["POSITIVE_PROBABILITY"], axis=1),
        helper.make_node("Clip", ["POSITIVE_PROBABILITY", "CLIP_MIN", "CLIP_MAX"], ["CLIPPED_PROBABILITY"]),
        helper.make_node("Sub", ["ONE", "CLIPPED_PROBABILITY"], ["ONE_MINUS_PROBABILITY"]),
        helper.make_node("Div", ["CLIPPED_PROBABILITY", "ONE_MINUS_PROBABILITY"], ["ODDS"]),
        helper.make_node("Log", ["ODDS"], ["RAW_LOGIT"]),
        helper.make_node("Mul", ["RAW_LOGIT", "SLOPE"], ["SCALED_LOGIT"]),
        helper.make_node("Add", ["SCALED_LOGIT", "INTERCEPT"], ["CALIBRATED_LOGIT"]),
        helper.make_node("Sigmoid", ["CALIBRATED_LOGIT"], ["SUPPORT_PROBABILITY"]),
    ]
    graph = helper.make_graph(nodes, "regulated_support_platt_calibrator", [input_info], [output_info], initializers)
    model = helper.make_model(
        graph,
        producer_name="regulated_ml_platform",
        opset_imports=[helper.make_opsetid("", ONNX_TARGET_OPSET)],
    )
    # Newer ONNX packages may emit a newer IR by default than the ONNX Runtime
    # backend in the validated Triton server can load. Pin the small custom graph
    # to the validated runtime contract instead of relying on library defaults.
    model.ir_version = TRITON_ONNX_MAX_IR_VERSION
    ir_version = _require_triton_ir_compatibility(model, "support_calibrator")
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return ir_version


def _base_config(feature_count: int, probability_output: str, max_batch_size: int) -> str:
    return f'''name: "support_base"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{ name: "FEATURES" data_type: TYPE_FP32 dims: [ {feature_count} ] }}
]
output [
  {{ name: "{probability_output}" data_type: TYPE_FP32 dims: [ 2 ] }}
]
dynamic_batching {{
  preferred_batch_size: [ 8, 32, 64 ]
  max_queue_delay_microseconds: 500
}}
instance_group [
  {{ count: 1 kind: KIND_CPU }}
]
'''


def _calibrator_config(max_batch_size: int) -> str:
    return f'''name: "support_calibrator"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{ name: "RAW_PROBABILITIES" data_type: TYPE_FP32 dims: [ 2 ] }}
]
output [
  {{ name: "SUPPORT_PROBABILITY" data_type: TYPE_FP32 dims: [ 1 ] }}
]
dynamic_batching {{
  preferred_batch_size: [ 8, 32, 64 ]
  max_queue_delay_microseconds: 250
}}
instance_group [
  {{ count: 1 kind: KIND_CPU }}
]
'''


def _ensemble_config(feature_count: int, probability_output: str, max_batch_size: int) -> str:
    return f'''name: "support_ensemble"
platform: "ensemble"
max_batch_size: {max_batch_size}
input [
  {{ name: "FEATURES" data_type: TYPE_FP32 dims: [ {feature_count} ] }}
]
output [
  {{ name: "SUPPORT_PROBABILITY" data_type: TYPE_FP32 dims: [ 1 ] }}
]
ensemble_scheduling {{
  step [
    {{
      model_name: "support_base"
      model_version: -1
      input_map {{ key: "FEATURES" value: "FEATURES" }}
      output_map {{ key: "{probability_output}" value: "RAW_PROBABILITIES" }}
    }},
    {{
      model_name: "support_calibrator"
      model_version: -1
      input_map {{ key: "RAW_PROBABILITIES" value: "RAW_PROBABILITIES" }}
      output_map {{ key: "SUPPORT_PROBABILITY" value: "SUPPORT_PROBABILITY" }}
    }}
  ]
}}
'''


def export_triton_repository(
    model_path: Path,
    metadata_path: Path,
    sample_path: Path,
    output_root: Path,
    max_batch_size: int = 128,
) -> dict[str, Any]:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    calibrated = joblib.load(model_path)
    if not hasattr(calibrated, "base_model") or not hasattr(calibrated, "calibrator"):
        raise TypeError("Expected a calibrated model with base_model and calibrator attributes")
    if not hasattr(calibrated.base_model, "named_steps"):
        raise TypeError("Expected the calibrated base model to be a fitted sklearn Pipeline")

    preprocessor = calibrated.base_model.named_steps["preprocess"]
    estimator = calibrated.base_model.named_steps["model"]
    sample = pd.read_csv(sample_path)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = sorted(set(feature_columns).difference(sample.columns))
    if missing:
        raise ValueError(f"Sample data is missing serving features: {missing}")
    transformed = _dense_float32(preprocessor.transform(sample[feature_columns].head(8)))
    feature_count = int(transformed.shape[1])

    output_root.mkdir(parents=True, exist_ok=True)
    repository = output_root / "model_repository"
    base_dir = repository / "support_base"
    calibrator_dir = repository / "support_calibrator"
    ensemble_dir = repository / "support_ensemble"
    for directory in [base_dir / "1", calibrator_dir / "1", ensemble_dir / "1"]:
        directory.mkdir(parents=True, exist_ok=True)
    ensemble_version_marker = ensemble_dir / "1" / "version.txt"
    ensemble_version_marker.write_text(
        "Triton ensemble version 1; scheduling is defined in ../config.pbtxt\n",
        encoding="utf-8",
    )

    base_model = convert_sklearn(
        estimator,
        initial_types=[("FEATURES", FloatTensorType([None, feature_count]))],
        options={id(estimator): {"zipmap": False}},
        target_opset=ONNX_TARGET_OPSET,
    )
    base_ir_version = _require_triton_ir_compatibility(base_model, "support_base")
    probability_output = _probability_output_name(base_model)
    base_path = base_dir / "1" / "model.onnx"
    base_path.write_bytes(base_model.SerializeToString())

    calibrator_path = calibrator_dir / "1" / "model.onnx"
    calibrator_ir_version = _build_calibrator_onnx(
        calibrated.calibration_slope,
        calibrated.calibration_intercept,
        calibrator_path,
    )

    (base_dir / "config.pbtxt").write_text(_base_config(feature_count, probability_output, max_batch_size), encoding="utf-8")
    (calibrator_dir / "config.pbtxt").write_text(_calibrator_config(max_batch_size), encoding="utf-8")
    (ensemble_dir / "config.pbtxt").write_text(_ensemble_config(feature_count, probability_output, max_batch_size), encoding="utf-8")

    preprocessor_path = output_root / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    family = _model_family(estimator)
    contract = {
        "contract_version": "triton-serving-contract-v1",
        "platform_version": "1.1.0",
        "model_release_version": metadata.get("model_version"),
        "policy_version": metadata.get("policy_version"),
        "feature_schema_version": metadata.get("feature_schema_version"),
        "feature_columns": feature_columns,
        "transformed_feature_count": feature_count,
        "estimator_class": estimator.__class__.__name__,
        "model_family": family,
        "accelerator_default": "cpu",
        "gpu_profile_allowed_without_benchmark": False,
        "gpu_runtime_evidence": "not_available",
        "max_batch_size": max_batch_size,
        "preferred_batch_sizes": [8, 32, 64],
        "max_queue_delay_microseconds": 500,
        "onnx_opset": ONNX_TARGET_OPSET,
        "triton_onnx_max_ir_version": TRITON_ONNX_MAX_IR_VERSION,
        "support_base_onnx_ir_version": base_ir_version,
        "support_calibrator_onnx_ir_version": calibrator_ir_version,
        "base_probability_output": probability_output,
        "calibration": {
            "method": "platt_scaling",
            "slope": calibrated.calibration_slope,
            "intercept": calibrated.calibration_intercept,
        },
        "artifacts": {
            "support_base_onnx_sha256": _sha256(base_path),
            "support_calibrator_onnx_sha256": _sha256(calibrator_path),
            "support_ensemble_version_marker_sha256": _sha256(ensemble_version_marker),
            "preprocessor_sha256": _sha256(preprocessor_path),
        },
    }
    contract_path = output_root / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--metadata", default="models/metadata.json")
    parser.add_argument("--sample", default="data/processed/features.csv")
    parser.add_argument("--output-root", default="models/triton")
    parser.add_argument("--max-batch-size", type=int, default=128)
    args = parser.parse_args()
    contract = export_triton_repository(
        Path(args.model), Path(args.metadata), Path(args.sample), Path(args.output_root), args.max_batch_size
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
