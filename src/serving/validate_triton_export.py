from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def validate_export(
    model_path: Path,
    sample_path: Path,
    triton_root: Path,
    output_path: Path,
    sample_size: int = 256,
    tolerance: float = 5e-5,
) -> dict[str, Any]:
    import onnxruntime as ort

    model = joblib.load(model_path)
    contract = json.loads((triton_root / "contract.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(triton_root / "preprocessor.joblib")
    frame = pd.read_csv(sample_path).head(sample_size)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    native = np.asarray(model.predict_proba(frame[feature_columns])[:, 1], dtype=np.float64)
    transformed = _dense_float32(preprocessor.transform(frame[feature_columns]))

    base_session = ort.InferenceSession(
        str(triton_root / "model_repository" / "support_base" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    base_output_name = contract["base_probability_output"]
    base_outputs = base_session.run([base_output_name], {"FEATURES": transformed})
    raw_probabilities = np.asarray(base_outputs[0], dtype=np.float32)
    if raw_probabilities.shape != (len(frame), 2):
        raise ValueError(f"Unexpected base probability shape: {raw_probabilities.shape}")

    calibrator_session = ort.InferenceSession(
        str(triton_root / "model_repository" / "support_calibrator" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    calibrated = calibrator_session.run(
        ["SUPPORT_PROBABILITY"], {"RAW_PROBABILITIES": raw_probabilities}
    )[0].reshape(-1)
    absolute_error = np.abs(native - calibrated.astype(np.float64))
    threshold = float(json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))["threshold"])
    native_decision = native >= threshold
    onnx_decision = calibrated >= threshold
    mismatch_count = int(np.sum(native_decision != onnx_decision))
    report = {
        "status": "PASS" if float(absolute_error.max(initial=0.0)) <= tolerance and mismatch_count == 0 else "FAIL",
        "sample_size": int(len(frame)),
        "tolerance": tolerance,
        "max_absolute_probability_error": float(absolute_error.max(initial=0.0)),
        "mean_absolute_probability_error": float(absolute_error.mean()) if len(absolute_error) else 0.0,
        "policy_threshold": threshold,
        "policy_decision_mismatches": mismatch_count,
        "native_probability_min": float(native.min(initial=1.0)),
        "native_probability_max": float(native.max(initial=0.0)),
        "onnx_probability_min": float(calibrated.min(initial=1.0)),
        "onnx_probability_max": float(calibrated.max(initial=0.0)),
        "providers": {
            "base": base_session.get_providers(),
            "calibrator": calibrator_session.get_providers(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--sample", default="data/processed/features.csv")
    parser.add_argument("--triton-root", default="models/triton")
    parser.add_argument("--output", default="reports/triton_onnx_parity.json")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    validate_export(
        Path(args.model), Path(args.sample), Path(args.triton_root), Path(args.output), args.sample_size, args.tolerance
    )


if __name__ == "__main__":
    main()
