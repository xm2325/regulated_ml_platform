from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile)) if values else 0.0


def _measure(call: Callable[[], Any], repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        call()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def benchmark(
    model_path: Path,
    sample_path: Path,
    triton_root: Path,
    output_path: Path,
    batch_sizes: list[int],
    repeats: int = 25,
    warmup: int = 5,
) -> dict[str, Any]:
    import onnxruntime as ort

    model = joblib.load(model_path)
    preprocessor = joblib.load(triton_root / "preprocessor.joblib")
    contract = json.loads((triton_root / "contract.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(sample_path)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    base_session = ort.InferenceSession(
        str(triton_root / "model_repository" / "support_base" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    calibrator_session = ort.InferenceSession(
        str(triton_root / "model_repository" / "support_calibrator" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    base_output = contract["base_probability_output"]
    results: list[dict[str, Any]] = []

    def native_predict(current_batch: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict_proba(current_batch)[:, 1], dtype=np.float64)

    def onnx_predict(current_batch: pd.DataFrame) -> np.ndarray:
        transformed = _dense_float32(preprocessor.transform(current_batch))
        raw = base_session.run([base_output], {"FEATURES": transformed})[0]
        return calibrator_session.run(
            ["SUPPORT_PROBABILITY"], {"RAW_PROBABILITIES": np.asarray(raw, dtype=np.float32)}
        )[0].reshape(-1)

    for batch_size in batch_sizes:
        batch = frame.iloc[np.arange(batch_size) % len(frame)][feature_columns].reset_index(drop=True)
        native_call = partial(native_predict, batch)
        onnx_call = partial(onnx_predict, batch)

        native_reference = native_call()
        onnx_reference = onnx_call()
        max_error = float(np.max(np.abs(native_reference - onnx_reference)))
        native_ms = _measure(native_call, repeats, warmup)
        onnx_ms = _measure(onnx_call, repeats, warmup)
        native_p50 = median(native_ms)
        onnx_p50 = median(onnx_ms)
        results.append(
            {
                "batch_size": batch_size,
                "native": {
                    "p50_ms": float(native_p50),
                    "p95_ms": _percentile(native_ms, 95),
                    "rows_per_second_at_p50": float(batch_size / max(native_p50 / 1000.0, 1e-12)),
                },
                "onnx_cpu": {
                    "p50_ms": float(onnx_p50),
                    "p95_ms": _percentile(onnx_ms, 95),
                    "rows_per_second_at_p50": float(batch_size / max(onnx_p50 / 1000.0, 1e-12)),
                },
                "onnx_to_native_throughput_ratio": float(native_p50 / max(onnx_p50, 1e-12)),
                "max_absolute_probability_error": max_error,
            }
        )

    report = {
        "benchmark_type": "native_sklearn_vs_onnxruntime_cpu",
        "runtime_evidence": "real_cpu",
        "not_a_triton_runtime_benchmark": True,
        "repeats": repeats,
        "warmup": warmup,
        "results": results,
        "interpretation": (
            "This measures native sklearn versus the exported ONNX execution path on CPU. "
            "It does not claim Triton server, CUDA, TensorRT, or GPU performance."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--sample", default="data/processed/features.csv")
    parser.add_argument("--triton-root", default="models/triton")
    parser.add_argument("--output", default="reports/onnx_cpu_benchmark.json")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()]
    benchmark(
        Path(args.model),
        Path(args.sample),
        Path(args.triton_root),
        Path(args.output),
        batch_sizes,
        args.repeats,
        args.warmup,
    )


if __name__ == "__main__":
    main()
