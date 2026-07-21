from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import typing


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_onnx_ir(
    model_path: pathlib.Path,
    contract_path: pathlib.Path,
    artifact_key: str,
    max_ir_version: int,
    output_path: pathlib.Path,
) -> dict[str, typing.Any]:
    import onnx

    model = onnx.load(model_path)
    original_ir_version = int(model.ir_version)
    if original_ir_version > max_ir_version:
        model.ir_version = max_ir_version
        onnx.checker.check_model(model)
        onnx.save(model, model_path)

    effective = onnx.load(model_path)
    onnx.checker.check_model(effective)
    effective_ir_version = int(effective.ir_version)
    if effective_ir_version > max_ir_version:
        raise ValueError(
            f"ONNX model IR version {effective_ir_version} exceeds runtime maximum {max_ir_version}"
        )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.setdefault("artifacts", {})[artifact_key] = _sha256(model_path)
    compatibility = contract.setdefault("runtime_compatibility", {})
    compatibility.update(
        {
            "triton_onnxruntime_max_ir_version": max_ir_version,
            "normalized_artifact": artifact_key,
            "original_ir_version": original_ir_version,
            "effective_ir_version": effective_ir_version,
            "normalization_applied": original_ir_version != effective_ir_version,
            "opset_imports": [
                {"domain": item.domain or "ai.onnx", "version": int(item.version)}
                for item in effective.opset_import
            ],
        }
    )
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    report = {
        "status": "PASS",
        "model_path": str(model_path),
        "artifact_key": artifact_key,
        "original_ir_version": original_ir_version,
        "effective_ir_version": effective_ir_version,
        "max_ir_version": max_ir_version,
        "normalization_applied": original_ir_version != effective_ir_version,
        "sha256": contract["artifacts"][artifact_key],
        "opset_imports": compatibility["opset_imports"],
        "boundary": (
            "IR-version normalization changes the ONNX container-format compatibility level only. "
            "Probability and policy-decision parity must still pass after normalization before release."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--artifact-key", required=True)
    parser.add_argument("--max-ir-version", type=int, default=10)
    parser.add_argument("--output", default="reports/onnx_ir_compatibility.json")
    args = parser.parse_args()

    report = normalize_onnx_ir(
        pathlib.Path(args.model),
        pathlib.Path(args.contract),
        args.artifact_key,
        args.max_ir_version,
        pathlib.Path(args.output),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
