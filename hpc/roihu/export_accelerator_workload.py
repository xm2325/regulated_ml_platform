#!/usr/bin/env python3
"""Export the deterministic accelerator-qualification MLP to ONNX.

The ONNX artifact is an intermediate for target-side TensorRT compilation.  It is
not the platform's champion model and must never be promoted by this workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from accelerator_workload import (
    DEFAULT_BATCH_SIZES,
    INPUT_FEATURES,
    INPUT_NAME,
    MAX_BATCH_SIZE,
    OUTPUT_FEATURES,
    OUTPUT_NAME,
    build_model,
    make_input,
    sha256_file,
    validate_source_commit,
    verify_source_archive,
    workload_contract,
)

SCHEMA_VERSION = "regulated-ml-platform.roihu-onnx-export/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--opset", type=int, choices=range(17, 21), default=18)
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _triton_config() -> str:
    preferred = ", ".join(str(value) for value in DEFAULT_BATCH_SIZES if value > 1)
    return f'''# Synthetic accelerator-qualification workload; not a production/champion model.
name: "accelerator_qualification"
platform: "tensorrt_plan"
max_batch_size: {MAX_BATCH_SIZE}
input [
  {{
    name: "{INPUT_NAME}"
    data_type: TYPE_FP32
    dims: [ {INPUT_FEATURES} ]
  }}
]
output [
  {{
    name: "{OUTPUT_NAME}"
    data_type: TYPE_FP32
    dims: [ {OUTPUT_FEATURES} ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
dynamic_batching {{
  preferred_batch_size: [ {preferred} ]
  max_queue_delay_microseconds: 1000
}}
'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source_root = args.source_root.expanduser().resolve(strict=True)
    script_path = Path(__file__).resolve(strict=True)
    try:
        script_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError("executed exporter is not inside the verified source root") from exc
    archive = verify_source_archive(args.source_archive, args.expected_source_sha256)
    commit = validate_source_commit(args.source_git_commit)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    onnx_path = output_dir / "accelerator_qualification.onnx"
    config_path = output_dir / "config.pbtxt"

    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    model = build_model(torch)
    example = make_input(torch, 1)
    torch.onnx.export(
        model,
        example,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        dynamo=False,
    )
    os.chmod(onnx_path, 0o600)
    config_path.write_text(_triton_config(), encoding="utf-8")
    os.chmod(config_path, 0o600)

    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(graph)
    onnx_validation: dict[str, Any] = {"status": "PASS", "onnx_package": onnx.__version__}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "workload": workload_contract(model),
        "source": {
            "archive": archive,
            "git_commit": commit,
            "executed_files": {
                "export_accelerator_workload.py": sha256_file(script_path),
                "accelerator_workload.py": sha256_file(source_root / "hpc/roihu/accelerator_workload.py"),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "platform_machine": platform.machine(),
            "torch": torch.__version__,
            "opset": args.opset,
        },
        "artifacts": {
            "onnx": {"name": onnx_path.name, "sha256": sha256_file(onnx_path), "size_bytes": onnx_path.stat().st_size},
            "triton_config": {
                "name": config_path.name,
                "sha256": sha256_file(config_path),
                "size_bytes": config_path.stat().st_size,
            },
        },
        "onnx_checker": onnx_validation,
        "claim_boundary": {
            "synthetic_accelerator_qualification_only": True,
            "production_model_claim_allowed": False,
            "champion_model_claim_allowed": False,
            "production_capacity_claim_allowed": False,
            "promotion_decision_allowed": False,
        },
    }


def main() -> int:
    os.umask(0o077)
    args = build_parser().parse_args()
    manifest_path = args.output_dir.parent / f"{args.output_dir.name}.failure.json"
    try:
        payload = run(args)
        manifest_path = args.output_dir / "export_manifest.json"
        _atomic_json(manifest_path, payload)
        print(json.dumps({"status": "PASS", "manifest": str(manifest_path)}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - persist fail-closed export evidence
        failure = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ERROR",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "claim_boundary": {
                "production_model_claim_allowed": False,
                "champion_model_claim_allowed": False,
                "production_capacity_claim_allowed": False,
                "promotion_decision_allowed": False,
            },
        }
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _atomic_json(manifest_path, failure)
        except Exception as write_exc:  # noqa: BLE001
            print(f"unable to write export failure evidence: {type(write_exc).__name__}", file=sys.stderr)
        print(f"ONNX export failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
