from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

_model: torch.jit.ScriptModule | None = None
_device: torch.device | None = None


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    body = record.get("body", record.get("data"))
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def handle(data: list[dict[str, Any]], context: Any) -> list[dict[str, Any]]:
    global _device, _model
    if _model is None:
        properties = context.system_properties
        gpu_id = properties.get("gpu_id")
        if not torch.cuda.is_available() or gpu_id is None:
            raise RuntimeError("the TorchServe smoke requires an assigned CUDA device")
        _device = torch.device(f"cuda:{gpu_id}")
        serialized = context.manifest["model"]["serializedFile"]
        _model = torch.jit.load(
            str(Path(properties["model_dir"]) / serialized),
            map_location=_device,
        )
        _model.eval()

    payload = _payload(data[0])
    batch = int(payload.get("batch", 256))
    if not 1 <= batch <= 1024:
        raise ValueError("batch must be between 1 and 1024")
    assert _device is not None
    inputs = torch.ones((batch, 256), dtype=torch.float32, device=_device)
    with torch.inference_mode():
        output = _model(inputs)
        checksum = float(output.sum().item())
    return [
        {
            "batch": batch,
            "device": _device.type,
            "output_shape": list(output.shape),
            "output_checksum": checksum,
        }
    ]
