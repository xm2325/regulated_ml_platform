import hashlib
import json

import onnx
from onnx import TensorProto, helper

from src.serving.normalize_onnx_ir import normalize_onnx_ir


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalize_onnx_ir_lowers_container_version_and_refreshes_contract(tmp_path):
    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 13

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"artifacts": {"model_sha256": "stale"}}), encoding="utf-8")
    report_path = tmp_path / "compatibility.json"

    report = normalize_onnx_ir(model_path, contract_path, "model_sha256", 10, report_path)

    normalized = onnx.load(model_path)
    onnx.checker.check_model(normalized)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["original_ir_version"] == 13
    assert report["effective_ir_version"] == 10
    assert report["normalization_applied"] is True
    assert normalized.ir_version == 10
    assert contract["artifacts"]["model_sha256"] == _sha256(model_path)
    assert contract["runtime_compatibility"]["effective_ir_version"] == 10
    assert contract["runtime_compatibility"]["opset_imports"][0]["version"] == 18


def test_normalize_onnx_ir_is_noop_when_already_compatible(tmp_path):
    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    before = _sha256(model_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")

    report = normalize_onnx_ir(model_path, contract_path, "model_sha256", 10, tmp_path / "report.json")

    assert report["normalization_applied"] is False
    assert report["effective_ir_version"] == 10
    assert _sha256(model_path) == before
