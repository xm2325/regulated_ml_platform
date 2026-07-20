from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from src.serving.canary import CanaryController
from src.serving.predictor import ModelPredictor
from src.serving.schemas import PredictionRequest

REQUEST = PredictionRequest(
    customer_id="C_CANARY_TEST",
    request_id="canary-test-request-0001",
    age=45,
    annual_income=52000,
    cash_balance=24000,
    investment_balance=15000,
    debt_balance=3500,
    risk_score=0.45,
    recent_activity_count=5,
    account_type="isa",
    employment_status="employed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class FakeRegistryProvider:
    def __init__(self, version: str = "2") -> None:
        self.version = version

    def sync(self, alias: str, output_dir: str | Path) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        sources = {
            "model.joblib": Path("models/model.joblib"),
            "metadata.json": Path("models/metadata.json"),
            "model_metrics.json": Path("reports/model_metrics.json"),
            "promotion_gate.json": Path("reports/promotion_gate.json"),
        }
        for name, source in sources.items():
            shutil.copy2(source, destination / name)
        checksums = {name: _sha256(destination / name) for name in sources}
        provenance = {
            "registered_model_name": "regulated-targeted-support-model",
            "alias": alias,
            "registry_version": self.version,
            "run_id": f"run-{self.version}",
            "sha256": checksums,
        }
        (destination / "registry_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        return provenance


class FakeRuntime:
    def __init__(self) -> None:
        self.predictor = ModelPredictor()
        self.reload_calls = 0

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        result = self.predictor.predict(request)
        result.update(
            {
                "model_source": "registry",
                "runtime_state": "ready_registry",
                "registry_model_name": "regulated-targeted-support-model",
                "registry_alias": "champion",
                "registry_model_version": "1",
            }
        )
        return result

    def status(self) -> dict[str, Any]:
        return {
            "requested_source": "registry",
            "active_source": "registry",
            "runtime_state": "ready_registry",
            "registry_version": "1",
        }

    def reload_from_registry(self, force: bool = False) -> dict[str, Any]:
        self.reload_calls += 1
        return {"status": "reloaded", "force": force}


class DivergentPredictor:
    def __init__(self, base: ModelPredictor) -> None:
        self.base = base
        self.model_version = base.model_version

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        result = dict(self.base.predict(request))
        result["support_probability"] = 0.99 if result["support_probability"] < 0.5 else 0.01
        result["recommended_action"] = "risk_review" if result["recommended_action"] != "risk_review" else "no_support"
        result["review_route"] = "manual_review"
        return result


def _request(index: int) -> PredictionRequest:
    return REQUEST.model_copy(update={"customer_id": f"C_CANARY_{index:03d}", "request_id": f"canary-request-{index:04d}"})


def test_canary_assignment_is_stable_and_serves_both_arms(tmp_path: Path) -> None:
    controller = CanaryController(
        FakeRuntime(),
        provider=FakeRegistryProvider(),
        enabled=True,
        traffic_percent=50,
        min_requests=4,
        min_challenger_requests=1,
        auto_promote=False,
        cache_dir=tmp_path,
    )
    first = [controller._assigned_role(_request(i)) for i in range(40)]
    second = [controller._assigned_role(_request(i)) for i in range(40)]
    assert first == second
    assert "champion" in first
    assert "challenger" in first


def test_identical_verified_challenger_passes_online_gate(tmp_path: Path) -> None:
    controller = CanaryController(
        FakeRuntime(),
        provider=FakeRegistryProvider(),
        enabled=True,
        traffic_percent=100,
        min_requests=4,
        min_challenger_requests=4,
        auto_promote=False,
        cache_dir=tmp_path,
    )
    for i in range(6):
        result = controller.predict(_request(i))
        assert result["served_model_role"] == "challenger"
        assert result["registry_alias"] == "challenger"
        assert result["registry_model_version"] == "2"
    evaluation = controller.evaluate()
    assert evaluation["decision"] == "PASS"
    assert evaluation["metrics"]["action_disagreement_rate"] == 0
    assert evaluation["metrics"]["probability_delta_p95"] == 0


def test_disagreement_stop_condition_blocks_further_challenger_traffic(tmp_path: Path) -> None:
    controller = CanaryController(
        FakeRuntime(),
        provider=FakeRegistryProvider(),
        enabled=True,
        traffic_percent=100,
        min_requests=4,
        min_challenger_requests=4,
        auto_promote=False,
        cache_dir=tmp_path,
    )
    assert controller._challenger is not None
    controller._challenger = DivergentPredictor(controller._challenger)  # type: ignore[assignment]
    for i in range(6):
        controller.predict(_request(i))
    evaluation = controller.evaluate()
    assert evaluation["decision"] == "STOP"
    assert evaluation["metrics"]["action_disagreement_rate"] > 0.05
    controller._stop_canary(evaluation["reasons"])
    status = controller.status()
    assert status["state"] == "stopped"
    assert status["last_transition"] == "automatic_stop"
    result = controller.predict(_request(99))
    assert result["served_model_role"] == "champion"
    assert result["canary_assignment"] == "champion"
