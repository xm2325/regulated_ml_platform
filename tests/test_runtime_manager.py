from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.serving.runtime_manager import ModelRuntimeManager, RuntimeModelError
from src.serving.schemas import PredictionRequest


REQUEST = PredictionRequest(
    customer_id="C_RUNTIME",
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
    def __init__(self, version: str = "1") -> None:
        self.version = version
        self.corrupt_after_hash = False
        self.raise_error = False

    def sync(self, alias: str, output_dir: str | Path) -> dict[str, Any]:
        if self.raise_error:
            raise RuntimeError("registry unavailable")
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
        if self.corrupt_after_hash:
            with (destination / "metadata.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")
        return provenance


def test_registry_runtime_loads_champion_and_enriches_prediction(tmp_path: Path) -> None:
    provider = FakeRegistryProvider("1")
    manager = ModelRuntimeManager(provider=provider, model_source="registry", cache_dir=tmp_path, strict_startup=True)
    status = manager.status()
    assert status["active_source"] == "registry"
    assert status["runtime_state"] == "ready_registry"
    assert status["registry_version"] == "1"
    result = manager.predict(REQUEST)
    assert result["model_source"] == "registry"
    assert result["registry_alias"] == "champion"
    assert result["registry_model_version"] == "1"


def test_registry_runtime_atomically_switches_version(tmp_path: Path) -> None:
    provider = FakeRegistryProvider("1")
    manager = ModelRuntimeManager(provider=provider, model_source="registry", cache_dir=tmp_path, strict_startup=True)
    first_predictor = manager.current_predictor()
    provider.version = "2"
    result = manager.reload_from_registry()
    assert result["status"] == "reloaded"
    assert manager.status()["registry_version"] == "2"
    assert manager.current_predictor() is not first_predictor
    assert manager.predict(REQUEST)["registry_model_version"] == "2"


def test_failed_candidate_never_replaces_current_model(tmp_path: Path) -> None:
    provider = FakeRegistryProvider("1")
    manager = ModelRuntimeManager(provider=provider, model_source="registry", cache_dir=tmp_path, strict_startup=True)
    current = manager.current_predictor()
    provider.version = "2"
    provider.corrupt_after_hash = True
    with pytest.raises(RuntimeModelError, match="checksum"):
        manager.reload_from_registry()
    assert manager.status()["registry_version"] == "1"
    assert manager.current_predictor() is current


def test_registry_startup_falls_back_to_local_when_not_strict(tmp_path: Path) -> None:
    provider = FakeRegistryProvider()
    provider.raise_error = True
    manager = ModelRuntimeManager(provider=provider, model_source="registry", cache_dir=tmp_path, strict_startup=False)
    status = manager.status()
    assert status["active_source"] == "local"
    assert status["runtime_state"] == "degraded_local_fallback"
    assert status["reload_failures"] == 1
    assert manager.predict(REQUEST)["model_source"] == "local"


def test_last_verified_registry_cache_survives_registry_outage(tmp_path: Path) -> None:
    provider = FakeRegistryProvider("7")
    first = ModelRuntimeManager(provider=provider, model_source="registry", cache_dir=tmp_path, strict_startup=True)
    assert first.status()["registry_version"] == "7"

    unavailable = FakeRegistryProvider("8")
    unavailable.raise_error = True
    restarted = ModelRuntimeManager(provider=unavailable, model_source="registry", cache_dir=tmp_path, strict_startup=False)
    status = restarted.status()
    assert status["active_source"] == "registry"
    assert status["registry_version"] == "7"
    assert status["runtime_state"] == "degraded_registry_cached"
    assert restarted.predict(REQUEST)["registry_model_version"] == "7"
