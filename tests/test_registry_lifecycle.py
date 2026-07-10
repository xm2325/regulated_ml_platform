from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.registry.lifecycle import RegistryConfig, RegistryError, promote_challenger, registry_status, rollback_champion


@dataclass
class FakeVersion:
    version: str
    run_id: str = "run"
    source: str = "mlflow-artifacts:/model"
    status: str = "READY"
    tags: dict[str, str] = field(default_factory=dict)


class FakeClient:
    def __init__(self) -> None:
        self.versions = {"1": FakeVersion("1", run_id="run-1"), "2": FakeVersion("2", run_id="run-2")}
        self.aliases: dict[str, str] = {"champion": "1", "challenger": "2"}

    def get_model_version_by_alias(self, _: str, alias: str) -> FakeVersion:
        if alias not in self.aliases:
            raise KeyError(alias)
        return self.versions[self.aliases[alias]]

    def set_registered_model_alias(self, _: str, alias: str, version: str) -> None:
        self.aliases[alias] = str(version)

    def delete_registered_model_alias(self, _: str, alias: str) -> None:
        self.aliases.pop(alias, None)

    def set_model_version_tag(self, _: str, version: str, key: str, value: str) -> None:
        self.versions[str(version)].tags[key] = value


PASS_GATE: dict[str, Any] = {
    "status": "PASS",
    "release_recommendation": "eligible_for_controlled_release",
}


def test_registry_status_reports_missing_aliases() -> None:
    client = FakeClient()
    client.aliases.pop("challenger")
    status = registry_status(RegistryConfig(), client)
    assert status["aliases"]["champion"]["present"] is True
    assert status["aliases"]["challenger"]["present"] is False


def test_promotion_requires_passing_gate() -> None:
    with pytest.raises(RegistryError, match="Promotion blocked"):
        promote_challenger(
            RegistryConfig(),
            {"status": "REVIEW", "release_recommendation": "hold_for_model_risk_review"},
            FakeClient(),
        )


def test_promotion_saves_previous_champion_as_rollback() -> None:
    client = FakeClient()
    result = promote_challenger(RegistryConfig(), PASS_GATE, client)
    assert result["promoted_version"] == "2"
    assert client.aliases["champion"] == "2"
    assert client.aliases["rollback"] == "1"
    assert "challenger" not in client.aliases
    assert client.versions["2"].tags["validation_status"] == "approved"


def test_expected_challenger_prevents_stale_promotion() -> None:
    with pytest.raises(RegistryError, match="challenger moved"):
        promote_challenger(RegistryConfig(), PASS_GATE, FakeClient(), expected_challenger_version="7")


def test_rollback_restores_safe_version_and_quarantines_failed_champion() -> None:
    client = FakeClient()
    promote_challenger(RegistryConfig(), PASS_GATE, client)
    result = rollback_champion(RegistryConfig(), "calibration drift", client)
    assert result["restored_version"] == "1"
    assert result["failed_version"] == "2"
    assert client.aliases["champion"] == "1"
    assert client.aliases["challenger"] == "2"
    assert client.aliases["rollback"] == "1"
    assert client.versions["2"].tags["rollback_reason"] == "calibration drift"
    assert client.versions["2"].tags["lifecycle_status"] == "challenger_after_rollback"
    assert client.versions["1"].tags["lifecycle_status"] == "champion_after_rollback"


def test_rollback_requires_reason() -> None:
    with pytest.raises(RegistryError, match="non-empty reason"):
        rollback_champion(RegistryConfig(), " ", FakeClient())
