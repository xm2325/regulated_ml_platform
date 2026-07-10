from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.serving.predictor import feature_row
from src.serving.schemas import PredictionRequest


class RegistryError(RuntimeError):
    """Raised when a controlled registry transition cannot be completed."""


@dataclass(frozen=True)
class RegistryConfig:
    tracking_uri: str = "http://localhost:5000"
    registry_uri: str | None = None
    experiment_name: str = "regulated-targeted-support"
    registered_model_name: str = "regulated-targeted-support-model"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    rollback_alias: str = "rollback"

    @classmethod
    def from_env(cls) -> RegistryConfig:
        tracking = os.getenv("MLFLOW_TRACKING_URI", cls.tracking_uri)
        return cls(
            tracking_uri=tracking,
            registry_uri=os.getenv("MLFLOW_REGISTRY_URI") or tracking,
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", cls.experiment_name),
            registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", cls.registered_model_name),
            champion_alias=os.getenv("MLFLOW_CHAMPION_ALIAS", cls.champion_alias),
            challenger_alias=os.getenv("MLFLOW_CHALLENGER_ALIAS", cls.challenger_alias),
            rollback_alias=os.getenv("MLFLOW_ROLLBACK_ALIAS", cls.rollback_alias),
        )


@dataclass(frozen=True)
class AliasSnapshot:
    alias: str
    present: bool
    version: str | None = None
    run_id: str | None = None
    source: str | None = None
    status: str | None = None
    tags: dict[str, str] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegistryError(f"Expected a JSON object in {path}")
    return value


def write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def configure_mlflow(config: RegistryConfig) -> tuple[Any, Any]:
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:  # pragma: no cover - registry image supplies MLflow
        raise RegistryError("MLflow is not installed. Install requirements-mlflow.lock.") from exc
    registry_uri = config.registry_uri or config.tracking_uri
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    return mlflow, MlflowClient(tracking_uri=config.tracking_uri, registry_uri=registry_uri)


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, str | int | float | bool]:
    output: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            output.update(_flatten(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            output[name] = "null" if item is None else item
        elif isinstance(item, (list, tuple)):
            output[name] = json.dumps(item, sort_keys=True)[:5000]
    return output


def _alias(client: Any, model_name: str, alias: str) -> AliasSnapshot:
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return AliasSnapshot(alias=alias, present=False)
    return AliasSnapshot(
        alias=alias,
        present=True,
        version=str(version.version),
        run_id=str(version.run_id),
        source=str(version.source),
        status=str(version.status),
        tags=dict(version.tags or {}),
    )


def registry_status(config: RegistryConfig, client: Any | None = None) -> dict[str, Any]:
    if client is None:
        _, client = configure_mlflow(config)
    aliases = [config.champion_alias, config.challenger_alias, config.rollback_alias]
    return {
        "checked_at": utc_now(),
        "tracking_uri": config.tracking_uri,
        "registry_uri": config.registry_uri or config.tracking_uri,
        "experiment_name": config.experiment_name,
        "registered_model_name": config.registered_model_name,
        "aliases": {name: asdict(_alias(client, config.registered_model_name, name)) for name in aliases},
    }


def _wait_ready(client: Any, model_name: str, version: str, attempts: int = 60) -> Any:
    latest = None
    for _ in range(attempts):
        latest = client.get_model_version(model_name, version)
        status = str(latest.status).upper()
        if status == "READY":
            return latest
        if status == "FAILED_REGISTRATION":
            raise RegistryError(f"Model version {version} failed registration")
        time.sleep(1)
    raise RegistryError(
        f"Model version {version} did not become READY; last status={getattr(latest, 'status', 'unknown')}"
    )


def register_release(
    config: RegistryConfig,
    model_path: str | Path,
    metadata_path: str | Path,
    metrics_path: str | Path,
    gate_path: str | Path,
    model_version: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    mlflow, client = configure_mlflow(config)
    paths = [Path(value) for value in [model_path, metadata_path, metrics_path, gate_path]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RegistryError(f"Required release artifact not found: {', '.join(missing)}")
    model_file, metadata_file, metrics_file, gate_file = paths
    metadata, metrics_blob, gate = map(load_json, [metadata_file, metrics_file, gate_file])
    best_model = str(metrics_blob["best_model"])
    release_metrics = metrics_blob["models"][best_model]
    release_version = model_version or str(metadata.get("model_version", "unknown"))
    loaded_model = joblib.load(model_file)

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=f"{best_model}-{release_version}") as run:
        run_id = str(run.info.run_id)
        params = {
            "release.model_version": release_version,
            "release.best_model": best_model,
            "release.gate_status": str(gate.get("status", "UNKNOWN")),
            "release.policy_threshold": float(metrics_blob["policy_threshold"]),
            **_flatten(metadata, "metadata"),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(
            {key: float(value) for key, value in release_metrics.items() if isinstance(value, (int, float))}
        )
        for path in paths:
            mlflow.log_artifact(str(path), artifact_path="release")
        mlflow.sklearn.log_model(
            sk_model=loaded_model,
            name="model",
            code_paths=["src"],
            serialization_format="cloudpickle",
        )

    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=config.registered_model_name,
        await_registration_for=120,
    )
    ready = _wait_ready(client, config.registered_model_name, str(registered.version))
    version = str(ready.version)
    tags = {
        "release_version": release_version,
        "validation_status": "passed" if gate.get("status") == "PASS" else "review_required",
        "promotion_gate_status": str(gate.get("status", "UNKNOWN")),
        "lifecycle_status": "challenger",
        "git_commit": str(metadata.get("git_commit", "unknown")),
        "policy_version": str(metadata.get("policy_version", "unknown")),
        "feature_schema_version": str(metadata.get("feature_schema_version", "unknown")),
        "registered_at": utc_now(),
    }
    for key, value in tags.items():
        client.set_model_version_tag(config.registered_model_name, version, key, value)
    client.set_registered_model_alias(config.registered_model_name, config.challenger_alias, version)
    result = {
        "action": "register",
        "registered_at": utc_now(),
        "run_id": run_id,
        "model_name": config.registered_model_name,
        "model_version": version,
        "release_version": release_version,
        "assigned_alias": config.challenger_alias,
        "gate_status": gate.get("status"),
        "artifact_source": str(ready.source),
        "status": registry_status(config, client),
    }
    write_json(output_path, result)
    return result


def gate_allows_promotion(gate: dict[str, Any]) -> bool:
    return gate.get("status") == "PASS" and gate.get("release_recommendation") == "eligible_for_controlled_release"


def promote_challenger(
    config: RegistryConfig,
    gate: dict[str, Any],
    client: Any | None = None,
    expected_challenger_version: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not gate_allows_promotion(gate):
        raise RegistryError("Promotion blocked: the release gate is not PASS and eligible_for_controlled_release")
    if client is None:
        _, client = configure_mlflow(config)
    before = registry_status(config, client)
    challenger = client.get_model_version_by_alias(config.registered_model_name, config.challenger_alias)
    version = str(challenger.version)
    if expected_challenger_version and version != expected_challenger_version:
        raise RegistryError(
            f"Promotion blocked: challenger moved from expected version {expected_challenger_version} to {version}"
        )
    champion = _alias(client, config.registered_model_name, config.champion_alias)
    if champion.present and champion.version != version:
        client.set_registered_model_alias(config.registered_model_name, config.rollback_alias, str(champion.version))
        client.set_model_version_tag(
            config.registered_model_name, str(champion.version), "lifecycle_status", "rollback_candidate"
        )
    client.set_registered_model_alias(config.registered_model_name, config.champion_alias, version)
    for key, value in {
        "validation_status": "approved",
        "lifecycle_status": "champion",
        "promoted_at": utc_now(),
    }.items():
        client.set_model_version_tag(config.registered_model_name, version, key, value)
    try:
        client.delete_registered_model_alias(config.registered_model_name, config.challenger_alias)
    except Exception:
        pass
    result = {
        "action": "promote",
        "promoted_at": utc_now(),
        "model_name": config.registered_model_name,
        "promoted_version": version,
        "previous_champion_version": champion.version,
        "gate_status": gate.get("status"),
        "before": before,
        "after": registry_status(config, client),
    }
    write_json(output_path, result)
    return result


def rollback_champion(
    config: RegistryConfig,
    reason: str,
    client: Any | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not reason.strip():
        raise RegistryError("Rollback requires a non-empty reason")
    if client is None:
        _, client = configure_mlflow(config)
    before = registry_status(config, client)
    current = client.get_model_version_by_alias(config.registered_model_name, config.champion_alias)
    safe = client.get_model_version_by_alias(config.registered_model_name, config.rollback_alias)
    current_version, safe_version = str(current.version), str(safe.version)
    if current_version == safe_version:
        raise RegistryError("Rollback alias already points to the current champion")
    client.set_registered_model_alias(config.registered_model_name, config.champion_alias, safe_version)
    client.set_registered_model_alias(config.registered_model_name, config.challenger_alias, current_version)
    for key, value in {
        "validation_status": "rolled_back",
        "lifecycle_status": "challenger_after_rollback",
        "rollback_reason": reason,
        "rolled_back_at": utc_now(),
    }.items():
        client.set_model_version_tag(config.registered_model_name, current_version, key, value)
    for key, value in {"lifecycle_status": "champion_after_rollback", "restored_at": utc_now()}.items():
        client.set_model_version_tag(config.registered_model_name, safe_version, key, value)
    result = {
        "action": "rollback",
        "rolled_back_at": utc_now(),
        "model_name": config.registered_model_name,
        "reason": reason,
        "failed_version": current_version,
        "restored_version": safe_version,
        "before": before,
        "after": registry_status(config, client),
    }
    write_json(output_path, result)
    return result


def sync_alias(
    config: RegistryConfig,
    alias: str,
    output_dir: str | Path,
    client: Any | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if client is None:
        _, client = configure_mlflow(config)
    version = client.get_model_version_by_alias(config.registered_model_name, alias)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="registry-sync-") as temp_dir:
        for name in ["model.joblib", "metadata.json", "model_metrics.json", "promotion_gate.json"]:
            source = Path(client.download_artifacts(str(version.run_id), f"release/{name}", temp_dir))
            final = destination / name
            temporary = destination / f".{name}.tmp"
            shutil.copy2(source, temporary)
            temporary.replace(final)
            copied[name] = str(final)
    provenance = {
        "synced_at": utc_now(),
        "tracking_uri": config.tracking_uri,
        "registered_model_name": config.registered_model_name,
        "alias": alias,
        "registry_version": str(version.version),
        "run_id": str(version.run_id),
        "source": str(version.source),
        "files": copied,
    }
    (destination / "registry_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_json(output_path, provenance)
    return provenance


def verify_alias(
    config: RegistryConfig,
    alias: str,
    request_path: str | Path,
    client: Any | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    request = PredictionRequest.model_validate(load_json(request_path))
    with tempfile.TemporaryDirectory(prefix="registry-verify-") as temp_dir:
        provenance = sync_alias(config, alias, temp_dir, client=client)
        model = joblib.load(Path(temp_dir) / "model.joblib")
        probability = float(model.predict_proba(feature_row(request))[0, 1])
    finite = probability == probability and 0.0 <= probability <= 1.0
    if not finite:
        raise RegistryError("Downloaded registry model produced an invalid probability")
    result = {
        "action": "verify",
        "verified_at": utc_now(),
        "alias": alias,
        "registry_version": provenance["registry_version"],
        "run_id": provenance["run_id"],
        "customer_id": request.customer_id,
        "support_probability": round(probability, 6),
        "prediction_is_finite": finite,
    }
    write_json(output_path, result)
    return result
