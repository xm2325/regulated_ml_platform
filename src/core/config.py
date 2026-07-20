from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceSettings:
    service_name: str = os.getenv("SERVICE_NAME", "regulated-ai-mlops-platform")
    platform_version: str = os.getenv("PLATFORM_VERSION", "0.8.0")
    service_version: str = os.getenv("SERVICE_VERSION", "0.8.0")
    model_release_version: str = os.getenv("MODEL_RELEASE_VERSION", "0.6.0")
    policy_version: str = os.getenv("POLICY_VERSION", "targeted-support-policy-v3")
    feature_schema_version: str = os.getenv("FEATURE_SCHEMA_VERSION", "financial_customer_features_v4")
    environment: str = os.getenv("APP_ENV", "local")
    git_commit: str = os.getenv("GIT_COMMIT", "unknown")

    model_path: str = os.getenv("MODEL_PATH", "models/model.joblib")
    metadata_path: str = os.getenv("METADATA_PATH", "models/metadata.json")
    audit_log_path: str | None = os.getenv("AUDIT_LOG_PATH")

    model_source: str = os.getenv("MODEL_SOURCE", "local")
    registry_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    registry_uri: str = os.getenv("MLFLOW_REGISTRY_URI", "http://localhost:5000")
    registry_model_name: str = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "regulated-targeted-support-model")
    registry_alias: str = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
    registry_cache_dir: str = os.getenv("REGISTRY_CACHE_DIR", "/tmp/regulated-ai-registry-cache")
    registry_reload_interval_seconds: float = float(os.getenv("REGISTRY_RELOAD_INTERVAL_SECONDS", "30"))
    registry_strict_startup: bool = _env_bool("REGISTRY_STRICT_STARTUP", False)
    registry_hot_reload_enabled: bool = _env_bool("REGISTRY_HOT_RELOAD_ENABLED", True)


settings = ServiceSettings()
