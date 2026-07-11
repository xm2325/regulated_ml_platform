from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceSettings:
    service_name: str = os.getenv("SERVICE_NAME", "regulated-ai-mlops-platform")
    service_version: str = os.getenv("SERVICE_VERSION", "0.8.0")
    policy_version: str = os.getenv("POLICY_VERSION", "targeted-support-policy-v3")
    feature_schema_version: str = os.getenv("FEATURE_SCHEMA_VERSION", "financial_customer_features_v4")
    environment: str = os.getenv("APP_ENV", "local")
    model_path: str = os.getenv("MODEL_PATH", "models/model.joblib")
    metadata_path: str = os.getenv("METADATA_PATH", "models/metadata.json")
    audit_log_path: str | None = os.getenv("AUDIT_LOG_PATH")


settings = ServiceSettings()
