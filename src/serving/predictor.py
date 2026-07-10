from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.core.config import settings
from src.serving.policy import TargetedSupportPolicy
from src.serving.schemas import PredictionRequest

MODEL_PATH = Path(settings.model_path)
METADATA_PATH = Path(settings.metadata_path)

_MODEL: Any | None = None
_METADATA: dict[str, Any] | None = None
_POLICY = TargetedSupportPolicy()


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    global _MODEL, _METADATA
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found: {MODEL_PATH}. Run training first.")
        _MODEL = joblib.load(MODEL_PATH)
    if _METADATA is None:
        if not METADATA_PATH.exists():
            raise RuntimeError(f"Metadata file not found: {METADATA_PATH}. Run training first.")
        _METADATA = json.loads(METADATA_PATH.read_text())
    return _MODEL, _METADATA


def reset_artifact_cache() -> None:
    global _MODEL, _METADATA
    _MODEL = None
    _METADATA = None


def _feature_row(request: PredictionRequest) -> pd.DataFrame:
    accessible_total = request.cash_balance + request.investment_balance
    row = {
        "age": request.age,
        "annual_income": request.annual_income,
        "cash_balance": request.cash_balance,
        "investment_balance": request.investment_balance,
        "debt_balance": request.debt_balance,
        "risk_score": request.risk_score,
        "recent_activity_count": request.recent_activity_count,
        "cash_ratio": request.cash_balance / max(accessible_total, 1.0),
        "debt_to_income": request.debt_balance / max(request.annual_income, 1.0),
        "accessible_total": accessible_total,
        "account_type": request.account_type,
        "employment_status": request.employment_status,
    }
    return pd.DataFrame([row])


def _confidence(probability: float) -> str:
    margin = abs(probability - 0.5)
    if margin >= 0.3:
        return "high"
    if margin >= 0.15:
        return "medium"
    return "low"


def _reason_codes(request: PredictionRequest, probability: float, threshold: float) -> list[str]:
    accessible_total = request.cash_balance + request.investment_balance
    cash_ratio = request.cash_balance / max(accessible_total, 1.0)
    debt_to_income = request.debt_balance / max(request.annual_income, 1.0)
    reasons: list[str] = []
    if cash_ratio > 0.55:
        reasons.append("high_cash_ratio")
    if accessible_total >= 10000:
        reasons.append("sufficient_accessible_assets")
    if debt_to_income > 0.65:
        reasons.append("high_debt_to_income")
    if request.risk_score < 0.25:
        reasons.append("low_risk_capacity")
    elif request.risk_score > 0.75:
        reasons.append("higher_risk_capacity")
    if request.recent_activity_count <= 1:
        reasons.append("limited_recent_activity")
    if probability < threshold:
        reasons.append("below_policy_threshold")
    return reasons or ["combined_model_factors"]


def predict_request(request: PredictionRequest) -> dict[str, Any]:
    model, metadata = load_artifacts()
    probability = float(model.predict_proba(_feature_row(request))[0, 1])
    threshold = float(metadata["policy_threshold"])
    policy_decision = _POLICY.decide(request, probability, threshold)
    return {
        "customer_id": request.customer_id,
        "decision_id": f"decision_{uuid.uuid4().hex[:20]}",
        "support_probability": round(probability, 6),
        "recommended_action": policy_decision.action,
        "confidence": _confidence(probability),
        "reason_codes": _reason_codes(request, probability, threshold),
        "policy_reasons": policy_decision.policy_reasons,
        "hard_safety_gate_triggered": policy_decision.hard_safety_gate_triggered,
        "model_version": str(metadata["model_version"]),
        "policy_version": policy_decision.policy_version,
        "feature_schema_version": str(metadata["feature_schema_version"]),
        "policy_threshold": threshold,
    }


def explain_request(request: PredictionRequest) -> dict[str, Any]:
    result = predict_request(request)
    return {
        "customer_id": request.customer_id,
        "decision_id": result["decision_id"],
        "reason_codes": result["reason_codes"],
        "policy_reasons": result["policy_reasons"],
        "explanation": (
            "The model score is separate from the versioned decision policy. "
            "Reason codes describe model-relevant inputs; policy reasons describe the deterministic action rule."
        ),
        "model_version": result["model_version"],
        "policy_version": result["policy_version"],
    }


def model_probability(model: Any, request: PredictionRequest) -> float:
    return float(np.asarray(model.predict_proba(_feature_row(request)))[0, 1])
