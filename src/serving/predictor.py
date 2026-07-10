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


def feature_row(request: PredictionRequest) -> pd.DataFrame:
    accessible_total = request.cash_balance + request.investment_balance
    row = {
        "age": request.age,
        "annual_income": request.annual_income,
        "cash_balance": request.cash_balance,
        "investment_balance": request.investment_balance,
        "debt_balance": request.debt_balance,
        "risk_score": request.risk_score,
        "recent_activity_count": request.recent_activity_count,
        "accessible_total": accessible_total,
        "cash_ratio": request.cash_balance / max(accessible_total, 1.0),
        "debt_to_income": request.debt_balance / max(request.annual_income, 1.0),
        "wealth_to_income": accessible_total / max(request.annual_income, 1.0),
        "account_type": request.account_type,
        "employment_status": request.employment_status,
    }
    return pd.DataFrame([row])


class ModelPredictor:
    """Single artifact-loading path shared by online, batch, explain, and shadow scoring."""

    def __init__(
        self,
        model_path: str | Path = settings.model_path,
        metadata_path: str | Path = settings.metadata_path,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.policy = TargetedSupportPolicy()
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise RuntimeError(f"Model artifact not found: {self.model_path}. Run training first.")
        if not self.metadata_path.exists():
            raise RuntimeError(f"Metadata file not found: {self.metadata_path}. Run training first.")
        self.model = joblib.load(self.model_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    @property
    def policy_threshold(self) -> float:
        return float(self.metadata.get("policy_threshold", self.metadata.get("threshold", 0.5)))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", settings.service_version))

    @property
    def feature_schema_version(self) -> str:
        return str(self.metadata.get("feature_schema_version", settings.feature_schema_version))

    @staticmethod
    def confidence(probability: float) -> str:
        margin = abs(probability - 0.5)
        if margin >= 0.3:
            return "high"
        if margin >= 0.15:
            return "medium"
        return "low"

    @staticmethod
    def reason_codes(request: PredictionRequest, probability: float, threshold: float | None = None) -> list[str]:
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
        resolved_threshold = 0.5 if threshold is None else threshold
        if probability < resolved_threshold:
            reasons.append("below_policy_threshold")
        return reasons or ["combined_model_factors"]

    def probability(self, request: PredictionRequest) -> float:
        if self.model is None:
            raise RuntimeError("Model artifact is not loaded")
        return float(np.asarray(self.model.predict_proba(feature_row(request)))[0, 1])

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        probability = self.probability(request)
        threshold = self.policy_threshold
        policy_decision = self.policy.decide(request, probability, threshold)
        return {
            "customer_id": request.customer_id,
            "decision_id": f"decision_{uuid.uuid4().hex[:20]}",
            "support_probability": round(probability, 6),
            "recommended_action": policy_decision.action,
            "confidence": self.confidence(probability),
            "reason_codes": self.reason_codes(request, probability, threshold),
            "policy_reasons": policy_decision.policy_reasons,
            "hard_safety_gate_triggered": policy_decision.hard_safety_gate_triggered,
            "model_version": self.model_version,
            "policy_version": policy_decision.policy_version,
            "feature_schema_version": self.feature_schema_version,
            "policy_threshold": threshold,
        }


_PREDICTOR: ModelPredictor | None = None


def get_predictor() -> ModelPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = ModelPredictor()
    return _PREDICTOR


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    predictor = get_predictor()
    return predictor.model, predictor.metadata


def reset_artifact_cache() -> None:
    global _PREDICTOR
    _PREDICTOR = None


def predict_request(request: PredictionRequest) -> dict[str, Any]:
    return get_predictor().predict(request)


def explain_request(request: PredictionRequest) -> dict[str, Any]:
    result = predict_request(request)
    return {
        "customer_id": request.customer_id,
        "decision_id": result["decision_id"],
        "reason_codes": result["reason_codes"],
        "policy_reasons": result["policy_reasons"],
        "explanation": (
            "The calibrated model score is separate from the versioned decision policy. "
            "Reason codes describe model-relevant inputs; policy reasons describe the deterministic action rule."
        ),
        "model_version": result["model_version"],
        "policy_version": result["policy_version"],
    }


def model_probability(model: Any, request: PredictionRequest) -> float:
    return float(np.asarray(model.predict_proba(feature_row(request)))[0, 1])
