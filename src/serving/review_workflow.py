from __future__ import annotations

import hashlib
import json
from typing import Any

from src.serving.schemas import PredictionRequest


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def audit_event_id(request: PredictionRequest, result: dict[str, Any]) -> str:
    payload = {
        "decision_id": result.get("decision_id"),
        "customer_id": request.customer_id,
        "model_version": result.get("model_version"),
        "policy_version": result.get("policy_version"),
        "feature_schema_version": result.get("feature_schema_version"),
        "support_probability": result.get("support_probability"),
        "recommended_action": result.get("recommended_action"),
        "policy_threshold": result.get("policy_threshold"),
    }
    return f"audit_{_stable_hash(payload)}"


def review_route(request: PredictionRequest, prediction: dict[str, Any]) -> dict[str, Any]:
    probability = float(prediction["support_probability"])
    threshold = float(prediction["policy_threshold"])
    accessible_total = float(request.cash_balance + request.investment_balance)
    debt_to_income = float(request.debt_balance / max(request.annual_income, 1.0))
    margin = abs(probability - threshold)

    reasons: list[str] = []
    if prediction.get("hard_safety_gate_triggered"):
        reasons.append("hard_safety_gate")
    if prediction.get("confidence") == "low":
        reasons.append("low_confidence_score")
    if margin <= 0.05:
        reasons.append("near_policy_threshold")
    if debt_to_income > 0.65:
        reasons.append("debt_pressure_review")
    if accessible_total >= 100000 and probability >= threshold:
        reasons.append("high_value_customer_review")
    if request.age >= 75 and probability >= threshold:
        reasons.append("older_customer_safeguard")

    route = "manual_review" if reasons else "auto_serve"
    reviewer_notes = (
        "Check affordability, vulnerability indicators from approved systems, and message suitability. Do not treat the model output as financial advice."
        if route == "manual_review"
        else "No manual-review trigger fired for this synthetic request."
    )
    return {
        "customer_id": request.customer_id,
        "review_route": route,
        "review_reasons": reasons,
        "reviewer_notes": reviewer_notes,
        "model_version": str(prediction.get("model_version", "unknown")),
        "audit_event_id": audit_event_id(request, prediction),
    }
