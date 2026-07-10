from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.serving.policy import TargetedSupportPolicy
from src.serving.predictor import load_artifacts, model_probability
from src.serving.review_workflow import audit_event_id
from src.serving.schemas import PredictionRequest

CANDIDATE_PATH = Path("models/candidate_model.joblib")
CANDIDATE_METADATA_PATH = Path("models/candidate_metadata.json")
_POLICY = TargetedSupportPolicy()
_CANDIDATE: Any | None = None
_CANDIDATE_METADATA: dict[str, Any] | None = None


def _load_candidate() -> tuple[Any | None, dict[str, Any]]:
    global _CANDIDATE, _CANDIDATE_METADATA
    if _CANDIDATE is None and CANDIDATE_PATH.exists():
        _CANDIDATE = joblib.load(CANDIDATE_PATH)
    if _CANDIDATE_METADATA is None:
        if CANDIDATE_METADATA_PATH.exists():
            _CANDIDATE_METADATA = json.loads(CANDIDATE_METADATA_PATH.read_text())
        else:
            _CANDIDATE_METADATA = {"candidate_model_name": "unavailable"}
    return _CANDIDATE, _CANDIDATE_METADATA


def shadow_predict(request: PredictionRequest, champion_result: dict[str, Any]) -> dict[str, Any]:
    candidate, candidate_metadata = _load_candidate()
    champion_model, metadata = load_artifacts()
    champion_probability = model_probability(champion_model, request)
    threshold = float(metadata["policy_threshold"])

    if candidate is None:
        payload = {
            "customer_id": request.customer_id,
            "champion_model_version": str(metadata["model_version"]),
            "candidate_model_name": str(candidate_metadata.get("candidate_model_name", "unavailable")),
            "status": "candidate_unavailable",
            "champion_probability": round(champion_probability, 6),
            "candidate_probability": None,
            "probability_delta": None,
            "champion_action": champion_result["recommended_action"],
            "candidate_action": None,
            "action_changed": False,
        }
        payload["audit_event_id"] = audit_event_id(request, {**champion_result, **payload})
        return payload

    candidate_probability = model_probability(candidate, request)
    candidate_decision = _POLICY.decide(request, candidate_probability, threshold)
    payload = {
        "customer_id": request.customer_id,
        "champion_model_version": str(metadata["model_version"]),
        "candidate_model_name": str(candidate_metadata.get("candidate_model_name", "candidate")),
        "status": "candidate_available",
        "champion_probability": round(champion_probability, 6),
        "candidate_probability": round(candidate_probability, 6),
        "probability_delta": round(candidate_probability - champion_probability, 6),
        "champion_action": champion_result["recommended_action"],
        "candidate_action": candidate_decision.action,
        "action_changed": candidate_decision.action != champion_result["recommended_action"],
    }
    payload["audit_event_id"] = audit_event_id(request, {**champion_result, **payload})
    return payload
