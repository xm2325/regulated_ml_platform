from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.serving.predictor import ModelPredictor
from src.serving.review_workflow import audit_event_id
from src.serving.schemas import PredictionRequest


def _select_candidate(champion_name: str, candidate_dir: Path) -> tuple[str, Path | None]:
    preferred = ["logistic_regression", "random_forest"]
    available = {path.stem: path for path in candidate_dir.glob("*.joblib")}
    for name in preferred:
        if name != champion_name and name in available:
            return name, available[name]
    for name, path in sorted(available.items()):
        if name != champion_name:
            return name, path
    return "none", None


def shadow_predict(request: PredictionRequest, champion: ModelPredictor | None = None) -> dict[str, Any]:
    """Compare the champion and candidate without changing the user-facing action."""
    predictor = champion or ModelPredictor()
    champion_result = predictor.predict(request)
    candidate_dir = predictor.model_path.parent / "candidates"
    champion_name = str(predictor.metadata.get("model_name", "champion"))
    candidate_name, candidate_path = _select_candidate(champion_name, candidate_dir)

    audit_payload = dict(champion_result)
    audit_payload["model_version"] = f"shadow-{predictor.model_version}"

    if candidate_path is None:
        return {
            "customer_id": request.customer_id,
            "champion_model_version": predictor.model_version,
            "candidate_model_name": candidate_name,
            "status": "candidate_unavailable",
            "champion_probability": champion_result["support_probability"],
            "candidate_probability": None,
            "probability_delta": None,
            "champion_action": champion_result["recommended_action"],
            "candidate_action": None,
            "action_changed": False,
            "audit_event_id": audit_event_id(request, audit_payload),
        }

    candidate = joblib.load(candidate_path)
    frame = predictor.request_to_frame(request)
    candidate_probability = float(candidate.predict_proba(frame)[:, 1][0])
    candidate_probability = float(np.clip(candidate_probability, 0.0, 1.0))
    candidate_action = predictor.action(request, candidate_probability)
    delta = candidate_probability - float(champion_result["support_probability"])
    return {
        "customer_id": request.customer_id,
        "champion_model_version": predictor.model_version,
        "candidate_model_name": candidate_name,
        "status": "candidate_available",
        "champion_probability": champion_result["support_probability"],
        "candidate_probability": round(candidate_probability, 4),
        "probability_delta": round(delta, 4),
        "champion_action": champion_result["recommended_action"],
        "candidate_action": candidate_action,
        "action_changed": candidate_action != champion_result["recommended_action"],
        "audit_event_id": audit_event_id(request, audit_payload),
    }
