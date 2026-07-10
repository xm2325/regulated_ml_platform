from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.monitoring.data_quality import check_data_quality
from src.serving.predictor import ModelPredictor
from src.serving.review_workflow import review_route
from src.serving.schemas import PredictionRequest

REQUEST_FIELDS = ["customer_id", "age", "annual_income", "cash_balance", "investment_balance", "debt_balance", "risk_score", "recent_activity_count", "account_type", "employment_status"]


def make_scoring_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["accessible_total"] = output["cash_balance"] + output["investment_balance"]
    output["cash_ratio"] = output["cash_balance"] / np.maximum(output["accessible_total"], 1.0)
    output["debt_to_income"] = output["debt_balance"] / np.maximum(output["annual_income"], 1.0)
    output["wealth_to_income"] = output["accessible_total"] / np.maximum(output["annual_income"], 1.0)
    return output[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def score_frame(frame: pd.DataFrame, predictor: ModelPredictor | None = None) -> pd.DataFrame:
    predictor = predictor or ModelPredictor()
    quality_columns = REQUEST_FIELDS + (["observation_date"] if "observation_date" in frame.columns else [])
    quality = check_data_quality(frame[quality_columns].copy())
    request_frame = frame[REQUEST_FIELDS].copy()
    if quality["status"] != "PASS":
        raise ValueError(f"Data quality check failed: {quality}")
    if predictor.model is None:
        raise RuntimeError("Model artifact is missing. Run src.models.train first.")
    probabilities = predictor.model.predict_proba(make_scoring_features(request_frame))[:, 1]
    rows: list[dict[str, Any]] = []
    for record, probability in zip(request_frame.to_dict(orient="records"), probabilities, strict=False):
        request = PredictionRequest(**record)
        prob = float(np.clip(probability, 0.0, 1.0))
        policy = predictor.policy.decide(request, prob, predictor.policy_threshold)
        base = {"support_probability": round(prob, 4), "recommended_action": policy.action, "confidence": predictor.confidence(prob), "model_version": predictor.model_version, "policy_version": policy.policy_version, "feature_schema_version": predictor.feature_schema_version, "policy_threshold": round(predictor.policy_threshold, 4), "hard_safety_gate_triggered": policy.hard_safety_gate_triggered}
        review = review_route(request, base)
        rows.append({"customer_id": request.customer_id, **base, "reason_codes": ";".join(predictor.reason_codes(request, prob)), "policy_reasons": ";".join(policy.policy_reasons), "review_route": review["review_route"], "review_reasons": ";".join(review["review_reasons"])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/batch_predictions.csv")
    args = parser.parse_args()
    scored = score_frame(pd.read_csv(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    print(f"Wrote {len(scored):,} predictions to {output}")


if __name__ == "__main__":
    main()
