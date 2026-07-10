from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.core.config import settings
from src.core.telemetry import log_event
from src.serving.predictor import ModelPredictor
from src.serving.review_workflow import review_route
from src.serving.schemas import ExplainResponse, PredictionRequest, PredictionResponse, ReviewRouteResponse, ShadowPredictionResponse

app = FastAPI(
    title="Regulated AI MLOps Platform",
    version=settings.service_version,
    description="A synthetic regulated-finance MLOps reference implementation with explicit model, policy, review, and audit layers.",
)
predictor = ModelPredictor()

PREDICTION_REQUEST_COUNT = Counter("prediction_request_count", "Number of prediction requests")
PREDICTION_ERROR_COUNT = Counter("prediction_error_count", "Number of prediction errors")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction request latency")
HTTP_REQUESTS = Counter("regulated_ai_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("regulated_ai_http_request_duration_seconds", "HTTP request duration", ["method", "path"])
DECISION_COUNT = Counter("regulated_ai_decisions_total", "Model-policy decisions", ["action", "review_route"])
MODEL_READY = Gauge("regulated_ai_model_ready", "Whether the production model is loaded")
MODEL_READY.set(1 if predictor.model is not None else 0)


@app.middleware("http")
async def request_observability(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        log_event("http_request_failed", request_id=request_id, method=request.method, path=request.url.path)
        raise
    finally:
        duration = time.perf_counter() - start
        HTTP_REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, request.url.path).observe(duration)
        log_event(
            "http_request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=status,
            duration_ms=round(duration * 1000, 3),
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Service-Version"] = settings.service_version
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if predictor.model is not None else "model_missing",
        "service_version": settings.service_version,
        "model_version": predictor.model_version,
        "policy_threshold": predictor.policy_threshold,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {"ready": predictor.model is not None, "model_loaded": predictor.model is not None, "model_version": predictor.model_version}


@app.get("/version")
def version() -> dict[str, Any]:
    return {
        "service_version": settings.service_version,
        "model_version": predictor.model_version,
        "policy_version": predictor.policy.version,
        "feature_schema_version": predictor.feature_schema_version,
        "environment": settings.environment,
    }


@app.get("/decision-contract")
def decision_contract() -> dict[str, Any]:
    return {
        "contract_version": "decision-contract-v1",
        "model_version": predictor.model_version,
        "policy_version": predictor.policy.version,
        "feature_schema_version": predictor.feature_schema_version,
        "score_range": [0.0, 1.0],
        "actions": ["no_support", "cash_buffer_warning", "investment_support", "risk_review"],
        "review_routes": ["auto_serve", "manual_review"],
        "audit_fields": ["decision_id", "audit_event_id", "model_version", "policy_version", "feature_schema_version"],
        "boundary": "Synthetic engineering demonstration; not financial advice.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, Any]:
    PREDICTION_REQUEST_COUNT.inc()
    start = time.perf_counter()
    try:
        result = predictor.predict(request)
        DECISION_COUNT.labels(result["recommended_action"], result["review_route"]).inc()
        return result
    except Exception as exc:
        PREDICTION_ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail="Prediction failed. Inspect structured service logs using the request ID.") from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictionRequest) -> dict[str, Any]:
    result = predictor.predict(request)
    return {
        "customer_id": request.customer_id,
        "decision_id": result["decision_id"],
        "reason_codes": result["reason_codes"],
        "policy_reasons": result["policy_reasons"],
        "explanation": (
            "The model score is interpreted by a separate policy layer. Reason codes describe input and score signals; "
            "policy reasons describe why the action was chosen. This is not financial advice."
        ),
        "model_version": result["model_version"],
        "policy_version": result["policy_version"],
        "audit_event_id": result["audit_event_id"],
    }


@app.post("/review-route", response_model=ReviewRouteResponse)
def route_for_review(request: PredictionRequest) -> dict[str, Any]:
    return review_route(request, predictor.predict(request))


@app.post("/shadow-predict", response_model=ShadowPredictionResponse)
def shadow_predict(request: PredictionRequest) -> dict[str, Any]:
    from src.serving.shadow import shadow_predict as run_shadow_predict

    return run_shadow_predict(request, predictor)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
