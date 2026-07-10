from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from src.core.audit import emit_decision_audit
from src.core.config import settings
from src.core.telemetry import log_event
from src.serving.predictor import explain_request, load_artifacts, predict_request
from src.serving.review_workflow import audit_event_id, review_route
from src.serving.schemas import (
    ExplainResponse,
    PredictionRequest,
    PredictionResponse,
    ReviewRouteResponse,
    ShadowPredictionResponse,
)
from src.serving.shadow import shadow_predict

REQUEST_COUNT = Counter("regulated_ai_requests_total", "Total API requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("regulated_ai_request_latency_seconds", "Request latency", ["method", "path"])
PREDICTION_COUNT = Counter("regulated_ai_predictions_total", "Prediction results", ["action", "confidence"])
REVIEW_ROUTE_COUNT = Counter("regulated_ai_review_routes_total", "Review routing decisions", ["route"])
HARD_SAFETY_GATE_COUNT = Counter("regulated_ai_hard_safety_gate_total", "Hard safety-gate triggers")


def _decision_contract() -> dict[str, Any]:
    _, metadata = load_artifacts()
    return {
        "contract_version": "model-contract-v1",
        "model_version": str(metadata["model_version"]),
        "policy_version": str(metadata["policy_version"]),
        "feature_schema_version": str(metadata["feature_schema_version"]),
        "policy_threshold": float(metadata["policy_threshold"]),
        "allowed_actions": ["no_support", "cash_buffer_warning", "investment_support", "risk_review"],
        "review_routes": ["auto_serve", "manual_review"],
        "decision_fields": [
            "decision_id",
            "audit_event_id",
            "model_version",
            "policy_version",
            "feature_schema_version",
            "support_probability",
            "recommended_action",
            "review_route",
        ],
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    log_event("service_start", service=settings.service_name, version=settings.service_version, environment=settings.environment)
    yield
    log_event("service_stop", service=settings.service_name, version=settings.service_version)


app = FastAPI(
    title="Regulated AI MLOps Platform",
    version=settings.service_version,
    description=(
        "Production-style synthetic financial AI service with separate model and policy layers, human review routing, audit identifiers, "
        "Prometheus metrics, and champion-challenger shadow scoring."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency = time.perf_counter() - start
        REQUEST_COUNT.labels(method=request.method, path=request.url.path, status="500").inc()
        REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(latency)
        log_event("request_failed", request_id=request_id, method=request.method, path=request.url.path, latency_ms=round(latency * 1000, 3))
        raise
    latency = time.perf_counter() - start
    REQUEST_COUNT.labels(method=request.method, path=request.url.path, status=str(response.status_code)).inc()
    REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(latency)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    log_event("request_complete", request_id=request_id, method=request.method, path=request.url.path, status=response.status_code, latency_ms=round(latency * 1000, 3))
    return response


@app.exception_handler(ValidationError)
async def validation_error_handler(_: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _, metadata = load_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "service": settings.service_name,
        "service_version": settings.service_version,
        "model_version": metadata["model_version"],
        "policy_version": metadata["policy_version"],
        "feature_schema_version": metadata["feature_schema_version"],
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        load_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, Any]:
    _, metadata = load_artifacts()
    return {
        "service_version": settings.service_version,
        "model_version": metadata["model_version"],
        "policy_version": metadata["policy_version"],
        "feature_schema_version": metadata["feature_schema_version"],
        "policy_threshold": metadata["policy_threshold"],
        "selected_model": metadata["selected_model"],
    }


@app.get("/decision-contract")
def decision_contract() -> dict[str, Any]:
    return _decision_contract()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    result = predict_request(payload)
    review = review_route(payload, result)
    result.update(
        {
            "review_route": review["review_route"],
            "review_reasons": review["review_reasons"],
            "audit_event_id": review["audit_event_id"],
        }
    )
    PREDICTION_COUNT.labels(action=result["recommended_action"], confidence=result["confidence"]).inc()
    REVIEW_ROUTE_COUNT.labels(route=result["review_route"]).inc()
    if result["hard_safety_gate_triggered"]:
        HARD_SAFETY_GATE_COUNT.inc()
    emit_decision_audit(
        {
            "audit_event_id": result["audit_event_id"],
            "decision_id": result["decision_id"],
            "request_id": request.state.request_id,
            "customer_id": payload.customer_id,
            "model_version": result["model_version"],
            "policy_version": result["policy_version"],
            "feature_schema_version": result["feature_schema_version"],
            "recommended_action": result["recommended_action"],
            "review_route": result["review_route"],
            "hard_safety_gate_triggered": result["hard_safety_gate_triggered"],
        }
    )
    return PredictionResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: PredictionRequest) -> ExplainResponse:
    result = explain_request(payload)
    prediction = predict_request(payload)
    result["audit_event_id"] = audit_event_id(payload, prediction)
    return ExplainResponse(**result)


@app.post("/review-route", response_model=ReviewRouteResponse)
def route_for_review(payload: PredictionRequest) -> ReviewRouteResponse:
    prediction = predict_request(payload)
    return ReviewRouteResponse(**review_route(payload, prediction))


@app.post("/shadow-predict", response_model=ShadowPredictionResponse)
def shadow(payload: PredictionRequest) -> ShadowPredictionResponse:
    champion = predict_request(payload)
    return ShadowPredictionResponse(**shadow_predict(payload, champion))


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
