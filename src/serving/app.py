from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.core.config import settings
from src.core.telemetry import log_event
from src.serving.canary import CanaryController
from src.serving.review_workflow import review_route
from src.serving.runtime_manager import ModelRuntimeManager
from src.serving.schemas import ExplainResponse, PredictionRequest, PredictionResponse, ReviewRouteResponse, ShadowPredictionResponse

runtime = ModelRuntimeManager()
canary = CanaryController(runtime)


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start_background_reload()
    canary.start_background_control()
    try:
        yield
    finally:
        canary.stop_background_control()
        runtime.stop_background_reload()


app = FastAPI(
    title="Regulated AI MLOps Platform",
    version=settings.service_version,
    description="A synthetic regulated-finance MLOps reference implementation with explicit model, policy, review, registry, canary, and audit layers.",
    lifespan=lifespan,
)

PREDICTION_REQUEST_COUNT = Counter("prediction_request_count", "Number of prediction requests")
PREDICTION_ERROR_COUNT = Counter("prediction_error_count", "Number of prediction errors")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction request latency")
HTTP_REQUESTS = Counter("regulated_ai_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("regulated_ai_http_request_duration_seconds", "HTTP request duration", ["method", "path"])
DECISION_COUNT = Counter("regulated_ai_decisions_total", "Model-policy decisions", ["action", "review_route"])
MODEL_READY = Gauge("regulated_ai_model_ready", "Whether a verified serving model is loaded")
REGISTRY_ACTIVE = Gauge("regulated_ai_registry_model_active", "Whether the active model came from the registry champion alias")
RUNTIME_DEGRADED = Gauge("regulated_ai_runtime_degraded", "Whether registry serving is using a degraded fallback state")
MODEL_RELOAD_ATTEMPTS = Gauge("regulated_ai_model_reload_attempts", "Registry model reload attempts in this process")
MODEL_RELOAD_SUCCESSES = Gauge("regulated_ai_model_reload_successes", "Successful registry model reloads in this process")
MODEL_RELOAD_FAILURES = Gauge("regulated_ai_model_reload_failures", "Failed registry model reloads in this process")
CANARY_ENABLED = Gauge("regulated_ai_canary_enabled", "Whether canary routing is enabled")
CANARY_STATE = Gauge("regulated_ai_canary_state", "Canary state code: 0 disabled, 1 waiting, 2 warming, 3 healthy, 4 stopped, 5 promoted")
CANARY_TRAFFIC_PERCENT = Gauge("regulated_ai_canary_traffic_percent", "Configured percentage of requests assigned to challenger")
CANARY_CHALLENGER_SERVED = Gauge("regulated_ai_canary_challenger_served", "Challenger-served requests in the active evidence window")
CANARY_ACTION_DISAGREEMENT_RATE = Gauge("regulated_ai_canary_action_disagreement_rate", "Champion/challenger action disagreement rate")
CANARY_PROBABILITY_DELTA_P95 = Gauge("regulated_ai_canary_probability_delta_p95", "P95 absolute champion/challenger probability delta")
CANARY_CHALLENGER_ERROR_RATE = Gauge("regulated_ai_canary_challenger_error_rate", "Challenger prediction error rate")
CANARY_LATENCY_RATIO = Gauge("regulated_ai_canary_latency_ratio", "Mean challenger latency divided by mean champion latency")
CANARY_ASSIGNMENTS = Counter("regulated_ai_canary_assignments_total", "Served canary assignments", ["served_role"])


def _runtime_status() -> dict[str, Any]:
    status = runtime.status()
    predictor = runtime.current_predictor()
    MODEL_READY.set(1 if predictor.model is not None else 0)
    REGISTRY_ACTIVE.set(1 if status["active_source"] == "registry" else 0)
    RUNTIME_DEGRADED.set(1 if str(status["runtime_state"]).startswith("degraded") else 0)
    MODEL_RELOAD_ATTEMPTS.set(status["reload_attempts"])
    MODEL_RELOAD_SUCCESSES.set(status["reload_successes"])
    MODEL_RELOAD_FAILURES.set(status["reload_failures"])
    return status


def _canary_status() -> dict[str, Any]:
    status = canary.status()
    metrics = status["metrics"]
    state_codes = {
        "disabled": 0,
        "disabled_non_registry_runtime": 0,
        "waiting_for_challenger": 1,
        "warming": 2,
        "healthy": 3,
        "stopped": 4,
        "promoted": 5,
    }
    CANARY_ENABLED.set(1 if status["enabled"] else 0)
    CANARY_STATE.set(state_codes.get(str(status["state"]), -1))
    CANARY_TRAFFIC_PERCENT.set(float(status["traffic_percent"]))
    CANARY_CHALLENGER_SERVED.set(float(metrics["challenger_served"]))
    CANARY_ACTION_DISAGREEMENT_RATE.set(float(metrics["action_disagreement_rate"]))
    CANARY_PROBABILITY_DELTA_P95.set(float(metrics["probability_delta_p95"] or 0.0))
    CANARY_CHALLENGER_ERROR_RATE.set(float(metrics["challenger_error_rate"]))
    CANARY_LATENCY_RATIO.set(float(metrics["latency_ratio"] or 0.0))
    return status


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
    status = _runtime_status()
    predictor = runtime.current_predictor()
    degraded = str(status["runtime_state"]).startswith("degraded")
    return {
        "status": "degraded" if degraded else ("ok" if predictor.model is not None else "model_missing"),
        "service_version": settings.service_version,
        "model_version": predictor.model_version,
        "model_source": status["active_source"],
        "runtime_state": status["runtime_state"],
        "registry_model_version": status["registry_version"],
        "policy_threshold": predictor.policy_threshold,
        "canary_state": _canary_status()["state"],
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    status = _runtime_status()
    predictor = runtime.current_predictor()
    return {
        "ready": predictor.model is not None,
        "model_loaded": predictor.model is not None,
        "degraded": str(status["runtime_state"]).startswith("degraded"),
        "runtime_state": status["runtime_state"],
        "model_source": status["active_source"],
        "model_version": predictor.model_version,
        "registry_model_version": status["registry_version"],
        "canary_state": _canary_status()["state"],
    }


@app.get("/version")
def version() -> dict[str, Any]:
    status = _runtime_status()
    predictor = runtime.current_predictor()
    canary_status = _canary_status()
    return {
        "platform_version": settings.platform_version,
        "service_version": settings.service_version,
        "model_release_version": predictor.model_version,
        "policy_version": predictor.policy.version,
        "feature_schema_version": predictor.feature_schema_version,
        "environment": settings.environment,
        "git_commit": settings.git_commit,
        "model_source": status["active_source"],
        "runtime_state": status["runtime_state"],
        "registry_model_name": status["registry_model_name"],
        "registry_alias": status["registry_alias"],
        "registry_model_version": status["registry_version"],
        "registry_run_id": status["registry_run_id"],
        "last_reload_status": status["last_reload_status"],
        "canary_enabled": canary_status["enabled"],
        "canary_state": canary_status["state"],
        "challenger_registry_version": canary_status["challenger_registry_version"],
    }


@app.get("/runtime/model")
def runtime_model() -> dict[str, Any]:
    status = _runtime_status()
    status.pop("last_reload_error", None)
    status["model_release_version"] = runtime.current_predictor().model_version
    status["platform_version"] = settings.platform_version
    status["service_version"] = settings.service_version
    return status


@app.get("/canary/status")
def canary_status() -> dict[str, Any]:
    return _canary_status()


@app.get("/canary/evaluate")
def canary_evaluate() -> dict[str, Any]:
    return canary.evaluate()


@app.get("/decision-contract")
def decision_contract() -> dict[str, Any]:
    status = _runtime_status()
    predictor = runtime.current_predictor()
    return {
        "contract_version": "decision-contract-v3",
        "model_version": predictor.model_version,
        "model_source": status["active_source"],
        "registry_model_version": status["registry_version"],
        "policy_version": predictor.policy.version,
        "feature_schema_version": predictor.feature_schema_version,
        "score_range": [0.0, 1.0],
        "actions": ["no_support", "cash_buffer_warning", "investment_support", "risk_review"],
        "review_routes": ["auto_serve", "manual_review"],
        "serving_roles": ["champion", "challenger", "champion_fallback"],
        "audit_fields": [
            "decision_id",
            "audit_event_id",
            "model_version",
            "model_source",
            "registry_model_version",
            "served_model_role",
            "canary_assignment",
            "comparison_champion_registry_version",
            "comparison_challenger_registry_version",
            "policy_version",
            "feature_schema_version",
        ],
        "boundary": "Synthetic engineering demonstration; not financial advice.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, Any]:
    PREDICTION_REQUEST_COUNT.inc()
    start = time.perf_counter()
    try:
        result = canary.predict(request)
        DECISION_COUNT.labels(result["recommended_action"], result["review_route"]).inc()
        CANARY_ASSIGNMENTS.labels(result["served_model_role"]).inc()
        return result
    except Exception as exc:
        PREDICTION_ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail="Prediction failed. Inspect structured service logs using the request ID.") from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictionRequest) -> dict[str, Any]:
    result = canary.predict(request)
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
        "model_source": result["model_source"],
        "runtime_state": result["runtime_state"],
        "registry_model_version": result["registry_model_version"],
    }


@app.post("/review-route", response_model=ReviewRouteResponse)
def route_for_review(request: PredictionRequest) -> dict[str, Any]:
    return review_route(request, canary.predict(request))


@app.post("/shadow-predict", response_model=ShadowPredictionResponse)
def shadow_predict(request: PredictionRequest) -> dict[str, Any]:
    from src.serving.shadow import shadow_predict as run_shadow_predict

    return run_shadow_predict(request, runtime.current_predictor())


@app.get("/metrics")
def metrics() -> Response:
    _runtime_status()
    _canary_status()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
