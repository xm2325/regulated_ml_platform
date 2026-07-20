from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src.core.config import settings
from src.core.telemetry import log_event
from src.registry.lifecycle import RegistryConfig, promote_challenger
from src.serving.predictor import ModelPredictor
from src.serving.runtime_manager import MlflowRegistryBundleProvider, RegistryBundleProvider, RuntimeModelError, _sha256
from src.serving.schemas import PredictionRequest


class ChampionRuntime(Protocol):
    def predict(self, request: PredictionRequest) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...

    def reload_from_registry(self, force: bool = False) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CanaryLimits:
    min_requests: int
    min_challenger_requests: int
    max_action_disagreement_rate: float
    max_probability_delta_p95: float
    max_challenger_error_rate: float
    max_latency_ratio: float
    max_manual_review_rate_increase: float


@dataclass
class CanaryRecord:
    assigned_role: str
    champion_error: bool
    challenger_error: bool
    action_changed: bool
    abs_probability_delta: float | None
    champion_latency_ms: float | None
    challenger_latency_ms: float | None
    champion_manual_review: bool
    challenger_manual_review: bool


@dataclass
class CanaryState:
    enabled: bool
    state: str
    traffic_percent: float
    challenger_registry_version: str | None = None
    challenger_run_id: str | None = None
    challenger_release_version: str | None = None
    loaded_at: float | None = None
    last_refresh_status: str | None = None
    last_refresh_error: str | None = None
    last_evaluation_decision: str | None = None
    last_evaluation_reasons: list[str] | None = None
    last_transition: str | None = None
    promoted_registry_version: str | None = None


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


class CanaryController:
    """Run a verified challenger beside the champion and control canary transitions."""

    REQUIRED_FILES = ("model.joblib", "metadata.json", "model_metrics.json", "promotion_gate.json")

    def __init__(
        self,
        runtime: ChampionRuntime,
        provider: RegistryBundleProvider | None = None,
        enabled: bool | None = None,
        traffic_percent: float | None = None,
        min_requests: int | None = None,
        min_challenger_requests: int | None = None,
        auto_promote: bool | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._runtime = runtime
        self._provider = provider
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cache_root = Path(cache_dir or settings.registry_cache_dir) / "canary"
        self._traffic_percent = min(max(float(settings.canary_traffic_percent if traffic_percent is None else traffic_percent), 0.0), 100.0)
        self._enabled = settings.canary_enabled if enabled is None else enabled
        self._auto_promote = settings.canary_auto_promote_enabled if auto_promote is None else auto_promote
        self._limits = CanaryLimits(
            min_requests=max(1, settings.canary_min_requests if min_requests is None else min_requests),
            min_challenger_requests=max(
                1,
                settings.canary_min_challenger_requests if min_challenger_requests is None else min_challenger_requests,
            ),
            max_action_disagreement_rate=settings.canary_max_action_disagreement_rate,
            max_probability_delta_p95=settings.canary_max_probability_delta_p95,
            max_challenger_error_rate=settings.canary_max_challenger_error_rate,
            max_latency_ratio=settings.canary_max_latency_ratio,
            max_manual_review_rate_increase=settings.canary_max_manual_review_rate_increase,
        )
        self._records: deque[CanaryRecord] = deque(maxlen=max(settings.canary_window_size, self._limits.min_requests))
        self._challenger: ModelPredictor | None = None
        self._challenger_gate: dict[str, Any] | None = None
        self._state = CanaryState(
            enabled=self._enabled,
            state="disabled" if not self._enabled else "waiting_for_challenger",
            traffic_percent=self._traffic_percent,
        )
        if self._enabled:
            self.refresh_challenger(force=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_background_control(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._control_loop, name="canary-controller", daemon=True)
            self._thread.start()

    def stop_background_control(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=min(max(settings.canary_evaluation_interval_seconds, 1.0), 5.0))

    def _control_loop(self) -> None:
        last_refresh = 0.0
        while not self._stop_event.wait(max(settings.canary_evaluation_interval_seconds, 0.5)):
            now = time.monotonic()
            if now - last_refresh >= max(settings.canary_refresh_interval_seconds, 1.0):
                try:
                    self.refresh_challenger()
                except Exception:
                    pass
                last_refresh = now
            evaluation = self.evaluate()
            if evaluation["decision"] == "STOP":
                self._stop_canary(evaluation["reasons"])
            elif evaluation["decision"] == "PASS" and self._auto_promote:
                self._promote_canary(evaluation)

    def _provider_instance(self) -> RegistryBundleProvider:
        return self._provider or MlflowRegistryBundleProvider()

    def _validate_bundle(self, bundle: Path, provenance: dict[str, Any]) -> tuple[ModelPredictor, dict[str, Any]]:
        for name in self.REQUIRED_FILES:
            if not (bundle / name).is_file():
                raise RuntimeModelError(f"Canary bundle is missing required file: {name}")
        expected_hashes = provenance.get("sha256")
        if not isinstance(expected_hashes, dict):
            raise RuntimeModelError("Canary provenance does not contain SHA-256 evidence")
        for name in self.REQUIRED_FILES:
            expected = str(expected_hashes.get(name, ""))
            if not expected or _sha256(bundle / name) != expected:
                raise RuntimeModelError(f"Canary bundle checksum validation failed for {name}")
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        gate = json.loads((bundle / "promotion_gate.json").read_text(encoding="utf-8"))
        if metadata.get("feature_schema_version") != settings.feature_schema_version:
            raise RuntimeModelError(
                f"Canary feature schema mismatch: expected {settings.feature_schema_version}, got {metadata.get('feature_schema_version')}"
            )
        if gate.get("status") != "PASS" or gate.get("release_recommendation") != "eligible_for_controlled_release":
            raise RuntimeModelError("Canary challenger is not eligible for controlled release")
        predictor = ModelPredictor(bundle / "model.joblib", bundle / "metadata.json")
        return predictor, gate

    def refresh_challenger(self, force: bool = False) -> dict[str, Any]:
        if not self._enabled:
            return {"status": "disabled"}
        runtime_status = self._runtime.status()
        if runtime_status.get("requested_source") != "registry":
            with self._lock:
                self._state.state = "disabled_non_registry_runtime"
            return {"status": "disabled", "reason": "canary requires MODEL_SOURCE=registry"}
        if self._state.state == "promoted" and not force:
            return {"status": "unchanged", "reason": "canary already promoted"}

        self._cache_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".challenger-", dir=self._cache_root))
        try:
            provenance = self._provider_instance().sync(settings.registry_challenger_alias, staging)
            version = str(provenance.get("registry_version", ""))
            if not version:
                raise RuntimeModelError("Canary challenger is missing registry_version provenance")
            with self._lock:
                if not force and version == self._state.challenger_registry_version and self._challenger is not None:
                    self._state.last_refresh_status = "unchanged"
                    self._state.last_refresh_error = None
                    return {"status": "unchanged", "registry_version": version}
            challenger, gate = self._validate_bundle(staging, provenance)
            with self._lock:
                self._challenger = challenger
                self._challenger_gate = gate
                self._records.clear()
                self._state.state = "warming"
                self._state.challenger_registry_version = version
                self._state.challenger_run_id = str(provenance.get("run_id") or "") or None
                self._state.challenger_release_version = challenger.model_version
                self._state.loaded_at = time.time()
                self._state.last_refresh_status = "loaded"
                self._state.last_refresh_error = None
                self._state.last_evaluation_decision = "WAIT"
                self._state.last_evaluation_reasons = ["minimum evidence not yet collected"]
                self._state.last_transition = None
                self._state.promoted_registry_version = None
            log_event("canary_challenger_loaded", registry_version=version, release_version=challenger.model_version)
            return {"status": "loaded", "registry_version": version, "model_version": challenger.model_version}
        except Exception as exc:
            with self._lock:
                if self._challenger is None:
                    self._state.state = "waiting_for_challenger"
                self._state.last_refresh_status = "failed"
                self._state.last_refresh_error = f"{type(exc).__name__}: {exc}"[:500]
            log_event("canary_challenger_refresh_failed", error_type=type(exc).__name__)
            if force:
                return {"status": "waiting_for_challenger", "error_type": type(exc).__name__}
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _assigned_role(self, request: PredictionRequest) -> str:
        with self._lock:
            active = self._enabled and self._challenger is not None and self._state.state in {"warming", "healthy"}
        if not active or self._traffic_percent <= 0:
            return "champion"
        key = request.request_id or request.customer_id
        digest = hashlib.sha256(f"{settings.canary_assignment_seed}:{key}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 10_000
        return "challenger" if bucket < round(self._traffic_percent * 100) else "champion"

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        assigned_role = self._assigned_role(request)
        champion_result: dict[str, Any] | None = None
        challenger_result: dict[str, Any] | None = None
        champion_error = False
        challenger_error = False

        champion_start = time.perf_counter()
        try:
            champion_result = self._runtime.predict(request)
        except Exception:
            champion_error = True
        champion_latency_ms = (time.perf_counter() - champion_start) * 1000

        with self._lock:
            challenger = self._challenger
            challenger_version = self._state.challenger_registry_version
            state = self._state.state

        challenger_latency_ms: float | None = None
        if challenger is not None and state in {"warming", "healthy"}:
            challenger_start = time.perf_counter()
            try:
                challenger_result = challenger.predict(request)
            except Exception:
                challenger_error = True
            challenger_latency_ms = (time.perf_counter() - challenger_start) * 1000

        if champion_result is None:
            raise RuntimeError("Champion prediction failed during canary request")

        action_changed = bool(
            challenger_result is not None
            and challenger_result.get("recommended_action") != champion_result.get("recommended_action")
        )
        delta = None
        if challenger_result is not None:
            delta = abs(float(challenger_result["support_probability"]) - float(champion_result["support_probability"]))

        served_role = assigned_role
        if assigned_role == "challenger" and challenger_result is None:
            served_role = "champion_fallback"
        served = dict(challenger_result if served_role == "challenger" else champion_result)

        self._record(
            CanaryRecord(
                assigned_role=served_role,
                champion_error=champion_error,
                challenger_error=challenger_error,
                action_changed=action_changed,
                abs_probability_delta=delta,
                champion_latency_ms=champion_latency_ms,
                challenger_latency_ms=challenger_latency_ms,
                champion_manual_review=champion_result.get("review_route") == "manual_review",
                challenger_manual_review=bool(challenger_result and challenger_result.get("review_route") == "manual_review"),
            )
        )

        runtime_status = self._runtime.status()
        served["served_model_role"] = served_role
        served["canary_state"] = self.status()["state"]
        served["canary_assignment"] = assigned_role
        served["comparison_champion_registry_version"] = runtime_status.get("registry_version")
        served["comparison_challenger_registry_version"] = challenger_version
        if served_role == "challenger":
            served["model_source"] = "registry"
            served["registry_model_name"] = settings.registry_model_name
            served["registry_alias"] = settings.registry_challenger_alias
            served["registry_model_version"] = challenger_version
            served["runtime_state"] = runtime_status.get("runtime_state", "ready_registry")
        return served

    def _record(self, record: CanaryRecord) -> None:
        with self._lock:
            self._records.append(record)

    def evaluate(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
            state = self._state.state
            challenger_version = self._state.challenger_registry_version
        metrics = self._metrics(records)
        reasons: list[str] = []
        if state in {"disabled", "disabled_non_registry_runtime", "waiting_for_challenger", "promoted", "stopped"}:
            decision = "WAIT" if state not in {"promoted", "stopped"} else state.upper()
            return {"decision": decision, "reasons": [f"canary state is {state}"], "metrics": metrics}
        if metrics["comparisons"] < self._limits.min_requests:
            reasons.append(f"comparisons {metrics['comparisons']} < minimum {self._limits.min_requests}")
        if metrics["challenger_served"] < self._limits.min_challenger_requests:
            reasons.append(
                f"challenger served {metrics['challenger_served']} < minimum {self._limits.min_challenger_requests}"
            )
        if reasons:
            decision = "WAIT"
        else:
            if metrics["action_disagreement_rate"] > self._limits.max_action_disagreement_rate:
                reasons.append("action disagreement rate exceeded limit")
            if metrics["probability_delta_p95"] is not None and metrics["probability_delta_p95"] > self._limits.max_probability_delta_p95:
                reasons.append("p95 probability delta exceeded limit")
            if metrics["challenger_error_rate"] > self._limits.max_challenger_error_rate:
                reasons.append("challenger error rate exceeded limit")
            if metrics["latency_ratio"] is not None and metrics["latency_ratio"] > self._limits.max_latency_ratio:
                reasons.append("challenger/champion mean latency ratio exceeded limit")
            if metrics["manual_review_rate_increase"] > self._limits.max_manual_review_rate_increase:
                reasons.append("manual review rate increase exceeded limit")
            decision = "STOP" if reasons else "PASS"
        with self._lock:
            self._state.last_evaluation_decision = decision
            self._state.last_evaluation_reasons = reasons or ["all configured online safety limits passed"]
            if decision == "PASS" and self._state.state == "warming":
                self._state.state = "healthy"
        return {"decision": decision, "reasons": reasons, "metrics": metrics, "challenger_registry_version": challenger_version}

    def _metrics(self, records: list[CanaryRecord]) -> dict[str, Any]:
        comparisons = sum(1 for item in records if not item.challenger_error and item.abs_probability_delta is not None)
        challenger_served = sum(1 for item in records if item.assigned_role == "challenger")
        champion_served = sum(1 for item in records if item.assigned_role in {"champion", "champion_fallback"})
        disagreements = sum(1 for item in records if item.action_changed)
        challenger_errors = sum(1 for item in records if item.challenger_error)
        deltas = [float(item.abs_probability_delta) for item in records if item.abs_probability_delta is not None]
        champion_latencies = [float(item.champion_latency_ms) for item in records if item.champion_latency_ms is not None]
        challenger_latencies = [float(item.challenger_latency_ms) for item in records if item.challenger_latency_ms is not None]
        champion_manual = sum(1 for item in records if item.champion_manual_review)
        challenger_manual = sum(1 for item in records if item.challenger_manual_review)
        champion_mean = sum(champion_latencies) / len(champion_latencies) if champion_latencies else None
        challenger_mean = sum(challenger_latencies) / len(challenger_latencies) if challenger_latencies else None
        latency_ratio = None
        if champion_mean is not None and challenger_mean is not None and champion_mean > 0:
            latency_ratio = challenger_mean / champion_mean
        return {
            "window_requests": len(records),
            "comparisons": comparisons,
            "champion_served": champion_served,
            "challenger_served": challenger_served,
            "action_disagreements": disagreements,
            "action_disagreement_rate": _safe_rate(disagreements, comparisons),
            "probability_delta_p95": _percentile95(deltas),
            "challenger_errors": challenger_errors,
            "challenger_error_rate": _safe_rate(challenger_errors, len(records)),
            "champion_mean_latency_ms": champion_mean,
            "challenger_mean_latency_ms": challenger_mean,
            "latency_ratio": latency_ratio,
            "champion_manual_review_rate": _safe_rate(champion_manual, comparisons),
            "challenger_manual_review_rate": _safe_rate(challenger_manual, comparisons),
            "manual_review_rate_increase": _safe_rate(challenger_manual, comparisons) - _safe_rate(champion_manual, comparisons),
        }

    def _stop_canary(self, reasons: list[str]) -> None:
        with self._lock:
            if self._state.state in {"stopped", "promoted"}:
                return
            self._state.state = "stopped"
            self._state.last_transition = "automatic_stop"
            self._state.last_evaluation_reasons = list(reasons)
        log_event("canary_automatic_stop", reasons=reasons, registry_version=self._state.challenger_registry_version)

    def _promote_canary(self, evaluation: dict[str, Any]) -> None:
        with self._lock:
            if self._state.state not in {"healthy", "warming"} or self._challenger_gate is None:
                return
            expected_version = self._state.challenger_registry_version
            gate = dict(self._challenger_gate)
        config = RegistryConfig(
            tracking_uri=settings.registry_tracking_uri,
            registry_uri=settings.registry_uri,
            registered_model_name=settings.registry_model_name,
            champion_alias=settings.registry_alias,
            challenger_alias=settings.registry_challenger_alias,
        )
        try:
            result = promote_challenger(
                config,
                gate,
                expected_challenger_version=expected_version,
            )
            self._runtime.reload_from_registry(force=True)
        except Exception as exc:
            with self._lock:
                self._state.last_transition = "promotion_failed"
                self._state.last_refresh_error = f"{type(exc).__name__}: {exc}"[:500]
            log_event("canary_promotion_failed", error_type=type(exc).__name__, registry_version=expected_version)
            return
        with self._lock:
            self._state.state = "promoted"
            self._state.last_transition = "automatic_promotion"
            self._state.promoted_registry_version = str(result["promoted_version"])
            self._state.last_evaluation_decision = "PASS"
            self._state.last_evaluation_reasons = evaluation.get("reasons") or ["all configured online safety limits passed"]
        log_event("canary_automatic_promotion", registry_version=result["promoted_version"])

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = asdict(self._state)
            records = list(self._records)
            limits = asdict(self._limits)
        state["metrics"] = self._metrics(records)
        state["limits"] = limits
        state["auto_promote_enabled"] = self._auto_promote
        state["assignment_seed"] = settings.canary_assignment_seed
        state.pop("last_refresh_error", None)
        return state
