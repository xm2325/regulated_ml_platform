from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from src.core.config import settings
from src.core.telemetry import log_event
from src.registry.lifecycle import RegistryConfig, sync_alias
from src.serving.predictor import ModelPredictor
from src.serving.schemas import PredictionRequest


class RuntimeModelError(RuntimeError):
    """Raised when a candidate serving bundle cannot be trusted or loaded."""


class RegistryBundleProvider(Protocol):
    def sync(self, alias: str, output_dir: str | Path) -> dict[str, Any]: ...


class MlflowRegistryBundleProvider:
    def __init__(self) -> None:
        self.config = RegistryConfig(
            tracking_uri=settings.registry_tracking_uri,
            registry_uri=settings.registry_uri,
            registered_model_name=settings.registry_model_name,
            champion_alias=settings.registry_alias,
        )

    def sync(self, alias: str, output_dir: str | Path) -> dict[str, Any]:
        return sync_alias(self.config, alias, output_dir)


@dataclass
class RuntimeStatus:
    requested_source: str
    active_source: str
    runtime_state: str
    registry_model_name: str | None = None
    registry_alias: str | None = None
    registry_version: str | None = None
    registry_run_id: str | None = None
    loaded_at: str | None = None
    last_reload_at: str | None = None
    last_reload_status: str | None = None
    last_reload_error: str | None = None
    reload_attempts: int = 0
    reload_successes: int = 0
    reload_failures: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _smoke_request() -> PredictionRequest:
    return PredictionRequest(
        customer_id="runtime_smoke_test",
        age=45,
        annual_income=52000,
        cash_balance=24000,
        investment_balance=15000,
        debt_balance=3500,
        risk_score=0.45,
        recent_activity_count=5,
        account_type="isa",
        employment_status="employed",
    )


class ModelRuntimeManager:
    """Own the active predictor and switch registry versions without interrupting in-flight requests."""

    REQUIRED_FILES = ("model.joblib", "metadata.json", "model_metrics.json", "promotion_gate.json")

    def __init__(
        self,
        provider: RegistryBundleProvider | None = None,
        model_source: str | None = None,
        cache_dir: str | Path | None = None,
        strict_startup: bool | None = None,
    ) -> None:
        requested_source = (model_source or settings.model_source).strip().lower()
        if requested_source not in {"local", "registry"}:
            raise RuntimeModelError("MODEL_SOURCE must be 'local' or 'registry'")

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._reload_thread: threading.Thread | None = None
        self._provider = provider
        self._cache_root = Path(cache_dir or settings.registry_cache_dir)
        self._strict_startup = settings.registry_strict_startup if strict_startup is None else strict_startup
        self._requested_source = requested_source

        local_predictor = ModelPredictor(settings.model_path, settings.metadata_path)
        self._predictor = local_predictor
        self._status = RuntimeStatus(
            requested_source=requested_source,
            active_source="local",
            runtime_state="ready_local" if requested_source == "local" else "degraded_local_fallback",
            loaded_at=_utc_now(),
        )

        if requested_source == "registry":
            self._cache_root.mkdir(parents=True, exist_ok=True)
            self._load_cached_bundle_if_available()
            try:
                self.reload_from_registry(force=True)
            except Exception as exc:
                self._mark_reload_failure(exc)
                if self._strict_startup and self._status.active_source != "registry":
                    raise RuntimeModelError("Registry startup failed and strict startup is enabled") from exc

    def current_predictor(self) -> ModelPredictor:
        with self._lock:
            return self._predictor

    def status(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._status)

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        predictor = self.current_predictor()
        result = predictor.predict(request)
        status = self.status()
        result.update(
            {
                "model_source": status["active_source"],
                "runtime_state": status["runtime_state"],
                "registry_model_name": status["registry_model_name"],
                "registry_alias": status["registry_alias"],
                "registry_model_version": status["registry_version"],
            }
        )
        return result

    def start_background_reload(self) -> None:
        if self._requested_source != "registry" or not settings.registry_hot_reload_enabled:
            return
        if settings.registry_reload_interval_seconds <= 0:
            return
        with self._lock:
            if self._reload_thread and self._reload_thread.is_alive():
                return
            self._stop_event.clear()
            self._reload_thread = threading.Thread(target=self._reload_loop, name="registry-model-reloader", daemon=True)
            self._reload_thread.start()

    def stop_background_reload(self) -> None:
        self._stop_event.set()
        thread = self._reload_thread
        if thread and thread.is_alive():
            thread.join(timeout=min(max(settings.registry_reload_interval_seconds, 1.0), 5.0))

    def _reload_loop(self) -> None:
        interval = max(settings.registry_reload_interval_seconds, 1.0)
        while not self._stop_event.wait(interval):
            try:
                self.reload_from_registry()
            except Exception as exc:  # pragma: no cover - exercised by live integration tests
                self._mark_reload_failure(exc)

    def reload_from_registry(self, force: bool = False) -> dict[str, Any]:
        if self._requested_source != "registry":
            return {"status": "disabled", "reason": "MODEL_SOURCE is not registry"}
        provider = self._provider or MlflowRegistryBundleProvider()
        with self._lock:
            self._status.reload_attempts += 1

        self._cache_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".registry-candidate-", dir=self._cache_root))
        try:
            provenance = provider.sync(settings.registry_alias, staging)
            registry_version = str(provenance.get("registry_version", ""))
            if not registry_version:
                raise RuntimeModelError("Registry bundle is missing registry_version provenance")

            with self._lock:
                if not force and self._status.registry_version == registry_version and self._status.active_source == "registry":
                    self._status.last_reload_at = _utc_now()
                    self._status.last_reload_status = "unchanged"
                    self._status.last_reload_error = None
                    return {"status": "unchanged", "registry_version": registry_version}

            candidate = self._validate_candidate_bundle(staging, provenance)
            final_dir = self._publish_verified_cache(staging, registry_version, provenance)
            candidate = ModelPredictor(final_dir / "model.joblib", final_dir / "metadata.json")
            probability = candidate.probability(_smoke_request())
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeModelError("Verified cache produced an invalid smoke probability")

            with self._lock:
                self._predictor = candidate
                self._status.active_source = "registry"
                self._status.runtime_state = "ready_registry"
                self._status.registry_model_name = str(provenance.get("registered_model_name") or settings.registry_model_name)
                self._status.registry_alias = str(provenance.get("alias") or settings.registry_alias)
                self._status.registry_version = registry_version
                self._status.registry_run_id = str(provenance.get("run_id") or "") or None
                self._status.loaded_at = _utc_now()
                self._status.last_reload_at = self._status.loaded_at
                self._status.last_reload_status = "reloaded"
                self._status.last_reload_error = None
                self._status.reload_successes += 1

            log_event(
                "registry_model_reload_succeeded",
                registry_model_name=self._status.registry_model_name,
                registry_alias=self._status.registry_alias,
                registry_version=registry_version,
                model_release_version=candidate.model_version,
            )
            return {"status": "reloaded", "registry_version": registry_version, "model_version": candidate.model_version}
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _validate_candidate_bundle(self, bundle: Path, provenance: dict[str, Any]) -> ModelPredictor:
        for name in self.REQUIRED_FILES:
            if not (bundle / name).is_file():
                raise RuntimeModelError(f"Registry bundle is missing required file: {name}")

        expected_hashes = provenance.get("sha256")
        if not isinstance(expected_hashes, dict):
            raise RuntimeModelError("Registry provenance does not contain SHA-256 evidence")
        for name in self.REQUIRED_FILES:
            expected = str(expected_hashes.get(name, ""))
            if not expected or _sha256(bundle / name) != expected:
                raise RuntimeModelError(f"Registry bundle checksum validation failed for {name}")

        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        gate = json.loads((bundle / "promotion_gate.json").read_text(encoding="utf-8"))
        if metadata.get("feature_schema_version") != settings.feature_schema_version:
            raise RuntimeModelError(
                f"Feature schema mismatch: expected {settings.feature_schema_version}, got {metadata.get('feature_schema_version')}"
            )
        if gate.get("status") != "PASS" or gate.get("release_recommendation") != "eligible_for_controlled_release":
            raise RuntimeModelError("Registry champion bundle is not eligible for controlled release")

        candidate = ModelPredictor(bundle / "model.joblib", bundle / "metadata.json")
        probability = candidate.probability(_smoke_request())
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeModelError("Registry candidate failed smoke prediction validation")
        return candidate

    def _publish_verified_cache(self, staging: Path, registry_version: str, provenance: dict[str, Any]) -> Path:
        releases = self._cache_root / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        final_dir = releases / registry_version
        if final_dir.exists():
            shutil.rmtree(final_dir)
        staging.replace(final_dir)

        pointer = {
            "registry_version": registry_version,
            "bundle_dir": str(final_dir),
            "registered_model_name": provenance.get("registered_model_name") or settings.registry_model_name,
            "alias": provenance.get("alias") or settings.registry_alias,
            "run_id": provenance.get("run_id"),
            "updated_at": _utc_now(),
        }
        temporary = self._cache_root / ".current.json.tmp"
        temporary.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self._cache_root / "current.json")
        return final_dir

    def _load_cached_bundle_if_available(self) -> None:
        pointer_path = self._cache_root / "current.json"
        if not pointer_path.is_file():
            return
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            bundle = Path(pointer["bundle_dir"])
            provenance = json.loads((bundle / "registry_provenance.json").read_text(encoding="utf-8"))
            candidate = self._validate_candidate_bundle(bundle, provenance)
        except Exception as exc:
            log_event("registry_cached_model_rejected", error_type=type(exc).__name__)
            return

        with self._lock:
            self._predictor = candidate
            self._status.active_source = "registry"
            self._status.runtime_state = "ready_registry_cached"
            self._status.registry_model_name = str(pointer.get("registered_model_name") or settings.registry_model_name)
            self._status.registry_alias = str(pointer.get("alias") or settings.registry_alias)
            self._status.registry_version = str(pointer.get("registry_version"))
            self._status.registry_run_id = str(pointer.get("run_id") or "") or None
            self._status.loaded_at = _utc_now()
            self._status.last_reload_status = "loaded_cached_registry"

    def _mark_reload_failure(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._status.reload_failures += 1
            self._status.last_reload_at = _utc_now()
            self._status.last_reload_status = "failed"
            self._status.last_reload_error = error[:500]
            if self._requested_source == "registry":
                self._status.runtime_state = (
                    "degraded_registry_cached" if self._status.active_source == "registry" else "degraded_local_fallback"
                )
        log_event("registry_model_reload_failed", error_type=type(exc).__name__, active_source=self._status.active_source)
