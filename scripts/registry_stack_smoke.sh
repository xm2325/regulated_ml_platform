#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.registry.yml}
OUT_DIR=${OUT_DIR:-reports/registry_integration}
mkdir -p "$OUT_DIR"

cleanup() {
  status=$?
  docker compose -f "$COMPOSE_FILE" logs --no-color > "$OUT_DIR/compose.log" 2>&1 || true
  docker compose -f "$COMPOSE_FILE" down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

docker compose -f "$COMPOSE_FILE" down -v --remove-orphans >/dev/null 2>&1 || true
docker compose -f "$COMPOSE_FILE" up -d --build postgres minio minio-init mlflow

for _ in {1..90}; do
  if curl --fail --silent http://127.0.0.1:5000/health >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:5000/health > "$OUT_DIR/mlflow_health.txt"

run_registry() {
  docker compose -f "$COMPOSE_FILE" --profile registry run --rm registry-cli "$@"
}

run_registry register \
  --model-version 0.6.0-ci-a \
  --output "$OUT_DIR/register_a.json"
run_registry promote \
  --gate reports/promotion_gate.json \
  --output "$OUT_DIR/promote_a.json"

run_registry register \
  --model-version 0.6.0-ci-b \
  --output "$OUT_DIR/register_b.json"
run_registry promote \
  --gate reports/promotion_gate.json \
  --output "$OUT_DIR/promote_b.json"

run_registry rollback \
  --reason "CI rollback drill after second controlled promotion" \
  --output "$OUT_DIR/rollback_b.json"
run_registry status --output "$OUT_DIR/status_before_api.json"
run_registry verify \
  --alias champion \
  --request examples/review_request.json \
  --output "$OUT_DIR/verify_before_api.json"

# Start the same API image in registry mode. Strict startup requires a valid champion.
docker compose -f "$COMPOSE_FILE" --profile runtime up -d --build api-registry-runtime
for _ in {1..90}; do
  if curl --fail --silent http://127.0.0.1:8002/ready > "$OUT_DIR/api_ready_initial.json" 2>/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8002/version > "$OUT_DIR/api_version_initial.json"
initial_version=$(python -c 'import json; print(json.load(open("reports/registry_integration/api_version_initial.json"))["registry_model_version"])')

# Register and promote a third version while the API stays up. The background poller must switch it in memory.
run_registry register \
  --model-version 0.6.0-ci-c \
  --output "$OUT_DIR/register_c.json"
run_registry promote \
  --gate reports/promotion_gate.json \
  --output "$OUT_DIR/promote_c.json"
promoted_version=$(python -c 'import json; print(json.load(open("reports/registry_integration/promote_c.json"))["promoted_version"])')

for _ in {1..60}; do
  curl --fail --silent http://127.0.0.1:8002/version > "$OUT_DIR/api_version_after_promote.json"
  current=$(python -c 'import json; print(json.load(open("reports/registry_integration/api_version_after_promote.json"))["registry_model_version"])')
  if [[ "$current" == "$promoted_version" ]]; then
    break
  fi
  sleep 2
done
curl --fail --silent -X POST http://127.0.0.1:8002/predict \
  -H 'Content-Type: application/json' \
  -d @examples/review_request.json > "$OUT_DIR/api_prediction_after_promote.json"
curl --fail --silent http://127.0.0.1:8002/runtime/model > "$OUT_DIR/api_runtime_after_promote.json"

# Roll back the registry alias without restarting the API; the poller must restore the former champion.
run_registry rollback \
  --reason "CI live API rollback drill" \
  --output "$OUT_DIR/rollback_c.json"
for _ in {1..60}; do
  curl --fail --silent http://127.0.0.1:8002/version > "$OUT_DIR/api_version_after_rollback.json"
  current=$(python -c 'import json; print(json.load(open("reports/registry_integration/api_version_after_rollback.json"))["registry_model_version"])')
  if [[ "$current" == "$initial_version" ]]; then
    break
  fi
  sleep 2
done
run_registry status --output "$OUT_DIR/status_final.json"
run_registry sync \
  --alias champion \
  --output-dir "$OUT_DIR/synced_champion" \
  --output "$OUT_DIR/sync.json"

python - <<'PY'
import json
from pathlib import Path

root = Path("reports/registry_integration")
initial = json.loads((root / "api_version_initial.json").read_text())
after_promote = json.loads((root / "api_version_after_promote.json").read_text())
after_rollback = json.loads((root / "api_version_after_rollback.json").read_text())
prediction = json.loads((root / "api_prediction_after_promote.json").read_text())
runtime = json.loads((root / "api_runtime_after_promote.json").read_text())
promote = json.loads((root / "promote_c.json").read_text())
status = json.loads((root / "status_final.json").read_text())

assert initial["model_source"] == "registry"
assert initial["runtime_state"] == "ready_registry"
assert after_promote["registry_model_version"] == str(promote["promoted_version"])
assert after_promote["registry_model_version"] != initial["registry_model_version"]
assert prediction["model_source"] == "registry"
assert prediction["registry_model_version"] == after_promote["registry_model_version"]
assert runtime["reload_successes"] >= 2
assert after_rollback["registry_model_version"] == initial["registry_model_version"]
assert after_rollback["registry_model_version"] != after_promote["registry_model_version"]
assert status["aliases"]["champion"]["version"] == after_rollback["registry_model_version"]
for name in ["model.joblib", "metadata.json", "model_metrics.json", "promotion_gate.json", "registry_provenance.json"]:
    path = root / "synced_champion" / name
    assert path.is_file() and path.stat().st_size > 0, path
print(json.dumps({
    "status": "PASS",
    "initial_registry_version": initial["registry_model_version"],
    "hot_reloaded_registry_version": after_promote["registry_model_version"],
    "rolled_back_registry_version": after_rollback["registry_model_version"],
    "reload_successes": runtime["reload_successes"],
}, indent=2))
PY
