#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.registry.yml}
OUT_DIR=${OUT_DIR:-reports/registry_integration}
mkdir -p "$OUT_DIR"

capture_runtime_diagnostics() {
  docker compose -f "$COMPOSE_FILE" --profile runtime --profile registry ps -a > "$OUT_DIR/compose_ps.txt" 2>&1 || true
  docker compose -f "$COMPOSE_FILE" --profile runtime logs --no-color api-registry-runtime > "$OUT_DIR/api_runtime.log" 2>&1 || true
}

cleanup() {
  status=$?
  capture_runtime_diagnostics
  docker compose -f "$COMPOSE_FILE" --profile runtime --profile registry logs --no-color > "$OUT_DIR/compose.log" 2>&1 || true
  docker compose -f "$COMPOSE_FILE" --profile runtime --profile registry down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

docker compose -f "$COMPOSE_FILE" --profile runtime --profile registry down -v --remove-orphans >/dev/null 2>&1 || true
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

# Establish a safe champion A and leave B as the challenger for live canary traffic.
run_registry register \
  --model-version 0.6.0-ci-a \
  --output "$OUT_DIR/register_a.json"
run_registry promote \
  --gate reports/promotion_gate.json \
  --output "$OUT_DIR/promote_a.json"
run_registry register \
  --model-version 0.6.0-ci-b \
  --output "$OUT_DIR/register_b.json"
run_registry status --output "$OUT_DIR/status_before_api.json"
run_registry verify \
  --alias champion \
  --request examples/review_request.json \
  --output "$OUT_DIR/verify_champion_before_api.json"
run_registry verify \
  --alias challenger \
  --request examples/review_request.json \
  --output "$OUT_DIR/verify_challenger_before_api.json"

if ! docker compose -f "$COMPOSE_FILE" --profile runtime up -d --build api-registry-runtime > "$OUT_DIR/api_compose_up.log" 2>&1; then
  cat "$OUT_DIR/api_compose_up.log"
  capture_runtime_diagnostics
  exit 1
fi
api_ready=0
for _ in {1..60}; do
  if curl --fail --silent http://127.0.0.1:8002/ready > "$OUT_DIR/api_ready_initial.json" 2>/dev/null; then
    api_ready=1
    break
  fi
  sleep 2
done
if [[ "$api_ready" != "1" ]]; then
  capture_runtime_diagnostics
  echo "Registry-backed API did not become ready." >&2
  cat "$OUT_DIR/compose_ps.txt" >&2 || true
  cat "$OUT_DIR/api_runtime.log" >&2 || true
  exit 1
fi
curl --fail --silent http://127.0.0.1:8002/version > "$OUT_DIR/api_version_initial.json"
curl --fail --silent http://127.0.0.1:8002/canary/status > "$OUT_DIR/canary_status_initial.json"
initial_version=$(python -c 'import json; print(json.load(open("reports/registry_integration/api_version_initial.json"))["registry_model_version"])')
challenger_version=$(python -c 'import json; print(json.load(open("reports/registry_integration/register_b.json"))["model_version"])')
container_id_before=$(docker compose -f "$COMPOSE_FILE" --profile runtime ps -q api-registry-runtime)
printf '%s\n' "$container_id_before" > "$OUT_DIR/api_container_id_before.txt"

# Send deterministic pseudonymous customers through the real API. The API scores both models,
# serves the challenger only to its assigned bucket, and accumulates online safety evidence.
: > "$OUT_DIR/canary_predictions.ndjson"
for i in $(seq 1 24); do
  python - "$i" <<'PY' > /tmp/canary-request.json
import json
import sys
from pathlib import Path
request = json.loads(Path("examples/review_request.json").read_text())
i = int(sys.argv[1])
request["customer_id"] = f"C_CANARY_{i:03d}"
request["request_id"] = f"canary-request-{i:04d}"
print(json.dumps(request))
PY
  curl --fail --silent -X POST http://127.0.0.1:8002/predict \
    -H 'Content-Type: application/json' \
    -d @/tmp/canary-request.json >> "$OUT_DIR/canary_predictions.ndjson"
  printf '\n' >> "$OUT_DIR/canary_predictions.ndjson"
done

# Background controller should evaluate PASS and promote B without an API restart.
canary_promoted=0
for _ in {1..60}; do
  curl --fail --silent http://127.0.0.1:8002/canary/status > "$OUT_DIR/canary_status_after_traffic.json"
  state=$(python -c 'import json; print(json.load(open("reports/registry_integration/canary_status_after_traffic.json"))["state"])')
  if [[ "$state" == "promoted" ]]; then
    canary_promoted=1
    break
  fi
  sleep 2
done
if [[ "$canary_promoted" != "1" ]]; then
  capture_runtime_diagnostics
  cat "$OUT_DIR/canary_status_after_traffic.json" >&2 || true
  echo "Canary controller did not promote the healthy challenger." >&2
  exit 1
fi

for _ in {1..60}; do
  curl --fail --silent http://127.0.0.1:8002/version > "$OUT_DIR/api_version_after_canary_promote.json"
  current=$(python -c 'import json; print(json.load(open("reports/registry_integration/api_version_after_canary_promote.json"))["registry_model_version"])')
  if [[ "$current" == "$challenger_version" ]]; then
    break
  fi
  sleep 2
done
curl --fail --silent -X POST http://127.0.0.1:8002/predict \
  -H 'Content-Type: application/json' \
  -d @examples/review_request.json > "$OUT_DIR/api_prediction_after_canary_promote.json"
curl --fail --silent http://127.0.0.1:8002/runtime/model > "$OUT_DIR/api_runtime_after_canary_promote.json"
run_registry status --output "$OUT_DIR/status_after_canary_promote.json"

# Simulate a post-promotion incident. Registry rollback must restore A while the same API container stays alive.
run_registry rollback \
  --reason "CI post-canary incident rollback drill" \
  --output "$OUT_DIR/rollback_after_canary.json"
for _ in {1..60}; do
  curl --fail --silent http://127.0.0.1:8002/version > "$OUT_DIR/api_version_after_rollback.json"
  current=$(python -c 'import json; print(json.load(open("reports/registry_integration/api_version_after_rollback.json"))["registry_model_version"])')
  if [[ "$current" == "$initial_version" ]]; then
    break
  fi
  sleep 2
done
container_id_after=$(docker compose -f "$COMPOSE_FILE" --profile runtime ps -q api-registry-runtime)
printf '%s\n' "$container_id_after" > "$OUT_DIR/api_container_id_after.txt"
run_registry status --output "$OUT_DIR/status_final.json"
run_registry sync \
  --alias champion \
  --output-dir "$OUT_DIR/synced_champion" \
  --output "$OUT_DIR/sync.json"

capture_runtime_diagnostics
python - <<'PY'
import json
from pathlib import Path

root = Path("reports/registry_integration")
initial = json.loads((root / "api_version_initial.json").read_text())
initial_canary = json.loads((root / "canary_status_initial.json").read_text())
after_canary = json.loads((root / "canary_status_after_traffic.json").read_text())
after_promote = json.loads((root / "api_version_after_canary_promote.json").read_text())
prediction = json.loads((root / "api_prediction_after_canary_promote.json").read_text())
runtime = json.loads((root / "api_runtime_after_canary_promote.json").read_text())
register_b = json.loads((root / "register_b.json").read_text())
after_rollback = json.loads((root / "api_version_after_rollback.json").read_text())
status = json.loads((root / "status_final.json").read_text())
container_before = (root / "api_container_id_before.txt").read_text().strip()
container_after = (root / "api_container_id_after.txt").read_text().strip()
predictions = [json.loads(line) for line in (root / "canary_predictions.ndjson").read_text().splitlines() if line.strip()]

assert initial["model_source"] == "registry"
assert initial["runtime_state"] == "ready_registry"
assert initial_canary["state"] in {"warming", "healthy"}
assert initial_canary["challenger_registry_version"] == str(register_b["model_version"])
assert len(predictions) == 24
assert any(item["served_model_role"] == "challenger" for item in predictions)
assert any(item["served_model_role"] == "champion" for item in predictions)
assert all(item["comparison_champion_registry_version"] == initial["registry_model_version"] for item in predictions)
assert all(item["comparison_challenger_registry_version"] == str(register_b["model_version"]) for item in predictions)
assert after_canary["state"] == "promoted"
assert after_canary["last_transition"] == "automatic_promotion"
assert after_canary["promoted_registry_version"] == str(register_b["model_version"])
assert after_canary["metrics"]["challenger_served"] >= 2
assert after_canary["metrics"]["action_disagreement_rate"] <= after_canary["limits"]["max_action_disagreement_rate"]
assert after_promote["registry_model_version"] == str(register_b["model_version"])
assert after_promote["registry_model_version"] != initial["registry_model_version"]
assert prediction["model_source"] == "registry"
assert prediction["registry_model_version"] == after_promote["registry_model_version"]
assert runtime["reload_successes"] >= 2
assert after_rollback["registry_model_version"] == initial["registry_model_version"]
assert status["aliases"]["champion"]["version"] == initial["registry_model_version"]
assert container_before and container_before == container_after
for name in ["model.joblib", "metadata.json", "model_metrics.json", "promotion_gate.json", "registry_provenance.json"]:
    path = root / "synced_champion" / name
    assert path.is_file() and path.stat().st_size > 0, path
print(json.dumps({
    "status": "PASS",
    "initial_champion_registry_version": initial["registry_model_version"],
    "canary_challenger_registry_version": str(register_b["model_version"]),
    "challenger_served_requests": after_canary["metrics"]["challenger_served"],
    "action_disagreement_rate": after_canary["metrics"]["action_disagreement_rate"],
    "automatic_promotion": after_canary["last_transition"],
    "rolled_back_registry_version": after_rollback["registry_model_version"],
    "same_api_container": container_before == container_after,
    "reload_successes": runtime["reload_successes"],
}, indent=2))
PY
