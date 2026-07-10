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
  --output "$OUT_DIR/rollback.json"
run_registry status --output "$OUT_DIR/status.json"
run_registry verify \
  --alias champion \
  --request examples/review_request.json \
  --output "$OUT_DIR/verify.json"
run_registry sync \
  --alias champion \
  --output-dir "$OUT_DIR/synced_champion" \
  --output "$OUT_DIR/sync.json"

python - <<'PY'
import json
from pathlib import Path

root = Path("reports/registry_integration")
status = json.loads((root / "status.json").read_text())
verify = json.loads((root / "verify.json").read_text())
aliases = status["aliases"]
assert aliases["champion"]["present"] is True
assert aliases["challenger"]["present"] is True
assert aliases["rollback"]["present"] is True
assert aliases["champion"]["version"] == aliases["rollback"]["version"]
assert aliases["champion"]["version"] != aliases["challenger"]["version"]
assert verify["prediction_is_finite"] is True
for name in ["model.joblib", "metadata.json", "model_metrics.json", "promotion_gate.json", "registry_provenance.json"]:
    path = root / "synced_champion" / name
    assert path.is_file() and path.stat().st_size > 0, path
print(json.dumps({"status": "PASS", "aliases": aliases, "verify": verify}, indent=2))
PY
