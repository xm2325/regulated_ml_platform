#!/usr/bin/env bash
set -euo pipefail

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.06-py3}"
TRITON_NAME="${TRITON_NAME:-regulated-ai-triton-cpu}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-18000}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-18002}"
OUT_DIR="${TRITON_RUNTIME_OUT_DIR:-reports/triton_cpu_runtime}"

mkdir -p "${OUT_DIR}"

docker rm -f "${TRITON_NAME}" >/dev/null 2>&1 || true

docker run -d --rm \
  --name "${TRITON_NAME}" \
  -p "${TRITON_HTTP_PORT}:8000" \
  -p "${TRITON_METRICS_PORT}:8002" \
  -v "$PWD/models/triton/model_repository:/models:ro" \
  "${TRITON_IMAGE}" \
  tritonserver \
    --model-repository=/models \
    --strict-model-config=true \
    --allow-metrics=true \
    --allow-gpu-metrics=false \
    >"${OUT_DIR}/container_id.txt"

cleanup() {
  docker logs "${TRITON_NAME}" >"${OUT_DIR}/triton_server.log" 2>&1 || true
  docker rm -f "${TRITON_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != "1" ]]; then
  echo "Triton did not become ready" >&2
  exit 1
fi

curl --fail --silent \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" \
  >"${OUT_DIR}/health_ready.txt"

curl --fail --silent \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/support_ensemble/ready" \
  >"${OUT_DIR}/support_ensemble_ready.txt"

curl --fail --silent \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/support_base/config" \
  >"${OUT_DIR}/support_base_runtime_config.json"

curl --fail --silent \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/support_calibrator/config" \
  >"${OUT_DIR}/support_calibrator_runtime_config.json"

curl --fail --silent \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/support_ensemble/config" \
  >"${OUT_DIR}/support_ensemble_runtime_config.json"

python -m src.operations.benchmark_triton_http \
  --triton-url "http://127.0.0.1:${TRITON_HTTP_PORT}" \
  --model models/model.joblib \
  --sample data/processed/features.csv \
  --triton-root models/triton \
  --output "${OUT_DIR}/triton_http_benchmark.json" \
  --batch-sizes 1,8,32,64,128 \
  --repeats 8 \
  --warmup 2

curl --fail --silent \
  "http://127.0.0.1:${TRITON_METRICS_PORT}/metrics" \
  >"${OUT_DIR}/triton_metrics.prom"

grep -q 'nv_inference_' "${OUT_DIR}/triton_metrics.prom"

python -m src.operations.triton_runtime_evidence \
  --base-config "${OUT_DIR}/support_base_runtime_config.json" \
  --calibrator-config "${OUT_DIR}/support_calibrator_runtime_config.json" \
  --benchmark "${OUT_DIR}/triton_http_benchmark.json" \
  --metrics "${OUT_DIR}/triton_metrics.prom" \
  --output "${OUT_DIR}/runtime_evidence.json"

printf 'Triton CPU runtime evidence completed successfully.\n'
