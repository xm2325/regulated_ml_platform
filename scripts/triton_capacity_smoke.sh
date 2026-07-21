#!/usr/bin/env bash
set -euo pipefail

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.06-py3}"
TRITON_SDK_IMAGE="${TRITON_SDK_IMAGE:-nvcr.io/nvidia/tritonserver:25.06-py3-sdk}"
TRITON_NAME="${TRITON_NAME:-regulated-ai-triton-capacity}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-18100}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-18102}"
OUT_DIR="${TRITON_CAPACITY_OUT_DIR:-reports/triton_capacity}"

mkdir -p "${OUT_DIR}"
docker rm -f "${TRITON_NAME}" >/dev/null 2>&1 || true

docker run -d \
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

capture_container_evidence() {
  docker inspect "${TRITON_NAME}" >"${OUT_DIR}/triton_container_inspect.json" 2>/dev/null || true
  docker logs "${TRITON_NAME}" >"${OUT_DIR}/triton_server.log" 2>&1 || true
}

cleanup() {
  capture_container_evidence
  docker rm -f "${TRITON_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" >/dev/null; then
    ready=1
    break
  fi
  running="$(docker inspect -f '{{.State.Running}}' "${TRITON_NAME}" 2>/dev/null || echo false)"
  if [[ "${running}" != "true" ]]; then
    capture_container_evidence
    echo "Triton container exited before becoming ready" >&2
    cat "${OUT_DIR}/triton_server.log" >&2 || true
    exit 1
  fi
  sleep 2
done

if [[ "${ready}" != "1" ]]; then
  capture_container_evidence
  echo "Triton did not become ready" >&2
  cat "${OUT_DIR}/triton_server.log" >&2 || true
  exit 1
fi

curl --fail --silent \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/support_ensemble/ready" \
  >"${OUT_DIR}/support_ensemble_ready.txt"

python -m src.operations.benchmark_triton_concurrency \
  --triton-url "http://127.0.0.1:${TRITON_HTTP_PORT}" \
  --metrics-url "http://127.0.0.1:${TRITON_METRICS_PORT}/metrics" \
  --model models/model.joblib \
  --sample data/processed/features.csv \
  --triton-root models/triton \
  --output "${OUT_DIR}/concurrency_benchmark.json" \
  --concurrency-levels 1,4,8,16,32 \
  --rounds-per-level 12 \
  --request-batch-size 1

curl --fail --silent \
  "http://127.0.0.1:${TRITON_METRICS_PORT}/metrics" \
  >"${OUT_DIR}/triton_metrics_after_concurrency.prom"

grep -q 'nv_inference_exec_count' "${OUT_DIR}/triton_metrics_after_concurrency.prom"

docker pull "${TRITON_SDK_IMAGE}"
docker image inspect "${TRITON_SDK_IMAGE}" >"${OUT_DIR}/triton_sdk_image_inspect.json"

set +e
docker run --rm --network host \
  -v "$PWD/${OUT_DIR}:/output" \
  "${TRITON_SDK_IMAGE}" \
  perf_analyzer \
    -m support_ensemble \
    -i http \
    -u "127.0.0.1:${TRITON_HTTP_PORT}" \
    --concurrency-range=1:16:3 \
    --percentile=95 \
    --measurement-interval=1000 \
    --stability-percentage=20 \
    --max-trials=5 \
    --warmup-request-count=20 \
    --collect-metrics \
    --metrics-url="127.0.0.1:${TRITON_METRICS_PORT}/metrics" \
    --verbose-csv \
    -f /output/perf_analyzer.csv \
    >"${OUT_DIR}/perf_analyzer.log" 2>&1
perf_status=$?
set -e
cat "${OUT_DIR}/perf_analyzer.log"
if [[ "${perf_status}" -ne 0 ]]; then
  echo "Triton Perf Analyzer failed" >&2
  exit "${perf_status}"
fi

test -s "${OUT_DIR}/perf_analyzer.csv"

python -m src.operations.triton_capacity_plan \
  --benchmark "${OUT_DIR}/concurrency_benchmark.json" \
  --policy config/triton_capacity_policy.yaml \
  --perf-analyzer-csv "${OUT_DIR}/perf_analyzer.csv" \
  --output-json "${OUT_DIR}/capacity_plan.json" \
  --output-md "${OUT_DIR}/capacity_plan.md"

printf 'Triton v1.2 concurrency, batching, Perf Analyzer, and capacity evidence completed successfully.\n'
