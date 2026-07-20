#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-regulated-ai}"
IMAGE_NAME="${IMAGE_NAME:-regulated-ai-mlops}"
IMAGE_TAG="${IMAGE_TAG:?IMAGE_TAG must be set}"
PORT="${PORT:-18000}"

cleanup() {
  local status=$?
  if kubectl cluster-info >/dev/null 2>&1; then
    echo "--- Kubernetes resources ---"
    kubectl get all -A || true
    echo "--- Pod descriptions ---"
    kubectl describe pods -l app=regulated-ai-api || true
    echo "--- Application logs ---"
    kubectl logs deployment/regulated-ai-api --all-containers=true || true
    echo "--- Recent events ---"
    kubectl get events --all-namespaces --sort-by=.lastTimestamp | tail -80 || true
  fi
  if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
    kill "${PORT_FORWARD_PID}" 2>/dev/null || true
  fi
  kind delete cluster --name "${CLUSTER_NAME}" || true
  exit "${status}"
}
trap cleanup EXIT

kind create cluster --name "${CLUSTER_NAME}" --wait 120s
kind load docker-image "${IMAGE_NAME}:${IMAGE_TAG}" --name "${CLUSTER_NAME}"

helm lint helm/regulated-ai
helm template regulated-ai helm/regulated-ai \
  --set image.repository="${IMAGE_NAME}" \
  --set image.tag="${IMAGE_TAG}" \
  --set image.pullPolicy=Never \
  --set replicaCount=1 \
  --set autoscaling.enabled=false \
  --set podDisruptionBudget.enabled=false \
  --set networkPolicy.enabled=false > /tmp/rendered-chart.yaml
kubectl apply --dry-run=server -f /tmp/rendered-chart.yaml

# Registry + canary remain opt-in, but the full configuration path is rendered and validated on every CI run.
helm template regulated-ai-registry helm/regulated-ai \
  --set image.repository="${IMAGE_NAME}" \
  --set image.tag="${IMAGE_TAG}" \
  --set image.pullPolicy=Never \
  --set replicaCount=1 \
  --set autoscaling.enabled=false \
  --set podDisruptionBudget.enabled=false \
  --set registryRuntime.enabled=true \
  --set registryRuntime.canary.enabled=true \
  --set registryRuntime.canary.trafficPercent=5 \
  --set networkPolicy.enabled=true \
  --set networkPolicy.mlflowEgress.enabled=true > /tmp/rendered-registry-chart.yaml
kubectl apply --dry-run=server -f /tmp/rendered-registry-chart.yaml

grep -q 'name: MODEL_SOURCE' /tmp/rendered-registry-chart.yaml
grep -q 'value: "registry"' /tmp/rendered-registry-chart.yaml
grep -q 'name: MLFLOW_CHALLENGER_ALIAS' /tmp/rendered-registry-chart.yaml
grep -q 'name: CANARY_ENABLED' /tmp/rendered-registry-chart.yaml
grep -q 'name: CANARY_TRAFFIC_PERCENT' /tmp/rendered-registry-chart.yaml
grep -q 'name: CANARY_AUTO_PROMOTE_ENABLED' /tmp/rendered-registry-chart.yaml
grep -q 'mountPath: /var/lib/regulated-ai/registry-cache' /tmp/rendered-registry-chart.yaml
grep -q 'fsGroupChangePolicy: OnRootMismatch' /tmp/rendered-registry-chart.yaml

# GPU and scheduled-training controls are configuration-contract tested only: the hosted runner has no NVIDIA GPU.
helm template regulated-ai-gpu helm/regulated-ai \
  --set image.repository="${IMAGE_NAME}" \
  --set image.tag="${IMAGE_TAG}" \
  --set image.pullPolicy=Never \
  --set inferenceScheduling.gpu.enabled=true \
  --set trainingJob.enabled=true \
  --set trainingJob.suspend=true \
  --set trainingJob.gpu.enabled=true > /tmp/rendered-gpu-training-chart.yaml
kubectl apply --dry-run=server -f /tmp/rendered-gpu-training-chart.yaml

grep -q 'kind: ResourceQuota' /tmp/rendered-gpu-training-chart.yaml
grep -q 'kind: PriorityClass' /tmp/rendered-gpu-training-chart.yaml
grep -q 'priorityClassName: "regulated-ai-inference"' /tmp/rendered-gpu-training-chart.yaml
grep -q 'priorityClassName: "regulated-ai-training"' /tmp/rendered-gpu-training-chart.yaml
grep -q 'nvidia.com/gpu' /tmp/rendered-gpu-training-chart.yaml
grep -q 'accelerator: nvidia-a100' /tmp/rendered-gpu-training-chart.yaml
grep -q 'kind: CronJob' /tmp/rendered-gpu-training-chart.yaml
grep -q 'concurrencyPolicy: Forbid' /tmp/rendered-gpu-training-chart.yaml
grep -q 'suspend: true' /tmp/rendered-gpu-training-chart.yaml

helm upgrade --install regulated-ai helm/regulated-ai \
  --set image.repository="${IMAGE_NAME}" \
  --set image.tag="${IMAGE_TAG}" \
  --set image.pullPolicy=Never \
  --set replicaCount=1 \
  --set autoscaling.enabled=false \
  --set podDisruptionBudget.enabled=false \
  --set networkPolicy.enabled=false \
  --wait \
  --timeout 3m

kubectl rollout status deployment/regulated-ai-api --timeout=180s
kubectl port-forward service/regulated-ai-api "${PORT}:80" >/tmp/port-forward.log 2>&1 &
PORT_FORWARD_PID=$!

for _ in {1..30}; do
  if curl --fail --silent "http://127.0.0.1:${PORT}/ready"; then
    curl --fail --silent "http://127.0.0.1:${PORT}/version"
    curl --fail --silent "http://127.0.0.1:${PORT}/canary/status"
    curl --fail --silent -X POST "http://127.0.0.1:${PORT}/predict" \
      -H 'Content-Type: application/json' \
      --data @examples/review_request.json
    exit 0
  fi
  sleep 2
done

cat /tmp/port-forward.log || true
exit 1
