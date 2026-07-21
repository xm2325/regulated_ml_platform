#!/usr/bin/env bash
# SMOKE_ONLY target-side exercise. It cannot issue GPU_ELIGIBLE because Perf
# Analyzer uses random input and the run is below formal duration/sample gates.
# Use gpumedium_full_qualification.sbatch for governed same-input qualification.
#SBATCH --account=project_2012997
#SBATCH --job-name=regulated-ml-triton-gh200
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=roihu-triton-%j.out

set -euo pipefail
umask 077

fail() {
  printf 'Roihu TensorRT/Triton job failed closed: %s\n' "$1" >&2
  exit 1
}

require_environment() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "required environment variable ${name} is not set"
}

require_project_path() {
  local path="$1"
  local root="$2"
  case "${path}" in
    "${root}/${SLURM_JOB_ACCOUNT}"/*) ;;
    *) fail "path must resolve below ${root}/${SLURM_JOB_ACCOUNT}" ;;
  esac
}

require_project_storage_path() {
  local path="$1"
  case "${path}" in
    "/projappl/${SLURM_JOB_ACCOUNT}"/*|"/scratch/${SLURM_JOB_ACCOUNT}"/*) ;;
    *) fail "container path must resolve below this project's /projappl or /scratch" ;;
  esac
}

for variable in \
  SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_SUBMIT_DIR TMPDIR \
  TRITON_SERVER_SIF TRITON_SERVER_SIF_SHA256 TRITON_SDK_SIF TRITON_SDK_SIF_SHA256 TRITON_CONTAINER_RELEASE \
  ONNX_PATH ONNX_SHA256 EXPORT_MANIFEST_PATH EXPORT_MANIFEST_SHA256 SOURCE_GIT_COMMIT EVIDENCE_ROOT; do
  require_environment "${variable}"
done

[[ "${SLURM_JOB_ACCOUNT}" =~ ^project_[0-9]+$ ]] || fail "SLURM_JOB_ACCOUNT must be a CSC project identifier"
[[ "${SLURM_JOB_PARTITION}" == "gpumedium" || "${SLURM_JOB_PARTITION}" == "gputest" ]] \
  || fail "use gpumedium, or gputest only for an already-proven short exercise"
for digest_name in TRITON_SERVER_SIF_SHA256 TRITON_SDK_SIF_SHA256 ONNX_SHA256 EXPORT_MANIFEST_SHA256; do
  [[ "${!digest_name}" =~ ^[0-9a-fA-F]{64}$ ]] || fail "${digest_name} must contain exactly 64 hexadecimal characters"
  digest_value="${!digest_name}"
  printf -v "${digest_name}" '%s' "${digest_value,,}"
done
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "SOURCE_GIT_COMMIT must be a full Git commit"
SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT,,}"
[[ "${TRITON_CONTAINER_RELEASE}" =~ ^[0-9]{2}\.[0-9]{2}$ ]] \
  || fail "TRITON_CONTAINER_RELEASE must use the NGC year.month form, for example 25.06"
TRT_PRECISION="${TRT_PRECISION:-bf16}"
case "${TRT_PRECISION}" in
  fp32) TRT_PRECISION_FLAG="" ;;
  fp16) TRT_PRECISION_FLAG="--fp16" ;;
  bf16) TRT_PRECISION_FLAG="--bf16" ;;
  *) fail "TRT_PRECISION must be fp32, fp16, or bf16" ;;
esac

TRITON_SERVER_SIF="$(realpath -e -- "${TRITON_SERVER_SIF}")"
TRITON_SDK_SIF="$(realpath -e -- "${TRITON_SDK_SIF}")"
ONNX_PATH="$(realpath -e -- "${ONNX_PATH}")"
EXPORT_MANIFEST_PATH="$(realpath -e -- "${EXPORT_MANIFEST_PATH}")"
SLURM_SUBMIT_DIR="$(realpath -e -- "${SLURM_SUBMIT_DIR}")"
for image in "${TRITON_SERVER_SIF}" "${TRITON_SDK_SIF}"; do
  require_project_storage_path "${image}"
  [[ -f "${image}" && "${image}" == *.sif ]] || fail "container inputs must be regular .sif files"
done
for artifact in "${ONNX_PATH}" "${EXPORT_MANIFEST_PATH}"; do
  require_project_path "${artifact}" "/scratch"
  [[ -f "${artifact}" ]] || fail "ONNX inputs must be regular files"
done
require_project_path "${SLURM_SUBMIT_DIR}" "/scratch"
TRITON_CONFIG_PATH="$(realpath -e -- "$(dirname -- "${ONNX_PATH}")/config.pbtxt")"
require_project_path "${TRITON_CONFIG_PATH}" "/scratch"

mkdir -p -- "${EVIDENCE_ROOT}"
EVIDENCE_ROOT="$(realpath -e -- "${EVIDENCE_ROOT}")"
require_project_path "${EVIDENCE_ROOT}" "/scratch"
EVIDENCE_DIR="${EVIDENCE_ROOT}/triton-tensorrt-gh200-${SLURM_JOB_ID}"
[[ ! -e "${EVIDENCE_DIR}" ]] || fail "refusing to overwrite an existing evidence directory"
mkdir -m 700 -- "${EVIDENCE_DIR}"

verify_digest() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local observed
  observed="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || fail "${label} SHA-256 mismatch"
  printf '%s  %s\n' "${observed}" "$(basename -- "${path}")" >"${EVIDENCE_DIR}/${label}.sha256"
}

verify_digest "${TRITON_SERVER_SIF}" "${TRITON_SERVER_SIF_SHA256}" "triton-server-sif"
verify_digest "${TRITON_SDK_SIF}" "${TRITON_SDK_SIF_SHA256}" "triton-sdk-sif"
verify_digest "${ONNX_PATH}" "${ONNX_SHA256}" "accelerator-qualification-onnx"
verify_digest "${EXPORT_MANIFEST_PATH}" "${EXPORT_MANIFEST_SHA256}" "onnx-export-manifest"

command -v apptainer >/dev/null 2>&1 || fail "Apptainer is not available on this Roihu node"
[[ "$(uname -m)" == "aarch64" ]] || fail "target-side TensorRT compilation requires the Roihu-GPU aarch64 environment"
gpu_identity="$(nvidia-smi --query-gpu=uuid,name,compute_cap --format=csv,noheader | head -n 1)"
[[ "${gpu_identity}" == *GH200* ]] || fail "allocated accelerator is not an NVIDIA GH200"
[[ "${gpu_identity}" == *9.0* ]] || fail "allocated accelerator does not report compute capability 9.0"

python3 - \
  "${EXPORT_MANIFEST_PATH}" "${EXPORT_MANIFEST_SHA256}" "${ONNX_PATH}" "${ONNX_SHA256}" \
  "${TRITON_CONFIG_PATH}" "${SOURCE_GIT_COMMIT}" "${EVIDENCE_DIR}/input_validation.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_manifest_sha = sys.argv[2]
onnx_path = Path(sys.argv[3])
expected_onnx_sha = sys.argv[4]
config_path = Path(sys.argv[5])
expected_commit = sys.argv[6]
output_path = Path(sys.argv[7])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
checks = {
    "manifest_digest_verified": sha256(manifest_path) == expected_manifest_sha,
    "manifest_passed": manifest.get("status") == "PASS",
    "source_commit_matches": manifest.get("source", {}).get("git_commit") == expected_commit,
    "workload_is_accelerator_qualification": manifest.get("workload", {}).get("classification")
    == "synthetic_accelerator_qualification_only",
    "workload_is_not_production": manifest.get("workload", {}).get("production_model") is False,
    "workload_is_not_champion": manifest.get("workload", {}).get("champion_model") is False,
    "onnx_digest_verified": sha256(onnx_path) == expected_onnx_sha,
    "onnx_digest_matches_manifest": manifest.get("artifacts", {}).get("onnx", {}).get("sha256") == expected_onnx_sha,
    "config_digest_matches_manifest": manifest.get("artifacts", {}).get("triton_config", {}).get("sha256")
    == sha256(config_path),
}
payload = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "workload_weights_sha256": manifest.get("workload", {}).get("weights_sha256"),
}
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output_path)
if payload["status"] != "PASS":
    raise SystemExit("ONNX/export-manifest validation failed closed")
PY

apptainer inspect --json "${TRITON_SERVER_SIF}" >"${EVIDENCE_DIR}/triton-server-sif-inspect.json"
apptainer inspect --json "${TRITON_SDK_SIF}" >"${EVIDENCE_DIR}/triton-sdk-sif-inspect.json"
grep -Fq -- "${TRITON_CONTAINER_RELEASE}" "${EVIDENCE_DIR}/triton-server-sif-inspect.json" \
  || fail "server SIF metadata does not contain the expected NGC container release"
grep -Fq -- "${TRITON_CONTAINER_RELEASE}" "${EVIDENCE_DIR}/triton-sdk-sif-inspect.json" \
  || fail "SDK SIF metadata does not contain the expected NGC container release"

WORK_DIR="$(mktemp -d -- "${TMPDIR%/}/regulated-ml-triton-${SLURM_JOB_ID}.XXXXXX")"
MODEL_REPOSITORY="${WORK_DIR}/model_repository"
MODEL_VERSION_DIR="${MODEL_REPOSITORY}/accelerator_qualification/1"
mkdir -p -- "${MODEL_VERSION_DIR}"
cp -- "${ONNX_PATH}" "${WORK_DIR}/accelerator_qualification.onnx"
cp -- "${TRITON_CONFIG_PATH}" "${MODEL_REPOSITORY}/accelerator_qualification/config.pbtxt"

SAMPLER_PID=""
SERVER_PID=""

write_job_manifest() {
  local return_code="$1"
  local status="ERROR"
  [[ "${return_code}" -eq 0 ]] && status="SMOKE_PASS"
  python3 - "${EVIDENCE_DIR}" "${status}" "${return_code}" "${SOURCE_GIT_COMMIT}" \
    "${TRITON_CONTAINER_RELEASE}" "${TRT_PRECISION}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
status = sys.argv[2]
return_code = int(sys.argv[3])
commit = sys.argv[4]
container_release = sys.argv[5]
precision = sys.argv[6]
artifacts = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "job_manifest.json":
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    artifacts.append({"path": str(path.relative_to(root)), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size})
payload = {
    "schema_version": "regulated-ml-platform.roihu-triton-sbatch-evidence/v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "exit_code": return_code,
    "source_git_commit": commit,
    "triton_container_release": container_release,
    "tensorrt_precision": precision,
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    },
    "artifacts": artifacts,
    "claim_boundary": {
        "synthetic_accelerator_qualification_only": True,
        "production_model_claim_allowed": False,
        "champion_model_claim_allowed": False,
        "production_capacity_claim_allowed": False,
        "semantic_parity_against_pytorch_established": False,
        "promotion_decision_allowed": False,
        "gpu_eligibility_decision_allowed": False,
    },
}
temporary = root / ".job_manifest.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(root / "job_manifest.json")
PY
}

cleanup() {
  local return_code="$?"
  trap - EXIT
  set +e
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1
    wait "${SERVER_PID}" >/dev/null 2>&1
  fi
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill "${SAMPLER_PID}" >/dev/null 2>&1
    wait "${SAMPLER_PID}" >/dev/null 2>&1
  fi
  if [[ -d "${MODEL_REPOSITORY}" && ! -e "${EVIDENCE_DIR}/model_repository" ]]; then
    cp -a -- "${MODEL_REPOSITORY}" "${EVIDENCE_DIR}/model_repository"
  fi
  write_job_manifest "${return_code}"
  local manifest_code="$?"
  rm -rf -- "${WORK_DIR}"
  if [[ "${return_code}" -eq 0 && "${manifest_code}" -ne 0 ]]; then
    return_code="${manifest_code}"
  fi
  exit "${return_code}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

nvidia-smi >"${EVIDENCE_DIR}/nvidia-smi-full.txt"
nvidia-smi --query-gpu=timestamp,index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
  --format=csv -l 1 >"${EVIDENCE_DIR}/nvidia-smi-timeseries.csv" 2>"${EVIDENCE_DIR}/nvidia-smi-timeseries.err" &
SAMPLER_PID="$!"
sleep 1
kill -0 "${SAMPLER_PID}" 2>/dev/null || fail "nvidia-smi sampler did not start"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,compute_cap --format=csv,noheader \
  >"${EVIDENCE_DIR}/nvidia-smi-snapshot.csv"

APPTAINER_ARGS=(--cleanenv --containall --nv --bind "${WORK_DIR}:/work")
TRTEXEC_PATH=/usr/src/tensorrt/bin/trtexec
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" uname -m >"${EVIDENCE_DIR}/triton-sdk-architecture.txt"
grep -qx 'aarch64' "${EVIDENCE_DIR}/triton-sdk-architecture.txt" || fail "SDK SIF is not executable as aarch64 on Roihu-GPU"
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SERVER_SIF}" uname -m >"${EVIDENCE_DIR}/triton-server-architecture.txt"
grep -qx 'aarch64' "${EVIDENCE_DIR}/triton-server-architecture.txt" || fail "server SIF is not executable as aarch64 on Roihu-GPU"
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" test -x "${TRTEXEC_PATH}" \
  || fail "SDK SIF does not provide the expected TensorRT executable"
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" "${TRTEXEC_PATH}" --help \
  >"${EVIDENCE_DIR}/trtexec-version-and-help.txt" 2>&1

TRTEXEC_COMMAND=(
  "${TRTEXEC_PATH}"
  --onnx=/work/accelerator_qualification.onnx
  --saveEngine=/work/model_repository/accelerator_qualification/1/model.plan
  --minShapes=INPUT__0:1x1024
  --optShapes=INPUT__0:64x1024
  --maxShapes=INPUT__0:256x1024
  --timingCacheFile=/work/timing.cache
  --builderOptimizationLevel=3
  --profilingVerbosity=detailed
  --dumpLayerInfo
)
if [[ -n "${TRT_PRECISION_FLAG}" ]]; then
  TRTEXEC_COMMAND+=("${TRT_PRECISION_FLAG}")
fi
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" "${TRTEXEC_COMMAND[@]}" \
  >"${EVIDENCE_DIR}/trtexec-build-and-validation.log" 2>&1
[[ -s "${MODEL_VERSION_DIR}/model.plan" ]] || fail "trtexec did not produce a TensorRT plan"
sha256sum -- "${MODEL_VERSION_DIR}/model.plan" >"${EVIDENCE_DIR}/model-plan.sha256"
cp -- "${WORK_DIR}/timing.cache" "${EVIDENCE_DIR}/tensorrt-timing.cache"

port_seed=$((SLURM_JOB_ID % 10000))
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-$((20000 + port_seed))}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-$((30000 + port_seed))}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-$((40000 + port_seed))}"
for port in "${TRITON_HTTP_PORT}" "${TRITON_GRPC_PORT}" "${TRITON_METRICS_PORT}"; do
  [[ "${port}" =~ ^[0-9]+$ && "${port}" -ge 1024 && "${port}" -le 65535 ]] || fail "Triton ports must be between 1024 and 65535"
done
[[ "${TRITON_HTTP_PORT}" != "${TRITON_GRPC_PORT}" && "${TRITON_HTTP_PORT}" != "${TRITON_METRICS_PORT}" \
  && "${TRITON_GRPC_PORT}" != "${TRITON_METRICS_PORT}" ]] || fail "Triton ports must be distinct"

apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SERVER_SIF}" tritonserver \
  --model-repository=/work/model_repository \
  --model-control-mode=none \
  --strict-model-config=true \
  --exit-on-error=true \
  --http-address=127.0.0.1 \
  --http-port="${TRITON_HTTP_PORT}" \
  --grpc-address=127.0.0.1 \
  --grpc-port="${TRITON_GRPC_PORT}" \
  --metrics-address=127.0.0.1 \
  --metrics-port="${TRITON_METRICS_PORT}" \
  --allow-http=true \
  --allow-grpc=true \
  --allow-metrics=true \
  --allow-gpu-metrics=true \
  >"${EVIDENCE_DIR}/triton-server.log" 2>&1 &
SERVER_PID="$!"

ready=0
for _ in $(seq 1 90); do
  if curl --fail --silent --show-error "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" \
    >"${EVIDENCE_DIR}/server-ready.txt"; then
    ready=1
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || fail "Triton server exited before becoming ready"
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "Triton server did not become ready within the bounded wait"

curl --fail --silent --show-error "http://127.0.0.1:${TRITON_HTTP_PORT}/v2" \
  >"${EVIDENCE_DIR}/triton-server-metadata.json"
curl --fail --silent --show-error "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/accelerator_qualification/ready" \
  >"${EVIDENCE_DIR}/model-ready.txt"
curl --fail --silent --show-error "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/accelerator_qualification/config" \
  >"${EVIDENCE_DIR}/loaded-model-config.json"

for batch_size in 1 16 64 256; do
  apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" perf_analyzer \
    -m accelerator_qualification \
    -i http \
    -u "127.0.0.1:${TRITON_HTTP_PORT}" \
    --shape INPUT__0:1024 \
    --input-data=random \
    --batch-size="${batch_size}" \
    --concurrency-range=1:4:1 \
    --measurement-interval=1000 \
    --stability-percentage=20 \
    --max-trials=5 \
    --percentile=99 \
    --verbose-csv \
    -f "/work/perf-analyzer-batch-${batch_size}.csv" \
    >"${EVIDENCE_DIR}/perf-analyzer-batch-${batch_size}.log" 2>&1
  [[ -s "${WORK_DIR}/perf-analyzer-batch-${batch_size}.csv" ]] || fail "Perf Analyzer produced no CSV for batch ${batch_size}"
  grep -q 'Inferences/Second' "${WORK_DIR}/perf-analyzer-batch-${batch_size}.csv" \
    || fail "Perf Analyzer CSV is missing its throughput column"
  cp -- "${WORK_DIR}/perf-analyzer-batch-${batch_size}.csv" "${EVIDENCE_DIR}/"
done

curl --fail --silent --show-error "http://127.0.0.1:${TRITON_METRICS_PORT}/metrics" \
  >"${EVIDENCE_DIR}/triton-metrics.prom"
grep -q 'nv_inference_' "${EVIDENCE_DIR}/triton-metrics.prom" || fail "Triton metrics contain no inference observations"
cp -a -- "${MODEL_REPOSITORY}" "${EVIDENCE_DIR}/model_repository"

python3 - "${EVIDENCE_DIR}" "${TRITON_CONTAINER_RELEASE}" "${TRT_PRECISION}" "${SOURCE_GIT_COMMIT}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
container_release = sys.argv[2]
precision = sys.argv[3]
commit = sys.argv[4]
server_metadata = json.loads((root / "triton-server-metadata.json").read_text(encoding="utf-8"))
triton_server_version = str(server_metadata.get("version", ""))
trtexec_log = (root / "trtexec-build-and-validation.log").read_text(encoding="utf-8", errors="replace")
nvidia_log = (root / "nvidia-smi-full.txt").read_text(encoding="utf-8", errors="replace")
trt_match = re.search(r"TensorRT(?:\.trtexec)?\s*(?:\[TensorRT\s*)?v?([0-9][0-9.]*)", trtexec_log)
cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_log)
plan_path = root / "model_repository/accelerator_qualification/1/model.plan"
plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
checks = {
    "triton_server_version_is_runtime_semver": bool(re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", triton_server_version)),
    "triton_container_release_is_ngc_year_month": bool(re.fullmatch(r"[0-9]{2}\.[0-9]{2}", container_release)),
    "tensorrt_version_observed_in_raw_log": trt_match is not None,
    "cuda_version_observed_in_raw_log": cuda_match is not None,
    "target_side_plan_exists": plan_path.is_file() and plan_path.stat().st_size > 0,
    "all_perf_analyzer_batches_present": all((root / f"perf-analyzer-batch-{batch}.csv").is_file() for batch in (1, 16, 64, 256)),
    "gpu_metrics_observed": "nv_gpu_" in (root / "triton-metrics.prom").read_text(encoding="utf-8", errors="replace"),
}
payload = {
    "schema_version": "regulated-ml-platform.roihu-triton-smoke/v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "SMOKE_PASS" if all(checks.values()) else "FAIL",
    "evidence_tier": "SMOKE_ONLY",
    "source_git_commit": commit,
    "software": {
        "triton_server_version": triton_server_version,
        "triton_container_release": container_release,
        "version_scheme_note": "Triton server uses 2.x semantic versions; NGC container releases use year.month and are intentionally separate.",
        "tensorrt_version_raw": trt_match.group(1) if trt_match else None,
        "cuda_driver_supported_version": cuda_match.group(1) if cuda_match else None,
    },
    "engine": {
        "precision": precision,
        "target_platform": "Roihu-GPU GH200 aarch64 compute capability 9.0",
        "plan_sha256": plan_sha256,
        "portable_across_gpu_or_tensorrt_versions": False,
    },
    "checks": checks,
    "claim_boundary": {
        "synthetic_accelerator_qualification_only": True,
        "production_model_claim_allowed": False,
        "champion_model_claim_allowed": False,
        "production_capacity_claim_allowed": False,
        "semantic_parity_against_pytorch_established": False,
        "promotion_decision_allowed": False,
        "gpu_eligibility_decision_allowed": False,
        "note": "Perf Analyzer used random contract-valid input; add reviewed semantic parity and workload policy before promotion.",
    },
}
output = root / "runtime_evidence.json"
temporary = root / ".runtime_evidence.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output)
if payload["status"] != "SMOKE_PASS":
    raise SystemExit("Triton runtime evidence checks failed closed")
PY

printf 'Roihu GH200 TensorRT/Triton SMOKE_ONLY evidence: %s\n' "${EVIDENCE_DIR}"
