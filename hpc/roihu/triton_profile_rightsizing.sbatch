#!/usr/bin/env bash
# Short, source-bound Triton profiling/right-sizing exercise. The default is a
# one-GPU gputest job; submitters may override --gres and --cpus-per-task for
# 2/4-GPU same-node comparisons. This path is SMOKE_ONLY.
#SBATCH --account=project_2012997
#SBATCH --job-name=regulated-ml-rightsize
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=roihu-rightsize-%j.out

set -euo pipefail
umask 077

fail() {
  printf 'Roihu Triton right-sizing failed closed: %s\n' "$1" >&2
  exit 1
}

require_environment() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "required environment variable ${name} is not set"
}

require_project_storage_path() {
  local path="$1"
  case "${path}" in
    "/projappl/${SLURM_JOB_ACCOUNT}"/*|"/scratch/${SLURM_JOB_ACCOUNT}"/*) ;;
    *) fail "path must resolve below this project's /projappl or /scratch" ;;
  esac
}

for variable in \
  SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK TMPDIR \
  SOURCE_ARCHIVE SOURCE_SHA256 SOURCE_GIT_COMMIT \
  TRITON_SERVER_SIF TRITON_SERVER_SIF_SHA256 \
  TRITON_SDK_SIF TRITON_SDK_SIF_SHA256 \
  MODEL_PLAN MODEL_PLAN_SHA256 TRITON_CONFIG_PATH TRITON_CONFIG_SHA256 \
  EVIDENCE_ROOT GPU_COUNT PARENT_FORMAL_JOB_ID; do
  require_environment "${variable}"
done

[[ "${SLURM_JOB_ACCOUNT}" =~ ^project_[0-9]+$ ]] || fail "invalid CSC project account"
[[ "${SLURM_JOB_PARTITION}" == "gputest" || "${SLURM_JOB_PARTITION}" == "gpumedium" ]] \
  || fail "right-sizing must run in gputest or gpumedium"
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit must be a full lowercase Git SHA-1"
[[ "${SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source SHA-256"
[[ "${TRITON_SERVER_SIF_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid server SIF SHA-256"
[[ "${TRITON_SDK_SIF_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SDK SIF SHA-256"
[[ "${MODEL_PLAN_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid model plan SHA-256"
[[ "${TRITON_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid Triton config SHA-256"
[[ "${GPU_COUNT}" =~ ^(1|2|4)$ ]] || fail "GPU_COUNT must be 1, 2, or 4"
[[ "${PARENT_FORMAL_JOB_ID}" =~ ^[0-9]+$ ]] || fail "parent formal job ID must be numeric"
[[ "${SLURM_CPUS_PER_TASK}" -eq $((GPU_COUNT * 72)) ]] \
  || fail "request exactly 72 Grace CPU cores per GH200"

SOURCE_ARCHIVE="$(realpath -e -- "${SOURCE_ARCHIVE}")"
TRITON_SERVER_SIF="$(realpath -e -- "${TRITON_SERVER_SIF}")"
TRITON_SDK_SIF="$(realpath -e -- "${TRITON_SDK_SIF}")"
MODEL_PLAN="$(realpath -e -- "${MODEL_PLAN}")"
TRITON_CONFIG_PATH="$(realpath -e -- "${TRITON_CONFIG_PATH}")"
for path in "${SOURCE_ARCHIVE}" "${TRITON_SERVER_SIF}" "${TRITON_SDK_SIF}" "${MODEL_PLAN}" "${TRITON_CONFIG_PATH}"; do
  require_project_storage_path "${path}"
done

verify_digest() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || fail "SHA-256 mismatch for ${path}"
}
verify_digest "${SOURCE_ARCHIVE}" "${SOURCE_SHA256}"
verify_digest "${TRITON_SERVER_SIF}" "${TRITON_SERVER_SIF_SHA256}"
verify_digest "${TRITON_SDK_SIF}" "${TRITON_SDK_SIF_SHA256}"
verify_digest "${MODEL_PLAN}" "${MODEL_PLAN_SHA256}"
verify_digest "${TRITON_CONFIG_PATH}" "${TRITON_CONFIG_SHA256}"

archive_root="regulated_ml_platform-${SOURCE_GIT_COMMIT}"
marker="$(tar -xOzf "${SOURCE_ARCHIVE}" "${archive_root}/.regulated-ml-source-commit")"
[[ "${marker}" == "${SOURCE_GIT_COMMIT}" ]] || fail "source archive marker mismatch"

mkdir -p -- "${EVIDENCE_ROOT}"
EVIDENCE_ROOT="$(realpath -e -- "${EVIDENCE_ROOT}")"
require_project_storage_path "${EVIDENCE_ROOT}"
EVIDENCE_DIR="${EVIDENCE_ROOT}/triton-rightsizing-${GPU_COUNT}gpu-${SLURM_JOB_ID}"
[[ ! -e "${EVIDENCE_DIR}" ]] || fail "refusing to overwrite existing evidence"
mkdir -m 700 -- "${EVIDENCE_DIR}"

[[ -r /etc/profile.d/zz-csc-env.sh ]] || fail "CSC environment initializer unavailable"
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module --force purge
module load python-pytorch/2.10
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
module -t list 2>"${EVIDENCE_DIR}/module-list.txt"
[[ "$(uname -m)" == "aarch64" ]] || fail "Roihu GPU aarch64 is required"
command -v apptainer >/dev/null 2>&1 || fail "Apptainer unavailable"

mapfile -t GPU_IDENTITIES < <(
  nvidia-smi --query-gpu=index,uuid,name,compute_cap --format=csv,noheader,nounits
)
[[ "${#GPU_IDENTITIES[@]}" -eq "${GPU_COUNT}" ]] \
  || fail "allocated GPU count does not match GPU_COUNT"
for identity in "${GPU_IDENTITIES[@]}"; do
  [[ "${identity}" == *GH200* && "${identity}" == *9.0* ]] \
    || fail "every allocated GPU must be a full GH200 with compute capability 9.0"
done
printf '%s\n' "${GPU_IDENTITIES[@]}" >"${EVIDENCE_DIR}/gpu-identities.csv"

WORK_DIR="$(mktemp -d -- "${TMPDIR%/}/regulated-ml-rightsize-${SLURM_JOB_ID}.XXXXXX")"
declare -a SERVER_PIDS=()
TELEMETRY_PID=""

cleanup() {
  local return_code="$?"
  trap - EXIT
  set +e
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" >/dev/null 2>&1
    wait "${TELEMETRY_PID}" >/dev/null 2>&1
  fi
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1
    wait "${pid}" >/dev/null 2>&1
  done
  rm -rf -- "${WORK_DIR}"
  exit "${return_code}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

PARSER_PATH="${WORK_DIR}/summarize_triton_rightsizing.py"
tar -xOzf "${SOURCE_ARCHIVE}" \
  "${archive_root}/hpc/roihu/summarize_triton_rightsizing.py" >"${PARSER_PATH}"
chmod 600 "${PARSER_PATH}"
python3 "${PARSER_PATH}" --help >"${EVIDENCE_DIR}/parser-help.txt"

for ((index = 0; index < GPU_COUNT; index++)); do
  repository="${WORK_DIR}/model-repository-${index}"
  mkdir -p -- "${repository}/accelerator_qualification/1"
  cp -- "${MODEL_PLAN}" "${repository}/accelerator_qualification/1/model.plan"
  cp -- "${TRITON_CONFIG_PATH}" "${repository}/accelerator_qualification/config.pbtxt"
done
mkdir -p -- "${WORK_DIR}/nsys-tmp"

port_seed=$((SLURM_JOB_ID % 1000))
HTTP_BASE=$((20000 + port_seed * 10))
GRPC_BASE=$((30000 + port_seed * 10))
METRICS_BASE=$((40000 + port_seed * 10))
ENABLE_NSYS="${ENABLE_NSYS:-0}"
[[ "${ENABLE_NSYS}" =~ ^(0|1)$ ]] || fail "ENABLE_NSYS must be 0 or 1"
if [[ "${ENABLE_NSYS}" == "1" && "${GPU_COUNT}" != "1" ]]; then
  fail "Nsight mode is limited to the one-GPU profiling point"
fi

for ((index = 0; index < GPU_COUNT; index++)); do
  http_port=$((HTTP_BASE + index))
  grpc_port=$((GRPC_BASE + index))
  metrics_port=$((METRICS_BASE + index))
  bind_args=(--cleanenv --containall --nv --bind "${WORK_DIR}:/work")
  server_command=(
    tritonserver
    "--model-repository=/work/model-repository-${index}"
    --model-control-mode=none
    --strict-model-config=true
    --exit-on-error=true
    --http-address=127.0.0.1 "--http-port=${http_port}"
    --grpc-address=127.0.0.1 "--grpc-port=${grpc_port}"
    --metrics-address=127.0.0.1 "--metrics-port=${metrics_port}"
    --allow-http=true --allow-grpc=true --allow-metrics=true --allow-gpu-metrics=true
  )
  if [[ "${ENABLE_NSYS}" == "1" ]]; then
    env -u LD_PRELOAD \
      APPTAINERENV_CUDA_VISIBLE_DEVICES="${index}" \
      APPTAINERENV_TMPDIR=/work/nsys-tmp \
      srun --overlap --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --argos=no \
      apptainer exec "${bind_args[@]}" "${TRITON_SERVER_SIF}" \
      /usr/local/cuda/bin/nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --duration=45 \
        --stop-on-exit=false \
        --kill=sigterm \
        --force-overwrite=true \
        --output=/work/nsys-profile \
        "${server_command[@]}" \
        >"${EVIDENCE_DIR}/triton-server-${index}.log" 2>&1 &
  else
    APPTAINERENV_CUDA_VISIBLE_DEVICES="${index}" \
      apptainer exec "${bind_args[@]}" "${TRITON_SERVER_SIF}" \
      "${server_command[@]}" \
      >"${EVIDENCE_DIR}/triton-server-${index}.log" 2>&1 &
  fi
  SERVER_PIDS+=("$!")
done

for ((index = 0; index < GPU_COUNT; index++)); do
  ready=0
  http_port=$((HTTP_BASE + index))
  for _ in $(seq 1 60); do
    if curl --fail --silent "http://127.0.0.1:${http_port}/v2/health/ready" >/dev/null; then
      ready=1
      break
    fi
    kill -0 "${SERVER_PIDS[index]}" 2>/dev/null \
      || fail "Triton server ${index} exited before readiness"
    sleep 1
  done
  [[ "${ready}" -eq 1 ]] || fail "Triton server ${index} readiness deadline exceeded"
  curl --fail --silent "http://127.0.0.1:${http_port}/v2" \
    >"${EVIDENCE_DIR}/triton-server-${index}-metadata.json"
done

printf '%s\n' \
  'timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw' \
  >"${EVIDENCE_DIR}/nvidia-smi.csv"
nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -lms 200 \
  >>"${EVIDENCE_DIR}/nvidia-smi.csv" 2>"${EVIDENCE_DIR}/nvidia-smi.err" &
TELEMETRY_PID="$!"
sleep 1
kill -0 "${TELEMETRY_PID}" 2>/dev/null || fail "GPU telemetry sampler failed to start"

declare -a CLIENT_PIDS=()
for ((index = 0; index < GPU_COUNT; index++)); do
  http_port=$((HTTP_BASE + index))
  apptainer exec \
    --cleanenv --containall \
    --bind "${EVIDENCE_DIR}:/evidence" \
    "${TRITON_SDK_SIF}" \
    perf_analyzer \
      -m accelerator_qualification \
      -i http \
      -u "127.0.0.1:${http_port}" \
      -b 64 \
      --concurrency-range=4 \
      --percentile=95 \
      --measurement-interval=3000 \
      --stability-percentage=20 \
      --max-trials=3 \
      --verbose-csv \
      -f "/evidence/perf-server-${index}.csv" \
      >"${EVIDENCE_DIR}/perf-server-${index}.log" 2>&1 &
  CLIENT_PIDS+=("$!")
done

client_status=0
for pid in "${CLIENT_PIDS[@]}"; do
  wait "${pid}" || client_status=1
done
[[ "${client_status}" -eq 0 ]] || fail "one or more Perf Analyzer clients failed"

kill "${TELEMETRY_PID}" >/dev/null 2>&1
wait "${TELEMETRY_PID}" >/dev/null 2>&1 || true
TELEMETRY_PID=""

for ((index = 0; index < GPU_COUNT; index++)); do
  metrics_port=$((METRICS_BASE + index))
  curl --fail --silent "http://127.0.0.1:${metrics_port}/metrics" \
    >"${EVIDENCE_DIR}/triton-metrics-${index}.prom"
  grep -q 'nv_inference_' "${EVIDENCE_DIR}/triton-metrics-${index}.prom" \
    || fail "server ${index} emitted no inference metrics"
done

if [[ "${ENABLE_NSYS}" == "1" ]]; then
  wait "${SERVER_PIDS[0]}" || true
  SERVER_PIDS=()
  [[ -s "${WORK_DIR}/nsys-profile.nsys-rep" ]] || fail "Nsight report was not produced"
  cp -- "${WORK_DIR}/nsys-profile.nsys-rep" "${EVIDENCE_DIR}/nsys-profile.nsys-rep"
  for report in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum osrt_sum; do
    output_name="${report//_/-}"
    apptainer exec \
      --cleanenv --containall \
      --bind "${EVIDENCE_DIR}:/evidence" \
      "${TRITON_SERVER_SIF}" \
      /usr/local/cuda/bin/nsys stats \
        --report "${report}" \
        --format csv \
        --output "/evidence/nsys-${output_name}" \
        --force-overwrite=true \
        /evidence/nsys-profile.nsys-rep \
        >"${EVIDENCE_DIR}/nsys-${output_name}.log" 2>&1
  done
  cp -- "${EVIDENCE_DIR}/nsys-cuda-gpu-kern-sum.csv" \
    "${EVIDENCE_DIR}/nsys-cuda-kernel-summary.csv"
  cp -- "${EVIDENCE_DIR}/nsys-cuda-api-sum.csv" \
    "${EVIDENCE_DIR}/nsys-cuda-api-summary.csv"
else
  for pid in "${SERVER_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1
  done
  for pid in "${SERVER_PIDS[@]}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  SERVER_PIDS=()
fi

summary_args=(
  run
  --run-dir "${EVIDENCE_DIR}"
  --output "${EVIDENCE_DIR}/summary.json"
  --gpu-count "${GPU_COUNT}"
  --source-commit "${SOURCE_GIT_COMMIT}"
  --source-archive-sha256 "${SOURCE_SHA256}"
  --model-plan-sha256 "${MODEL_PLAN_SHA256}"
  --parent-formal-job-id "${PARENT_FORMAL_JOB_ID}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --partition "${SLURM_JOB_PARTITION}"
  --batch-size 64
  --concurrency 4
  --minimum-telemetry-samples 10
)
if [[ "${ENABLE_NSYS}" == "1" ]]; then
  summary_args+=(--profiler-enabled)
fi
python3 "${PARSER_PATH}" "${summary_args[@]}"

find "${EVIDENCE_DIR}" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${EVIDENCE_DIR}/SHA256SUMS"
printf 'Roihu Triton right-sizing SMOKE_ONLY evidence completed: %s\n' "${EVIDENCE_DIR}"
