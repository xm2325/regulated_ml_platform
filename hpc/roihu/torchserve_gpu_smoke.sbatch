#!/usr/bin/env bash
# Bounded compatibility smoke for the archived TorchServe 0.12.0 release.
# This is not a benchmark or a production-serving recommendation.
#SBATCH --account=project_2012997
#SBATCH --job-name=regulated-ml-ts-smoke
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH --output=roihu-torchserve-smoke-%j.out

set -euo pipefail
umask 077

TORCHSERVE_RUNTIME_MODE="${TORCHSERVE_RUNTIME_MODE:-host}"

fail() {
  printf 'Roihu TorchServe smoke failed closed: %s\n' "$1" >&2
  exit 1
}

for variable in \
  SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK TMPDIR \
  SOURCE_ARCHIVE SOURCE_SHA256 SOURCE_GIT_COMMIT \
  TORCHSERVE_WHEEL TORCHSERVE_WHEEL_SHA256 \
  MODEL_ARCHIVER_WHEEL MODEL_ARCHIVER_WHEEL_SHA256 \
  STAGED_JAVA_HOME STAGED_JAVA_MANIFEST_SHA256 EVIDENCE_ROOT; do
  [[ -n "${!variable:-}" ]] || fail "required environment variable ${variable} is not set"
done

case "${TORCHSERVE_RUNTIME_MODE}" in
  host|apptainer) ;;
  *) fail "TORCHSERVE_RUNTIME_MODE must be host or apptainer" ;;
esac
[[ "${SLURM_JOB_ACCOUNT}" =~ ^project_[0-9]+$ ]] || fail "invalid project account"
[[ "${SLURM_JOB_PARTITION}" == "gputest" ]] || fail "TorchServe smoke must use gputest"
[[ "${SLURM_CPUS_PER_TASK}" -eq 72 ]] || fail "one full GH200 requires 72 Grace CPU cores"
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit must be a full SHA-1"
for digest in \
  "${SOURCE_SHA256}" "${TORCHSERVE_WHEEL_SHA256}" \
  "${MODEL_ARCHIVER_WHEEL_SHA256}" "${STAGED_JAVA_MANIFEST_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256"
done

require_project_path() {
  case "$1" in
    "/projappl/${SLURM_JOB_ACCOUNT}"/*|"/scratch/${SLURM_JOB_ACCOUNT}"/*) ;;
    *) fail "all inputs and evidence must remain in project storage" ;;
  esac
}

SOURCE_ARCHIVE="$(realpath -e -- "${SOURCE_ARCHIVE}")"
TORCHSERVE_WHEEL="$(realpath -e -- "${TORCHSERVE_WHEEL}")"
MODEL_ARCHIVER_WHEEL="$(realpath -e -- "${MODEL_ARCHIVER_WHEEL}")"
STAGED_JAVA_HOME="$(realpath -e -- "${STAGED_JAVA_HOME}")"
for path in \
  "${SOURCE_ARCHIVE}" "${TORCHSERVE_WHEEL}" "${MODEL_ARCHIVER_WHEEL}" \
  "${STAGED_JAVA_HOME}"; do
  require_project_path "${path}"
done

verify_digest() {
  local observed
  observed="$(sha256sum -- "$1" | awk '{print $1}')"
  [[ "${observed}" == "$2" ]] || fail "SHA-256 mismatch for $1"
}
verify_digest "${SOURCE_ARCHIVE}" "${SOURCE_SHA256}"
verify_digest "${TORCHSERVE_WHEEL}" "${TORCHSERVE_WHEEL_SHA256}"
verify_digest "${MODEL_ARCHIVER_WHEEL}" "${MODEL_ARCHIVER_WHEEL_SHA256}"
verify_digest \
  "${STAGED_JAVA_HOME}/SHA256SUMS" "${STAGED_JAVA_MANIFEST_SHA256}"
(
  cd "${STAGED_JAVA_HOME}"
  sha256sum --check --quiet SHA256SUMS
) || fail "staged OpenJDK file manifest verification failed"

archive_root="regulated_ml_platform-${SOURCE_GIT_COMMIT}"
marker="$(tar -xOzf "${SOURCE_ARCHIVE}" "${archive_root}/.regulated-ml-source-commit")"
[[ "${marker}" == "${SOURCE_GIT_COMMIT}" ]] || fail "source archive marker mismatch"

mkdir -p -- "${EVIDENCE_ROOT}"
EVIDENCE_ROOT="$(realpath -e -- "${EVIDENCE_ROOT}")"
require_project_path "${EVIDENCE_ROOT}"
EVIDENCE_DIR="${EVIDENCE_ROOT}/torchserve-smoke-${SLURM_JOB_ID}"
[[ ! -e "${EVIDENCE_DIR}" ]] || fail "refusing to overwrite evidence"
mkdir -m 700 -- "${EVIDENCE_DIR}"

if [[ "${TORCHSERVE_RUNTIME_MODE}" == "host" ]]; then
  export CSC_ENV_INIT_NON_INTERACTIVE=yes
  set +u
  source /etc/profile.d/zz-csc-env.sh
  set -u
  module --force purge
  module load python-pytorch/2.10
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="/usr/bin/python"
  [[ -x "${PYTHON_BIN}" ]] || fail "Apptainer Python is unavailable"
fi
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
JAVA_HOME="${STAGED_JAVA_HOME}"
export JAVA_HOME
[[ -x "${JAVA_HOME}/bin/java" ]] || fail "JAVA_HOME does not contain an executable java"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
[[ "$(uname -m)" == "aarch64" ]] || fail "Roihu aarch64 is required"
mapfile -t gpu_identity < <(
  nvidia-smi --query-gpu=index,uuid,name,compute_cap --format=csv,noheader,nounits
)
[[ "${#gpu_identity[@]}" -eq 1 && "${gpu_identity[0]}" == *GH200* ]] \
  || fail "exactly one GH200 is required"
printf '%s\n' "${gpu_identity[0]}" >"${EVIDENCE_DIR}/gpu-identity.csv"

WORK_DIR="${EVIDENCE_ROOT}/.torchserve-work-${SLURM_JOB_ID}"
[[ ! -e "${WORK_DIR}" ]] || fail "refusing to reuse a TorchServe work directory"
mkdir -m 700 -- "${WORK_DIR}"
mkdir -m 700 -- "${WORK_DIR}/tmp"
WORKER_LAUNCHER=""
if [[ "${TORCHSERVE_RUNTIME_MODE}" == "host" ]]; then
  NODE_LOCAL_TMPDIR="$(realpath -e -- "${TMPDIR}")"
  [[ "${NODE_LOCAL_TMPDIR}" == /tmp/* ]] || fail "Slurm TMPDIR must be node-local /tmp"
  WORKER_LAUNCHER="${NODE_LOCAL_TMPDIR}/regulated-ml-worker-${SLURM_JOB_ID}"
fi
export TEMP="${WORK_DIR}/tmp"
export TMPDIR="${WORK_DIR}/tmp"
TELEMETRY_PID=""
cleanup() {
  local return_code="$?"
  trap - EXIT
  set +e
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" >/dev/null 2>&1
    wait "${TELEMETRY_PID}" >/dev/null 2>&1
  fi
  if [[ -x "${WORK_DIR}/venv/bin/torchserve" ]]; then
    env -u LD_PRELOAD "${WORK_DIR}/venv/bin/torchserve" --stop >/dev/null 2>&1
  fi
  if [[ -n "${WORKER_LAUNCHER}" ]]; then
    rm -f -- "${WORKER_LAUNCHER}"
  fi
  cp -a -- "${WORK_DIR}/logs" "${EVIDENCE_DIR}/" 2>/dev/null
  rm -rf -- "${WORK_DIR}"
  exit "${return_code}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

for source_file in torchserve_smoke_tools.py torchserve_smoke_handler.py; do
  tar -xOzf "${SOURCE_ARCHIVE}" \
    "${archive_root}/hpc/roihu/${source_file}" >"${WORK_DIR}/${source_file}"
done
"${PYTHON_BIN}" "${WORK_DIR}/torchserve_smoke_tools.py" --help \
  >"${EVIDENCE_DIR}/tools-help.txt"

if [[ "${TORCHSERVE_RUNTIME_MODE}" == "host" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${WORK_DIR}/venv"
  "${WORK_DIR}/venv/bin/python" -m pip install \
    --disable-pip-version-check --no-index --no-deps \
    "${TORCHSERVE_WHEEL}" "${MODEL_ARCHIVER_WHEEL}" \
    >"${EVIDENCE_DIR}/offline-install.log"
else
  # NVIDIA's 24.02 runtime omits Debian's ensurepip package. Keep TorchServe in
  # a job-scoped prefix while retaining the container's governed PyTorch/CUDA.
  python_series="$("${PYTHON_BIN}" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  prefix_site="${WORK_DIR}/venv/lib/python${python_series}/site-packages"
  mkdir -m 700 -p -- "${WORK_DIR}/venv/bin" "${prefix_site}"
  "${PYTHON_BIN}" -m pip install \
    --disable-pip-version-check --no-index --no-deps --ignore-installed \
    --target "${prefix_site}" \
    "${TORCHSERVE_WHEEL}" "${MODEL_ARCHIVER_WHEEL}" \
    >"${EVIDENCE_DIR}/offline-install.log"
  ln -s -- "${PYTHON_BIN}" "${WORK_DIR}/venv/bin/python"
  cat >"${WORK_DIR}/venv/bin/torch-model-archiver" <<EOF
#!/usr/bin/bash
exec "${PYTHON_BIN}" -c \
  'from model_archiver.model_packaging import generate_model_archive; generate_model_archive()' \
  "\$@"
EOF
  cat >"${WORK_DIR}/venv/bin/torchserve" <<EOF
#!/usr/bin/bash
exec "${PYTHON_BIN}" -c 'from ts.model_server import start; start()' "\$@"
EOF
  chmod 700 \
    "${WORK_DIR}/venv/bin/torch-model-archiver" \
    "${WORK_DIR}/venv/bin/torchserve"
  export PYTHONPATH="${prefix_site}"
fi

"${WORK_DIR}/venv/bin/python" "${WORK_DIR}/torchserve_smoke_tools.py" \
  build-model --output "${WORK_DIR}/model.pt"
mkdir -p -- "${WORK_DIR}/model-store" "${WORK_DIR}/logs"
"${WORK_DIR}/venv/bin/torch-model-archiver" \
  --model-name regulated_smoke \
  --version 1.0 \
  --serialized-file "${WORK_DIR}/model.pt" \
  --handler "${WORK_DIR}/torchserve_smoke_handler.py" \
  --export-path "${WORK_DIR}/model-store" \
  --force

if [[ "${TORCHSERVE_RUNTIME_MODE}" == "host" ]]; then
  # Roihu's module Python resolves through /appl on Lustre. Keep the host-mode
  # compatibility launcher for diagnosis; the governed passing path uses the
  # isolated ARM64 PyTorch Apptainer runtime.
  worker_python_target="$(readlink -f -- "${WORK_DIR}/venv/bin/python")"
  worker_site_packages="$(
    "${WORK_DIR}/venv/bin/python" -c \
      'import sysconfig; print(sysconfig.get_paths()["purelib"])'
  )"
  [[ -x "${worker_python_target}" ]] || fail "resolved worker Python is not executable"
  [[ "${worker_site_packages}" == "${WORK_DIR}/venv/"* ]] \
    || fail "worker site-packages escaped the controlled venv"
  cat >"${WORKER_LAUNCHER}" <<EOF
#!/usr/bin/bash
set -euo pipefail
unset LD_PRELOAD
export VIRTUAL_ENV="${WORK_DIR}/venv"
export PYTHONPATH="${worker_site_packages}"
exec -a "${WORK_DIR}/venv/bin/python" "${worker_python_target}" "\$@"
EOF
  chmod 700 "${WORKER_LAUNCHER}"
  sha256sum "${WORKER_LAUNCHER}" >"${EVIDENCE_DIR}/worker-launcher.sha256"
fi

printf '%s\n' \
  'inference_address=http://127.0.0.1:8080' \
  'management_address=http://127.0.0.1:8081' \
  'metrics_address=http://127.0.0.1:8082' \
  'disable_token_authorization=true' \
  'enable_model_api=false' \
  'number_of_netty_threads=4' \
  'job_queue_size=32' \
  'async_logging=true' \
  >"${WORK_DIR}/config.properties"
if [[ "${TORCHSERVE_RUNTIME_MODE}" == "host" ]]; then
  printf '%s\n' 'vmargs=-Djdk.lang.Process.launchMechanism=FORK' \
    >>"${WORK_DIR}/config.properties"
fi

export LOG_LOCATION="${WORK_DIR}/logs"
cd "${WORK_DIR}"
"${JAVA_HOME}/bin/java" -version >"${EVIDENCE_DIR}/java-version.txt" 2>&1
if [[ "${TORCHSERVE_RUNTIME_MODE}" == "host" ]]; then
  env -u LD_PRELOAD \
    JAVA_HOME="${JAVA_HOME}" \
    TS_WORKER_PYTHON="${WORKER_LAUNCHER}" \
    "${WORK_DIR}/venv/bin/python" -c \
    'import os, sys; sys.executable = os.environ["TS_WORKER_PYTHON"]; from ts.model_server import start; start()' \
    --start \
    --ncs \
    --model-store "${WORK_DIR}/model-store" \
    --models regulated_smoke=regulated_smoke.mar \
    --ts-config "${WORK_DIR}/config.properties" \
    >"${EVIDENCE_DIR}/torchserve-start.log" 2>&1
else
  env -u LD_PRELOAD JAVA_HOME="${JAVA_HOME}" \
    "${WORK_DIR}/venv/bin/torchserve" \
    --start \
    --ncs \
    --model-store "${WORK_DIR}/model-store" \
    --models regulated_smoke=regulated_smoke.mar \
    --ts-config "${WORK_DIR}/config.properties" \
    >"${EVIDENCE_DIR}/torchserve-start.log" 2>&1
fi

ready=0
for _ in $(seq 1 60); do
  if curl --fail --silent --max-time 3 \
    http://127.0.0.1:8080/ping \
    >"${EVIDENCE_DIR}/ping.json"; then
    ready=1
    break
  fi
  sleep 1
done
[[ "${ready}" -eq 1 ]] || fail "TorchServe readiness deadline exceeded"

printf '%s\n' \
  'timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw' \
  >"${EVIDENCE_DIR}/nvidia-smi.csv"
nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -lms 200 \
  >>"${EVIDENCE_DIR}/nvidia-smi.csv" 2>"${EVIDENCE_DIR}/nvidia-smi.err" &
TELEMETRY_PID="$!"

prediction_ready=0
for _ in $(seq 1 60); do
  if curl --fail --silent \
    --max-time 3 \
    --header 'Content-Type: application/json' \
    --data '{"batch":256}' \
    http://127.0.0.1:8080/predictions/regulated_smoke \
    >"${EVIDENCE_DIR}/response.json"; then
    prediction_ready=1
    break
  fi
  sleep 1
done
[[ "${prediction_ready}" -eq 1 ]] || fail "TorchServe model worker readiness deadline exceeded"
for _ in $(seq 1 39); do
  curl --fail --silent \
    --max-time 3 \
    --header 'Content-Type: application/json' \
    --data '{"batch":256}' \
    http://127.0.0.1:8080/predictions/regulated_smoke \
    >"${EVIDENCE_DIR}/response.json"
done
sleep 2
kill "${TELEMETRY_PID}" >/dev/null 2>&1
wait "${TELEMETRY_PID}" >/dev/null 2>&1 || true
TELEMETRY_PID=""

curl --fail --silent --max-time 3 http://127.0.0.1:8082/metrics \
  >"${EVIDENCE_DIR}/torchserve-metrics.prom"
grep -q 'ts_' "${EVIDENCE_DIR}/torchserve-metrics.prom" \
  || fail "TorchServe emitted no metrics"

torchserve_version="$("${WORK_DIR}/venv/bin/torchserve" --version | awk '{print $NF}')"
pytorch_version="$("${WORK_DIR}/venv/bin/python" -c 'import torch; print(torch.__version__)')"
cuda_version="$("${WORK_DIR}/venv/bin/python" -c 'import torch; print(torch.version.cuda)')"
java_version="$("${JAVA_HOME}/bin/java" -version 2>&1 | head -n 1 | awk '{gsub(/"/, "", $3); print $3}')"

"${WORK_DIR}/venv/bin/python" "${WORK_DIR}/torchserve_smoke_tools.py" summarize \
  --response "${EVIDENCE_DIR}/response.json" \
  --telemetry "${EVIDENCE_DIR}/nvidia-smi.csv" \
  --model "${WORK_DIR}/model.pt" \
  --output "${EVIDENCE_DIR}/summary.json" \
  --batch 256 \
  --source-commit "${SOURCE_GIT_COMMIT}" \
  --source-archive-sha256 "${SOURCE_SHA256}" \
  --torchserve-wheel-sha256 "${TORCHSERVE_WHEEL_SHA256}" \
  --model-archiver-wheel-sha256 "${MODEL_ARCHIVER_WHEEL_SHA256}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --partition "${SLURM_JOB_PARTITION}" \
  --torchserve-version "${torchserve_version}" \
  --pytorch-version "${pytorch_version}" \
  --cuda-version "${cuda_version}" \
  --java-version "${java_version}"

grep -q '"status": "SMOKE_PASS"' "${EVIDENCE_DIR}/summary.json" \
  || fail "TorchServe summary did not pass"
sha256sum "${EVIDENCE_DIR}"/* >"${EVIDENCE_DIR}/SHA256SUMS"
printf 'SMOKE_ONLY: archived TorchServe compatibility proved; no performance or production claim.\n'
