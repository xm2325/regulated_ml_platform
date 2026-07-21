#!/usr/bin/env bash
# Formal candidate-profile qualification: >=300 s CPU, >=300 s Triton HTTP,
# >=1000 timed samples/path, >=1000 same-input parity rows, and 200 ms GPU
# telemetry. A dependent CPU finalizer must verify sacct COMPLETED before the gate.
#SBATCH --account=project_2012997
#SBATCH --job-name=regulated-ml-gh200-full
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --output=roihu-full-qualification-%j.out

set -euo pipefail
umask 077
export PYTHONSAFEPATH=1

fail() {
  printf 'Roihu full qualification failed closed: %s\n' "$1" >&2
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
  SOURCE_ARCHIVE SOURCE_SHA256 SOURCE_GIT_COMMIT \
  TRITON_SERVER_SIF TRITON_SERVER_SIF_SHA256 TRITON_SDK_SIF TRITON_SDK_SIF_SHA256 TRITON_CONTAINER_RELEASE \
  ONNX_PATH ONNX_SHA256 EXPORT_MANIFEST_PATH EXPORT_MANIFEST_SHA256 EVIDENCE_ROOT; do
  require_environment "${variable}"
done

[[ "${SLURM_JOB_ACCOUNT}" =~ ^project_[0-9]+$ ]] || fail "SLURM_JOB_ACCOUNT must be a CSC project identifier"
[[ "${SLURM_JOB_PARTITION}" == "gpumedium" ]] || fail "formal qualification must run in gpumedium"
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "SOURCE_GIT_COMMIT must be a full Git commit"
SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT,,}"
[[ "${TRITON_CONTAINER_RELEASE}" =~ ^[0-9]{2}\.[0-9]{2}$ ]] || fail "container release must use NGC year.month form"
for digest_name in SOURCE_SHA256 TRITON_SERVER_SIF_SHA256 TRITON_SDK_SIF_SHA256 ONNX_SHA256 EXPORT_MANIFEST_SHA256; do
  [[ "${!digest_name}" =~ ^[0-9a-fA-F]{64}$ ]] || fail "${digest_name} must be a 64-character SHA-256"
  digest_value="${!digest_name}"
  printf -v "${digest_name}" '%s' "${digest_value,,}"
done

SOURCE_ARCHIVE="$(realpath -e -- "${SOURCE_ARCHIVE}")"
TRITON_SERVER_SIF="$(realpath -e -- "${TRITON_SERVER_SIF}")"
TRITON_SDK_SIF="$(realpath -e -- "${TRITON_SDK_SIF}")"
ONNX_PATH="$(realpath -e -- "${ONNX_PATH}")"
EXPORT_MANIFEST_PATH="$(realpath -e -- "${EXPORT_MANIFEST_PATH}")"
SLURM_SUBMIT_DIR="$(realpath -e -- "${SLURM_SUBMIT_DIR}")"
TRITON_CONFIG_PATH="$(realpath -e -- "$(dirname -- "${ONNX_PATH}")/config.pbtxt")"
require_project_path "${SOURCE_ARCHIVE}" "/projappl"
require_project_storage_path "${TRITON_SERVER_SIF}"
require_project_storage_path "${TRITON_SDK_SIF}"
require_project_path "${ONNX_PATH}" "/scratch"
require_project_path "${EXPORT_MANIFEST_PATH}" "/scratch"
require_project_path "${TRITON_CONFIG_PATH}" "/scratch"
require_project_path "${SLURM_SUBMIT_DIR}" "/scratch"
[[ -f "${SOURCE_ARCHIVE}" && "${SOURCE_ARCHIVE}" == *.tar.gz ]] || fail "source must be a regular .tar.gz archive"
[[ -f "${TRITON_SERVER_SIF}" && "${TRITON_SERVER_SIF}" == *.sif ]] || fail "server input must be a regular SIF"
[[ -f "${TRITON_SDK_SIF}" && "${TRITON_SDK_SIF}" == *.sif ]] || fail "SDK input must be a regular SIF"

mkdir -p -- "${EVIDENCE_ROOT}"
EVIDENCE_ROOT="$(realpath -e -- "${EVIDENCE_ROOT}")"
require_project_path "${EVIDENCE_ROOT}" "/scratch"
EVIDENCE_DIR="${EVIDENCE_ROOT}/full-qualification-${SLURM_JOB_ID}"
[[ ! -e "${EVIDENCE_DIR}" ]] || fail "refusing to overwrite an existing qualification directory"
mkdir -m 700 -- "${EVIDENCE_DIR}" "${EVIDENCE_DIR}/raw" "${EVIDENCE_DIR}/release" "${EVIDENCE_DIR}/validator"

verify_digest() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local observed
  observed="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || fail "${label} SHA-256 mismatch"
  printf '%s  %s\n' "${observed}" "$(basename -- "${path}")" >"${EVIDENCE_DIR}/raw/${label}.sha256"
}
verify_digest "${SOURCE_ARCHIVE}" "${SOURCE_SHA256}" source-archive
verify_digest "${TRITON_SERVER_SIF}" "${TRITON_SERVER_SIF_SHA256}" triton-server-sif
verify_digest "${TRITON_SDK_SIF}" "${TRITON_SDK_SIF_SHA256}" triton-sdk-sif
verify_digest "${ONNX_PATH}" "${ONNX_SHA256}" onnx
verify_digest "${EXPORT_MANIFEST_PATH}" "${EXPORT_MANIFEST_SHA256}" export-manifest

[[ -r /etc/profile.d/zz-csc-env.sh ]] || fail "CSC non-interactive environment initializer is unavailable"
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module --force purge
module load python-pytorch/2.10
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
module -t list 2>"${EVIDENCE_DIR}/raw/module-list.txt"
command -v apptainer >/dev/null 2>&1 || fail "Apptainer is unavailable"
[[ "$(uname -m)" == "aarch64" ]] || fail "formal target-side qualification requires Roihu-GPU aarch64"
gpu_identity="$(nvidia-smi --query-gpu=uuid,name,compute_cap --format=csv,noheader | head -n 1)"
[[ "${gpu_identity}" == *GH200* && "${gpu_identity}" == *9.0* ]] || fail "a full GH200 compute capability 9.0 was not observed"

WORK_DIR="$(mktemp -d -- "${TMPDIR%/}/regulated-ml-full-${SLURM_JOB_ID}.XXXXXX")"
SERVER_PID=""
TELEMETRY_PID=""

cleanup() {
  local return_code="$?"
  trap - EXIT
  set +e
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" >/dev/null 2>&1
    wait "${TELEMETRY_PID}" >/dev/null 2>&1
  fi
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1
    wait "${SERVER_PID}" >/dev/null 2>&1
  fi
  python3 - "${EVIDENCE_DIR}" "${return_code}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "status": "AWAITING_COMPLETED_SACCT" if int(sys.argv[2]) == 0 else "ERROR",
    "qualification_job_exit_code": int(sys.argv[2]),
    "gpu_eligibility_decision_issued": False,
    "note": "Use the dependent CPU finalizer; a running job cannot attest its own COMPLETED state.",
}
path = root / "pre_finalize_status.json"
temporary = root / ".pre_finalize_status.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(path)
PY
  local status_code="$?"
  rm -rf -- "${WORK_DIR}"
  if [[ "${return_code}" -eq 0 && "${status_code}" -ne 0 ]]; then
    return_code="${status_code}"
  fi
  exit "${return_code}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

SOURCE_EXTRACT_ROOT="${WORK_DIR}/source"
mkdir -m 700 -- "${SOURCE_EXTRACT_ROOT}"
EXPECTED_TOP_LEVEL="regulated_ml_platform-${SOURCE_GIT_COMMIT}"
SOURCE_ROOT="$(python3 - "${SOURCE_ARCHIVE}" "${SOURCE_EXTRACT_ROOT}" "${EXPECTED_TOP_LEVEL}" <<'PY'
from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

archive = Path(sys.argv[1]).resolve(strict=True)
destination = Path(sys.argv[2]).resolve(strict=True)
expected = sys.argv[3]
if not sys.flags.safe_path or "" in sys.path:
    raise SystemExit("Python safe-path provenance guard is not active")
count = 0
expanded = 0
top = set()
with tarfile.open(archive, "r:gz") as handle:
    for member in handle:
        count += 1
        expanded += max(0, member.size)
        if count > 50_000 or expanded > 1_000_000_000:
            raise SystemExit("source archive exceeds extraction policy")
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
            raise SystemExit("unsafe source archive path")
        if member.issym() or member.islnk() or member.isdev() or member.isfifo() or not (member.isdir() or member.isfile()):
            raise SystemExit("disallowed source archive member")
        top.add(pure.parts[0])
        target = destination.joinpath(*pure.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
        else:
            source = handle.extractfile(member)
            if source is None:
                raise SystemExit("unreadable source member")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            target.chmod(0o600)
if top != {expected}:
    raise SystemExit("source root does not attest the requested full commit")
root = destination / expected
required = [
    root / "hpc/roihu/accelerator_workload.py",
    root / "hpc/roihu/qualify_triton_http.py",
    root / "hpc/roihu/finalize_roihu_qualification.py",
    root / "src/operations/roihu_gpu_evidence.py",
    root / "config/roihu_gpu_evidence_policy.yaml",
]
if not all(path.is_file() for path in required):
    raise SystemExit("source archive lacks formal Roihu qualification components")
print(root)
PY
)"
export PYTHONPATH="${SOURCE_ROOT}/hpc/roihu"
srun --ntasks=1 \
  env -u PYTHONOPTIMIZE PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${SOURCE_ROOT}/hpc/roihu" \
  python3 -c 'import sys; from pathlib import Path; import accelerator_workload; expected = Path(sys.argv[1]).resolve(strict=True); observed = Path(accelerator_workload.__file__).resolve(strict=True); valid = sys.flags.safe_path and "" not in sys.path and observed == expected; sys.exit("Python source provenance guard failed") if not valid else None; print(observed)' \
  "${SOURCE_ROOT}/hpc/roihu/accelerator_workload.py" \
  >"${EVIDENCE_DIR}/raw/python-source-provenance.txt"

python3 - "${EXPORT_MANIFEST_PATH}" "${EXPORT_MANIFEST_SHA256}" "${ONNX_PATH}" "${ONNX_SHA256}" \
  "${TRITON_CONFIG_PATH}" "${SOURCE_GIT_COMMIT}" "${EVIDENCE_DIR}/raw/input-validation.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path, onnx_path, config_path = Path(sys.argv[1]), Path(sys.argv[3]), Path(sys.argv[5])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
checks = {
    "manifest_digest": sha(manifest_path) == sys.argv[2],
    "manifest_pass": manifest.get("status") == "PASS",
    "onnx_digest": sha(onnx_path) == sys.argv[4] == manifest.get("artifacts", {}).get("onnx", {}).get("sha256"),
    "config_digest": sha(config_path) == manifest.get("artifacts", {}).get("triton_config", {}).get("sha256"),
    "source_commit": manifest.get("source", {}).get("git_commit") == sys.argv[6],
    "candidate_only": manifest.get("workload", {}).get("classification") == "synthetic_accelerator_qualification_only",
    "not_champion": manifest.get("workload", {}).get("champion_model") is False,
}
payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
output = Path(sys.argv[7])
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output)
if payload["status"] != "PASS":
    raise SystemExit("export inputs failed validation")
PY

apptainer inspect --json "${TRITON_SERVER_SIF}" >"${EVIDENCE_DIR}/raw/triton-server-sif-inspect.json"
apptainer inspect --json "${TRITON_SDK_SIF}" >"${EVIDENCE_DIR}/raw/triton-sdk-sif-inspect.json"
grep -Fq -- "${TRITON_CONTAINER_RELEASE}" "${EVIDENCE_DIR}/raw/triton-server-sif-inspect.json" \
  || fail "server SIF metadata lacks expected NGC release"
grep -Fq -- "${TRITON_CONTAINER_RELEASE}" "${EVIDENCE_DIR}/raw/triton-sdk-sif-inspect.json" \
  || fail "SDK SIF metadata lacks expected NGC release"

MODEL_REPOSITORY="${WORK_DIR}/model_repository"
MODEL_VERSION_DIR="${MODEL_REPOSITORY}/accelerator_qualification/1"
mkdir -p -- "${MODEL_VERSION_DIR}"
cp -- "${ONNX_PATH}" "${WORK_DIR}/accelerator_qualification.onnx"
cp -- "${TRITON_CONFIG_PATH}" "${MODEL_REPOSITORY}/accelerator_qualification/config.pbtxt"
APPTAINER_ARGS=(--cleanenv --containall --nv --bind "${WORK_DIR}:/work")
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONSAFEPATH=1
export APPTAINERENV_PYTHONPATH="/work/source/${EXPECTED_TOP_LEVEL}/hpc/roihu"
TRTEXEC_PATH=/usr/src/tensorrt/bin/trtexec
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" uname -m >"${EVIDENCE_DIR}/raw/sdk-architecture.txt"
grep -qx aarch64 "${EVIDENCE_DIR}/raw/sdk-architecture.txt" || fail "SDK SIF is not aarch64"
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SERVER_SIF}" uname -m >"${EVIDENCE_DIR}/raw/server-architecture.txt"
grep -qx aarch64 "${EVIDENCE_DIR}/raw/server-architecture.txt" || fail "server SIF is not aarch64"
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" test -x "${TRTEXEC_PATH}" \
  || fail "SDK SIF does not provide the expected TensorRT executable"
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" "${TRTEXEC_PATH}" --help \
  >"${EVIDENCE_DIR}/raw/trtexec-version-and-help.txt" 2>&1
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" nvcc --version \
  >"${EVIDENCE_DIR}/raw/nvcc-version.txt" 2>&1
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" "${TRTEXEC_PATH}" \
  --onnx=/work/accelerator_qualification.onnx \
  --saveEngine=/work/model_repository/accelerator_qualification/1/model.plan \
  --minShapes=INPUT__0:1x1024 \
  --optShapes=INPUT__0:64x1024 \
  --maxShapes=INPUT__0:256x1024 \
  --timingCacheFile=/work/timing.cache \
  --builderOptimizationLevel=3 \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --noTF32 \
  >"${EVIDENCE_DIR}/raw/trtexec-build-and-validation.log" 2>&1
[[ -s "${MODEL_VERSION_DIR}/model.plan" ]] || fail "no TensorRT FP32 plan was produced"

port_seed=$((SLURM_JOB_ID % 10000))
HTTP_PORT=$((20000 + port_seed))
GRPC_PORT=$((30000 + port_seed))
METRICS_PORT=$((40000 + port_seed))
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SERVER_SIF}" tritonserver \
  --model-repository=/work/model_repository \
  --model-control-mode=none \
  --strict-model-config=true \
  --exit-on-error=true \
  --http-address=127.0.0.1 --http-port="${HTTP_PORT}" \
  --grpc-address=127.0.0.1 --grpc-port="${GRPC_PORT}" \
  --metrics-address=127.0.0.1 --metrics-port="${METRICS_PORT}" \
  --allow-http=true --allow-grpc=true --allow-metrics=true --allow-gpu-metrics=true \
  >"${EVIDENCE_DIR}/raw/triton-server.log" 2>&1 &
SERVER_PID="$!"
ready=0
for _ in $(seq 1 90); do
  if curl --fail --silent --show-error "http://127.0.0.1:${HTTP_PORT}/v2/health/ready" >/dev/null; then
    ready=1
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || fail "Triton exited before readiness"
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "Triton readiness deadline exceeded"
curl --fail --silent --show-error "http://127.0.0.1:${HTTP_PORT}/v2" >"${EVIDENCE_DIR}/raw/triton-server-metadata.json"
curl --fail --silent --show-error "http://127.0.0.1:${HTTP_PORT}/v2/models/accelerator_qualification/ready" >/dev/null

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=72
FIXTURE_DIR="${WORK_DIR}/fixtures"
srun --ntasks=1 --cpu-bind=cores \
  env -u PYTHONOPTIMIZE PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${SOURCE_ROOT}/hpc/roihu" \
  python3 "${SOURCE_ROOT}/hpc/roihu/qualify_triton_http.py" \
  --mode cpu \
  --output "${EVIDENCE_DIR}/raw/cpu-benchmark.json" \
  --fixture-dir "${FIXTURE_DIR}" \
  --duration-seconds 300 \
  --minimum-samples 1000 \
  --batch-size 64 \
  --parity-rows 1024 \
  --cpu-threads 72 \
  --warmup 5

nvidia-smi --query-gpu=timestamp,index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
  --format=csv,nounits -lms 200 >"${EVIDENCE_DIR}/raw/nvidia-smi.csv" 2>"${EVIDENCE_DIR}/raw/nvidia-smi.err" &
TELEMETRY_PID="$!"
sleep 1
kill -0 "${TELEMETRY_PID}" 2>/dev/null || fail "200 ms nvidia-smi sampler did not start"
apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" python3 \
  /work/source/"${EXPECTED_TOP_LEVEL}"/hpc/roihu/qualify_triton_http.py \
  --mode gpu \
  --output /work/gpu-benchmark.json \
  --fixture-dir /work/fixtures \
  --duration-seconds 300 \
  --minimum-samples 1000 \
  --batch-size 64 \
  --concurrency 4 \
  --warmup 5 \
  --url "127.0.0.1:${HTTP_PORT}"
kill "${TELEMETRY_PID}" >/dev/null 2>&1
wait "${TELEMETRY_PID}" >/dev/null 2>&1 || true
TELEMETRY_PID=""
cp -- "${WORK_DIR}/gpu-benchmark.json" "${EVIDENCE_DIR}/raw/gpu-benchmark.json"

apptainer exec "${APPTAINER_ARGS[@]}" "${TRITON_SDK_SIF}" python3 \
  /work/source/"${EXPECTED_TOP_LEVEL}"/hpc/roihu/qualify_triton_http.py \
  --mode parity \
  --output /work/parity.json \
  --fixture-dir /work/fixtures \
  --parity-rows 1024 \
  --batch-size 64 \
  --url "127.0.0.1:${HTTP_PORT}" \
  --maximum-absolute-probability-error 0.00005 \
  --decision-threshold 0.5
cp -- "${WORK_DIR}/parity.json" "${EVIDENCE_DIR}/raw/parity.json"
cp -a -- "${FIXTURE_DIR}" "${EVIDENCE_DIR}/raw/fixtures"
curl --fail --silent --show-error "http://127.0.0.1:${METRICS_PORT}/metrics" >"${EVIDENCE_DIR}/raw/triton-metrics.prom"
grep -q 'nv_inference_' "${EVIDENCE_DIR}/raw/triton-metrics.prom" || fail "Triton emitted no inference metrics"
nvidia-smi >"${EVIDENCE_DIR}/raw/nvidia-smi-full.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits \
  >"${EVIDENCE_DIR}/raw/nvidia-smi-snapshot.csv"

python3 - "${EVIDENCE_DIR}" "${TRITON_CONTAINER_RELEASE}" <<'PY'
from __future__ import annotations

import csv
import json
import os
import platform
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
release = sys.argv[2]
snapshot = next(csv.reader((root / "raw/nvidia-smi-snapshot.csv").open(encoding="utf-8")))
if len(snapshot) != 6:
    raise SystemExit("unexpected nvidia-smi snapshot shape")
_, uuid, name, driver, memory_total, compute = [value.strip() for value in snapshot]
server = json.loads((root / "raw/triton-server-metadata.json").read_text(encoding="utf-8"))
sdk_inspect = json.loads((root / "raw/triton-sdk-sif-inspect.json").read_text(encoding="utf-8"))
labels = sdk_inspect.get("data", {}).get("attributes", {}).get("labels", {})
tensorrt_version = str(labels.get("com.nvidia.tensorrt.version", ""))
nvcc = (root / "raw/nvcc-version.txt").read_text(encoding="utf-8", errors="replace")
cuda = re.search(r"release\s+([0-9]+(?:\.[0-9]+)+)", nvcc)
if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", tensorrt_version) or not cuda:
    raise SystemExit("container CUDA toolkit or TensorRT version missing from raw runtime evidence")
payload = {
    "hardware": {
        "gpu_product_name": name,
        "compute_capability": compute,
        "cpu_architecture": platform.machine(),
        "gpu_count": 1,
        "gpu_memory_total_mib": float(memory_total),
        "gpu_uuid": uuid,
        "driver_version": driver,
    },
    "software": {
        "cuda_version": cuda.group(1),
        "tensorrt_version": tensorrt_version,
        "triton_server_version": str(server.get("version", "")),
        "triton_container_release": release,
    },
    "version_scheme_note": "Triton runtime 2.x and NGC year.month release are independent fields.",
}
output = root / "raw/software-versions.json"
temporary = root / "raw/.software-versions.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output)
PY

cp -- "${SOURCE_ARCHIVE}" "${EVIDENCE_DIR}/release/source.tar.gz"
cp -- "${MODEL_VERSION_DIR}/model.plan" "${EVIDENCE_DIR}/release/model.plan"
if ! ln -- "${TRITON_SERVER_SIF}" "${EVIDENCE_DIR}/release/triton-server.sif" 2>/dev/null; then
  cp --reflink=auto -- "${TRITON_SERVER_SIF}" "${EVIDENCE_DIR}/release/triton-server.sif"
fi
cp -- "${SOURCE_ROOT}/hpc/roihu/finalize_roihu_qualification.py" "${EVIDENCE_DIR}/validator/"
cp -- "${SOURCE_ROOT}/src/operations/roihu_gpu_evidence.py" "${EVIDENCE_DIR}/validator/"
cp -- "${SOURCE_ROOT}/config/roihu_gpu_evidence_policy.yaml" "${EVIDENCE_DIR}/validator/"

printf 'Qualification runtime complete; no eligibility decision is issued until sacct completion.\n'
printf 'Evidence directory: %s\n' "${EVIDENCE_DIR}"
printf 'Submit finalize_roihu_qualification.sbatch with QUALIFICATION_JOB_ID=%s and this evidence path.\n' "${SLURM_JOB_ID}"
