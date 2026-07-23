#!/usr/bin/env bash
# Outer Slurm contract for the isolated ARM64 PyTorch/TorchServe runtime.
# The inner source-bound gate remains responsible for inference and evidence.
#SBATCH --account=project_2012997
#SBATCH --job-name=regulated-ml-ts-apptainer
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:12:00
#SBATCH --output=roihu-torchserve-apptainer-%j.out

set -euo pipefail
umask 077

fail() {
  printf 'Roihu TorchServe Apptainer wrapper failed closed: %s\n' "$1" >&2
  exit 1
}

for variable in \
  SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK TMPDIR \
  SOURCE_ARCHIVE SOURCE_SHA256 SOURCE_GIT_COMMIT \
  TORCHSERVE_WHEEL TORCHSERVE_WHEEL_SHA256 \
  MODEL_ARCHIVER_WHEEL MODEL_ARCHIVER_WHEEL_SHA256 \
  STAGED_JAVA_HOME STAGED_JAVA_MANIFEST_SHA256 EVIDENCE_ROOT \
  PYTORCH_SIF PYTORCH_SIF_SHA256; do
  [[ -n "${!variable:-}" ]] || fail "required environment variable ${variable} is not set"
done

[[ "${SLURM_JOB_ACCOUNT}" =~ ^project_[0-9]+$ ]] || fail "invalid project account"
[[ "${SLURM_JOB_PARTITION}" == "gputest" ]] || fail "gputest is required"
[[ "${SLURM_CPUS_PER_TASK}" -eq 72 ]] || fail "one GH200 requires 72 Grace cores"
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source commit"
for digest in "${SOURCE_SHA256}" "${PYTORCH_SIF_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256"
done

require_project_path() {
  case "$1" in
    "/projappl/${SLURM_JOB_ACCOUNT}"/*|"/scratch/${SLURM_JOB_ACCOUNT}"/*) ;;
    *) fail "runtime inputs must remain in project storage" ;;
  esac
}

SOURCE_ARCHIVE="$(realpath -e -- "${SOURCE_ARCHIVE}")"
PYTORCH_SIF="$(realpath -e -- "${PYTORCH_SIF}")"
require_project_path "${SOURCE_ARCHIVE}"
require_project_path "${PYTORCH_SIF}"
[[ "$(sha256sum "${SOURCE_ARCHIVE}" | awk '{print $1}')" == "${SOURCE_SHA256}" ]] \
  || fail "source archive SHA-256 mismatch"
[[ "$(sha256sum "${PYTORCH_SIF}" | awk '{print $1}')" == "${PYTORCH_SIF_SHA256}" ]] \
  || fail "PyTorch SIF SHA-256 mismatch"

archive_root="regulated_ml_platform-${SOURCE_GIT_COMMIT}"
marker="$(tar -xOzf "${SOURCE_ARCHIVE}" "${archive_root}/.regulated-ml-source-commit")"
[[ "${marker}" == "${SOURCE_GIT_COMMIT}" ]] || fail "source marker mismatch"
INNER_SCRIPT="${TMPDIR}/torchserve-inner-${SLURM_JOB_ID}.sbatch"
tar -xOzf "${SOURCE_ARCHIVE}" \
  "${archive_root}/hpc/roihu/torchserve_gpu_smoke.sbatch" >"${INNER_SCRIPT}"
chmod 700 "${INNER_SCRIPT}"
trap 'rm -f -- "${INNER_SCRIPT}"' EXIT

env_args=(
  --env "TORCHSERVE_RUNTIME_MODE=apptainer"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  --env "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT}"
  --env "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
  --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
  --env "TMPDIR=${TMPDIR}"
)
for variable in \
  SOURCE_ARCHIVE SOURCE_SHA256 SOURCE_GIT_COMMIT \
  TORCHSERVE_WHEEL TORCHSERVE_WHEEL_SHA256 \
  MODEL_ARCHIVER_WHEEL MODEL_ARCHIVER_WHEEL_SHA256 \
  STAGED_JAVA_HOME STAGED_JAVA_MANIFEST_SHA256 EVIDENCE_ROOT; do
  env_args+=(--env "${variable}=${!variable}")
done

exec apptainer exec \
  --nv \
  --cleanenv \
  "${env_args[@]}" \
  --bind "/scratch/${SLURM_JOB_ACCOUNT}:/scratch/${SLURM_JOB_ACCOUNT}" \
  --bind "/projappl/${SLURM_JOB_ACCOUNT}:/projappl/${SLURM_JOB_ACCOUNT}" \
  --bind "${TMPDIR}:${TMPDIR}" \
  "${PYTORCH_SIF}" \
  /bin/bash "${INNER_SCRIPT}"
