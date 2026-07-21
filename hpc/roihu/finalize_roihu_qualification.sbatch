#!/usr/bin/env bash
# Submit from Roihu-CPU with --dependency=afterok:<qualification-job-id>.
#SBATCH --account=project_2012997
#SBATCH --job-name=regulated-ml-gpu-finalize
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=roihu-gpu-finalize-%j.out

set -euo pipefail
umask 077
export PYTHONSAFEPATH=1

fail() {
  printf 'Roihu evidence finalizer failed closed: %s\n' "$1" >&2
  exit 2
}

for variable in SLURM_JOB_ACCOUNT QUALIFICATION_JOB_ID QUALIFICATION_EVIDENCE_DIR SOURCE_GIT_COMMIT; do
  [[ -n "${!variable:-}" ]] || fail "${variable} is required"
done
[[ "${SLURM_JOB_ACCOUNT}" =~ ^project_[0-9]+$ ]] || fail "invalid CSC project account"
[[ "${QUALIFICATION_JOB_ID}" =~ ^[0-9]+$ ]] || fail "qualification job ID must be numeric"
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit must be a lowercase full Git commit"
QUALIFICATION_EVIDENCE_DIR="$(realpath -e -- "${QUALIFICATION_EVIDENCE_DIR}")"
case "${QUALIFICATION_EVIDENCE_DIR}" in
  "/scratch/${SLURM_JOB_ACCOUNT}"/*) ;;
  *) fail "qualification evidence must resolve below this project's scratch directory" ;;
esac
[[ -f "${QUALIFICATION_EVIDENCE_DIR}/validator/finalize_roihu_qualification.py" ]] \
  || fail "digest-staged finalizer is missing"

[[ -r /etc/profile.d/zz-csc-env.sh ]] || fail "CSC non-interactive environment initializer is unavailable"
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module --force purge
module load python-data
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
unset PYTHONPATH PYTHONHOME
python3 -c 'import sys; assert sys.flags.safe_path and "" not in sys.path'
python3 "${QUALIFICATION_EVIDENCE_DIR}/validator/finalize_roihu_qualification.py" \
  --evidence-root "${QUALIFICATION_EVIDENCE_DIR}" \
  --job-id "${QUALIFICATION_JOB_ID}" \
  --source-commit "${SOURCE_GIT_COMMIT}"
