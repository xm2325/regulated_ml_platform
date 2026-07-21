#!/usr/bin/env bash
# Submit from Roihu-GPU with:
#   sbatch --account=project_... --export=ALL,SOURCE_ARCHIVE=...,SOURCE_SHA256=...,SOURCE_GIT_COMMIT=...,PYTHON_WHEELHOUSE=...,EVIDENCE_ROOT=... hpc/roihu/gputest_pytorch.sbatch
#SBATCH --account=project_2012997
#SBATCH --job-name=regulated-ml-gh200-qual
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=roihu-pytorch-%j.out

set -euo pipefail
umask 077

# Exclude the reusable Slurm submission directory from Python's implicit import
# path before any Python or pip process can start.
export PYTHONSAFEPATH=1

fail() {
  printf 'Roihu qualification job failed closed: %s\n' "$1" >&2
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
    *) fail "dependency path must resolve below this project's /projappl or /scratch" ;;
  esac
}

for variable in SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_SUBMIT_DIR SOURCE_ARCHIVE SOURCE_SHA256 SOURCE_GIT_COMMIT PYTHON_WHEELHOUSE EVIDENCE_ROOT TMPDIR; do
  require_environment "${variable}"
done

[[ "${SLURM_JOB_ACCOUNT}" =~ ^project_[0-9]+$ ]] || fail "SLURM_JOB_ACCOUNT must be a CSC project identifier"
[[ "${SLURM_JOB_PARTITION}" == "gputest" ]] || fail "this bounded exercise must run in the gputest partition"
[[ "${SOURCE_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] || fail "SOURCE_SHA256 must contain exactly 64 hexadecimal characters"
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "SOURCE_GIT_COMMIT must be a full Git commit"
SOURCE_SHA256="${SOURCE_SHA256,,}"
SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT,,}"

SOURCE_ARCHIVE="$(realpath -e -- "${SOURCE_ARCHIVE}")"
PYTHON_WHEELHOUSE="$(realpath -e -- "${PYTHON_WHEELHOUSE}")"
SLURM_SUBMIT_DIR="$(realpath -e -- "${SLURM_SUBMIT_DIR}")"
require_project_path "${SOURCE_ARCHIVE}" "/projappl"
require_project_storage_path "${PYTHON_WHEELHOUSE}"
require_project_path "${SLURM_SUBMIT_DIR}" "/scratch"
[[ -f "${SOURCE_ARCHIVE}" ]] || fail "SOURCE_ARCHIVE must be a regular file"
[[ "${SOURCE_ARCHIVE}" == *.tar.gz ]] || fail "SOURCE_ARCHIVE must use the .tar.gz format"
[[ -d "${PYTHON_WHEELHOUSE}" ]] || fail "PYTHON_WHEELHOUSE must be a directory"

mkdir -p -- "${EVIDENCE_ROOT}"
EVIDENCE_ROOT="$(realpath -e -- "${EVIDENCE_ROOT}")"
require_project_path "${EVIDENCE_ROOT}" "/scratch"
EVIDENCE_DIR="${EVIDENCE_ROOT}/pytorch-gh200-${SLURM_JOB_ID}"
[[ ! -e "${EVIDENCE_DIR}" ]] || fail "refusing to overwrite an existing evidence directory"
mkdir -m 700 -- "${EVIDENCE_DIR}"

observed_source_sha256="$(sha256sum -- "${SOURCE_ARCHIVE}" | awk '{print $1}')"
[[ "${observed_source_sha256}" == "${SOURCE_SHA256}" ]] || fail "source archive SHA-256 mismatch"
printf '%s  %s\n' "${observed_source_sha256}" "$(basename -- "${SOURCE_ARCHIVE}")" >"${EVIDENCE_DIR}/source.sha256"

[[ -r /etc/profile.d/zz-csc-env.sh ]] || fail "CSC non-interactive environment initializer is unavailable"
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module --force purge
module load python-pytorch/2.10
# Lock Python import and pip configuration before the first Python process.
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
unset PYTHONPATH PYTHONHOME
export PIP_CONFIG_FILE=/dev/null
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PIP_FIND_LINKS PIP_EXTRA_INDEX_URL
module -t list 2>"${EVIDENCE_DIR}/module-list.txt"
python3 --version >"${EVIDENCE_DIR}/python-version.txt" 2>&1
python3 -c 'import torch; print(torch.__version__); print(torch.version.cuda)' >"${EVIDENCE_DIR}/torch-version.txt"

WORK_DIR="$(mktemp -d -- "${TMPDIR%/}/regulated-ml-gh200-${SLURM_JOB_ID}.XXXXXX")"
SAMPLER_PID=""

write_job_manifest() {
  local return_code="$1"
  local status="ERROR"
  [[ "${return_code}" -eq 0 ]] && status="SMOKE_PASS"
  python3 - "${EVIDENCE_DIR}" "${status}" "${return_code}" "${SOURCE_GIT_COMMIT}" "${SOURCE_SHA256}" "${PYTHON_WHEELHOUSE}" <<'PY'
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
source_sha256 = sys.argv[5]
wheelhouse = Path(sys.argv[6])
artifacts = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "job_manifest.json":
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    artifacts.append({
        "path": str(path.relative_to(root)),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    })
payload = {
    "schema_version": "regulated-ml-platform.roihu-sbatch-evidence/v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "exit_code": return_code,
    "source_git_commit": commit,
    "source_archive_sha256": source_sha256,
    "dependency_inputs": {
        "python_wheelhouse": {
            "directory_name": wheelhouse.name,
            "validation_evidence": "python-wheelhouse-input.json",
            "environment_evidence": "python-wheelhouse-environment.json",
            "installation_completed": (root / "python-wheelhouse-environment.json").is_file(),
        },
    },
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
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill "${SAMPLER_PID}" >/dev/null 2>&1
    wait "${SAMPLER_PID}" >/dev/null 2>&1
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
expected_top_level = sys.argv[3]
members_seen = 0
expanded_bytes = 0
top_levels = set()
with tarfile.open(archive, mode="r:gz") as handle:
    for member in handle:
        members_seen += 1
        expanded_bytes += max(0, member.size)
        if members_seen > 50_000 or expanded_bytes > 1_000_000_000:
            raise SystemExit("source archive exceeds the bounded extraction policy")
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
            raise SystemExit("source archive contains an unsafe path")
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("source archive contains a disallowed special or linked member")
        if not (member.isdir() or member.isfile()):
            raise SystemExit("source archive contains an unsupported member type")
        top_levels.add(pure.parts[0])
        target = destination.joinpath(*pure.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            continue
        source = handle.extractfile(member)
        if source is None:
            raise SystemExit("unable to read a source archive member")
        with source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        target.chmod(0o600)
if top_levels != {expected_top_level}:
    raise SystemExit("source archive top-level directory does not attest the requested full Git commit")
root = destination / expected_top_level
required = [
    root / "hpc/roihu/accelerator_workload.py",
    root / "hpc/roihu/benchmark_pytorch_inference.py",
    root / "hpc/roihu/export_accelerator_workload.py",
    root / "hpc/roihu/onnx-wheelhouse.json",
    root / "hpc/roihu/requirements-onnx.lock",
]
if not all(path.is_file() for path in required):
    raise SystemExit("source archive is missing the Roihu accelerator-qualification runtime")
print(root)
PY
)"

PYTHON_WHEELHOUSE_CONTRACT="${SOURCE_ROOT}/hpc/roihu/onnx-wheelhouse.json"
PYTHON_WHEELHOUSE_LOCK="${SOURCE_ROOT}/hpc/roihu/requirements-onnx.lock"
python3 - \
  "${PYTHON_WHEELHOUSE_CONTRACT}" \
  "${PYTHON_WHEELHOUSE_LOCK}" \
  "${PYTHON_WHEELHOUSE}" \
  "${EVIDENCE_DIR}/python-wheelhouse-input.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


contract_path = Path(sys.argv[1]).resolve(strict=True)
lock_path = Path(sys.argv[2]).resolve(strict=True)
root = Path(sys.argv[3]).resolve(strict=True)
output_path = Path(sys.argv[4])
staged_contract = root / "onnx-wheelhouse.json"
staged_lock = root / "requirements-onnx.lock"
staged_checksums = root / "SHA256SUMS"
contract = json.loads(contract_path.read_text(encoding="utf-8"))
packages = contract.get("packages")
valid_package_list = (
    isinstance(packages, list)
    and len(packages) == 2
    and all(isinstance(item, dict) for item in packages)
    and {item.get("name") for item in packages} == {"onnx", "protobuf"}
)
packages = packages if isinstance(packages, list) else []
expected_files = {
    "onnx-wheelhouse.json",
    "requirements-onnx.lock",
    "SHA256SUMS",
    *(package.get("filename") for package in packages if isinstance(package.get("filename"), str)),
}
children = list(root.iterdir())
observed_files = {path.name for path in children}
regular_unlinked_files = all(path.is_file() and not path.is_symlink() for path in children)
package_evidence = []
package_digests_pass = True
checksum_lines = []
for package in packages:
    filename = package.get("filename")
    path = root / filename if isinstance(filename, str) else root / "INVALID"
    observed = sha256(path) if path.is_file() and not path.is_symlink() else None
    expected = package.get("sha256")
    passed = isinstance(expected, str) and observed == expected
    package_digests_pass = package_digests_pass and passed
    if isinstance(filename, str) and isinstance(expected, str):
        checksum_lines.append(f"{expected}  {filename}")
    package_evidence.append({
        "name": package.get("name"),
        "version": package.get("version"),
        "filename": filename,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "digest_status": "PASS" if passed else "FAIL",
        "size_bytes": path.stat().st_size if path.is_file() and not path.is_symlink() else None,
    })
expected_checksums = "\n".join(checksum_lines) + "\n"
checks = {
    "schema": contract.get("schema_version") == "regulated-ml-platform.roihu-python-wheelhouse/v1",
    "package_set": valid_package_list,
    "machine": contract.get("platform_machine") == platform.machine() == "aarch64",
    "python_version": platform.python_version().startswith(f'{contract.get("python_version")}.'),
    "python_cache_tag": contract.get("python_cache_tag") == sys.implementation.cache_tag == "cpython-312",
    "base_module": contract.get("base_module") == "python-pytorch/2.10",
    "python_safe_path": sys.flags.safe_path,
    "offline_batch": contract.get("runtime_contract", {}).get("network_allowed_in_batch_job") is False,
    "no_dependency_resolution": contract.get("runtime_contract", {}).get("install_dependencies") is False,
    "exact_file_inventory": regular_unlinked_files and observed_files == expected_files,
    "staged_contract_matches_source": staged_contract.is_file() and staged_contract.read_bytes() == contract_path.read_bytes(),
    "staged_lock_matches_source": staged_lock.is_file() and staged_lock.read_bytes() == lock_path.read_bytes(),
    "wheel_digests": package_digests_pass,
    "checksum_manifest": staged_checksums.is_file()
    and staged_checksums.read_text(encoding="utf-8") == expected_checksums,
}
payload = {
    "schema_version": "regulated-ml-platform.roihu-python-wheelhouse-input/v1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "source_contract_sha256": sha256(contract_path),
    "source_lock_sha256": sha256(lock_path),
    "packages": package_evidence,
    "environment": {"python": platform.python_version(), "platform_machine": platform.machine()},
}
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output_path)
if payload["status"] != "PASS":
    raise SystemExit("pinned Python wheelhouse failed validation")
PY

PYTHON_SITE_PACKAGES="${WORK_DIR}/python-site-packages"
python3 -m pip install \
  --disable-pip-version-check \
  --no-index \
  --no-deps \
  --no-cache-dir \
  --no-compile \
  --only-binary=:all: \
  --require-hashes \
  --find-links "${PYTHON_WHEELHOUSE}" \
  --target "${PYTHON_SITE_PACKAGES}" \
  --requirement "${PYTHON_WHEELHOUSE_LOCK}" \
  >"${EVIDENCE_DIR}/python-wheelhouse-install.log" 2>&1
export PYTHONPATH="${PYTHON_SITE_PACKAGES}"
python3 - "${PYTHON_WHEELHOUSE_CONTRACT}" "${EVIDENCE_DIR}/python-wheelhouse-environment.json" <<'PY'
from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import google.protobuf
import ml_dtypes
import numpy
import onnx
import torch
from packaging.requirements import Requirement

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {package["name"]: package["version"] for package in contract["packages"]}
package_versions = {
    "onnx": onnx.__version__,
    "protobuf": google.protobuf.__version__,
    "numpy": numpy.__version__,
    "typing_extensions": metadata.version("typing_extensions"),
    "ml_dtypes": ml_dtypes.__version__,
    "torch": torch.__version__,
}
checks = {
    "pinned_onnx_version": package_versions["onnx"] == expected.get("onnx"),
    "pinned_protobuf_version": package_versions["protobuf"] == expected.get("protobuf"),
    "onnx_import": onnx.__name__ == "onnx",
    "protobuf_import": google.protobuf.__name__ == "google.protobuf",
    "python_safe_path": sys.flags.safe_path,
    "implicit_current_directory_excluded": "" not in sys.path,
}
dependencies = []
for raw_requirement in metadata.distribution("onnx").requires or []:
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    installed = metadata.version(requirement.name)
    passed = installed in requirement.specifier
    dependencies.append({
        "name": requirement.name,
        "installed_version": installed,
        "required_specifier": str(requirement.specifier),
        "status": "PASS" if passed else "FAIL",
    })
checks["onnx_direct_dependencies"] = all(item["status"] == "PASS" for item in dependencies)
payload = {
    "schema_version": "regulated-ml-platform.roihu-python-wheelhouse-environment/v1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "installation": "offline_no_index_no_deps_require_hashes_to_node_local_tmpdir",
    "check_scope": "onnx-direct-dependencies-and-imports",
    "checks": checks,
    "direct_dependencies": dependencies,
    "platform_machine": platform.machine(),
    "python": platform.python_version(),
    "packages": package_versions,
}
output = Path(sys.argv[2])
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output)
if payload["status"] != "PASS":
    raise SystemExit("installed ONNX dependency contract failed validation")
PY

# Safe-path mode deliberately removes a script's directory from sys.path.
# Add only the digest-verified source module directory, ahead of the pinned
# node-local wheel target, for the two governed runtime entry points.
export PYTHONPATH="${SOURCE_ROOT}/hpc/roihu:${PYTHON_SITE_PACKAGES}"

nvidia-smi --query-gpu=timestamp,index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
  --format=csv -l 1 >"${EVIDENCE_DIR}/nvidia-smi-timeseries.csv" 2>"${EVIDENCE_DIR}/nvidia-smi-timeseries.err" &
SAMPLER_PID="$!"
sleep 1
kill -0 "${SAMPLER_PID}" 2>/dev/null || fail "nvidia-smi sampler did not start"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,compute_cap --format=csv,noheader \
  >"${EVIDENCE_DIR}/nvidia-smi-snapshot.csv"
gpu_identity="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1)"
[[ "${gpu_identity}" == *GH200* ]] || fail "allocated accelerator is not an NVIDIA GH200"
[[ "${gpu_identity}" == *9.0* ]] || fail "allocated accelerator does not report compute capability 9.0"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun --ntasks=1 --cpu-bind=cores python3 "${SOURCE_ROOT}/hpc/roihu/benchmark_pytorch_inference.py" \
  --output "${EVIDENCE_DIR}/pytorch_benchmark.json" \
  --source-root "${SOURCE_ROOT}" \
  --source-archive "${SOURCE_ARCHIVE}" \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --source-git-commit "${SOURCE_GIT_COMMIT}" \
  --batch-sizes 1,16,64,256 \
  --warmup 5 \
  --repetitions 30 \
  --cpu-threads "${SLURM_CPUS_PER_TASK}" \
  --expected-gpu-name GH200 \
  --expected-compute-capability 9.0

srun --ntasks=1 --cpu-bind=cores python3 "${SOURCE_ROOT}/hpc/roihu/export_accelerator_workload.py" \
  --output-dir "${EVIDENCE_DIR}/onnx" \
  --source-root "${SOURCE_ROOT}" \
  --source-archive "${SOURCE_ARCHIVE}" \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --source-git-commit "${SOURCE_GIT_COMMIT}" \
  --opset 18

python3 - "${EVIDENCE_DIR}/pytorch_benchmark.json" "${EVIDENCE_DIR}/onnx/export_manifest.json" "${EVIDENCE_DIR}/cross_artifact_validation.json" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

benchmark_path, export_path, output_path = map(Path, sys.argv[1:])
benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
export = json.loads(export_path.read_text(encoding="utf-8"))
checks = {
    "benchmark_passed": benchmark.get("status") == "PASS",
    "export_passed": export.get("status") == "PASS",
    "onnx_checker_passed": export.get("onnx_checker", {}).get("status") == "PASS",
    "same_workload_name": benchmark.get("workload", {}).get("name") == export.get("workload", {}).get("name"),
    "same_weights_sha256": benchmark.get("workload", {}).get("weights_sha256") == export.get("workload", {}).get("weights_sha256"),
    "benchmark_blocks_production_claim": benchmark.get("claim_boundary", {}).get("production_model_claim_allowed") is False,
    "export_blocks_production_claim": export.get("claim_boundary", {}).get("production_model_claim_allowed") is False,
}
payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output_path)
if payload["status"] != "PASS":
    raise SystemExit("cross-artifact workload validation failed closed")
PY

printf 'Roihu GH200 accelerator SMOKE_ONLY evidence: %s\n' "${EVIDENCE_DIR}"
