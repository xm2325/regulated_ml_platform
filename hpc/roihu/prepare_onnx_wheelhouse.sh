#!/usr/bin/env bash
# Prepare the pinned ARM64 Python wheelhouse on a Roihu login node.
#
# Downloads and the compatibility smoke test happen before Slurm submission.
# Compute jobs install only these digest-verified wheels into node-local storage.

set -euo pipefail
umask 077

fail() {
  printf 'Roihu Python wheelhouse preparation failed: %s\n' "$1" >&2
  exit 1
}

[[ "$#" -eq 1 ]] || fail "usage: $0 OUTPUT_DIRECTORY"
[[ "$(uname -m)" == "aarch64" ]] || fail "run this helper on a Roihu aarch64 login node"

# Prevent user site packages, inherited .pth files, or a caller-supplied Python
# path from influencing any contract-parsing Python process.
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
unset PYTHONPATH PYTHONHOME
export PIP_CONFIG_FILE=/dev/null

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
contract_path="${script_dir}/onnx-wheelhouse.json"
lock_path="${script_dir}/requirements-onnx.lock"
[[ -f "${contract_path}" ]] || fail "the pinned wheel contract is unavailable"
[[ -f "${lock_path}" ]] || fail "the hash-locked Python requirements are unavailable"

[[ -r /etc/profile.d/zz-csc-env.sh ]] || fail "CSC environment initializer is unavailable"
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module --force purge
module load python-pytorch/2.10

# Module activation is allowed to change PATH, but not Python import provenance.
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
unset PYTHONPATH PYTHONHOME
export PIP_CONFIG_FILE=/dev/null

output_dir="$1"
[[ ! -e "${output_dir}" ]] || fail "output directory already exists"
parent_dir="$(dirname -- "${output_dir}")"
mkdir -p -- "${parent_dir}"
parent_dir="$(realpath -e -- "${parent_dir}")"
output_dir="${parent_dir}/$(basename -- "${output_dir}")"

python3 - "${contract_path}" <<'PY'
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
packages = contract.get("packages")
checks = {
    "schema": contract.get("schema_version") == "regulated-ml-platform.roihu-python-wheelhouse/v1",
    "machine": contract.get("platform_machine") == platform.machine() == "aarch64",
    "python": platform.python_version().startswith(f'{contract.get("python_version")}.'),
    "cache_tag": contract.get("python_cache_tag") == sys.implementation.cache_tag == "cpython-312",
    "base_module": contract.get("base_module") == "python-pytorch/2.10",
    "python_safe_path": sys.flags.safe_path,
    "packages": isinstance(packages, list)
    and len(packages) == 2
    and all(isinstance(item, dict) for item in packages)
    and {item.get("name") for item in packages} == {"onnx", "protobuf"},
}
if not all(checks.values()):
    raise SystemExit(f"wheelhouse contract does not match this Roihu environment: {checks}")
for package in packages:
    if not isinstance(package.get("filename"), str) or not package["filename"].endswith(".whl"):
        raise SystemExit("wheelhouse contract contains an invalid filename")
    digest = package.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit("wheelhouse contract contains an invalid SHA-256")
PY

staging_dir="$(mktemp -d -- "${parent_dir}/.roihu-python-wheelhouse.XXXXXX")"
cleanup() {
  rm -rf -- "${staging_dir}"
}
trap cleanup EXIT

unset PIP_NO_INDEX PIP_FIND_LINKS PIP_EXTRA_INDEX_URL
python3 -m pip download \
  --disable-pip-version-check \
  --index-url https://pypi.org/simple \
  --no-deps \
  --no-cache-dir \
  --only-binary=:all: \
  --require-hashes \
  --dest "${staging_dir}" \
  --requirement "${lock_path}"

python3 - "${contract_path}" "${staging_dir}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2]).resolve(strict=True)
packages = contract["packages"]
expected = {package["filename"] for package in packages}
observed = {path.name for path in root.glob("*.whl") if path.is_file() and not path.is_symlink()}
if observed != expected:
    raise SystemExit(f"downloaded wheel inventory mismatch: expected={sorted(expected)} observed={sorted(observed)}")
checksum_lines = []
for package in packages:
    path = root / package["filename"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != package["sha256"]:
        raise SystemExit(f'{package["name"]} wheel SHA-256 mismatch')
    checksum_lines.append(f'{digest}  {path.name}')
(root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
PY

validation_site="${staging_dir}/.validation-site"
python3 -m pip install \
  --disable-pip-version-check \
  --no-index \
  --no-deps \
  --no-cache-dir \
  --no-compile \
  --only-binary=:all: \
  --require-hashes \
  --find-links "${staging_dir}" \
  --target "${validation_site}" \
  --requirement "${lock_path}" >/dev/null
PYTHONPATH="${validation_site}" python3 - "${contract_path}" "${staging_dir}/.validation.onnx" <<'PY'
from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

import onnx
import torch
from packaging.requirements import Requirement

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {package["name"]: package["version"] for package in contract["packages"]}
for name, version in expected.items():
    if metadata.version(name) != version:
        raise SystemExit(f"installed {name} version does not match the pinned contract")
for raw_requirement in metadata.distribution("onnx").requires or []:
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    installed = metadata.version(requirement.name)
    if installed not in requirement.specifier:
        raise SystemExit(f"ONNX dependency mismatch: {requirement.name} {installed} not in {requirement.specifier}")
output = Path(sys.argv[2])
model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU()).eval()
torch.onnx.export(model, torch.zeros(1, 4), output, opset_version=18, dynamo=False)
graph = onnx.load(str(output), load_external_data=False)
onnx.checker.check_model(graph)
PY
rm -rf -- "${validation_site}" "${staging_dir}/.validation.onnx"

cp -- "${contract_path}" "${staging_dir}/onnx-wheelhouse.json"
cp -- "${lock_path}" "${staging_dir}/requirements-onnx.lock"
mv -- "${staging_dir}" "${output_dir}"
trap - EXIT
printf 'Prepared pinned Roihu Python wheelhouse: %s\n' "${output_dir}"
