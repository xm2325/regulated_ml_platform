from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROIHU = Path("hpc/roihu")


def _load(name: str, path: Path):
    sys.path.insert(0, str(ROIHU.resolve()))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_deterministic_workload_contract_is_bounded_and_claim_scoped():
    workload = _load("roihu_accelerator_workload", ROIHU / "accelerator_workload.py")

    assert workload.parse_batch_sizes("1,16,64,256") == [1, 16, 64, 256]
    assert workload.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert workload.MAX_BATCH_SIZE == 256
    assert workload.WORKLOAD_CLASSIFICATION == "synthetic_accelerator_qualification_only"
    with pytest.raises(ValueError):
        workload.parse_batch_sizes("1,257")
    with pytest.raises(ValueError):
        workload.parse_batch_sizes("1,1")


@pytest.mark.parametrize(
    "script",
    [
        "gputest_pytorch.sbatch",
        "triton_tensorrt_apptainer.sbatch",
        "gpumedium_full_qualification.sbatch",
        "finalize_roihu_qualification.sbatch",
    ],
)
def test_slurm_scripts_are_valid_bash(script: str):
    subprocess.run(["bash", "-n", str(ROIHU / script)], check=True)


def test_gputest_wrapper_matches_roihu_gh200_and_is_smoke_only():
    script = (ROIHU / "gputest_pytorch.sbatch").read_text(encoding="utf-8")

    for contract in (
        "#SBATCH --account=project_2012997",
        "#SBATCH --partition=gputest",
        "#SBATCH --cpus-per-task=72",
        "#SBATCH --gres=gpu:gh200:1",
        "#SBATCH --time=00:15:00",
        "module load python-pytorch/2.10",
        "CSC_ENV_INIT_NON_INTERACTIVE=yes",
        "source /etc/profile.d/zz-csc-env.sh",
        "umask 077",
        "sha256sum",
        "regulated_ml_platform-${SOURCE_GIT_COMMIT}",
        "nvidia-smi",
        "-l 1",
        'status="SMOKE_PASS"',
        '"gpu_eligibility_decision_allowed": False',
        "PYTHON_WHEELHOUSE",
        "onnx-wheelhouse.json",
        "--no-index",
        "--no-deps",
        "--require-hashes",
        "PYTHONNOUSERSITE=1",
        "PYTHONSAFEPATH=1",
        "unset PYTHONPATH PYTHONHOME",
        "PIP_CONFIG_FILE=/dev/null",
        "onnx-direct-dependencies-and-imports",
        '"onnx_checker_passed"',
    ):
        assert contract in script
    assert "SOURCE_ARCHIVE" in script and "/projappl" in script
    assert "EVIDENCE_ROOT" in script and "/scratch" in script
    assert "pip download" not in script
    assert "docker pull" not in script
    assert script.index("export PYTHONNOUSERSITE=1") < script.index("python3")
    assert script.index("export PYTHONSAFEPATH=1") < script.index("python3")
    assert script.index("unset PYTHONPATH PYTHONHOME") < script.index("python3")
    assert 'export PYTHONPATH="${SOURCE_ROOT}/hpc/roihu:${PYTHON_SITE_PACKAGES}"' in script
    assert '"python_safe_path": sys.flags.safe_path' in script
    assert '"implicit_current_directory_excluded": "" not in sys.path' in script


def test_roihu_onnx_wheel_contract_is_arm64_pinned_and_prepared_outside_slurm():
    contract = json.loads((ROIHU / "onnx-wheelhouse.json").read_text(encoding="utf-8"))
    lock = (ROIHU / "requirements-onnx.lock").read_text(encoding="utf-8")
    helper = (ROIHU / "prepare_onnx_wheelhouse.sh").read_text(encoding="utf-8")
    wrapper = (ROIHU / "gputest_pytorch.sbatch").read_text(encoding="utf-8")

    assert contract["schema_version"] == "regulated-ml-platform.roihu-python-wheelhouse/v1"
    assert contract["python_cache_tag"] == "cpython-312"
    assert contract["platform_machine"] == "aarch64"
    packages = {package["name"]: package for package in contract["packages"]}
    assert packages["onnx"]["version"] == "1.22.0"
    assert packages["onnx"]["sha256"] == "ae5a563f281cd9d2845622cecf6c092a57e4ee1b138f66fdbbdd4200567a5e16"
    assert packages["protobuf"]["version"] == "5.29.6"
    assert packages["protobuf"]["sha256"] == "a8866b2cff111f0f863c1b3b9e7572dc7eaea23a7fae27f6fc613304046483e6"
    assert contract["runtime_contract"]["network_allowed_in_batch_job"] is False
    assert "onnx==1.22.0" in lock and "protobuf==5.29.6" in lock
    assert "sha256:ae5a563f281cd9d2845622cecf6c092a57e4ee1b138f66fdbbdd4200567a5e16" in lock
    assert "sha256:a8866b2cff111f0f863c1b3b9e7572dc7eaea23a7fae27f6fc613304046483e6" in lock
    assert "module load python-pytorch/2.10" in helper
    assert "pip download" in helper and "--only-binary=:all:" in helper and "--require-hashes" in helper
    assert "PIP_CONFIG_FILE=/dev/null" in helper and "https://pypi.org/simple" in helper
    assert "SHA256SUMS" in helper and '"$(uname -m)" == "aarch64"' in helper
    assert helper.index("export PYTHONNOUSERSITE=1") < helper.index("python3")
    assert helper.index("export PYTHONSAFEPATH=1") < helper.index("python3")
    assert helper.index("unset PYTHONPATH PYTHONHOME") < helper.index("python3")
    assert '"python_safe_path": sys.flags.safe_path' in helper
    assert "--no-index" in wrapper and "--no-deps" in wrapper and "--require-hashes" in wrapper
    assert "python3 -m pip check" not in wrapper
    assert 'metadata.distribution("onnx").requires' in wrapper
    assert "pip download" not in wrapper


def test_triton_smoke_is_loopback_digest_verified_and_not_gate_evidence():
    script = (ROIHU / "triton_tensorrt_apptainer.sbatch").read_text(encoding="utf-8")

    assert "SMOKE_ONLY" in script
    assert '"status": "SMOKE_PASS"' in script
    assert "--cleanenv --containall --nv" in script
    assert "trtexec" in script and "tritonserver" in script and "perf_analyzer" in script
    assert "TRTEXEC_PATH=/usr/src/tensorrt/bin/trtexec" in script
    assert 'test -x "${TRTEXEC_PATH}"' in script
    assert '-b "${batch_size}"' in script
    assert "--batch-size=" not in script
    assert "--http-address=127.0.0.1" in script
    assert "--grpc-address=127.0.0.1" in script
    assert "--metrics-address=127.0.0.1" in script
    assert '"/projappl/${SLURM_JOB_ACCOUNT}"/*|"/scratch/${SLURM_JOB_ACCOUNT}"/*' in script
    assert "triton_server_version" in script
    assert "triton_container_release" in script
    assert "semantic_parity_against_pytorch_established" in script
    assert '"gpu_eligibility_decision_allowed": False' in script
    assert "apptainer pull" not in script


def test_full_qualification_has_formal_duration_parity_and_telemetry_contracts():
    script = (ROIHU / "gpumedium_full_qualification.sbatch").read_text(encoding="utf-8")

    for contract in (
        "#SBATCH --partition=gpumedium",
        "#SBATCH --cpus-per-task=72",
        "#SBATCH --gres=gpu:gh200:1",
        "--duration-seconds 300",
        "--minimum-samples 1000",
        "--parity-rows 1024",
        "--decision-threshold 0.5",
        "--maximum-absolute-probability-error 0.00005",
        "-lms 200",
        "--format=csv,nounits",
        "--noTF32",
        "TRTEXEC_PATH=/usr/src/tensorrt/bin/trtexec",
        "CSC_ENV_INIT_NON_INTERACTIVE=yes",
        "source /etc/profile.d/zz-csc-env.sh",
        "PYTHONSAFEPATH=1",
        "unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE",
        'export PYTHONPATH="${SOURCE_ROOT}/hpc/roihu"',
        "python-source-provenance.txt",
        'env -u PYTHONOPTIMIZE PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${SOURCE_ROOT}/hpc/roihu"',
        "qualification-entrypoint-help.txt",
        "env -u PYTHONOPTIMIZE -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1",
        "APPTAINERENV_PYTHONSAFEPATH=1",
        'APPTAINERENV_PYTHONPATH="/work/source/${EXPECTED_TOP_LEVEL}/hpc/roihu"',
        "nvcc --version",
        "com.nvidia.tensorrt.version",
        "qualify_triton_http.py",
        "AWAITING_COMPLETED_SACCT",
    ):
        assert contract in script
    assert script.count("--duration-seconds 300") == 2
    assert "TRT_PRECISION" not in script
    assert "--bf16" not in script and "--fp16" not in script
    assert "apptainer pull" not in script


def test_http_client_enforces_formal_minima_and_same_fixture_parity():
    client = (ROIHU / "qualify_triton_http.py").read_text(encoding="utf-8")

    assert "duration_seconds < 300.0" in client
    assert "minimum_samples < 1000" in client
    assert "parity_rows < 1000" in client
    assert 'np.load(args.fixture_dir / "benchmark_input.npy"' in client
    assert 'np.load(args.fixture_dir / "parity_inputs.npy"' in client
    assert 'np.load(args.fixture_dir / "cpu_parity_probabilities.npy"' in client
    assert "binary_data=True" in client
    assert "policy_decision_mismatches" in client
    assert "maximum_absolute_probability_error" in client


def test_http_client_loads_attested_sibling_without_pythonpath(tmp_path: Path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"

    completed = subprocess.run(
        [sys.executable, str((ROIHU / "qualify_triton_http.py").resolve()), "--help"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run same-input CPU and Triton HTTP qualification" in completed.stdout


def test_finalizer_parses_completed_sacct_and_gpu_telemetry(tmp_path: Path):
    finalizer = _load("roihu_finalizer", ROIHU / "finalize_roihu_qualification.py")

    slurm = finalizer._parse_sacct("123|COMPLETED|0:0|gpumedium|r-gpu001|roihu|\n", "123")
    assert slurm == {
        "cluster": "roihu",
        "partition": "gpumedium",
        "job_id": "123",
        "node_list": "r-gpu001",
        "state": "COMPLETED",
        "exit_code": "0:0",
    }
    telemetry_path = tmp_path / "nvidia-smi.csv"
    telemetry_path.write_text(
        "timestamp, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
        "2026-07-21 12:00:00, 50, 1000, 98000\n"
        "2026-07-21 12:00:00.200, 70, 2000, 98000\n",
        encoding="utf-8",
    )
    telemetry = finalizer._telemetry(telemetry_path)
    assert telemetry["sample_count"] == 2
    assert telemetry["sustained_gpu_utilization_fraction"] == pytest.approx(0.60)
    assert telemetry["peak_gpu_memory_used_mib"] == 2000
    with pytest.raises(ValueError):
        finalizer._parse_sacct("123|RUNNING|0:0|gpumedium|r-gpu001|roihu|\n", "123")


def test_dependent_finalizer_invokes_the_governed_adapter():
    wrapper = (ROIHU / "finalize_roihu_qualification.sbatch").read_text(encoding="utf-8")
    adapter = (ROIHU / "finalize_roihu_qualification.py").read_text(encoding="utf-8")

    assert "#SBATCH --partition=test" in wrapper
    assert "CSC_ENV_INIT_NON_INTERACTIVE=yes" in wrapper
    assert "source /etc/profile.d/zz-csc-env.sh" in wrapper
    assert "module load python-data" in wrapper
    assert wrapper.index("export PYTHONSAFEPATH=1") < wrapper.index("python3")
    assert "unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE" in wrapper
    assert "Python safe-path provenance guard failed" in wrapper
    assert "assert sys.flags.safe_path" not in wrapper
    readme = (ROIHU / "README.md").read_text(encoding="utf-8")
    assert "--dependency=" in readme and "afterok:" in readme
    assert "sacct" in adapter
    assert '"manifest.json"' in adapter
    assert '"benchmark.json"' in adapter
    assert '"SHA256SUMS"' in adapter
    assert "validator/roihu_gpu_evidence.py" in adapter
    assert '"decision.json"' in adapter
