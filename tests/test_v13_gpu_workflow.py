from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "triton-gpu-evidence.yml"
RUNBOOK = ROOT / "docs" / "roihu_gpu_evidence.md"


def test_gpu_evidence_workflow_is_github_hosted_contract_ci_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in workflow
    assert "github.event.pull_request.head.sha || github.sha" in workflow
    assert "ref: ${{ env.SOURCE_SHA }}" in workflow
    assert "self-hosted" not in workflow
    assert "nvidia-smi" not in workflow
    assert "--gpus" not in workflow
    assert "docker run" not in workflow
    assert 'executes_gpu_workload": False' in workflow
    assert 'proves_real_gpu": False' in workflow
    assert "CSC Roihu Slurm gpumedium/gpularge" in workflow


def test_gpu_evidence_workflow_validates_contracts_and_builds_source_bundle() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m src.operations.roihu_gpu_evidence --help" in workflow
    assert "ruff format --check" in workflow
    assert "tests/test_v13_roihu_gpu_evidence.py" in workflow
    assert "tests/test_v13_roihu_bundle.py" in workflow
    assert 'bash scripts/prepare_roihu_source_bundle.sh "${RUNNER_TEMP}/roihu-source"' in workflow
    assert "sha256sum --check" in workflow
    assert "contract-not-gpu-evidence" in workflow
    assert '".github/workflows/platform.yml"' in workflow
    assert '"helm/regulated-ai/**"' in workflow


def test_roihu_runbook_matches_v13_runtime_entrypoints_and_claim_boundaries() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "scripts/prepare_roihu_source_bundle.sh" in runbook
    assert "gputest_pytorch.sbatch" in runbook
    assert "triton_tensorrt_apptainer.sbatch" in runbook
    assert "--partition=gpumedium" in runbook
    assert "--gres=gpu:gh200:1" in runbook
    assert "72" in runbook
    assert "--cleanenv --containall --nv" in runbook
    assert "python -m src.operations.roihu_gpu_evidence" in runbook
    assert "当前 tree-ensemble champion 仍为 `CPU_ONLY`" in runbook
    assert "绿色 workflow 不执行也不证明真实 GPU" in runbook
    assert "Roihu 使用 Slurm 和 Apptainer，不是 Kubernetes" in runbook
