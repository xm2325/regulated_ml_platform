import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "helm" / "regulated-ai"
TEMPLATES = CHART / "templates"
EVIDENCE_SHA256 = "a" * 64


def _template_text(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def _helm_template(*extra_args: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not available in this test environment")
    result = subprocess.run(
        [helm, "template", "regulated-ai", str(CHART), "--namespace", "ml-platform", *extra_args],
        check=False,
        capture_output=True,
        text=True,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def _governed_gpu_args() -> tuple[str, ...]:
    return (
        "--set",
        "tritonServing.enabled=true",
        "--set",
        "tritonServing.accelerator=gpu",
        "--set",
        "tritonServing.modelFamily=neural_network",
        "--set",
        "tritonServing.gpuEvidenceApproved=true",
        "--set",
        "tritonServing.replicas=2",
        "--set",
        "tritonServing.gpu.nodeSelector.accelerator=nvidia-gh200",
        "--set",
        "tritonServing.gpuProductionEvidence.acceleratorDecision=GPU_ELIGIBLE",
        "--set",
        "tritonServing.gpuProductionEvidence.runtimeEvidence=real_gpu",
        "--set-string",
        f"tritonServing.gpuProductionEvidence.reportSha256={EVIDENCE_SHA256}",
        "--set",
        "tritonServing.gpuProductionEvidence.clusterProfile=csc-roihu-gh200",
        "--set",
        "tritonServing.gpuProductionEvidence.acceleratorProduct=nvidia-gh200",
    )


def test_v13_gpu_controls_are_disabled_by_default() -> None:
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "gpuProductionEvidence:" in values
    assert "acceleratorDecision: NOT_EVALUATED" in values
    assert "runtimeEvidence: contract_only" in values
    assert "acceleratorProduct: unverified" in values
    assert "clusterProfile: unverified" in values
    assert "kedaAutoscaling:\n    enabled: false" in values
    assert "gpuResourceQuota:\n      enabled: false" in values


def test_platform_workflow_labels_its_gpu_render_as_a_non_runtime_fixture() -> None:
    workflow = (ROOT / ".github/workflows/platform.yml").read_text(encoding="utf-8")
    assert "ci-contract-fixture-a100" in workflow
    assert "CI fixture only: this render does not prove an A100 or other GPU runtime." in workflow


def test_gpu_production_templates_encode_evidence_and_capacity_guards() -> None:
    quota = _template_text("triton-gpu-resourcequota.yaml")
    keda = _template_text("triton-keda-scaledobject.yaml")
    assert "acceleratorDecision=GPU_ELIGIBLE" in quota
    assert "runtimeEvidence=real_gpu" in quota
    assert "reportSha256" in quota
    assert "hardGpuCount" in quota
    assert "maxReplicaCount" in quota
    assert "acceleratorDecision=GPU_ELIGIBLE" in keda
    assert "runtimeEvidence=real_gpu" in keda
    assert "stabilizationWindowSeconds" in keda
    assert "nv_inference_queue_duration_us" in keda
    assert "nv_gpu_utilization" in keda


def test_observability_and_isolation_contracts_are_least_privilege() -> None:
    network_policy = _template_text("triton-networkpolicy.yaml")
    service_monitor = _template_text("triton-servicemonitor.yaml")
    prometheus_rule = _template_text("triton-prometheusrule.yaml")
    assert "policyTypes:\n    - Ingress\n    - Egress" in network_policy
    assert "namespaceSelector:" in network_policy
    assert "podSelector:" in network_policy
    assert "port: 53" in network_policy
    assert "kind: ServiceMonitor" in service_monitor
    assert "port: metrics" in service_monitor
    assert "kind: PrometheusRule" in prometheus_rule
    assert "RegulatedAITritonNoReadyReplicas" in prometheus_rule
    assert "RegulatedAITritonQueueTimeHigh" in prometheus_rule
    assert "RegulatedAITritonGpuSaturated" in prometheus_rule


def test_helm_renders_complete_governed_gpu_contract() -> None:
    result = _helm_template(
        *_governed_gpu_args(),
        "--set",
        "tritonServing.productionControls.podDisruptionBudget.enabled=true",
        "--set",
        "tritonServing.productionControls.networkPolicy.enabled=true",
        "--set",
        "tritonServing.productionControls.gpuResourceQuota.enabled=true",
        "--set",
        "tritonServing.productionControls.serviceMonitor.enabled=true",
        "--set",
        "tritonServing.productionControls.prometheusRule.enabled=true",
        "--set",
        "tritonServing.kedaAutoscaling.enabled=true",
    )
    rendered = result.stdout
    for kind in (
        "PodDisruptionBudget",
        "NetworkPolicy",
        "ResourceQuota",
        "ServiceMonitor",
        "PrometheusRule",
        "ScaledObject",
    ):
        assert f"kind: {kind}" in rendered
    assert 'requests.nvidia.com/gpu: "8"' in rendered
    assert "minReplicaCount: 2" in rendered
    assert "maxReplicaCount: 8" in rendered
    assert "stabilizationWindowSeconds: 600" in rendered
    assert "regulated_ai_triton_queue_time_ms_per_request" in rendered
    assert f'regulated-ai/gpu-evidence-sha256: "{EVIDENCE_SHA256}"' in rendered
    assert 'regulated-ai/cluster-profile: "csc-roihu-gh200"' in rendered
    assert 'regulated-ai/accelerator-product: "nvidia-gh200"' in rendered


def test_keda_contract_rejects_non_real_gpu_evidence() -> None:
    args = list(_governed_gpu_args())
    runtime_index = args.index("tritonServing.gpuProductionEvidence.runtimeEvidence=real_gpu")
    args[runtime_index] = "tritonServing.gpuProductionEvidence.runtimeEvidence=contract_only"
    result = _helm_template(
        *args,
        "--set",
        "tritonServing.kedaAutoscaling.enabled=true",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "requires runtimeEvidence=real_gpu" in result.stderr


def test_gpu_deployment_rejects_boolean_approval_without_real_evidence() -> None:
    result = _helm_template(
        "--set",
        "tritonServing.enabled=true",
        "--set",
        "tritonServing.accelerator=gpu",
        "--set",
        "tritonServing.modelFamily=neural_network",
        "--set",
        "tritonServing.gpuEvidenceApproved=true",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "requires acceleratorDecision=GPU_ELIGIBLE" in result.stderr


def test_gpu_quota_must_cover_keda_maximum_replica_capacity() -> None:
    result = _helm_template(
        *_governed_gpu_args(),
        "--set",
        "tritonServing.kedaAutoscaling.enabled=true",
        "--set",
        "tritonServing.productionControls.gpuResourceQuota.enabled=true",
        "--set",
        "tritonServing.productionControls.gpuResourceQuota.hardGpuCount=7",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "lower than required autoscaling capacity=8" in result.stderr


def test_gh200_evidence_cannot_authorize_an_a100_node_selector() -> None:
    result = _helm_template(
        *_governed_gpu_args(),
        "--set",
        "tritonServing.gpu.nodeSelector.accelerator=nvidia-a100",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "deployment requires acceleratorProduct to match tritonServing.gpu.nodeSelector.accelerator" in result.stderr


def test_gpu_contract_requires_nvidia_extended_resource() -> None:
    result = _helm_template(
        *_governed_gpu_args(),
        "--set",
        "tritonServing.gpu.resourceName=example.com/gpu",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "deployment requires gpu.resourceName=nvidia.com/gpu" in result.stderr


def test_triton_pdb_rejects_single_replica_configuration() -> None:
    result = _helm_template(
        "--set",
        "tritonServing.enabled=true",
        "--set",
        "tritonServing.replicas=1",
        "--set",
        "tritonServing.productionControls.podDisruptionBudget.enabled=true",
        expect_success=False,
    )
    assert result.returncode != 0
    assert "requires tritonServing.replicas >= 2" in result.stderr
