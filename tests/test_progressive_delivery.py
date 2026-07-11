from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.operations.canary_gate import evaluate_gate, load_policy

ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _yaml(path: str) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_canary_gate_passes_healthy_observation_window() -> None:
    result = evaluate_gate(
        _json("examples/canary_metrics_pass.json"),
        load_policy(ROOT / "config/canary_gate.yaml"),
    )

    assert result["decision"] == "PASS"
    assert result["status"] == "PASS"
    assert result["failed_checks"] == []
    assert all(check["passed"] for check in result["checks"].values())


def test_canary_gate_rolls_back_regressed_observation_window() -> None:
    result = evaluate_gate(
        _json("examples/canary_metrics_rollback.json"),
        load_policy(ROOT / "config/canary_gate.yaml"),
    )

    assert result["decision"] == "ROLLBACK"
    assert result["release_recommendation"] == "abort_canary_and_restore_stable"
    assert {"availability", "error_rate", "p95_latency_ms", "drift_psi", "fairness_gap"} <= set(
        result["failed_checks"]
    )


def test_canary_gate_fails_closed_when_mandatory_metric_is_missing() -> None:
    metrics = _json("examples/canary_metrics_pass.json")
    del metrics["canary"]["drift_psi"]

    result = evaluate_gate(metrics, load_policy(ROOT / "config/canary_gate.yaml"))

    assert result["decision"] == "ROLLBACK"
    assert result["failed_checks"] == ["input_valid"]
    assert result["fail_closed"] is True
    assert any("canary.drift_psi" in error for error in result["input_errors"])


def test_rollout_has_automated_analysis_and_rollback_contract() -> None:
    rollout = _yaml("gitops/base/rollout.yaml")
    spec = rollout["spec"]
    strategy = spec["strategy"]["canary"]

    assert rollout["kind"] == "Rollout"
    assert spec["progressDeadlineAbort"] is True
    assert spec["rollbackWindow"]["revisions"] >= 3
    assert strategy["stableService"] == "regulated-ai-mlops-stable"
    assert strategy["canaryService"] == "regulated-ai-mlops-canary"
    assert strategy["abortScaleDownDelaySeconds"] > 0
    assert [step["setWeight"] for step in strategy["steps"] if "setWeight" in step] == [5, 25, 50, 100]
    assert len([step for step in strategy["steps"] if "analysis" in step]) == 3

    template = _yaml("gitops/base/analysis-template.yaml")
    metrics = {metric["name"]: metric for metric in template["spec"]["metrics"]}
    assert {
        "request-volume",
        "availability",
        "error-rate",
        "p95-latency",
        "population-stability-index",
        "fairness-gap",
    } <= metrics.keys()
    assert all(metric["failureLimit"] > 0 for metric in metrics.values())
    assert all("len(result) == 0" in metric["failureCondition"] for metric in metrics.values())


def test_environment_overlays_are_namespaced_and_resource_limited() -> None:
    expected = {
        "dev": "regulated-ml-dev",
        "preprod": "regulated-ml-preprod",
        "prod": "regulated-ml-prod",
    }
    for environment, namespace in expected.items():
        prefix = f"gitops/environments/{environment}"
        overlay = _yaml(f"{prefix}/kustomization.yaml")
        namespace_manifest = _yaml(f"{prefix}/namespace.yaml")
        quota = _yaml(f"{prefix}/resourcequota.yaml")

        assert overlay["namespace"] == namespace
        assert "../../base" in overlay["resources"]
        assert namespace_manifest["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"
        assert quota["kind"] == "ResourceQuota"
        assert int(quota["spec"]["hard"]["requests.nvidia.com/gpu"]) >= 1
        assert int(quota["spec"]["hard"]["pods"]) >= 1

        assert "../../gpu" not in overlay["resources"]
        assert [image["name"] for image in overlay["images"]] == [
            "ghcr.io/xm2325/regulated_ml_platform"
        ]
        image = overlay["images"][0]
        if environment == "prod":
            assert "newTag" not in image
            assert image["digest"] == "sha256:" + ("0" * 64)
        else:
            assert image["newTag"] == "0.8.0"


def test_gitops_reconciler_automates_nonprod_but_requires_manual_prod_sync() -> None:
    project = _yaml("gitops/argocd/project.yaml")
    assert project["kind"] == "AppProject"
    assert project["spec"]["sourceRepos"] == ["https://github.com/xm2325/regulated_ml_platform.git"]
    assert project["spec"]["destinations"][0]["namespace"] == "regulated-ml-*"

    nonprod = _yaml("gitops/argocd/nonprod-applicationset.yaml")
    assert nonprod["kind"] == "ApplicationSet"
    elements = nonprod["spec"]["generators"][0]["list"]["elements"]
    assert {element["environment"] for element in elements} == {"dev", "preprod"}
    template_spec = nonprod["spec"]["template"]["spec"]
    assert template_spec["source"]["repoURL"] == "https://github.com/xm2325/regulated_ml_platform.git"
    assert template_spec["syncPolicy"]["automated"] == {
        "prune": True,
        "selfHeal": True,
        "allowEmpty": False,
    }

    production = _yaml("gitops/argocd/prod-application.yaml")
    assert production["kind"] == "Application"
    assert production["spec"]["source"]["path"] == "gitops/environments/prod"
    assert production["spec"]["destination"]["namespace"] == "regulated-ml-prod"
    assert "automated" not in production["spec"]["syncPolicy"]


def test_new_delivery_manifests_use_release_version_without_latest_tags() -> None:
    manifests = list((ROOT / "gitops").rglob("*.yaml"))
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in manifests)

    assert "0.6.0" not in rendered
    assert ":latest" not in rendered
    assert "ghcr.io/xm2325/regulated_ml_platform:0.8.0" in rendered
    assert "ghcr.io/xm2325/regulated_ml_platform-triton:0.8.0" in rendered


def test_gpu_contract_is_validated_but_not_activated_by_environment_overlays() -> None:
    gpu = _yaml("gitops/gpu/kustomization.yaml")
    assert "triton-deployment.yaml" in gpu["resources"]
    for environment in ("dev", "preprod", "prod"):
        overlay = _yaml(f"gitops/environments/{environment}/kustomization.yaml")
        assert "../../gpu" not in overlay["resources"]
        assert all("triton" not in image["name"] for image in overlay["images"])


def test_triton_contract_requests_a100_gpu_and_is_hardened() -> None:
    deployment = _yaml("gitops/gpu/triton-deployment.yaml")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert pod["nodeSelector"]["cloud.google.com/gke-accelerator"] == "nvidia-tesla-a100"
    assert any(toleration["key"] == "nvidia.com/gpu" for toleration in pod["tolerations"])
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["startupProbe"]["httpGet"]["path"] == "/v2/health/ready"
    assert container["readinessProbe"]["httpGet"]["path"] == "/v2/health/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/v2/health/live"
    assert deployment["spec"]["strategy"]["rollingUpdate"] == {
        "maxUnavailable": 1,
        "maxSurge": 0,
    }

    model_config = (ROOT / "serving/triton/models/credit-risk/config.pbtxt").read_text(encoding="utf-8")
    assert "max_batch_size: 32" in model_config
    assert "dynamic_batching" in model_config
    assert "preferred_batch_size: [4, 8, 16, 32]" in model_config
    assert "max_queue_delay_microseconds: 5000" in model_config
    assert "kind: KIND_GPU" in model_config


def test_observability_configuration_covers_release_and_gpu_risk() -> None:
    prometheus = _yaml("monitoring/prometheus.yml")
    jobs = {job["job_name"] for job in prometheus["scrape_configs"]}
    assert "kubernetes-annotated-pods" in jobs
    assert "regulated-ai-rollout-services" in jobs
    assert "argo-rollouts-controller" in jobs
    assert "/etc/prometheus/rules/*.yml" in prometheus["rule_files"]

    rules = _yaml("monitoring/alert_rules.yml")
    alerts = {
        rule["alert"]
        for group in rules["groups"]
        for rule in group["rules"]
        if "alert" in rule
    }
    assert {
        "CanaryHighErrorRate",
        "CanaryAnalysisFailed",
        "RolloutDegraded",
        "ModelDriftPSIBreach",
        "ModelFairnessGapBreach",
        "TritonInferenceFailureRate",
        "GpuSaturation",
    } <= alerts

    dashboard = _json("monitoring/grafana/dashboards/progressive-delivery.json")
    assert dashboard["uid"] == "regulated-ai-progressive-delivery"
    assert len(dashboard["panels"]) >= 8
    assert {"argo-rollouts", "triton", "gpu"} <= set(dashboard["tags"])

    datasource = _yaml("monitoring/grafana/provisioning/datasources/prometheus.yml")
    assert datasource["datasources"][0]["uid"] == "prometheus"
