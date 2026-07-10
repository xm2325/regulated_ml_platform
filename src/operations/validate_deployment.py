from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_resources(root: Path) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if isinstance(document, dict):
                document["_source"] = str(path)
                resources.append(document)
    return resources


def validate(root: Path) -> dict[str, Any]:
    resources = load_resources(root)
    kinds = {resource.get("kind") for resource in resources}
    deployments = [resource for resource in resources if resource.get("kind") == "Deployment" and resource.get("metadata", {}).get("name") == "regulated-ai-mlops"]
    checks: dict[str, bool] = {"deployment_present": bool(deployments), "horizontal_pod_autoscaler_present": "HorizontalPodAutoscaler" in kinds, "pod_disruption_budget_present": "PodDisruptionBudget" in kinds, "network_policy_present": "NetworkPolicy" in kinds, "service_account_present": "ServiceAccount" in kinds}
    if deployments:
        deployment = deployments[0]
        pod_spec = deployment["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        container_security = container.get("securityContext", {})
        pod_security = pod_spec.get("securityContext", {})
        checks.update({"replicas_at_least_two": int(deployment["spec"].get("replicas", 0)) >= 2, "readiness_probe": "readinessProbe" in container, "liveness_probe": "livenessProbe" in container, "startup_probe": "startupProbe" in container, "resource_requests": bool(container.get("resources", {}).get("requests")), "resource_limits": bool(container.get("resources", {}).get("limits")), "run_as_non_root": bool(container_security.get("runAsNonRoot") or pod_security.get("runAsNonRoot")), "read_only_root_filesystem": bool(container_security.get("readOnlyRootFilesystem")), "privilege_escalation_disabled": container_security.get("allowPrivilegeEscalation") is False, "capabilities_dropped": "ALL" in container_security.get("capabilities", {}).get("drop", []), "seccomp_runtime_default": pod_security.get("seccompProfile", {}).get("type") == "RuntimeDefault", "service_account_configured": bool(pod_spec.get("serviceAccountName"))})
    failed = [name for name, passed in checks.items() if not passed]
    return {"status": "PASS" if not failed else "REVIEW", "checks": checks, "failed_checks": failed, "resource_count": len(resources), "kinds": sorted(str(kind) for kind in kinds if kind)}


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = ["# Deployment security and reliability validation", "", f"## Validation status: **{report['status']}**", "", "| Check | Result |", "|---|---|"]
    lines.extend(f"| {name} | {'PASS' if passed else 'REVIEW'} |" for name, passed in report["checks"].items())
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="k8s")
    parser.add_argument("--output-json", default="reports/deployment_validation.json")
    parser.add_argument("--output-md", default="reports/deployment_validation.md")
    args = parser.parse_args()
    report = validate(Path(args.root))
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
