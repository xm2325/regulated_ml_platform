from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_platform_workflow_exports_exact_tested_image() -> None:
    workflow = (ROOT / ".github/workflows/platform.yml").read_text(encoding="utf-8")
    assert "tested-container-image.tar.gz" in workflow
    assert "sha256sum" in workflow
    assert "docker save" in workflow
    assert "docker push" not in workflow
    assert "actions/deploy-pages" not in workflow


def test_release_workflow_validates_source_run_and_reuses_artifacts() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "workflow_run" in workflow
    assert ".event == \"push\"" in workflow
    assert ".head_branch == \"main\"" in workflow
    assert ".head_repository.full_name == env.GITHUB_REPOSITORY" in workflow
    assert "run-id: ${{ needs.release-context.outputs.run_id }}" in workflow
    assert "sha256sum --check" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert '"service-1.3.0" "platform-1.3.0"' in workflow


def test_runtime_image_links_back_to_source_repository() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    assert "org.opencontainers.image.source" in dockerfile
    assert "https://github.com/xm2325/regulated_ml_platform" in dockerfile


def test_v13_release_version_is_consistent_across_runtime_and_chart() -> None:
    version = "1.3.0"
    chart = yaml.safe_load((ROOT / "helm/regulated-ai/Chart.yaml").read_text(encoding="utf-8"))
    values = yaml.safe_load((ROOT / "helm/regulated-ai/values.yaml").read_text(encoding="utf-8"))
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert f'version = "{version}"' in pyproject
    assert f"PLATFORM_VERSION={version}" in dockerfile
    assert f"SERVICE_VERSION={version}" in dockerfile
    assert chart["version"] == version
    assert chart["appVersion"] == version
    assert values["image"]["tag"] == f"service-{version}"
    assert values["tritonServing"]["modelRepositoryImage"]["tag"] == f"model-0.6.0-serving-{version}"
