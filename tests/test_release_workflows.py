from pathlib import Path

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


def test_runtime_image_links_back_to_source_repository() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    assert "org.opencontainers.image.source" in dockerfile
    assert "https://github.com/xm2325/regulated_ml_platform" in dockerfile
