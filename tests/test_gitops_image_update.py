from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.update_gitops_image import update_overlay

ROOT = Path(__file__).resolve().parents[1]


def _temporary_prod_overlay(tmp_path: Path) -> Path:
    overlay = tmp_path / "gitops/environments/prod/kustomization.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        (ROOT / "gitops/environments/prod/kustomization.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return overlay


def test_update_overlay_replaces_only_selected_image_with_digest(tmp_path: Path) -> None:
    overlay = _temporary_prod_overlay(tmp_path)
    digest = "sha256:" + ("a" * 64)

    evidence = update_overlay(tmp_path, "prod", "api", digest)
    payload = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    images = {entry["name"]: entry for entry in payload["images"]}

    assert evidence["previous"] == "sha256:" + ("0" * 64)
    assert evidence["digest"] == digest
    assert images["ghcr.io/xm2325/regulated_ml_platform"]["digest"] == digest
    assert "newTag" not in images["ghcr.io/xm2325/regulated_ml_platform"]
    assert set(images) == {"ghcr.io/xm2325/regulated_ml_platform"}


@pytest.mark.parametrize(
    "digest",
    ["latest", "sha256:not-a-digest", "sha256:" + ("A" * 64), "sha512:" + ("a" * 64)],
)
def test_update_overlay_rejects_non_sha256_digest(tmp_path: Path, digest: str) -> None:
    _temporary_prod_overlay(tmp_path)

    with pytest.raises(ValueError, match="Digest must match sha256"):
        update_overlay(tmp_path, "prod", "api", digest)


def test_update_overlay_rejects_unsupported_component(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported component"):
        update_overlay(tmp_path, "dev", "triton", "sha256:" + ("b" * 64))
