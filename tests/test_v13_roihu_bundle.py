from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_source_bundle_is_clean_reproducible_and_self_identifying(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script_dir = repo / "scripts"
    script_dir.mkdir()
    script = script_dir / "prepare_roihu_source_bundle.sh"
    script.write_bytes((ROOT / "scripts/prepare_roihu_source_bundle.sh").read_bytes())
    (repo / "README.md").write_text("bounded Roihu bundle\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Bundle Test")
    _git(repo, "config", "user.email", "bundle-test@example.invalid")
    _git(repo, "add", "README.md", "scripts/prepare_roihu_source_bundle.sh")
    _git(repo, "commit", "-m", "Create fixture")

    output = tmp_path / "bundle"
    subprocess.run(["bash", str(script), str(output)], cwd=repo, check=True, capture_output=True, text=True)

    manifest = json.loads((output / "source-bundle.json").read_text(encoding="utf-8"))
    archive = output / manifest["archive_filename"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    assert manifest["archive_root"] == f"regulated_ml_platform-{manifest['source_commit']}"
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
        assert all(name == manifest["archive_root"] or name.startswith(f"{manifest['archive_root']}/") for name in names)
        marker = handle.extractfile(f"{manifest['archive_root']}/.regulated-ml-source-commit")
        assert marker is not None
        assert marker.read().decode() == manifest["source_commit"]

    second_output = tmp_path / "bundle-again"
    subprocess.run(["bash", str(script), str(second_output)], cwd=repo, check=True, capture_output=True, text=True)
    second_archive = second_output / manifest["archive_filename"]
    assert second_archive.read_bytes() == archive.read_bytes()


def test_source_bundle_rejects_uncommitted_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = ROOT / "scripts/prepare_roihu_source_bundle.sh"
    _git(repo, "init")
    _git(repo, "config", "user.name", "Bundle Test")
    _git(repo, "config", "user.email", "bundle-test@example.invalid")
    (repo / "tracked.txt").write_text("first\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "Create fixture")
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    result = subprocess.run(["bash", str(script), str(tmp_path / "bundle")], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 65
    assert "dirty worktree" in result.stderr
