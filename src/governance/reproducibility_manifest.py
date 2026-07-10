from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEY_PATHS = ["requirements-runtime.lock", "config/policy.yaml", "config/promotion_gate.yaml", "models/metadata.json", "reports/model_metrics.json", "reports/promotion_gate.json", "reports/sbom.cdx.json"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, Any]:
    files = []
    for relative in KEY_PATHS:
        path = root / relative
        files.append({"path": relative, "exists": path.exists(), "sha256": sha256(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None})
    return {"created_at": datetime.now(timezone.utc).isoformat(), "status": "PASS" if all(item["exists"] for item in files) else "REVIEW", "files": files}


def write_markdown(manifest: dict[str, Any], output: Path) -> None:
    lines = ["# Reproducibility manifest", "", f"Status: `{manifest['status']}`", "", "| File | Exists | SHA-256 | Bytes |", "|---|---|---|---:|"]
    lines.extend(f"| `{item['path']}` | {item['exists']} | `{item['sha256'] or 'n/a'}` | {item['size_bytes'] or 0} |" for item in manifest["files"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-json", default="reports/reproducibility_manifest.json")
    parser.add_argument("--output-md", default="docs/reproducibility_manifest.md")
    args = parser.parse_args()
    manifest = build_manifest(Path(args.root))
    Path(args.output_json).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(manifest, Path(args.output_md))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
