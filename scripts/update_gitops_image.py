"""Update an allow-listed GitOps image to an immutable OCI digest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

ENVIRONMENTS = ("dev", "preprod", "prod")
COMPONENT_IMAGES = {
    "api": "ghcr.io/xm2325/regulated_ml_platform",
}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def update_overlay(root: Path, environment: str, component: str, digest: str) -> dict[str, str]:
    """Replace one declared overlay tag with an allow-listed immutable digest."""

    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unsupported environment: {environment}")
    if component not in COMPONENT_IMAGES:
        raise ValueError(f"Unsupported component: {component}")
    if not _DIGEST.fullmatch(digest):
        raise ValueError("Digest must match sha256 followed by 64 lowercase hexadecimal characters")

    overlay = root / "gitops" / "environments" / environment / "kustomization.yaml"
    if not overlay.is_file():
        raise FileNotFoundError(f"Overlay not found: {overlay}")
    payload = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError(f"Overlay has no image allow-list: {overlay}")

    image_name = COMPONENT_IMAGES[component]
    image_entry: dict[str, Any] | None = None
    for candidate in payload["images"]:
        if isinstance(candidate, dict) and candidate.get("name") == image_name:
            image_entry = candidate
            break
    if image_entry is None:
        raise ValueError(f"Component {component!r} is not deployed by the {environment!r} overlay")

    previous = str(image_entry.get("digest") or image_entry.get("newTag") or "unversioned")
    image_entry["newName"] = image_name
    image_entry.pop("newTag", None)
    image_entry["digest"] = digest

    rendered = yaml.safe_dump(payload, sort_keys=False, width=120)
    temporary = overlay.with_suffix(".yaml.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(overlay)
    return {
        "environment": environment,
        "component": component,
        "image": image_name,
        "previous": previous,
        "digest": digest,
        "overlay": str(overlay),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pin an allow-listed GitOps image by digest")
    parser.add_argument("--environment", required=True, choices=ENVIRONMENTS)
    parser.add_argument("--component", required=True, choices=sorted(COMPONENT_IMAGES))
    parser.add_argument("--digest", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    evidence = update_overlay(args.root.resolve(), args.environment, args.component, args.digest)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
