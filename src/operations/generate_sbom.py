from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECT_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
    "prometheus-client",
    "PyYAML",
]


def build_sbom(
    packages: list[str] | None = None,
    *,
    application_version: str | None = None,
) -> dict[str, object]:
    release_version = application_version or (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    components = []
    for package in packages or DIRECT_PACKAGES:
        try:
            package_version, package_metadata = version(package), metadata(package)
        except PackageNotFoundError:
            continue
        normalized = package.lower().replace("_", "-")
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{normalized}@{package_version}",
            "name": package_metadata.get("Name", package),
            "version": package_version,
            "purl": f"pkg:pypi/{normalized}@{package_version}",
        }
        license_expression = package_metadata.get("License-Expression") or package_metadata.get("License")
        if license_expression and len(license_expression) < 200:
            component["licenses"] = [{"license": {"name": license_expression}}]
        components.append(component)
    components.sort(key=lambda item: str(item["name"]).lower())
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {
                "type": "application",
                "name": "regulated-ai-mlops-platform",
                "version": release_version,
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "stdlib-importlib-metadata-sbom",
                        "version": "1",
                    }
                ]
            },
        },
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/sbom.cdx.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_sbom(), indent=2), encoding="utf-8")
    print(f"Wrote CycloneDX SBOM to {output}")


if __name__ == "__main__":
    main()
