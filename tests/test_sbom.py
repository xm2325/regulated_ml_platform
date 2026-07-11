from src.operations.generate_sbom import build_sbom


def test_sbom_is_cyclonedx_and_lists_runtime_components():
    sbom = build_sbom(["fastapi", "numpy"])
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["metadata"]["component"]["version"] == "0.8.0"
    names = {component["name"].lower() for component in sbom["components"]}
    assert "fastapi" in names
    assert "numpy" in names
