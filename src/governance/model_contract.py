from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.serving.schemas import PredictionRequest, PredictionResponse


def build_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    return {"contract_version": "model-contract-v2", "model_name": metadata.get("model_name", "unknown"), "model_version": metadata.get("model_version", "unknown"), "policy_version": metadata.get("policy_version", "unknown"), "feature_schema_version": metadata.get("feature_schema_version", "unknown"), "threshold": metadata.get("threshold", 0.5), "input_schema": PredictionRequest.model_json_schema(), "output_schema": PredictionResponse.model_json_schema(), "decision_semantics": {"model_output": "Calibrated probability for the synthetic support-needed target.", "policy_output": "A deterministic policy maps the score and safety conditions to an action.", "review_output": "An independent routing layer assigns auto_serve or manual_review."}, "prohibited_uses": ["financial advice", "credit approval or decline", "customer vulnerability classification", "use with real personal data without review"]}


def write_markdown(contract: dict[str, Any], output: Path) -> None:
    lines = ["# Model and decision contract", "", "The model, feature schema, policy, review route, and audit fields are versioned separately.", "", f"- Contract: `{contract['contract_version']}`", f"- Model: `{contract['model_version']}`", f"- Policy: `{contract['policy_version']}`", f"- Feature schema: `{contract['feature_schema_version']}`", f"- Threshold: `{contract['threshold']}`", "", "## Prohibited uses", ""]
    lines.extend(f"- {item}" for item in contract["prohibited_uses"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="models/metadata.json")
    parser.add_argument("--output-json", default="models/model_contract.json")
    parser.add_argument("--output-md", default="docs/model_contract.md")
    args = parser.parse_args()
    contract = build_contract(json.loads(Path(args.metadata).read_text(encoding="utf-8")))
    Path(args.output_json).write_text(json.dumps(contract, indent=2), encoding="utf-8")
    write_markdown(contract, Path(args.output_md))
    print(json.dumps({"status": "PASS", "contract_version": contract["contract_version"]}, indent=2))


if __name__ == "__main__":
    main()
