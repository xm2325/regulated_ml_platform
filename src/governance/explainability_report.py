from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def build_explainability_report(model_path: Path, data_path: Path, sample_size: int = 600) -> dict[str, Any]:
    model = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    if len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=42)
    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = frame[TARGET].astype(int)
    result = permutation_importance(model, features, target, scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1)
    rows = [
        {"feature": name, "importance_mean_auc_drop": float(mean), "importance_std": float(std)}
        for name, mean, std in zip(features.columns, result.importances_mean, result.importances_std, strict=True)
    ]
    rows = sorted(rows, key=lambda row: row["importance_mean_auc_drop"], reverse=True)
    return {
        "method": "permutation_importance_on_synthetic_sample",
        "sample_rows": int(len(frame)),
        "scoring": "roc_auc",
        "top_features": rows[:10],
        "all_features": rows,
        "limitations": "Permutation importance is a predictive model-risk check. It does not establish causality and can be unstable when features are correlated.",
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = ["# Model explainability report", "", f"Method: `{report['method']}`", f"Sample rows: `{report['sample_rows']}`", f"Scoring: `{report['scoring']}`", "", "| Rank | Feature | Mean AUC drop | Std |", "|---:|---|---:|---:|"]
    for index, row in enumerate(report["top_features"], start=1):
        lines.append(f"| {index} | {row['feature']} | {row['importance_mean_auc_drop']:.5f} | {row['importance_std']:.5f} |")
    lines.extend(["", "## Limitations", "", report["limitations"]])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--output-json", default="reports/explainability_report.json")
    parser.add_argument("--output-md", default="reports/explainability_report.md")
    parser.add_argument("--sample-size", type=int, default=600)
    args = parser.parse_args()
    report = build_explainability_report(Path(args.model), Path(args.data), args.sample_size)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(json.dumps({"top_features": report["top_features"][:5]}, indent=2))


if __name__ == "__main__":
    main()
