from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TIME_COLUMN


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna().astype(float), current.dropna().astype(float)
    if ref.empty or cur.empty:
        return 0.0
    edges = np.unique(np.quantile(ref, q=np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    ref_dist = reference.astype(str).value_counts(normalize=True)
    cur_dist = current.astype(str).value_counts(normalize=True)
    keys = sorted(set(ref_dist.index).union(cur_dist.index))
    return float(sum(abs(ref_dist.get(key, 0.0) - cur_dist.get(key, 0.0)) for key in keys) / 2.0)


def build_drift_summary(reference: pd.DataFrame, current: pd.DataFrame, windows: dict[str, str] | None = None) -> dict[str, object]:
    numeric = [{"feature": column, "psi": (psi := population_stability_index(reference[column], current[column])), "status": "review" if psi >= 0.2 else "ok", "reference_missing_rate": float(reference[column].isna().mean()), "current_missing_rate": float(current[column].isna().mean())} for column in NUMERIC_FEATURES]
    categorical = [{"feature": column, "total_variation_distance": (distance := categorical_drift(reference[column], current[column])), "status": "review" if distance >= 0.2 else "ok"} for column in CATEGORICAL_FEATURES]
    any_review = any(row["status"] == "review" for row in numeric + categorical)
    return {"overall_status": "review" if any_review else "ok", "windows": windows or {}, "numeric": numeric, "categorical": categorical}


def write_html(summary: dict[str, object], output: Path) -> None:
    numeric_rows = "\n".join(f"<tr><td>{row['feature']}</td><td>{row['psi']:.4f}</td><td>{row['status']}</td></tr>" for row in summary["numeric"])
    categorical_rows = "\n".join(f"<tr><td>{row['feature']}</td><td>{row['total_variation_distance']:.4f}</td><td>{row['status']}</td></tr>" for row in summary["categorical"])
    output.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Drift report</title><style>body{{font-family:Arial;margin:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.5rem}}</style></head><body><h1>Drift report</h1><p><b>Overall status: {summary['overall_status']}</b></p><p>Windows: {summary.get('windows', {})}</p><h2>Numeric PSI</h2><table><tr><th>Feature</th><th>PSI</th><th>Status</th></tr>{numeric_rows}</table><h2>Categorical distance</h2><table><tr><th>Feature</th><th>Distance</th><th>Status</th></tr>{categorical_rows}</table></body></html>", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="reports/drift_report.html")
    parser.add_argument("--temporal-split", action="store_true")
    args = parser.parse_args()
    reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
    windows = None
    if args.temporal_split:
        ordered = reference.assign(_date=pd.to_datetime(reference[TIME_COLUMN])).sort_values("_date")
        n = len(ordered)
        reference = ordered.iloc[: int(0.60 * n)].drop(columns="_date")
        current = ordered.iloc[int(0.90 * n) :].drop(columns="_date")
        windows = {"reference_start": str(reference[TIME_COLUMN].min()), "reference_end": str(reference[TIME_COLUMN].max()), "current_start": str(current[TIME_COLUMN].min()), "current_end": str(current[TIME_COLUMN].max())}
    summary = build_drift_summary(reference, current, windows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_html(summary, output)
    (output.parent / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote drift report to {output}")


if __name__ == "__main__":
    main()
