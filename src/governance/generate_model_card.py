from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_model_card(metrics_path: Path) -> str:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = metrics["best_model"]
    result = metrics["models"][best]
    metadata = metrics.get("metadata", {})
    intervals = metrics.get("confidence_intervals", {})
    auc_interval = intervals.get("auc", {})
    precision_interval = intervals.get("precision_at_policy_threshold", {})
    return f"""# Model card: synthetic targeted-support model

## Decision summary

The selected `{best}` uses separate chronological windows for model selection, Platt calibration, policy-threshold selection, and out-of-time evaluation.

| Item | Value |
|---|---|
| Model version | `{metrics['model_version']}` |
| Policy version | `{metadata.get('policy_version', 'unknown')}` |
| Feature schema | `{metadata.get('feature_schema_version', 'unknown')}` |
| Frozen threshold | `{metrics['policy_threshold']:.2f}` |

| Out-of-time metric | Estimate | Bootstrap 95% interval |
|---|---:|---:|
| AUC | {result['auc']:.4f} | {auc_interval.get('lower_95', float('nan')):.4f}–{auc_interval.get('upper_95', float('nan')):.4f} |
| Brier score | {result['brier']:.4f} | {intervals.get('brier', {}).get('lower_95', float('nan')):.4f}–{intervals.get('brier', {}).get('upper_95', float('nan')):.4f} |
| ECE | {result['expected_calibration_error']:.4f} | not calculated |
| Policy precision | {result['precision_at_policy_threshold']:.4f} | {precision_interval.get('lower_95', float('nan')):.4f}–{precision_interval.get('upper_95', float('nan')):.4f} |
| Policy recall | {result['recall_at_policy_threshold']:.4f} | not calculated |

## Use boundary

The model supports only a synthetic engineering demonstration. It must not be used for financial advice, credit decisions, vulnerability classification, or decisions about real people.

## Decision separation

The model returns a calibrated probability. A versioned deterministic policy maps that probability and explicit safety conditions to an action. A separate routing layer decides whether manual review is required.

## Monitoring

Monitor schema validity, missingness, temporal drift, prediction and action distributions, review rate, latency, errors, delayed-label calibration, segment error rates, and audit delivery.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/model_metrics.json")
    parser.add_argument("--output", default="docs/model_card.md")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_model_card(Path(args.metrics)), encoding="utf-8")
    print(f"Wrote model card to {output}")


if __name__ == "__main__":
    main()
