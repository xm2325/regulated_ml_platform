from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_champion_challenger_report(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = [{"model_name": name, "auc": float(values["auc"]), "brier": float(values["brier"]), "expected_calibration_error": float(values.get("expected_calibration_error", 1.0)), "precision_at_policy_threshold": float(values.get("precision_at_policy_threshold", 0.0))} for name, values in metrics["models"].items()]
    rows = sorted(rows, key=lambda row: (row["auc"], -row["brier"]), reverse=True)
    champion, challenger = rows[0], rows[1] if len(rows) > 1 else None
    return {"status": "PROMOTE_CHAMPION" if challenger else "KEEP_CHAMPION", "reason": "Champion has the best out-of-time AUC among available candidates.", "champion": champion, "challenger": challenger, "auc_delta_vs_challenger": round(champion["auc"] - challenger["auc"], 6) if challenger else 0.0, "brier_delta_vs_challenger": round(champion["brier"] - challenger["brier"], 6) if challenger else 0.0, "all_candidates": rows}


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = ["# Champion-challenger report", "", f"Status: `{report['status']}`", f"Reason: {report['reason']}", "", "| Model | AUC | Brier | ECE | Policy precision |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['model_name']} | {row['auc']:.4f} | {row['brier']:.4f} | {row['expected_calibration_error']:.4f} | {row['precision_at_policy_threshold']:.4f} |" for row in report["all_candidates"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/model_metrics.json")
    parser.add_argument("--output-json", default="reports/champion_challenger_report.json")
    parser.add_argument("--output-md", default="reports/champion_challenger_report.md")
    args = parser.parse_args()
    report = build_champion_challenger_report(json.loads(Path(args.metrics).read_text(encoding="utf-8")))
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
