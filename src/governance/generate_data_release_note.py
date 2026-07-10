from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/features.csv")
    parser.add_argument("--output", default="docs/data_release_note.md")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    lines = ["# Data release note", "", f"Rows: {len(frame):,}", f"Columns: {len(frame.columns):,}", f"Date range: {frame['observation_date'].min()} to {frame['observation_date'].max()}", "", "## Target distribution", "", f"Positive support label rate: {frame['support_needed'].mean():.4f}", "", "## Limits", "", "This dated dataset is synthetic and is intended only for testing a controlled ML lifecycle."]
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
