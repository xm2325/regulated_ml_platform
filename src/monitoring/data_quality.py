from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.make_dataset import ACCOUNT_TYPES, EMPLOYMENT
from src.features.build_features import REQUIRED_COLUMNS

RAW_REQUIRED = REQUIRED_COLUMNS - {"support_needed"}
RANGE_RULES = {
    "age": (18, 100), "annual_income": (0, 250000), "cash_balance": (0, 500000),
    "investment_balance": (0, 1000000), "debt_balance": (0, 250000),
    "risk_score": (0, 1), "recent_activity_count": (0, 100),
}


def check_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    missing_columns = sorted(RAW_REQUIRED.difference(frame.columns))
    result: dict[str, Any] = {
        "rows": int(len(frame)), "missing_columns": missing_columns, "missing_rate_by_column": {},
        "range_violations": {}, "category_violations": {}, "date_violations": 0,
        "duplicate_customer_ids": 0, "status": "PASS",
    }
    if missing_columns:
        result["status"] = "FAIL"
        return result
    for column in sorted(RAW_REQUIRED):
        result["missing_rate_by_column"][column] = float(frame[column].isna().mean())
    for column, (low, high) in RANGE_RULES.items():
        bad = frame[column].isna() | ~np.isfinite(frame[column]) | (frame[column] < low) | (frame[column] > high)
        result["range_violations"][column] = int(bad.sum())
    result["category_violations"]["account_type"] = int((~frame["account_type"].isin(ACCOUNT_TYPES)).sum())
    result["category_violations"]["employment_status"] = int((~frame["employment_status"].isin(EMPLOYMENT)).sum())
    result["date_violations"] = int(pd.to_datetime(frame["observation_date"], errors="coerce").isna().sum())
    result["duplicate_customer_ids"] = int(frame["customer_id"].duplicated().sum())
    max_missing = max(result["missing_rate_by_column"].values()) if result["missing_rate_by_column"] else 1.0
    total_violations = sum(result["range_violations"].values()) + sum(result["category_violations"].values()) + result["date_violations"] + result["duplicate_customer_ids"]
    if max_missing > 0.01 or total_violations > 0:
        result["status"] = "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/data_quality_report.json")
    args = parser.parse_args()
    report = check_data_quality(pd.read_csv(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
