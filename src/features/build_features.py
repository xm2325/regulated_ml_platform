from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "customer_id", "observation_date", "age", "annual_income", "cash_balance",
    "investment_balance", "debt_balance", "risk_score", "recent_activity_count",
    "account_type", "employment_status", "support_needed",
}
NUMERIC_FEATURES = [
    "age", "annual_income", "cash_balance", "investment_balance", "debt_balance",
    "risk_score", "recent_activity_count", "accessible_total", "cash_ratio",
    "debt_to_income", "wealth_to_income",
]
CATEGORICAL_FEATURES = ["account_type", "employment_status"]
TARGET = "support_needed"
ID_COLUMN = "customer_id"
TIME_COLUMN = "observation_date"


def validate_input_schema(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")
    if pd.to_datetime(frame[TIME_COLUMN], errors="coerce").isna().any():
        raise ValueError(f"Column {TIME_COLUMN} contains invalid dates")
    for column in ["age", "annual_income", "cash_balance", "investment_balance", "debt_balance", "risk_score", "recent_activity_count"]:
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise ValueError(f"Column {column} contains missing or non-finite values")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    validate_input_schema(frame)
    output = frame.copy()
    output[TIME_COLUMN] = pd.to_datetime(output[TIME_COLUMN]).dt.strftime("%Y-%m-%d")
    output["accessible_total"] = output["cash_balance"] + output["investment_balance"]
    output["cash_ratio"] = output["cash_balance"] / np.maximum(output["accessible_total"], 1.0)
    output["debt_to_income"] = output["debt_balance"] / np.maximum(output["annual_income"], 1.0)
    output["wealth_to_income"] = output["accessible_total"] / np.maximum(output["annual_income"], 1.0)
    ordered = [ID_COLUMN, TIME_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    optional = [column for column in ["true_support_probability"] if column in output.columns]
    return output[ordered + optional].sort_values([TIME_COLUMN, ID_COLUMN]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/customers.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    args = parser.parse_args()
    features = build_features(pd.read_csv(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"Wrote {len(features):,} feature rows to {output}")


if __name__ == "__main__":
    main()
