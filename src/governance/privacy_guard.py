from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DIRECT_IDENTIFIER_COLUMNS = {"name", "full_name", "email", "phone", "address", "postcode", "national_insurance_number", "date_of_birth"}
ALLOWED_DIRECT_IDS = {"customer_id"}


def inspect_columns(frame: pd.DataFrame) -> dict[str, Any]:
    columns = set(frame.columns)
    hits = sorted(columns.intersection(DIRECT_IDENTIFIER_COLUMNS))
    quasi = [column for column in ["age", "annual_income", "employment_status"] if column in columns]
    unique_ratio = float(frame["customer_id"].nunique() / len(frame)) if "customer_id" in frame.columns and len(frame) else None
    return {"status": "PASS" if not hits else "FAIL", "rows": int(len(frame)), "columns": sorted(frame.columns.tolist()), "allowed_direct_ids": sorted(ALLOWED_DIRECT_IDS.intersection(columns)), "blocked_direct_identifier_hits": hits, "quasi_identifiers_present": quasi, "customer_id_unique_ratio": unique_ratio, "data_minimisation_note": "Synthetic data retains customer_id for traceability and excludes direct personal identifiers."}


def write_markdown(report: dict[str, Any], output: Path) -> None:
    output.write_text("\n".join(["# Privacy and data-minimisation report", "", f"Status: `{report['status']}`", f"Rows checked: `{report['rows']}`", "", f"Allowed audit identifiers: {', '.join(report['allowed_direct_ids']) or 'none'}", f"Blocked identifier hits: {', '.join(report['blocked_direct_identifier_hits']) or 'none'}", f"Quasi-identifiers: {', '.join(report['quasi_identifiers_present']) or 'none'}", "", report["data_minimisation_note"]]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/customers.csv")
    parser.add_argument("--output-json", default="reports/privacy_report.json")
    parser.add_argument("--output-md", default="reports/privacy_report.md")
    args = parser.parse_args()
    report = inspect_columns(pd.read_csv(args.input))
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
