from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACCOUNT_TYPES = ["current", "savings", "isa", "pension", "investment"]
EMPLOYMENT = ["employed", "self_employed", "student", "retired", "unemployed"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a dated synthetic cohort with a small, explicit temporal shift."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2025-01-01")
    day_offset = rng.integers(0, 365, size=n)
    observation_date = start + day_offset.astype("timedelta64[D]")
    time_fraction = day_offset / 364.0

    age = rng.integers(18, 76, size=n)
    annual_income = rng.lognormal(mean=10.75 + 0.06 * time_fraction, sigma=0.45, size=n).clip(12000, 180000)
    cash_balance = rng.lognormal(mean=9.4 + 0.08 * time_fraction, sigma=0.85, size=n).clip(200, 250000)
    investment_balance = rng.lognormal(mean=9.2, sigma=1.1, size=n).clip(0, 600000)
    debt_balance = rng.lognormal(mean=8.6 + 0.05 * time_fraction, sigma=1.0, size=n).clip(0, 90000)
    risk_score = rng.beta(2.2 + 0.25 * time_fraction, 3.0, size=n)
    recent_activity_count = rng.poisson(lam=4 + 0.5 * time_fraction, size=n)
    account_type = rng.choice(ACCOUNT_TYPES, size=n, p=[0.22, 0.25, 0.23, 0.14, 0.16])
    employment_status = rng.choice(EMPLOYMENT, size=n, p=[0.58, 0.12, 0.06, 0.16, 0.08])

    accessible_total = cash_balance + investment_balance
    cash_ratio = cash_balance / np.maximum(accessible_total, 1.0)
    debt_to_income = debt_balance / np.maximum(annual_income, 1.0)
    raw_score = (
        -2.0
        + (2.25 + 0.45 * time_fraction) * (cash_ratio > 0.55).astype(float)
        + 0.9 * (accessible_total > 25000).astype(float)
        + (0.85 - 0.35 * time_fraction) * (risk_score > 0.35).astype(float)
        + 0.5 * (recent_activity_count >= 3).astype(float)
        + 0.5 * ((age >= 30) & (age <= 65)).astype(float)
        - (1.2 + 0.35 * time_fraction) * (debt_to_income > 0.65).astype(float)
        - 0.5 * (employment_status == "unemployed").astype(float)
        + 0.2 * time_fraction
        + rng.normal(0.0, 0.75, size=n)
    )
    support_probability = sigmoid(raw_score)
    support_needed = rng.binomial(1, support_probability)

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "observation_date": pd.to_datetime(observation_date).strftime("%Y-%m-%d"),
            "age": age,
            "annual_income": annual_income.round(2),
            "cash_balance": cash_balance.round(2),
            "investment_balance": investment_balance.round(2),
            "debt_balance": debt_balance.round(2),
            "risk_score": risk_score.round(4),
            "recent_activity_count": recent_activity_count,
            "account_type": account_type,
            "employment_status": employment_status,
            "support_needed": support_needed,
            "true_support_probability": support_probability.round(4),
        }
    )
    return frame.sort_values(["observation_date", "customer_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", default="data/raw/customers.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_customers(args.n, args.seed)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} dated rows to {output}")


if __name__ == "__main__":
    main()
