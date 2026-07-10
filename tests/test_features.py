from src.data.make_dataset import generate_customers
from src.features.build_features import build_features


def test_build_features_has_expected_columns():
    features = build_features(generate_customers(n=50, seed=1))
    expected = {"customer_id", "age", "annual_income", "cash_balance", "investment_balance", "debt_balance", "risk_score", "recent_activity_count", "accessible_total", "cash_ratio", "debt_to_income", "wealth_to_income", "account_type", "employment_status", "support_needed"}
    assert expected.issubset(set(features.columns))
    assert features["cash_ratio"].between(0, 1).all()


def test_missing_required_column_raises():
    raw = generate_customers(n=10, seed=2).drop(columns=["age"])
    try:
        build_features(raw)
    except ValueError as exc:
        assert "age" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
