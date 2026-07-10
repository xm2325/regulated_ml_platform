from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.core.config import settings
from src.serving.schemas import PredictionRequest


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    policy_version: str
    policy_reasons: list[str]
    hard_safety_gate_triggered: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TargetedSupportPolicy:
    """Separate model score generation from business and safety policy decisions."""

    def __init__(self, version: str = settings.policy_version) -> None:
        self.version = version

    def decide(self, request: PredictionRequest, probability: float, threshold: float) -> PolicyDecision:
        accessible_total = request.cash_balance + request.investment_balance
        cash_ratio = request.cash_balance / max(accessible_total, 1.0)
        debt_to_income = request.debt_balance / max(request.annual_income, 1.0)

        if debt_to_income > 0.85:
            return PolicyDecision(
                action="risk_review",
                policy_version=self.version,
                policy_reasons=["hard_debt_pressure_gate"],
                hard_safety_gate_triggered=True,
            )
        if probability < threshold:
            return PolicyDecision(
                action="no_support",
                policy_version=self.version,
                policy_reasons=["score_below_policy_threshold"],
                hard_safety_gate_triggered=False,
            )
        if debt_to_income > 0.65:
            return PolicyDecision(
                action="risk_review",
                policy_version=self.version,
                policy_reasons=["debt_pressure_review"],
                hard_safety_gate_triggered=False,
            )
        if cash_ratio > 0.55 and accessible_total >= 10000:
            return PolicyDecision(
                action="investment_support",
                policy_version=self.version,
                policy_reasons=["high_cash_ratio", "minimum_accessible_assets_met"],
                hard_safety_gate_triggered=False,
            )
        return PolicyDecision(
            action="cash_buffer_warning",
            policy_version=self.version,
            policy_reasons=["support_score_above_threshold", "investment_gate_not_met"],
            hard_safety_gate_triggered=False,
        )
