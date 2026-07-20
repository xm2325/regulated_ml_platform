from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AccountType = Literal["current", "savings", "isa", "pension", "investment"]
EmploymentStatus = Literal["employed", "self_employed", "student", "retired", "unemployed"]
ReviewRoute = Literal["auto_serve", "manual_review"]
ShadowStatus = Literal["candidate_available", "candidate_unavailable"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1, description="Pseudonymous customer key supplied by the calling system.")
    request_id: str | None = Field(default=None, min_length=8, max_length=128)
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0, le=10_000_000)
    cash_balance: float = Field(..., ge=0, le=100_000_000)
    investment_balance: float = Field(..., ge=0, le=100_000_000)
    debt_balance: float = Field(..., ge=0, le=100_000_000)
    risk_score: float = Field(..., ge=0, le=1)
    recent_activity_count: int = Field(..., ge=0, le=10_000)
    account_type: AccountType
    employment_status: EmploymentStatus


class PredictionResponse(BaseModel):
    customer_id: str
    decision_id: str
    support_probability: float
    recommended_action: str
    confidence: str
    reason_codes: list[str]
    policy_reasons: list[str]
    hard_safety_gate_triggered: bool
    model_version: str
    policy_version: str
    feature_schema_version: str
    policy_threshold: float
    review_route: ReviewRoute
    review_reasons: list[str]
    audit_event_id: str
    model_source: Literal["local", "registry"] = "local"
    runtime_state: str = "ready_local"
    registry_model_name: str | None = None
    registry_alias: str | None = None
    registry_model_version: str | None = None


class ExplainResponse(BaseModel):
    customer_id: str
    decision_id: str
    reason_codes: list[str]
    policy_reasons: list[str]
    explanation: str
    model_version: str
    policy_version: str
    audit_event_id: str
    model_source: Literal["local", "registry"] = "local"
    runtime_state: str = "ready_local"
    registry_model_version: str | None = None


class ShadowPredictionResponse(BaseModel):
    customer_id: str
    champion_model_version: str
    candidate_model_name: str
    status: ShadowStatus
    champion_probability: float
    candidate_probability: float | None
    probability_delta: float | None
    champion_action: str
    candidate_action: str | None
    action_changed: bool
    audit_event_id: str


class ReviewRouteResponse(BaseModel):
    customer_id: str
    review_route: ReviewRoute
    review_reasons: list[str]
    reviewer_notes: str
    model_version: str
    audit_event_id: str
