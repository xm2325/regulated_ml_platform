# Regulated AI MLOps Platform

[![CI](https://github.com/xm2325/regulated_ml_platform/actions/workflows/ci.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/ci.yml)
[![Docker](https://github.com/xm2325/regulated_ml_platform/actions/workflows/docker-build.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/docker-build.yml)
[![Pages](https://github.com/xm2325/regulated_ml_platform/actions/workflows/pages.yml/badge.svg)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style ML platform that turns a model score into a controlled, reviewable, and traceable decision.**

[Open the evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [Read the release pack](docs/release_approval_pack.md) · [Inspect the model contract](docs/model_contract.md)

## The result first

The current synthetic-data run selects a random-forest champion and evaluates it on an independent test split. Model selection and threshold choice are completed on validation data before the test result is calculated.

| Result | Value |
|---|---:|
| AUC | 0.785 |
| Brier score | 0.189 |
| Policy precision | 0.822 |
| High-confidence precision | 0.879 |
| Automated promotion status | PASS |
| API p95 latency | 106.1 ms |
| Automated tests | 30 passed |

Run `make all` to reproduce the exact values and rebuild the dashboard.

## What the project proves

The main result is not one accuracy number. The repository shows how an ML system can connect six production concerns:

1. **Independent evaluation** — train/validation/test separation, validation-selected threshold, bootstrap confidence intervals, calibration, and segment checks.
2. **Decision control** — the model returns a probability; a separate versioned policy maps the score and safety gates to an action.
3. **Human review** — near-threshold, low-confidence, debt-pressure, older-customer, and high-value cases can be routed to manual review.
4. **Traceability** — every response carries model, policy, feature-schema, decision, and audit identifiers.
5. **Operational evidence** — drift, load, SLO, incident, privacy, data-quality, champion–challenger, and reproducibility reports.
6. **Secure deployment** — non-root container, read-only filesystem, Kubernetes probes, resource controls, HPA, PodDisruptionBudget, NetworkPolicy, Helm, blue–green, canary, KServe, and shadow scoring examples.

## One request, step by step

```text
validated request
      │
      ▼
feature construction ── feature_schema_version
      │
      ▼
champion model score ── model_version
      │
      ▼
versioned policy + hard safety gate ── policy_version
      │
      ├── auto_serve
      └── manual_review
      │
      ▼
decision_id + audit_event_id + Prometheus metrics + structured log
```

The model does not directly issue the final action. `src/serving/policy.py` owns the deterministic action policy, while `src/serving/review_workflow.py` owns review routing.

## Current platform layers

| Layer | Main implementation | Evidence |
|---|---|---|
| Data contract | Pydantic request schema, feature validation, data-quality gate | `src/serving/schemas.py`, `reports/data_quality_report.json` |
| Model lifecycle | MLflow-compatible tracking, validation selection, test evaluation | `src/models/train.py`, `reports/model_evaluation.md` |
| Decision policy | Separate policy version, threshold, hard safety gate | `src/serving/policy.py`, `config/policy.yaml` |
| Serving | FastAPI, OpenAPI, health/readiness, request IDs | `src/serving/app.py`, `docs/openapi.json` |
| Review and audit | Manual-review rules, redacted structured audit event | `src/serving/review_workflow.py`, `src/core/audit.py` |
| Monitoring | Prometheus, drift, data quality, latency, decision counts | `monitoring/`, `reports/` |
| Release control | Promotion gate, release pack, artifact checksums | `src/governance/`, `docs/release_approval_pack.md` |
| Deployment | Docker, Compose, Kubernetes, Helm, KServe | `docker/`, `k8s/`, `helm/`, `kserve/` |
| Presentation | Self-contained evidence dashboard and GitHub Pages workflow | `site/index.html`, `.github/workflows/pages.yml` |

## Reproduce the full platform

```bash
git clone https://github.com/xm2325/regulated_ml_platform.git
cd regulated_ml_platform

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make all
```

`make all` performs this sequence:

```text
generate synthetic data
→ build features
→ train and select models on validation data
→ freeze the threshold
→ evaluate once on the test split
→ generate governance and monitoring evidence
→ validate Kubernetes controls
→ build the release pack and artifact manifest
→ build the evidence dashboard
→ run the test suite
```

## Run the API

```bash
make serve
```

Useful endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and model health |
| `GET /ready` | Kubernetes readiness |
| `GET /version` | Service, model, policy, and schema versions |
| `GET /decision-contract` | Machine-readable decision semantics |
| `POST /predict` | Champion score, policy action, review route, and audit IDs |
| `POST /explain` | Model reason codes plus policy reasons |
| `POST /review-route` | Manual-review routing result |
| `POST /shadow-predict` | Champion–challenger comparison without changing the served action |
| `GET /metrics` | Prometheus metrics |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-request-0001' \
  -d @examples/review_request.json
```

The response separates model evidence from policy evidence:

```json
{
  "decision_id": "decision_...",
  "support_probability": 0.73,
  "recommended_action": "investment_support",
  "reason_codes": ["high_cash_ratio", "sufficient_accessible_assets"],
  "policy_reasons": ["high_cash_ratio", "minimum_accessible_assets_met"],
  "model_version": "0.5.0",
  "policy_version": "targeted-support-policy-v2",
  "feature_schema_version": "financial_customer_features_v3",
  "review_route": "auto_serve",
  "audit_event_id": "audit_..."
}
```

## Review the evidence in the right order

Start with the decision, then inspect the supporting layers:

1. `docs/release_approval_pack.md` — overall release decision and control status.
2. `reports/promotion_gate.md` — exact pass/review checks.
3. `reports/model_evaluation.md` — independent test metrics and uncertainty.
4. `reports/calibration_report.md` and `reports/fairness_report.md` — probability and segment behaviour.
5. `reports/drift_report.html`, `reports/load_test_report.md`, and `reports/incident_drill_report.md` — operational evidence.
6. `docs/model_contract.md` and `docs/reproducibility_manifest.md` — interface and artifact traceability.
7. `site/index.html` — a single-page summary generated from the committed reports.

## CI/CD and release checks

The GitHub workflows perform:

- unit, API, contract, governance, and deployment tests;
- lint and high-severity static security checks;
- synthetic artifact generation;
- Kubernetes security validation;
- Docker image build;
- evidence artifact upload;
- GitHub Pages publication.

## Repository map

```text
src/
  core/          configuration, structured logging, redacted audit events
  data/          synthetic dataset generation
  features/      feature contract and transformations
  models/        validation-based model selection and test evaluation
  serving/       API, policy, review routing, batch and shadow scoring
  monitoring/    data quality and drift
  governance/    promotion, privacy, contract, explainability, manifests
  operations/    load test, incident drill, deployment validation, site build
config/          versioned policy and promotion thresholds
docs/            model, data, architecture, runbook, SLO, rollback, release pack
reports/         committed machine-readable and human-readable evidence
site/            self-contained evidence dashboard
k8s/             production-style Kubernetes examples
helm/            parameterised deployment chart
.github/         CI, container, CodeQL, and Pages workflows
```

## Boundary

The dataset, outcome, and consumer action are synthetic. This repository demonstrates ML engineering, release control, monitoring, and governance design. It is not a validated financial-advice system and must not be used for real customer decisions.
