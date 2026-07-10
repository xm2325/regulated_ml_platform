# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)
[![Pages](https://img.shields.io/badge/evidence-dashboard-blue)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style ML system that converts a calibrated model probability into a controlled, reviewable, and traceable decision.**

[Open the evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [Read the release pack](docs/release_approval_pack.md) · [Inspect the decision contract](docs/model_contract.md)

## Result first

Version `0.6.0` now includes a runnable model registry in addition to the validated model evidence. For the dated synthetic cohort, candidate selection, probability calibration, policy-threshold selection, and final evaluation use separate chronological windows. The final result comes from the latest untouched window.

| Out-of-time result | Value |
|---|---:|
| AUC | 0.788 |
| AUC bootstrap 95% interval | 0.747–0.823 |
| Brier score | 0.180 |
| Expected calibration error | 0.075 |
| Policy precision | 0.875 |
| Policy recall | 0.710 |
| High-confidence precision | 0.883 |
| Policy threshold | 0.70 |
| API p95 latency | 94.8 ms |
| Automated promotion status | PASS |
| Automated tests | 40 passed |

The release decision is not based on AUC alone. Data quality, probability calibration, segment behaviour, privacy, drift, load, deployment configuration, incident response, reproducibility, and rollback evidence are checked before the release pack is marked `PASS`.

## What version 0.6.0 includes

| Change | Why it matters |
|---|---|
| Chronological five-window evaluation | Stops later observations from entering model fitting or policy selection |
| Dedicated Platt calibration window | Separates probability calibration from model selection and threshold choice |
| Out-of-time test | Measures behaviour on the latest synthetic cohort |
| Stronger group diagnostics | Reports precision, recall, false-positive rate, calibration error, and evidence sufficiency |
| One artifact-driven GitHub workflow | Generates data, models, reports, container input, and website once per commit |
| Docker and kind/Helm integration test | Checks the same image first as a container and then inside Kubernetes |
| CycloneDX SBOM and dependency audit | Records runtime packages and checks known dependency issues in CI |
| Dependabot for Python and Actions | Creates scheduled dependency update pull requests |
| Direct source tree | No bootstrap archive or self-modifying workflow is required |
| MLflow model registry | Tracks runs and model versions with runnable champion, challenger, and rollback aliases |
| PostgreSQL backend | Stores experiments, runs, registry versions, tags, and aliases outside the MLflow process |
| MinIO artifact store | Stores versioned models and release evidence through the MLflow artifact proxy |
| Controlled registry transition | Blocks promotion unless the promotion gate is PASS and records rollback provenance |

## Evaluation design

```text
2025-01-01                                                     2025-12-31
│                                                                     │
├──────── train ────────┤ selection ┤ calibration ┤ policy ┤ OOT test ┤
│       3,000 rows      │ 500 rows  │  500 rows  │500 rows│ 500 rows │
│                       │           │            │        │          │
fit preprocessing       choose      fit Platt    freeze   report only
and candidate models    champion    scaling      threshold
```

| Window | Rows | Date range | Purpose |
|---|---:|---|---|
| Train | 3,000 | 2025-01-01 to 2025-08-05 | Fit preprocessing and candidate models |
| Model selection | 500 | 2025-08-05 to 2025-09-15 | Select the champion using AUC and Brier score |
| Calibration | 500 | 2025-09-15 to 2025-10-19 | Fit Platt scaling on the selected base model |
| Policy validation | 500 | 2025-10-19 to 2025-11-27 | Select the decision threshold |
| Out-of-time test | 500 | 2025-11-27 to 2025-12-31 | Calculate the final reported metrics |

Calibration reduced the out-of-time Brier score from `0.1814` to `0.1795` and expected calibration error from `0.0832` to `0.0748`. AUC remained `0.7884`, as expected for a monotonic probability transformation.

## One request, step by step

```text
validated request
      │
      ▼
feature construction ── feature_schema_version
      │
      ▼
calibrated champion probability ── model_version
      │
      ▼
versioned policy + hard safety gate ── policy_version
      │
      ├── auto_serve
      └── manual_review
      │
      ▼
decision_id + audit_event_id + Prometheus metrics + redacted log
```

The model does not directly issue the final action. `src/serving/policy.py` maps the probability and explicit safety conditions to an action. `src/serving/review_workflow.py` independently decides whether the case can be served automatically or needs human review.

## Platform map

| Layer | Implementation | Evidence |
|---|---|---|
| Dated data contract | Synthetic cohort, date validation, duplicate-ID check | `src/data/`, `src/features/`, `reports/data_quality_report.json` |
| Model lifecycle | Candidate fitting, temporal selection, calibration, out-of-time evaluation | `src/models/train.py`, `src/models/calibration.py` |
| Model registry | MLflow runs, model versions, aliases, promotion gate, rollback and atomic sync | `src/registry/`, `docker-compose.registry.yml` |
| Decision control | Separate threshold, policy version, hard safety gate | `src/serving/policy.py`, `config/policy.yaml` |
| Serving | FastAPI, OpenAPI, readiness, request IDs | `src/serving/app.py`, `docs/openapi.json` |
| Review and audit | Manual-review rules and redacted audit events | `src/serving/review_workflow.py`, `src/core/audit.py` |
| Monitoring | Early-vs-late drift, validity, latency, prediction and review counts | `src/monitoring/`, `reports/` |
| Release control | Promotion gate, model card, SBOM, artifact checksums | `src/governance/`, `reports/sbom.cdx.json` |
| Deployment | Secure container, Kubernetes, Helm, HPA, PDB, NetworkPolicy | `docker/`, `k8s/`, `helm/` |
| CI/CD | One evidence job, tested image, kind/Helm test, Pages artifact reuse | `.github/workflows/platform.yml` |

## Reproduce the evidence

```bash
git clone https://github.com/xm2325/regulated_ml_platform.git
cd regulated_ml_platform

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

make evidence
make lint
make security
```

`make evidence` performs:

```text
generate dated synthetic data
→ build point-in-time features
→ fit candidate models
→ select champion on a later window
→ fit Platt calibration on a separate window
→ select policy threshold on another window
→ evaluate on the latest untouched window
→ generate calibration, segment, drift, privacy and release evidence
→ run batch and API load tests
→ validate deployment controls
→ generate CycloneDX SBOM and artifact manifest
→ build the evidence dashboard
→ run the full test suite
```

`make audit` checks the API runtime lock and `make audit-registry` checks the MLflow, PostgreSQL-driver, and S3-client lock. Install the registry client with `pip install -r requirements-mlflow.txt`.

## Run the model registry

```bash
make registry-up
make registry-register
make registry-promote
make registry-status
make registry-verify
```

The local stack uses MLflow `3.14.0`, PostgreSQL for tracking and registry metadata, and MinIO for model artifacts. A newly registered release receives the `challenger` alias. Promotion is blocked unless `reports/promotion_gate.json` is `PASS`; the previous champion is moved to `rollback` before the challenger becomes champion.

```text
register → challenger
PASS gate → previous champion becomes rollback → challenger becomes champion
incident rollback → rollback becomes champion → failed champion becomes challenger
```

`make registry-smoke` runs two registrations, two controlled promotions, one rollback, one alias-status check, one registry model download, and one real probability prediction. See `docs/model_registry.md` for the command contract and production boundary.

## Run the API

```bash
make serve
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and model health |
| `GET /ready` | Kubernetes readiness |
| `GET /version` | Service, model, policy, and feature-schema versions |
| `GET /decision-contract` | Machine-readable decision semantics |
| `POST /predict` | Calibrated probability, policy action, review route, and audit IDs |
| `POST /explain` | Model reason codes and policy reasons |
| `POST /review-route` | Human-review routing result |
| `POST /shadow-predict` | Champion–challenger comparison without changing the served action |
| `GET /metrics` | Prometheus metrics |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-request-0001' \
  -d @examples/review_request.json
```

## GitHub pipeline

The main workflow creates one evidence set and passes it to later jobs:

```text
evidence job
  ├── model-artifacts
  ├── release-evidence
  └── generated-site
          │
          ├── container smoke test → kind/Helm deployment test → GHCR
          ├── PostgreSQL + MinIO + MLflow registry lifecycle test
          └── GitHub Pages deployment
```

The container job builds one image, tests that image locally, loads the same image into kind, installs the Helm chart, calls `/ready`, `/version`, and `/predict`, and only then publishes tags on a `main` push.

## Read the evidence in this order

1. `docs/release_approval_pack.md` — overall decision and control status.
2. `reports/model_evaluation.md` — chronological windows, out-of-time metrics, and uncertainty.
3. `reports/calibration_report.md` — raw versus calibrated probability quality.
4. `reports/promotion_gate.md` — exact automated checks.
5. `reports/fairness_report.md` — group metrics and insufficient-evidence flags.
6. `reports/drift_report.html` — early reference window versus latest monitoring window.
7. `reports/deployment_validation.md` — container and Kubernetes configuration checks.
8. `reports/sbom.cdx.json` — direct runtime dependency inventory.
9. `site/index.html` — generated one-page evidence summary.

## Boundary

The dataset, target, drift, and consumer actions are synthetic. This repository demonstrates ML engineering, chronological evaluation, probability calibration, decision control, monitoring, deployment testing, and release evidence. It is not a validated financial-advice system and must not be used for real customer decisions.
