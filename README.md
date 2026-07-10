# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)

**A production-style ML system that converts a calibrated model probability into a controlled, reviewable, and traceable decision.**

## Result first

Version `0.6.0` uses a dated synthetic cohort. Candidate selection, probability calibration, policy-threshold selection, and final evaluation use separate chronological windows.

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
| Automated tests | 34 passed |

## Evaluation design

```text
train → model selection → calibration → policy validation → out-of-time test
3000       500              500             500                 500 rows
```

| Window | Date range | Purpose |
|---|---|---|
| Train | 2025-01-01 to 2025-08-05 | Fit preprocessing and candidate models |
| Model selection | 2025-08-05 to 2025-09-15 | Select the champion |
| Calibration | 2025-09-15 to 2025-10-19 | Fit Platt scaling |
| Policy validation | 2025-10-19 to 2025-11-27 | Select the decision threshold |
| Out-of-time test | 2025-11-27 to 2025-12-31 | Calculate final metrics |

Calibration reduced Brier score from `0.1814` to `0.1795` and expected calibration error from `0.0832` to `0.0748`.

## What version 0.6.0 adds

- direct source files, with no bootstrap archive required;
- chronological five-window evaluation;
- a dedicated Platt calibration stage;
- stronger group diagnostics with insufficient-evidence flags;
- one artifact-driven GitHub workflow;
- a Docker smoke test followed by a kind/Helm deployment test;
- a CycloneDX SBOM, `pip-audit`, CodeQL, and Dependabot.

## Decision path

```text
validated request
→ feature construction
→ calibrated model probability
→ versioned policy and hard safety gate
→ auto_serve or manual_review
→ decision_id, audit_event_id, metrics, and redacted log
```

The model does not directly issue the final action. `src/serving/policy.py` maps the probability and safety conditions to an action. `src/serving/review_workflow.py` independently assigns the review route.

## Reproduce

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

The optional MLflow integration can be installed with `pip install -r requirements-mlflow.txt`.

## GitHub pipeline

```text
evidence job
  ├── model-artifacts ──→ container smoke test ──→ kind/Helm test ──→ GHCR
  ├── release-evidence
  └── generated-site ──────────────────────────────────────────────→ Pages
```

Training and evidence generation run once per commit. Docker and Pages reuse the resulting artifacts.

## Evidence

1. `docs/release_approval_pack.md`
2. `reports/model_evaluation.md`
3. `reports/calibration_report.md`
4. `reports/promotion_gate.md`
5. `reports/fairness_report.md`
6. `reports/drift_report.html`
7. `reports/deployment_validation.md`
8. `reports/sbom.cdx.json`
9. `site/index.html`

## Boundary

The dataset, target, time shift, and consumer actions are synthetic. This repository demonstrates ML engineering, chronological evaluation, probability calibration, decision control, monitoring, deployment testing, and release evidence. It is not a validated financial-advice system.