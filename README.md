# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)
[![Pages](https://img.shields.io/badge/evidence-dashboard-blue)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style ML system that converts a calibrated model probability into a controlled, reviewable, and traceable decision.**

[Open the evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [Read the release pack](docs/release_approval_pack.md) · [Inspect the decision contract](docs/model_contract.md) · [Read registry runtime design](docs/registry_runtime.md)

## Result first

The engineering platform and API service are version `0.8.0`; the validated model release remains `0.6.0`. Version `0.8.0` changes how a verified model is selected and served. It does not retrain the model or change the reported out-of-time metrics.

| Out-of-time model result | Value |
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
| Automated tests | 49 passed |

The main v0.8 change is a registry-aware serving path:

```text
MLflow champion alias
        ↓
download release bundle
        ↓
SHA-256 + feature schema + promotion gate + smoke prediction
        ↓
last-known-good verified cache
        ↓
atomic in-memory predictor swap
        ↓
FastAPI serves the new registry version without restart
```

A failed download, checksum, schema check, release gate, or smoke prediction never replaces the active predictor.

## Live registry transition proved in CI

The integration test starts PostgreSQL, MinIO, MLflow, and a real API container. The API is kept running while aliases change.

```text
API starts on champion registry version 1
        ↓
register + promote registry version 3
        ↓
background poll detects champion change
        ↓
API hot-reloads version 3 without restart
        ↓
/predict reports registry_model_version = 3
        ↓
registry rollback restores version 1
        ↓
API hot-reloads version 1 without restart
```

The verified CI run recorded:

| Runtime evidence | Result |
|---|---|
| Initial active source | registry |
| Initial champion | registry version 1 |
| Promoted champion | registry version 3 |
| API after promotion | registry version 3 |
| API prediction provenance | registry version 3 |
| Reload successes after promotion | 2 |
| Reload failures | 0 |
| Registry rollback target | registry version 1 |
| API after rollback | registry version 1 |
| API restart required | no |

The registry versions above are CI-generated test versions. The model artifact inside them remains validated model release `0.6.0`.

## Evaluation design

Model selection, probability calibration, policy-threshold selection, and final evaluation use separate chronological windows.

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
| Model selection | 500 | 2025-08-05 to 2025-09-15 | Select the candidate using AUC and Brier score |
| Calibration | 500 | 2025-09-15 to 2025-10-19 | Fit Platt scaling on the selected base model |
| Policy validation | 500 | 2025-10-19 to 2025-11-27 | Select the decision threshold |
| Out-of-time test | 500 | 2025-11-27 to 2025-12-31 | Calculate final reported metrics only |

Calibration reduced the out-of-time Brier score from `0.1814` to `0.1795` and expected calibration error from `0.0832` to `0.0748`. AUC remained `0.7884`, as expected for a monotonic probability transformation.

## Model, service, and platform versions are separate

```text
platform_version      = 0.8.0   # lifecycle, CI, registry, deployment system
service_version       = 0.8.0   # FastAPI runtime contract
model_release_version = 0.6.0   # validated calibrated model artifact
registry_model_version          # immutable MLflow registry version, e.g. 1 or 3
policy_version         = targeted-support-policy-v3
feature_schema_version = financial_customer_features_v4
```

`GET /version`, prediction responses, audit events, and Prometheus metrics expose enough provenance to distinguish these layers.

## One request, step by step

```text
validated request
      │
      ▼
feature construction ── feature_schema_version
      │
      ▼
verified active predictor ── model_release_version + registry_model_version
      │
      ▼
calibrated probability
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

## Registry runtime safety rules

Before a registry candidate can replace the active predictor, the runtime checks:

1. all required files exist: model, metadata, metrics, and promotion gate;
2. each required file matches its SHA-256 evidence;
3. `feature_schema_version` exactly matches the service contract;
4. the promotion gate is `PASS` and `eligible_for_controlled_release`;
5. the model can be deserialized by the running service;
6. a fixed smoke request produces a finite probability in `[0, 1]`;
7. only after all checks pass is the in-memory predictor reference replaced.

Fallback order in registry mode:

```text
current verified registry predictor
        ↓ if process restarts
last verified cached registry bundle
        ↓ if no valid cache exists
packaged local model
```

`REGISTRY_STRICT_STARTUP=true` can block startup when no verified registry model can be loaded. With strict startup disabled, a usable fallback remains ready but `/health`, `/ready`, `/version`, and `/runtime/model` report the degraded state.

## Platform map

| Layer | Implementation | Evidence |
|---|---|---|
| Dated data contract | Synthetic cohort, date validation, duplicate-ID check | `src/data/`, `src/features/`, `reports/data_quality_report.json` |
| Model lifecycle | Candidate fitting, temporal selection, calibration, out-of-time evaluation | `src/models/train.py`, `src/models/calibration.py` |
| Registry control plane | MLflow aliases, promotion gate, rollback | `src/registry/`, `docker-compose.registry.yml` |
| Registry serving data plane | verified bundle cache, hot reload, fallback, runtime provenance | `src/serving/runtime_manager.py`, `docs/registry_runtime.md` |
| Decision control | Separate threshold, policy version, hard safety gate | `src/serving/policy.py`, `config/policy.yaml` |
| Serving | FastAPI, OpenAPI, readiness, request IDs | `src/serving/app.py`, `docs/openapi.json` |
| Review and audit | Manual-review rules and redacted audit events | `src/serving/review_workflow.py`, `src/core/audit.py` |
| Monitoring | Drift, validity, latency, decisions, registry state and reload metrics | `src/monitoring/`, `/metrics`, `reports/` |
| Release control | Promotion gate, model card, SBOM, artifact checksums | `src/governance/`, `reports/sbom.cdx.json` |
| Deployment | non-root container, Kubernetes, Helm, HPA, PDB, NetworkPolicy | `docker/`, `k8s/`, `helm/` |
| CI/CD | evidence, exact tested image, kind/Helm, live registry transition, separate publication | `.github/workflows/platform.yml`, `.github/workflows/release.yml` |

## Reproduce the model evidence

```bash
git clone https://github.com/xm2325/regulated_ml_platform.git
cd regulated_ml_platform

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

make evidence
make lint
make security
make audit
make audit-registry
make audit-registry-client
```

## Run the registry lifecycle locally

```bash
make registry-up
make registry-register
make registry-promote
make registry-status
make registry-verify
```

The local stack uses MLflow `3.14.0`, PostgreSQL for tracking and registry metadata, and MinIO for model artifacts.

```text
register → challenger
PASS gate → previous champion becomes rollback → challenger becomes champion
incident rollback → rollback becomes champion → failed champion becomes challenger
```

Run the full registry and live API transition test with:

```bash
make registry-smoke
```

See `docs/model_registry.md` for registry commands and `docs/registry_runtime.md` for serving behaviour.

## Run the API

Local packaged-model mode:

```bash
make serve
```

Registry-driven mode requires an accessible MLflow registry and can be configured with:

```text
MODEL_SOURCE=registry
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_REGISTRY_URI=http://mlflow:5000
MLFLOW_REGISTERED_MODEL_NAME=regulated-targeted-support-model
MLFLOW_CHAMPION_ALIAS=champion
REGISTRY_STRICT_STARTUP=true
REGISTRY_HOT_RELOAD_ENABLED=true
REGISTRY_RELOAD_INTERVAL_SECONDS=30
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health plus active model source and degraded state |
| `GET /ready` | Readiness without hiding fallback state |
| `GET /version` | Platform, service, model, policy, schema, registry alias/version/run ID |
| `GET /runtime/model` | Runtime source, reload state and counters; error text is not exposed |
| `GET /decision-contract` | Machine-readable decision semantics |
| `POST /predict` | Probability, action, review route, audit IDs, and model provenance |
| `POST /explain` | Model reason codes and policy reasons |
| `POST /review-route` | Human-review routing result |
| `POST /shadow-predict` | Candidate comparison without changing the served action |
| `GET /metrics` | Prometheus service and registry-runtime metrics |

## Helm registry mode

The Helm chart defaults to packaged local-model mode. Registry mode is opt-in:

```bash
helm upgrade --install regulated-ai helm/regulated-ai \
  --set registryRuntime.enabled=true \
  --set registryRuntime.trackingUri=http://mlflow.mlops.svc.cluster.local:5000 \
  --set registryRuntime.registryUri=http://mlflow.mlops.svc.cluster.local:5000 \
  --set networkPolicy.mlflowEgress.enabled=true
```

When registry mode is enabled, the chart mounts a writable `emptyDir` at `/var/lib/regulated-ai/registry-cache` while the container root filesystem remains read-only and the process remains UID `10001`. `fsGroup=10001` gives the non-root process access to the cache volume.

The example NetworkPolicy allows MLflow egress only when explicitly enabled. Production authentication, TLS, secrets, backup, object retention, and highly available registry services remain deployment responsibilities outside this demo.

## GitHub pipeline

Required validation and publication are separate:

```text
Platform workflow
  evidence
    ├── model-artifacts
    ├── release-evidence
    └── generated-site
          │
          ├── Docker live test → kind/Helm deployment test
          └── PostgreSQL + MinIO + MLflow + live API hot-reload/rollback test

successful main push only
          ↓
Release workflow
  ├── publish the exact tested image to GHCR
  └── deploy the generated evidence site to GitHub Pages
```

A Pages or GHCR account-setting error therefore does not change the result of the required engineering validation.

## Boundary

The dataset, target, drift, and consumer actions are synthetic. This repository demonstrates ML engineering, chronological evaluation, probability calibration, decision control, registry-driven serving, monitoring, deployment testing, and release evidence. It is not a validated financial-advice system and must not be used for real customer decisions.
