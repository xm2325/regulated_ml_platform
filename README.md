# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)
[![Pages](https://img.shields.io/badge/evidence-dashboard-blue)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style ML system that turns a calibrated model probability into a controlled, reviewable, traceable decision and manages model changes through verified registry, canary, promotion, and rollback paths.**

[Open the evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [Read the release pack](docs/release_approval_pack.md) · [Registry runtime](docs/registry_runtime.md) · [Canary runtime](docs/canary_runtime.md)

## Result first

The engineering platform and API service are version `0.9.0`; the validated calibrated model release remains `0.6.0`. Version `0.9.0` changes the online model-transition control path. It does not retrain the validated model or change its reported out-of-time metrics.

| Validated model result | Value |
|---|---:|
| AUC | 0.788 |
| AUC bootstrap 95% interval | 0.747–0.823 |
| Brier score | 0.180 |
| Expected calibration error | 0.075 |
| Policy precision | 0.875 |
| Policy recall | 0.710 |
| High-confidence precision | 0.883 |
| Policy threshold | 0.70 |
| API p95 latency from release evidence | 94.8 ms |
| Offline promotion gate | PASS |
| Automated tests | 53 passed |

The v0.9 online transition path is:

```text
validated challenger
        ↓
MLflow challenger alias
        ↓
checksum + schema + offline gate verification
        ↓
stable customer-level canary cohort
        ↓
champion + challenger score the same request
        ↓
configured percentage receives challenger result
        ↓
online safety evidence
        ├── WAIT: not enough evidence
        ├── STOP: a safety limit failed
        └── PASS: configured limits passed
                    ↓
             optional promotion
                    ↓
             challenger → champion
                    ↓
             incident rollback
                    ↓
             former champion restored
```

`CANARY_AUTO_PROMOTE_ENABLED` is `false` by default. The CI integration stack enables automatic promotion only to prove the complete state transition.

## What v0.9 adds

### Stable canary cohorts

Canary routing is deterministic per pseudonymous customer and challenger registry version:

```text
canary seed + challenger registry version + pseudonymous customer_id
        ↓
SHA-256 bucket
        ↓
champion or challenger cohort
```

A customer does not switch arms merely because a new request ID is created. A new challenger registry version can form a new cohort.

### Dual scoring with explicit served-model provenance

During `warming` and `healthy` states, both champion and challenger score the same validated request. Only the configured canary cohort receives the challenger result.

Each prediction can report:

```text
served_model_role
canary_assignment
model_release_version
registry_model_version
comparison_champion_registry_version
comparison_challenger_registry_version
policy_version
feature_schema_version
decision_id
audit_event_id
```

If challenger scoring fails for an assigned request, the API serves the champion and marks `served_model_role=champion_fallback`.

### Online safety gate

The canary controller tracks a bounded evidence window and checks:

| Online metric | Reason for checking it |
|---|---|
| minimum comparisons | avoids promotion on too little evidence |
| minimum challenger-served requests | proves real canary exposure occurred |
| action disagreement rate | detects downstream decision changes |
| p95 absolute probability delta | detects large score changes |
| challenger error rate | detects runtime failures |
| challenger/champion latency ratio | detects operational regression |
| manual-review-rate increase | detects extra human review burden |

These are online safety and behavioural checks. They do not prove that a challenger has better long-term outcomes than the champion. Delayed labels, business outcomes, causal analysis, fairness review, and human approval can still be required before a real production release.

### STOP, PROMOTE, and ROLLBACK states

```text
waiting_for_challenger
        ↓
      warming
      /     \
   STOP    enough evidence
    ↓          ↓
stopped     healthy
                ↓ optional controlled promotion
             promoted
                ↓ incident rollback
             rolled_back
```

A failed online gate removes the challenger from served traffic. A successful gate can remain `healthy` for manual approval or, when explicitly enabled, run the existing controlled MLflow promotion. A later registry rollback restores the former champion without restarting the API, and the public canary state reports `rolled_back` once the active champion no longer matches the promoted canary version.

## CI proves the real control path

The registry integration job starts real PostgreSQL, MinIO, MLflow, and FastAPI containers and executes:

```text
register A
→ promote A to champion
→ register B as challenger
→ start registry-backed API
→ verify A and B bundles
→ send 24 deterministic pseudonymous customers through /predict
→ verify both champion and challenger cohorts receive traffic
→ collect online comparison evidence
→ wait for automatic controlled promotion of B
→ verify API hot-reloads B as champion
→ run incident rollback
→ verify API hot-reloads A again
→ verify the API container ID never changed
→ verify canary state reports rolled_back
```

For this lifecycle test, registry versions A and B intentionally contain the same validated `0.6.0` model artifact. Zero prediction/action disagreement is therefore expected. The test verifies routing, provenance, online evidence, alias transitions, hot reload, and rollback; it does not claim B is a better model.

A separate unit test injects a deliberately divergent challenger and verifies that the online gate returns `STOP` and later requests remain on the champion.

## Model, service, platform, and registry versions are separate

```text
platform_version       = 0.9.0
service_version        = 0.9.0
model_release_version  = 0.6.0
registry_model_version = immutable MLflow integer, e.g. 1 or 2
policy_version         = targeted-support-policy-v3
feature_schema_version = financial_customer_features_v4
```

This separation prevents a registry integer from being mistaken for a validated model release version.

## Evaluation design

Model selection, probability calibration, policy-threshold selection, and final evaluation use separate chronological windows.

```text
2025-01-01                                                     2025-12-31
│                                                                     │
├──────── train ────────┤ selection ┤ calibration ┤ policy ┤ OOT test ┤
│       3,000 rows      │ 500 rows  │  500 rows  │500 rows│ 500 rows │
│                       │           │            │        │          │
fit preprocessing       choose      fit Platt    freeze   report only
and candidate models    candidate   scaling      threshold
```

| Window | Rows | Purpose |
|---|---:|---|
| Train | 3,000 | Fit preprocessing and candidate models |
| Model selection | 500 | Select candidate using later observations |
| Calibration | 500 | Fit Platt scaling on a separate window |
| Policy validation | 500 | Freeze the decision threshold |
| Out-of-time test | 500 | Calculate final reported metrics only |

Calibration reduced the out-of-time Brier score from `0.1814` to `0.1795` and expected calibration error from `0.0832` to `0.0748`. AUC remained `0.7884`, as expected for a monotonic probability transformation.

## Registry serving safety rules

Before a registry model can become an active predictor, the runtime checks:

1. required model, metadata, metrics, and promotion-gate files exist;
2. each required file matches SHA-256 evidence;
3. the feature schema matches the running service contract;
4. the offline promotion gate is `PASS` and release-eligible;
5. the model can be deserialized;
6. a smoke request produces a finite probability in `[0, 1]`;
7. only then is the in-memory predictor reference replaced.

Fallback order:

```text
current verified registry predictor
        ↓ process restart
last verified cached registry bundle
        ↓ no valid cache
packaged local validated model
```

`REGISTRY_STRICT_STARTUP=true` can block startup when no verified registry model is available. With strict startup disabled, a fallback can remain ready while health/version endpoints report a degraded state.

## Platform map

| Layer | Implementation | Main evidence |
|---|---|---|
| Dated data contract | synthetic cohort, date and duplicate checks | `src/data/`, `src/features/`, `reports/data_quality_report.json` |
| Model lifecycle | temporal selection, calibration, OOT evaluation | `src/models/` |
| Registry control plane | MLflow aliases, promotion, rollback | `src/registry/`, `docs/model_registry.md` |
| Registry serving | verified cache, hot reload, fallback | `src/serving/runtime_manager.py`, `docs/registry_runtime.md` |
| Canary control | deterministic routing, online gate, stop/promotion state | `src/serving/canary.py`, `docs/canary_runtime.md` |
| Decision control | versioned policy and hard safety gate | `src/serving/policy.py`, `config/policy.yaml` |
| Review and audit | manual-review rules and redacted audit events | `src/serving/review_workflow.py`, `src/core/audit.py` |
| Monitoring | drift, latency, decisions, registry and canary metrics | `src/monitoring/`, `/metrics` |
| Release control | promotion gate, model card, SBOM, checksums | `src/governance/`, `reports/` |
| Deployment | non-root container, Kubernetes, Helm, HPA, PDB, NetworkPolicy | `docker/`, `k8s/`, `helm/` |
| CI/CD | evidence, container/kind, real registry+canary lifecycle | `.github/workflows/platform.yml` |

## Run locally

Generate the model and governance evidence:

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

Run packaged local-model API mode:

```bash
make serve
```

Run the local MLflow registry lifecycle:

```bash
make registry-up
make registry-register
make registry-promote
make registry-status
make registry-verify
```

Run the full registry/API integration drill:

```bash
make registry-smoke
```

## API surface

| Endpoint | Purpose |
|---|---|
| `GET /health` | service, model source, degraded state, canary state |
| `GET /ready` | readiness without hiding fallback state |
| `GET /version` | platform/service/model/policy/schema/registry/canary provenance |
| `GET /runtime/model` | registry runtime state and reload counters |
| `GET /canary/status` | canary state, evidence metrics, limits, transition provenance |
| `GET /decision-contract` | machine-readable decision semantics |
| `POST /predict` | probability, action, review route, audit and served-model provenance |
| `POST /explain` | reason codes and policy reasons |
| `POST /review-route` | human-review routing |
| `POST /shadow-predict` | legacy local candidate comparison without changing served action |
| `GET /metrics` | Prometheus service, registry, and canary metrics |

## Helm registry and canary mode

The Helm chart defaults to packaged local-model mode. Registry and canary are opt-in.

```bash
helm upgrade --install regulated-ai helm/regulated-ai \
  --set registryRuntime.enabled=true \
  --set registryRuntime.trackingUri=http://mlflow.mlops.svc.cluster.local:5000 \
  --set registryRuntime.registryUri=http://mlflow.mlops.svc.cluster.local:5000 \
  --set registryRuntime.canary.enabled=true \
  --set registryRuntime.canary.trafficPercent=5 \
  --set networkPolicy.mlflowEgress.enabled=true
```

The chart keeps `registryRuntime.canary.autoPromoteEnabled=false` by default.

The container remains non-root with a read-only root filesystem. Registry cache data uses a writable volume at `/var/lib/regulated-ai/registry-cache` with UID/GID `10001` access.

## Multi-replica boundary

The reference canary controller stores its evidence window in process memory. The real CI proof uses one API container. For a multi-replica production service, automatic promotion should use one elected controller or an external deployment controller consuming centrally aggregated evidence. Different API replicas must not independently promote from different local evidence windows.

That is why automatic promotion is disabled by default in normal and Helm configuration. See [docs/canary_runtime.md](docs/canary_runtime.md) for the full boundary and failure matrix.

## GitHub validation path

```text
Platform workflow
  evidence
    ├── 53 automated tests
    ├── Ruff + Bandit
    ├── runtime / registry dependency audits
    ├── model + release evidence
    └── generated site
          │
          ├── Docker live test
          ├── kind + default Helm deployment
          ├── registry + canary Helm server-side dry-run
          └── PostgreSQL + MinIO + MLflow + live API canary/promotion/rollback
```

A successful `main` validation can then feed the separate release workflow for the exact tested image and generated Pages evidence.

## Boundary

The dataset, target, drift, and consumer actions are synthetic. This repository demonstrates ML engineering, chronological evaluation, probability calibration, decision control, registry-driven serving, canary routing, online safety gates, monitoring, deployment testing, and release evidence. It is not a validated financial-advice system and must not be used for real customer decisions.
