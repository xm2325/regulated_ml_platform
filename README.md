# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)
[![Pages](https://img.shields.io/badge/evidence-dashboard-blue)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style regulated ML reference platform covering chronological model evaluation, MLflow registry serving, controlled canary release, continuous-operations decisions, immutable environment promotion, Kubernetes workload isolation, SLO alerting, and rollback evidence.**

[Evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [Production operations](docs/production_operations.md) · [Canary runtime](docs/canary_runtime.md) · [Registry runtime](docs/registry_runtime.md) · [Incident runbooks](docs/runbooks/ml_platform_incidents.md)

## Result first

The engineering platform and API service are version `1.0.0`; the validated calibrated model release remains `0.6.0`.

`v1.0` does **not** claim a better model. It adds the production controls needed after a model already exists:

```text
monitor data + model + service
        ↓
continuous-ops decision
  ├── NO_TRAINING
  ├── BLOCKED_DATA_QUALITY
  ├── INVESTIGATE_MODEL
  └── TRAIN_CANDIDATE
             ↓
       challenger only
             ↓
 offline gate → canary → rollback-capable promotion
             ↓
 immutable release identity
             ↓
 dev → preprod → production approval boundary → prod
```

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

## What v1.0 adds

### 1. Continuous training is a controlled decision, not a cron command

`src/operations/continuous_ops.py` combines data quality, drift, optional live performance, and fairness evidence.

```text
bad data
→ BLOCKED_DATA_QUALITY
→ do not retrain on suspect data

performance/fairness degradation
→ INVESTIGATE_MODEL
→ segment failure before deciding on retraining

material drift + valid data
→ TRAIN_CANDIDATE
→ register challenger only
→ offline gate
→ canary required before champion

no trigger
→ NO_TRAINING
→ do not create a redundant release
```

The current CI evidence correctly returns `NO_TRAINING`:

```text
data quality                 PASS
max numeric PSI              0.03488773339483244
max categorical distance     0.06966666666666665
automatic promotion          false
```

Unit tests separately prove drift, bad-data, and degraded-model branches.

### 2. One immutable artifact moves between environments

A release identity is built from:

```text
Docker image sha256 digest
+ git commit
+ model release version
+ policy version
+ feature schema version
        ↓
release_id
```

A rebuild between environments changes the identity and is blocked.

The Helm chart supports immutable deployment as:

```text
repository@sha256:<digest>
```

Environment policy lives separately in:

```text
deploy/environments/dev-values.yaml
deploy/environments/preprod-values.yaml
deploy/environments/prod-values.yaml
```

This keeps replica/HPA/PDB/canary settings separate from artifact identity.

### 3. Real CI proves dev → preprod readiness, then blocks prod without approval

Each CI run derives its release identity from the **actual Docker image built and smoke-tested in that run**. The run-specific source of truth is the `environment-promotion-preprod/release_identity.json` Actions artifact.

The identity contains:

```text
image_digest           sha256:<actual tested image digest>
git_commit             <tested Git commit>
model_release_version  0.6.0
policy_version          targeted-support-policy-v3
feature_schema_version  financial_customer_features_v4
release_id              <derived immutable identity>
```

`dev → preprod` becomes `READY` only after the required technical checks pass.

The **same run-specific release identity** is then evaluated for `preprod → prod`. All nine technical gates must pass:

```text
automated tests
lint
security scan
dependency audit
container smoke
Helm/kind
registry integration
canary evidence
rollback drill
```

Even after those checks pass, the production control intentionally verifies:

```text
status          BLOCKED
approval_status pending
reason          manual production approval is required
```

This makes the distinction explicit:

```text
technically releasable ≠ authorized for production
```

The image embeds the Git commit, so its digest changes when the source commit changes. Digests are therefore stored as run-specific evidence rather than hardcoded into README text.

### 4. Inference and training no longer compete as identical Kubernetes workloads

The Helm chart now has separate workload controls:

```text
online inference   PriorityClass value 100000
scheduled training PriorityClass value   1000
```

It also includes:

- namespace `ResourceQuota` for requested/limited CPU and memory;
- explicit resource requests and limits;
- optional inference and training node selectors/tolerations;
- optional `nvidia.com/gpu` requests;
- A100 node-selection contract;
- a scheduled-training `CronJob` with `concurrencyPolicy: Forbid`;
- training disabled and suspended by default.

The CPU serving path is live tested in Docker and kind.

The GPU profile is Helm-rendered and Kubernetes server-side dry-run validated. Hosted CI has no NVIDIA GPU, so the repository does **not** claim runtime validation of CUDA, TensorRT, Triton, A100 throughput, GPU memory pressure, or GPU autoscaling.

### 5. SLI/SLO monitoring now has actionable alerts and runbooks

`observability/prometheus/regulated-ai-alerts.yaml` defines alerts for:

```text
error-budget burn
p95 prediction latency breach
runtime degraded state
registry reload failure
canary STOP
challenger error rate
champion/challenger decision disagreement
```

Every alert must have:

```text
severity
sustained duration
PromQL expression
summary
repo:// runbook link
```

`src/operations/validate_alerting.py` fails CI when this contract is broken.

Current generated evidence:

```text
alert rules      7
validation       PASS
missing runbooks 0
```

Operational response procedures are in [docs/runbooks/ml_platform_incidents.md](docs/runbooks/ml_platform_incidents.md).

## Existing controlled model-release path

The v0.9 canary path remains part of v1.0:

```text
verified challenger
        ↓
MLflow challenger alias
        ↓
checksum + schema + offline gate verification
        ↓
stable customer-level canary cohort
        ↓
champion + challenger score the same request
        ↓
configured cohort receives challenger result
        ↓
online safety evidence
  ├── WAIT
  ├── STOP
  └── PASS
        ↓ optional controlled promotion
challenger → champion
        ↓ incident
rollback to former champion
```

`CANARY_AUTO_PROMOTE_ENABLED=false` remains the normal/default setting.

The real registry integration starts PostgreSQL, MinIO, MLflow, and FastAPI, sends real API traffic, proves canary routing and model provenance, performs controlled promotion, then restores the former champion without restarting the API container.

The lifecycle test intentionally uses the same validated `0.6.0` artifact in two MLflow registry versions. Zero prediction/action disagreement in that test proves the release-control path, not challenger model superiority.

## Canary evidence and safety metrics

During `warming` and `healthy`, both champion and challenger score the same request. The canary controller tracks:

| Online metric | Control purpose |
|---|---|
| minimum comparisons | avoids decisions on too little evidence |
| minimum challenger-served requests | proves real exposure occurred |
| action disagreement rate | detects downstream decision changes |
| p95 absolute probability delta | detects large score changes |
| challenger error rate | detects runtime failures |
| challenger/champion latency ratio | detects operational regression |
| manual-review-rate increase | detects extra review burden |

A failed gate stops challenger serving. Terminal canary evidence is frozen so later champion traffic cannot overwrite the evidence used for the release decision.

## Version layers are separate

```text
platform_version       = 1.0.0
service_version        = 1.0.0
model_release_version  = 0.6.0
registry_model_version = immutable MLflow integer
policy_version         = targeted-support-policy-v3
feature_schema_version = financial_customer_features_v4
release_id             = immutable deployment identity across environments
```

This avoids treating an application release, MLflow registry integer, validated model version, policy version, schema version, and deployed container as the same concept.

## Evaluation design

Model selection, calibration, policy-threshold selection, and final reporting use separate chronological windows.

```text
2025-01-01                                                     2025-12-31
│                                                                     │
├──────── train ────────┤ selection ┤ calibration ┤ policy ┤ OOT test ┤
│       3,000 rows      │ 500 rows  │  500 rows  │500 rows│ 500 rows │
│                       │           │            │        │          │
fit preprocessing       choose      fit Platt    freeze   report only
and candidates           model       scaling      threshold
```

| Window | Rows | Purpose |
|---|---:|---|
| Train | 3,000 | fit preprocessing and candidate models |
| Model selection | 500 | select candidate on later observations |
| Calibration | 500 | fit Platt scaling separately |
| Policy validation | 500 | freeze decision threshold |
| Out-of-time test | 500 | calculate final metrics only |

Calibration reduced out-of-time Brier score from `0.1814` to `0.1795` and expected calibration error from `0.0832` to `0.0748`. AUC remained `0.7884`.

## Registry serving safety rules

Before a registry bundle can replace an active predictor, the runtime checks:

1. all required files exist;
2. SHA-256 evidence matches;
3. feature schema matches the service contract;
4. offline promotion gate is `PASS` and release-eligible;
5. deserialization succeeds;
6. a fixed smoke request produces a finite probability in `[0,1]`;
7. only then is the in-memory predictor replaced atomically.

Fallback order:

```text
current verified registry predictor
        ↓ process restart
last verified cached registry bundle
        ↓ no valid cache
packaged local validated model
```

## Platform map

| Layer | Implementation | Evidence |
|---|---|---|
| Data contract | dated synthetic cohort, range/category/date/duplicate validation | `src/data/`, `src/monitoring/data_quality.py` |
| Model lifecycle | temporal split, candidate selection, calibration, OOT evaluation | `src/models/` |
| Continuous operations | drift/DQ/performance/fairness decision controller | `src/operations/continuous_ops.py` |
| Registry control plane | MLflow aliases, promotion, rollback | `src/registry/` |
| Registry serving | verified cache, hot reload, fallback | `src/serving/runtime_manager.py` |
| Canary control | stable cohorts, dual scoring, online gate, STOP/PROMOTE/ROLLBACK | `src/serving/canary.py` |
| Decision control | versioned policy, hard safety gates, human-review routing | `src/serving/policy.py`, `src/serving/review_workflow.py` |
| GitOps-style promotion | immutable release identity and environment gates | `src/operations/gitops_promotion.py`, `deploy/environments/` |
| Observability | Prometheus metrics, SLO alerts, runbook contract | `observability/`, `docs/runbooks/` |
| Workload isolation | PriorityClasses, quotas, HPA/PDB, CPU/GPU scheduling profiles | `helm/regulated-ai/` |
| Security/release | non-root container, read-only root, SBOM, checksums, CodeQL/Bandit/audits | `docker/`, `src/governance/`, `.github/workflows/` |

## CI control graph

```text
evidence
  ├── data + features
  ├── train/calibrate/OOT evaluate
  ├── drift + data quality
  ├── continuous-ops decision
  ├── alert/runbook contract
  ├── governance evidence
  ├── tests + Ruff + Bandit + dependency audits
  └── artifacts
        │
        ├──────────────────────┐
        ↓                      ↓
container-and-kind      registry-integration
  ├ Docker live           ├ PostgreSQL
  ├ kind live             ├ MinIO
  ├ Helm dry-runs         ├ MLflow
  ├ GPU/training contract ├ canary traffic
  ├ image digest          ├ promotion/hot reload
  └ dev→preprod READY     └ rollback drill
        │                      │
        └──────────┬───────────┘
                   ↓
        production-promotion-control
                   ↓
          technical checks PASS
                   ↓
       approval pending → BLOCKED
```

The workflow demonstrates a GitOps-style immutable promotion contract using GitHub Actions. It does not claim to use Harness itself.

## Run locally

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

Run the local MLflow lifecycle:

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
| `GET /runtime/model` | registry runtime and reload counters |
| `GET /canary/status` | canary state, evidence, limits, transition provenance |
| `GET /canary/evaluate` | current online canary decision |
| `GET /decision-contract` | machine-readable decision semantics |
| `POST /predict` | probability, action, review route, audit and served-model provenance |
| `POST /explain` | reason codes and policy reasons |
| `POST /review-route` | human-review routing |
| `GET /metrics` | Prometheus service, registry, and canary metrics |

## Important production boundaries

This repository deliberately distinguishes implemented controls from capabilities that still require a real enterprise environment.

Not claimed as runtime-validated here:

- real customer financial data or financial-advice suitability;
- Harness itself;
- Argo CD/Flux controller execution;
- a service-mesh runtime demonstration;
- NVIDIA GPU runtime benchmarking;
- CUDA/TensorRT/Triton throughput or dynamic-batching benchmarks;
- GPU utilization-based autoscaling;
- a live streaming feature/data pipeline;
- autonomous production model promotion;
- multi-replica canary-controller leader election or centralized evidence storage.

The CPU serving path, Docker/kind deployment, MLflow registry lifecycle, canary routing, immutable environment identity, production approval block, CT/CM decision controls, alert/runbook contract, and rollback path are exercised by automated CI.

The dataset, target, drift, and customer actions are synthetic. This repository is an ML engineering reference implementation and must not be used for real financial decisions.
