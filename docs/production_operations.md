# Production operations: v1.0

## Purpose

Version `1.0.0` adds day-2 ML platform controls around the existing model, registry, canary, and rollback lifecycle. The validated model artifact remains release `0.6.0`; this version changes how data drift, retraining, environment promotion, compute contention, alerting, and production approval are controlled.

The reference path is:

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
       offline gate
             ↓
        canary evidence
             ↓
immutable release identity
             ↓
       dev → preprod
             ↓
technical production gates
             ↓
explicit production approval
             ↓
            prod
```

A retraining trigger and a production deployment are deliberately separate decisions.

## 1. Continuous training and monitoring decision

`src/operations/continuous_ops.py` combines monitoring evidence with `config/continuous_ops.yaml`.

### Decisions

| Decision | Meaning | Safe next action |
|---|---|---|
| `NO_TRAINING` | no configured trigger is active | continue monitoring |
| `BLOCKED_DATA_QUALITY` | input quality failed | quarantine/fix data before any retraining |
| `INVESTIGATE_MODEL` | observed performance or fairness crossed an operational floor/ceiling | open investigation and segment the failure |
| `TRAIN_CANDIDATE` | material drift is present and data quality is acceptable | train/register a challenger only |

A `TRAIN_CANDIDATE` decision does **not** change the champion. The policy requires:

```text
new training output
→ challenger alias
→ offline promotion gate
→ canary evidence
→ controlled promotion decision
```

`auto_promote_after_retraining=false` is part of the control contract.

### Current CI evidence

The current synthetic evidence produced:

```text
data quality       PASS
max numeric PSI    0.03488773339483244
max categorical TVD 0.06966666666666665
continuous decision NO_TRAINING
```

This is intentionally a no-op decision: a continuous-training system should be able to decide *not* to create a redundant model release.

Unit tests separately prove that material drift produces `TRAIN_CANDIDATE`, bad data produces `BLOCKED_DATA_QUALITY`, and poor performance/fairness produces `INVESTIGATE_MODEL`.

## 2. Immutable environment promotion

`src/operations/gitops_promotion.py` creates one release identity from:

```text
Docker image sha256 digest
+ git commit
+ model release version
+ policy version
+ feature schema version
        ↓
release_id
```

The same identity must move between environments. Rebuilding the image between dev, preprod, and prod is blocked because that creates a different artifact.

### Environment policy

Environment-specific values live in:

```text
deploy/environments/dev-values.yaml
deploy/environments/preprod-values.yaml
deploy/environments/prod-values.yaml
```

They describe environment policy such as replica count, HPA, PDB, canary percentage, and evidence requirements. They do not redefine the release artifact.

The Helm chart accepts `image.digest`; when present it renders:

```text
repository@sha256:<digest>
```

rather than a mutable tag.

### Verified CI release identity

The live container job built and tested this exact Docker image identity:

```text
image digest   sha256:a4df2766d951bce08dddca9f191c0bf47486b0e7b39ba1d282a5f1bd2ed503f8
release ID     2bf37ed424e112fa5efd
model release  0.6.0
policy         targeted-support-policy-v3
schema         financial_customer_features_v4
```

`dev → preprod` was `READY` only after all required checks were `PASS`.

The same release identity was then evaluated for `preprod → prod`. All nine technical checks were `PASS`:

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

but the result remained:

```text
status          BLOCKED
approval_status pending
reason          manual production approval is required
```

This is deliberate: technical eligibility is not equivalent to production authorization.

## 3. Kubernetes inference/training isolation

The Helm chart separates online inference and scheduled training concerns.

### Priority

```text
regulated-ai-inference  priority 100000
regulated-ai-training   priority 1000
```

Batch training must not have equal scheduling priority to customer-facing inference.

### Namespace compute budget

A `ResourceQuota` caps aggregate requested and limited CPU/memory. This prevents one workload family from silently consuming unlimited namespace capacity.

### Scheduled training

An optional `CronJob` is present but disabled and suspended by default. It uses:

- `concurrencyPolicy: Forbid`;
- lower training priority;
- explicit requests/limits;
- separate data and output PVCs;
- non-root security context;
- optional GPU scheduling.

A schedule therefore does not imply that every scheduled run must retrain. The continuous-ops decision should determine whether training is justified before a production workflow activates the training job.

## 4. GPU scheduling contract and its boundary

Both inference and training can request:

```text
nvidia.com/gpu: 1
nodeSelector:
  accelerator: nvidia-a100
```

with the corresponding GPU toleration.

CI renders this profile and performs Kubernetes server-side dry-run validation. It verifies the GPU resource request, A100 node selector, priority classes, quota, and training CronJob contract.

**CI does not contain an NVIDIA GPU.** Therefore this repository does not claim that CUDA, TensorRT, Triton, A100 throughput, GPU memory pressure, dynamic batching, or GPU autoscaling have been runtime-benchmarked. Those require a real GPU test environment and are a separate serving-performance milestone.

## 5. SLI/SLO alerting and runbooks

`observability/prometheus/regulated-ai-alerts.yaml` defines actionable alerts for:

- API error-budget burn;
- p95 prediction latency SLO breach;
- degraded registry runtime;
- registry reload failures;
- stopped canary;
- challenger error rate;
- champion/challenger decision disagreement.

Every rule must include:

```text
unique alert name
severity = page | ticket
sustained `for` duration
PromQL expression
summary
repo:// runbook link
```

`src/operations/validate_alerting.py` fails CI if this contract is broken. Current evidence reports `7` alerts, `PASS`, and zero validation failures.

The response procedures live in `docs/runbooks/ml_platform_incidents.md` and include evidence preservation, promotion freeze, rollback conditions, and incident-review fields.

## 6. CI control graph

```text
evidence
  ├── dated synthetic data + features
  ├── train/calibrate/OOT evaluate
  ├── data quality + drift
  ├── continuous-ops decision
  ├── alert/runbook validation
  ├── governance/release evidence
  ├── tests/lint/security/audits
  └── artifacts
        │
        ├───────────────┬─────────────────────┐
        ↓               ↓                     │
container-and-kind   registry-integration     │
  │                   │                       │
  ├ Docker live       ├ Postgres              │
  ├ kind live         ├ MinIO                  │
  ├ Helm dry-runs     ├ MLflow                 │
  ├ GPU contract      ├ real canary traffic   │
  ├ image digest      ├ controlled promote    │
  └ dev→preprod       └ rollback drill        │
        │               │                     │
        └───────┬───────┘                     │
                ↓                             │
      production-promotion-control            │
                ↓                             │
      all technical checks PASS               │
                ↓                             │
      approval pending → BLOCKED              │
```

The workflow demonstrates a GitOps-style immutable promotion contract using GitHub Actions. It does not claim use of Harness itself.

## 7. What this version still does not claim

- no real customer financial data or validated financial-advice use;
- no external enterprise GitOps controller such as Argo CD or Flux running in CI;
- no Harness deployment environment;
- no real NVIDIA GPU runtime benchmark;
- no Triton/TensorRT/CUDA serving benchmark yet;
- no service mesh runtime demonstration;
- no live upstream streaming feature pipeline;
- no autonomous production model promotion;
- no multi-replica canary controller election/central evidence store.

These boundaries are intentional so the repository distinguishes implemented evidence from architecture that still requires a production environment.
