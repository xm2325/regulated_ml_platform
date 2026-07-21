# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![Triton CPU runtime](https://github.com/xm2325/regulated_ml_platform/actions/workflows/triton-cpu-runtime.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/triton-cpu-runtime.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)
[![Pages](https://img.shields.io/badge/evidence-dashboard-blue)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style regulated ML reference platform covering chronological evaluation, calibrated model serving, MLflow registry lifecycle, controlled canary release, monitoring, immutable promotion, rollback, and a verified ONNX/Triton CPU inference path.**

[Evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [Real Triton runtime evidence](docs/triton_runtime_evidence.md) · [Triton serving design](docs/triton_serving.md) · [Production operations](docs/production_operations.md) · [Canary runtime](docs/canary_runtime.md) · [Registry runtime](docs/registry_runtime.md) · [Incident runbooks](docs/runbooks/ml_platform_incidents.md)

## Result first

Platform/service version: `1.1.0`. Validated calibrated model release: `0.6.0`.

```text
chronological model development
        ↓
calibration + frozen policy threshold
        ↓
MLflow registry / champion / challenger / rollback
        ↓
stable canary + online safety gate
        ↓
verified serving export
        ↓
base estimator ONNX + Platt calibrator ONNX
        ↓
real Triton support_ensemble on CPU
        ↓
runtime HTTP parity + batching + metrics evidence
        ↓
accelerator policy
   ┌────┴─────────────────────┐
   ↓                          ↓
current tree champion       future GPU-compatible model
CPU_ONLY                    real GPU evidence required
```

### Validated model

| Result | Value |
|---|---:|
| OOT AUC | 0.7884 |
| Brier score | 0.1795 |
| Expected calibration error | 0.0748 |
| Policy precision | 0.8755 |
| Policy recall | 0.7098 |
| High-confidence precision | 0.8828 |
| Frozen policy threshold | 0.70 |

### v1.1 evidence

| Evidence | Result |
|---|---:|
| Triton repository validation | PASS |
| Native ↔ local ONNX parity sample | 256 |
| Local ONNX policy-decision mismatches | 0 |
| Real Triton 25.06 CPU server | PASS |
| `support_base` / `support_calibrator` / `support_ensemble` | READY |
| Real HTTP batches | 1, 8, 32, 64, 128 |
| Real HTTP probability parity | PASS for all batches |
| Largest observed Triton probability error | `3.57e-7` |
| Runtime-loaded dynamic batching contract | PASS |
| Triton `nv_inference_*` metrics | PRESENT |
| Current accelerator decision | `CPU_ONLY` |
| Current tree-model forced-GPU rejection | PASS |
| Docker + kind + Helm path | PASS |
| MLflow registry/canary/promotion/rollback | PASS |
| Production approval-pending block | PASS |
| CodeQL | PASS |

## Real Triton CPU runtime evidence

A GitHub-hosted CI run starts the official NVIDIA Triton Inference Server `25.06` image with GPU metrics disabled, loads the generated model repository, and sends real V2 HTTP inference requests.

Reference hosted-CI run:

| Batch | Triton p50 | Triton p95 | Rows/s at p50 | Max abs probability error |
|---:|---:|---:|---:|---:|
| 1 | 2.03 ms | 2.78 ms | 493 | `3.82e-8` |
| 8 | 1.18 ms | 1.26 ms | 6,790 | `2.77e-7` |
| 32 | 2.08 ms | 2.16 ms | 15,371 | `3.57e-7` |
| 64 | 2.43 ms | 3.80 ms | 26,376 | `1.36e-7` |
| 128 | 4.00 ms | 4.07 ms | 32,000 | `1.76e-7` |

These are small hosted-runner reference measurements, not production capacity numbers. The important release evidence is that every tested batch preserved probability semantics within tolerance and the actual server loaded the intended batching configuration.

See [`docs/triton_runtime_evidence.md`](docs/triton_runtime_evidence.md) for the full evidence and failure analysis.

## Serving representation preserves calibration

The active artifact is a `RandomForestClassifier` wrapped by Platt calibration. Exporting only the underlying estimator would change the probabilities used by the frozen policy threshold.

v1.1 therefore builds:

```text
support_base
FEATURES → base estimator → [P(class0), P(class1)]

support_calibrator
P(class1) → clip → logit → slope/intercept → sigmoid

support_ensemble
support_base → support_calibrator → SUPPORT_PROBABILITY
```

Generated repository:

```text
models/triton/
├── contract.json
├── preprocessor.joblib
└── model_repository/
    ├── support_base/
    │   ├── config.pbtxt
    │   └── 1/model.onnx
    ├── support_calibrator/
    │   ├── config.pbtxt
    │   └── 1/model.onnx
    └── support_ensemble/
        ├── config.pbtxt
        └── 1/
```

`contract.json` records model/policy/schema versions, transformed feature count, calibration parameters, batching configuration, model family, ONNX compatibility metadata, accelerator boundary, and SHA-256 artifact hashes.

## Why real-server testing is a separate gate

Local ONNX Runtime parity alone did not prove deployability. Real Triton testing found two defects that static/local checks initially missed:

```text
1. Platt calibrator ONNX
   local ONNX Runtime: PASS
   Triton 25.06: FAIL
   reason: ONNX IR 13 > validated server limit 10

2. after IR compatibility fix
   support_base: READY
   support_calibrator: READY
   support_ensemble: FAIL
   reason: missing numeric model-version directory
```

The exporter/validator now enforce both constraints. Release evidence therefore distinguishes:

```text
ONNX file created
≠ local ONNX parity passed
≠ Triton repository structurally valid
≠ real Triton server loaded all models
≠ runtime HTTP parity passed
```

Each is a separate gate.

## Dynamic batching contract

Generated Triton configs use bounded defaults:

```text
max batch size          128
preferred batch sizes   8, 32, 64
base queue delay        500 microseconds
calibrator queue delay  250 microseconds
```

Required runtime CI confirms Triton actually loaded the batching configuration and executes inference at batches `1/8/32/64/128`.

This does **not** claim that concurrent requests were proven to coalesce into preferred dynamic batches. Production batching tuning still requires representative concurrency plus queue-time, execution-count, latency, and throughput evidence.

## Preprocessing boundary

Triton currently consumes the transformed `FP32` feature matrix:

```text
raw validated request
→ versioned fitted sklearn preprocessor
→ transformed FP32 matrix
→ Triton support_ensemble
```

The fitted `preprocessor.joblib` is versioned and hashed in the serving contract. The repository does not claim that raw categorical preprocessing currently executes inside Triton.

## Why the current champion remains CPU_ONLY

The validated champion is a calibrated tree ensemble. GPU capacity is not allocated simply because the platform supports A100 node scheduling.

```text
model family = tree_ensemble
        ↓
accelerator policy
        ↓
CPU_ONLY
```

A future GPU-compatible model must provide all required evidence before `GPU_ELIGIBLE`:

```text
approved model family
+ real GPU runtime evidence
+ probability parity PASS
+ policy-decision parity PASS
+ meaningful throughput speedup
+ acceptable p95 latency
+ useful/safe sustained GPU utilization
→ GPU_ELIGIBLE
```

Helm fails closed: the current tree model cannot be forced onto GPU merely by setting `accelerator=gpu`.

## GPU-aware scaling is evidence-gated

The reference scaling decision considers:

```text
accelerator eligibility
GPU utilization
Triton queue time
average batch size
replica limits
```

Examples:

```text
high queue pressure
→ SCALE_OUT candidate

low GPU + low queue + healthy batching
→ SCALE_IN candidate

low GPU + low queue + average batch ≈ 1
→ HOLD; inspect batching first

accelerator != GPU_ELIGIBLE
→ GPU_PROFILE_DISABLED
```

This is tested decision logic, not a claim that a live GPU autoscaler has been exercised.

## Production control chain

```text
monitor data/model/service
        ↓
continuous-ops decision
        ↓
challenger-only retraining path
        ↓
offline gate + MLflow registry
        ↓
stable canary + online safety gate
        ↓
controlled promotion / rollback
        ↓
immutable dev → preprod promotion
        ↓
technical production gates
        ↓
explicit production approval
```

Retraining, model promotion, environment promotion, and production authorization remain separate decisions.

## CI proof graph

```text
evidence
  ├── train/calibrate/OOT evaluate
  ├── ONNX export + compatibility + local parity
  ├── accelerator decision
  ├── monitoring/governance
  └── tests/lint/security/dependency audits
         │
         ├────────────┬────────────────────┬──────────────────┐
         ↓            ↓                    ↓                  ↓
  triton-contract  triton-cpu-runtime  container-and-kind  registry-integration
         │            │                    │                  │
  repo/Helm rules  real Triton server   Docker/kind        MLflow lifecycle
  GPU fail-closed  HTTP parity/metrics  immutable ID       canary/rollback
         └────────────┴──────────────┬─────┴──────────────────┘
                                     ↓
                         production-promotion-control
                                     ↓
                         approval pending → BLOCKED
```

## Run locally

```bash
git clone https://github.com/xm2325/regulated_ml_platform.git
cd regulated_ml_platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-onnx.txt

make evidence
make lint
make security
make audit
make audit-onnx
```

Build the ONNX/Triton serving package:

```bash
make triton
```

With Docker and access to the Triton image, run the real CPU runtime drill:

```bash
make triton-runtime-cpu
```

Run the MLflow registry/API lifecycle:

```bash
make registry-smoke
```

## Evidence boundaries

Automatically validated:

```text
chronological evaluation + calibration
MLflow registry + verified serving + rollback
canary safety controls
continuous monitoring/retraining decisions
Docker/Kubernetes/Helm controls
immutable environment promotion
ONNX export preserving calibration
ONNX/Triton runtime compatibility checks
native ↔ local ONNX probability/policy parity
real Triton CPU server readiness
real Triton HTTP probability parity for batches 1/8/32/64/128
runtime-loaded batching configuration
Triton inference metrics
accelerator eligibility gate
GPU-aware scaling decision logic
current tree-model GPU fail-closed control
```

Not claimed without separate runtime evidence:

```text
real bank customer data
current RandomForest acceleration on A100
CUDA/TensorRT performance
real GPU throughput improvement
live GPU autoscaling
live service-mesh operation
fully autonomous production promotion
```

`.github/workflows/triton-gpu-evidence.yml` is manual and requires a real self-hosted GPU runner. Its existence is not GPU evidence by itself.

The dataset, target, and customer actions are synthetic. This repository demonstrates ML engineering and production-control patterns and must not be used for real financial decisions.
