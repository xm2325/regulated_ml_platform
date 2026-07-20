# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)
[![Pages](https://img.shields.io/badge/evidence-dashboard-blue)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style regulated ML reference platform covering chronological evaluation, MLflow registry serving, controlled canary release, monitoring, immutable promotion, rollback, and an evidence-gated ONNX/Triton serving path.**

[Evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [Triton serving](docs/triton_serving.md) · [Production operations](docs/production_operations.md) · [Canary runtime](docs/canary_runtime.md) · [Registry runtime](docs/registry_runtime.md) · [Incident runbooks](docs/runbooks/ml_platform_incidents.md)

## Result first

Platform/service version: `1.1.0`. Validated calibrated model release: `0.6.0`.

v1.1 adds a verifiable serving representation and accelerator decision path without claiming unsupported GPU performance.

```text
validated calibrated champion
        ↓
fitted preprocessing contract
        ↓
base estimator ONNX
        ↓
Platt calibration ONNX
        ↓
Triton ensemble repository
        ↓
probability + policy parity
        ↓
CPU execution evidence
        ↓
accelerator policy
   ┌────┴──────────────┐
   ↓                   ↓
current tree model   future GPU-compatible model
   ↓                   ↓
CPU_ONLY            real GPU benchmark required
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

### v1.1 GitHub Actions evidence

| Evidence | Result |
|---|---:|
| Triton repository validation | PASS |
| Native ↔ ONNX parity sample | 256 |
| Maximum absolute probability error | `1.76e-7` |
| Mean absolute probability error | `4.65e-8` |
| Policy-decision mismatches | 0 |
| Current accelerator decision | `CPU_ONLY` |
| Alert/runbook contract | PASS, 10 alerts |
| Triton model-repository image build | PASS |
| CPU Triton Helm contract | PASS |
| Current tree-model forced-GPU rejection | PASS |
| Future GPU-compatible deployment contract | PASS |
| Docker + kind path | PASS |
| MLflow registry/canary/promotion/rollback | PASS |
| Production approval-pending block | PASS |
| CodeQL | PASS |

## v1.1: preserve model semantics during serving export

The active artifact is a Platt-calibrated classifier. Exporting only the underlying estimator would change the probabilities used by the frozen policy threshold.

v1.1 therefore generates:

```text
support_base
FEATURES → base estimator → [P(class0), P(class1)]

support_calibrator
P(class1) → clip → logit → slope/intercept → sigmoid

support_ensemble
support_base → support_calibrator → SUPPORT_PROBABILITY
```

Generated structure:

```text
models/triton/
├── contract.json
├── preprocessor.joblib
└── model_repository/
    ├── support_base/1/model.onnx
    ├── support_base/config.pbtxt
    ├── support_calibrator/1/model.onnx
    ├── support_calibrator/config.pbtxt
    └── support_ensemble/config.pbtxt
```

`contract.json` records model/policy/schema versions, feature count, batching configuration, calibration parameters, model family, accelerator boundary, and SHA-256 artifact hashes.

## Probability and policy parity are release gates

Required CI runs the same observations through native calibrated sklearn and the exported ONNX execution path.

Current evidence:

```text
sample size                     256
max absolute probability error  0.000000176
mean absolute probability error 0.0000000465
policy decision mismatches      0
status                          PASS
```

A conversion is not accepted merely because an `.onnx` file was created.

## CPU benchmark: not a Triton/GPU benchmark

Required hosted CI compares native sklearn with ONNX Runtime CPU on the same runner.

| Batch | Native p50 | ONNX Runtime CPU p50 | Throughput ratio |
|---:|---:|---:|---:|
| 1 | 46.09 ms | 2.32 ms | 19.85× |
| 8 | 46.41 ms | 2.15 ms | 21.58× |
| 32 | 46.78 ms | 2.76 ms | 16.96× |
| 64 | 46.97 ms | 2.91 ms | 16.16× |
| 128 | 67.15 ms | 3.37 ms | 19.92× |

The report explicitly records:

```text
runtime_evidence = real_cpu
not_a_triton_runtime_benchmark = true
```

These numbers are not presented as Triton server, CUDA, TensorRT, A100, or GPU benchmark results.

## Dynamic batching contract

Generated Triton configs use bounded batching defaults:

```text
max batch size          128
preferred batch sizes   8, 32, 64
base queue delay        500 microseconds
calibrator queue delay  250 microseconds
```

Prometheus recording rules combine request rate, average batch size, queue time, GPU utilization, and GPU memory utilization where those Triton metrics are available.

## Why the current champion remains CPU_ONLY

The validated champion is a calibrated `RandomForestClassifier`, classified as `tree_ensemble`.

The accelerator policy does not allocate GPU capacity just because the platform supports GPU nodes.

```text
model family = tree_ensemble
        ↓
accelerator policy
        ↓
CPU_ONLY
```

A future GPU-compatible model must provide real GPU runtime evidence, parity, throughput improvement, acceptable p95 latency, and useful/safe sustained GPU utilization before `GPU_ELIGIBLE`.

## Helm fails closed for GPU

The optional Triton deployment defaults to CPU and is disabled by default.

A GPU render requires both:

```text
gpuEvidenceApproved = true
AND
modelFamily is GPU-eligible
```

Required CI proves the current tree model cannot be forced onto GPU even when someone manually flips the GPU flag.

The Triton server image and immutable model-repository image are separate so server runtime provenance and model artifact provenance are not mixed.

## GPU-aware scaling decision

The reference controller combines:

```text
accelerator eligibility
GPU utilization
Triton queue time
average batch size
replica boundaries
```

Examples:

```text
high queue pressure
→ SCALE_OUT candidate

low GPU + low queue + healthy batching
→ SCALE_IN candidate

low GPU + low queue + batch ≈ 1
→ HOLD; inspect batching first

accelerator != GPU_ELIGIBLE
→ GPU_PROFILE_DISABLED
```

This is a tested control contract, not a claim that a live production GPU autoscaler has been exercised.

## Existing production control chain

v1.1 keeps the earlier lifecycle intact:

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
  ├── ONNX export + parity + CPU benchmark
  ├── accelerator decision
  ├── monitoring/governance
  └── tests/lint/security/dependency audits
         │
         ├─────────────┬───────────────────┐
         ↓             ↓                   ↓
  triton-contract  container-and-kind  registry-integration
         │             │                   │
  model-repo image  Docker/kind         Postgres/MinIO/MLflow
  CPU Helm contract immutable ID        canary/promote/rollback
  RF GPU rejection  dev→preprod
         └─────────────┴─────────┬─────────┘
                                 ↓
                  production-promotion-control
                                 ↓
                    approval pending → BLOCKED
```

## Real GPU evidence is separate

`.github/workflows/triton-gpu-evidence.yml` is manual and requires a self-hosted GPU runner.

Its existence does not mean GPU evidence has been produced. It is designed to run a real Triton server, HTTP inference benchmark/parity, and Triton/GPU metric capture only when real NVIDIA runtime is available.

For the current tree-ensemble model it still preserves `CPU_ONLY`; running on a GPU-capable host is not itself proof of GPU acceleration.

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

Run only the v1.1 serving evidence path after training:

```bash
make triton
```

Run the local API:

```bash
make serve
```

Run the full local MLflow registry/API lifecycle when Docker is available:

```bash
make registry-smoke
```

## Boundaries

Automatically validated:

```text
chronological evaluation + calibration
MLflow registry + verified serving + rollback
canary safety controls
continuous monitoring/retraining decisions
Docker/Kubernetes/Helm controls
immutable environment promotion
ONNX export preserving calibration
Triton model repository + batching contract
native↔ONNX probability/policy parity
CPU ONNX Runtime benchmark
accelerator eligibility gate
GPU-aware scaling decision logic
Triton/GPU monitoring and runbooks
current tree-model GPU fail-closed control
```

Not claimed without further runtime evidence:

```text
real bank customer data
current RandomForest acceleration on A100
CUDA/TensorRT performance
real GPU throughput improvement
live GPU autoscaling
live service-mesh operation
fully autonomous production promotion
```

The dataset, target, and customer actions are synthetic. This repository demonstrates ML engineering and production-control patterns and must not be used for real financial decisions.
