# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![Triton CPU runtime](https://github.com/xm2325/regulated_ml_platform/actions/workflows/triton-cpu-runtime.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/triton-cpu-runtime.yml)
[![Triton capacity](https://github.com/xm2325/regulated_ml_platform/actions/workflows/triton-capacity.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/triton-capacity.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)
[![Pages](https://img.shields.io/badge/evidence-dashboard-blue)](https://xm2325.github.io/regulated_ml_platform/)

**A production-style regulated ML reference platform with chronological evaluation, calibration, MLflow registry lifecycle, canary release, monitoring, immutable promotion, rollback, real Triton CPU inference, runtime dynamic-batching evidence, and bounded capacity planning.**

[Evidence dashboard](https://xm2325.github.io/regulated_ml_platform/) · [v1.2 capacity evidence](docs/triton_capacity_evidence.md) · [Real Triton runtime evidence](docs/triton_runtime_evidence.md) · [Triton serving design](docs/triton_serving.md) · [Production operations](docs/production_operations.md) · [Registry runtime](docs/registry_runtime.md) · [Incident runbooks](docs/runbooks/ml_platform_incidents.md)

## Result first

Platform/service version: `1.2.0`. Validated calibrated model release: `0.6.0`.

```text
chronological model development
        ↓
calibration + frozen policy threshold
        ↓
MLflow registry / champion / challenger / rollback
        ↓
stable canary + online safety gate
        ↓
versioned ONNX/Triton serving package
        ↓
real Triton 25.06 CPU server
        ↓
HTTP probability parity
        ↓
concurrent synchronized requests
        ↓
actual scheduler batching from Prometheus counters
        ↓
NVIDIA Triton Perf Analyzer
        ↓
SLO + safety-headroom capacity reference
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

## v1.2: runtime batching and capacity evidence

v1.1 proved that the calibrated model could run on a real Triton server. v1.2 tests what happens when requests overlap.

```text
custom concurrent HTTP benchmark
→ native ↔ Triton probability parity
→ HTTP correctness
→ inference_count / execution_count deltas
→ observed average backend batch size

NVIDIA Triton Perf Analyzer
→ optimized Triton client path
→ concurrency / latency / throughput sweep
→ server capacity source

capacity policy
→ SLO filter
→ safety headroom
→ bounded reference replica decision
```

The two clients intentionally have different roles. The Python/httpx benchmark is a **semantic and batching gate**. It is not used as the Triton server-capacity source because Python scheduling, JSON serialization, and the client HTTP stack add measurable overhead. Server capacity comes from NVIDIA Triton Perf Analyzer.

### Passing v1.2 reference evidence

A successful hosted-CI run observed real dynamic batching while preserving model semantics:

| Concurrency | Custom HTTP p95 | Custom rows/s | Avg `support_base` batch | Probability parity |
|---:|---:|---:|---:|---:|
| 1 | 2.08 ms | 485 | 1.00 | PASS |
| 4 | 4.03 ms | 913 | 2.82 | PASS |
| 8 | 7.07 ms | 981 | 1.92 | PASS |
| 16 | 39.59 ms | 867 | 1.76 | PASS |
| 32 | 27.00 ms | 1,042 | 1.70 | PASS |

All requests completed without HTTP failures; the largest observed absolute probability error was about `5.41e-7`.

The same live server was then measured with Triton Perf Analyzer:

| Concurrency | Inferences/s | p95 | p99 |
|---:|---:|---:|---:|
| 1 | 807 | 1.279 ms | 1.317 ms |
| 4 | 2,801 | 1.545 ms | 1.637 ms |
| 7 | 4,634 | 1.721 ms | 1.836 ms |
| 10 | 7,554 | 1.796 ms | 2.070 ms |
| 13 | 8,846 | 1.995 ms | 2.566 ms |
| 16 | 9,425 | 2.391 ms | 3.296 ms |

These are short shared-runner reference measurements, not production capacity promises.

### What proves batching actually happened

The generated Triton configs still declare:

```text
max batch size          128
preferred batch sizes   8, 32, 64
base queue delay        500 microseconds
calibrator queue delay  250 microseconds
```

v1.2 no longer stops at configuration validation. For each concurrency scenario it reads Prometheus counters before and after the workload:

```text
observed average batch size
= delta(nv_inference_count)
  / delta(nv_inference_exec_count)
```

For example, one passing run processed 48 single-row requests at concurrency 4 in 17 base-model backend executions, producing an observed average batch size of about `2.82`.

## Capacity decision contract

`config/triton_capacity_policy.yaml` separates measurement from policy.

The custom concurrent HTTP path must pass:

```text
HTTP correctness
+ probability parity
+ observed runtime batching
```

Perf Analyzer points are then checked against server latency objectives. The highest-throughput SLO-passing point becomes the measured server reference capacity.

```text
safe reference capacity per replica
= measured Perf Analyzer capacity
  × configured safety headroom

reference replicas
= ceil(reference target
       / safe reference capacity per replica)
```

The decision fails closed when semantic correctness fails, batching is ineffective, Perf Analyzer evidence is missing, too few server points pass the SLO, or the result exceeds the configured replica boundary.

The generated `capacity_report.html` keeps three separate evidence views:

```text
p95 latency vs concurrency
throughput vs concurrency
observed backend batch size vs concurrency
```

See [`docs/triton_capacity_evidence.md`](docs/triton_capacity_evidence.md) for the measurement contract and evidence boundaries.

## v1.1 serving safety remains intact

The active artifact is a `RandomForestClassifier` wrapped by Platt calibration. Exporting only the base estimator would change probabilities used by the frozen policy threshold.

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
        └── 1/version.txt
```

`contract.json` records model/policy/schema versions, transformed feature count, calibration parameters, batching configuration, ONNX compatibility metadata, model family, accelerator boundary, and artifact hashes.

### Real-server failures that became release controls

Real Triton CI found defects that local ONNX execution did not expose:

```text
1. custom Platt calibrator
   local ONNX Runtime: PASS
   Triton 25.06: FAIL
   reason: ONNX IR 13 exceeded validated runtime limit 10

2. after IR compatibility fix
   support_base: READY
   support_calibrator: READY
   support_ensemble: FAIL
   reason: missing numeric model-version directory

3. after adding support_ensemble/1/
   local workspace: valid
   cross-job GitHub artifact: directory disappeared
   reason: empty directories are not preserved
```

The exporter/validator now pins and records the validated ONNX IR limit, requires the ensemble version directory, and preserves it across artifact transfer with a hashed version marker.

## Preprocessing boundary

Triton currently consumes the transformed `FP32` feature matrix:

```text
raw validated request
→ versioned fitted sklearn preprocessor
→ transformed FP32 matrix
→ Triton support_ensemble
```

The fitted `preprocessor.joblib` is versioned and hashed. The project does not claim that raw categorical preprocessing currently executes inside Triton.

## Why the current champion remains CPU_ONLY

The validated champion is a calibrated tree ensemble.

```text
model family = tree_ensemble
        ↓
accelerator policy
        ↓
CPU_ONLY
```

A future GPU-compatible model must provide separate evidence before `GPU_ELIGIBLE`:

```text
approved model family
+ real GPU runtime evidence
+ probability parity PASS
+ policy-decision parity PASS
+ meaningful throughput improvement
+ acceptable p95 latency
+ useful/safe sustained GPU utilization
→ GPU_ELIGIBLE
```

Helm fails closed: the current tree model cannot be forced onto GPU merely by setting a GPU flag.

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
  ├── chronological train/calibrate/OOT evaluation
  ├── ONNX export + compatibility + local parity
  ├── monitoring/governance
  └── tests/lint/security/dependency audits
         │
         ├────────────┬───────────────────┬──────────────────┬──────────────────┐
         ↓            ↓                   ↓                  ↓                  ↓
 triton-contract  triton-cpu-runtime  triton-capacity  container-and-kind  registry-integration
         │            │                   │                  │                  │
 repo/Helm rules  real server parity  concurrency        Docker/kind        MLflow lifecycle
 GPU fail-closed  batching config     actual batching    immutable ID       canary/rollback
                                      Perf Analyzer
                                      capacity policy
         └────────────┴──────────────┬────┴──────────────────┴──────────────────┘
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

Run the real Triton CPU parity drill:

```bash
make triton-runtime-cpu
```

Run the v1.2 concurrency, Perf Analyzer, and capacity evidence path:

```bash
make triton-capacity
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
ONNX/Triton runtime compatibility
native ↔ local ONNX probability/policy parity
real Triton CPU server readiness
real Triton HTTP probability parity
runtime-loaded batching configuration
real concurrent HTTP correctness
Prometheus-derived actual average batch size
NVIDIA Triton Perf Analyzer execution
SLO/headroom capacity policy
bounded reference replica decision
accelerator eligibility gate
current tree-model GPU fail-closed control
```

Not claimed without separate runtime evidence:

```text
real bank customer data
production capacity guarantee
current RandomForest acceleration on A100
CUDA/TensorRT performance
real GPU throughput improvement
live GPU autoscaling
live service-mesh operation
fully autonomous production promotion
```

`.github/workflows/triton-gpu-evidence.yml` is manual and requires a real self-hosted GPU runner. Its existence is not GPU evidence by itself.

The dataset, target, and customer actions are synthetic. This repository demonstrates ML engineering and production-control patterns and must not be used for real financial decisions.
