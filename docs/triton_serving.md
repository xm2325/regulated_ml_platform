# Governed Triton serving path: v1.1

## Result first

v1.1 converts the validated calibrated model into a versioned Triton model repository and now verifies that repository on a **real Triton 25.06 CPU server**.

```text
validated sklearn champion
        ↓
fitted preprocessing contract
        ↓
base estimator ONNX
        ↓
Platt calibration ONNX
        ↓
Triton ensemble repository
        ↓
local probability/policy parity
        ↓
real Triton CPU readiness + HTTP parity
        ↓
accelerator policy
   ┌────┴───────────────────┐
   ↓                        ↓
current tree model        future GPU-compatible model
CPU_ONLY                  real GPU evidence required
```

Validated model release remains `0.6.0`; platform/service version `1.1.0` changes serving and deployment controls, not the validated model itself.

See [`triton_runtime_evidence.md`](triton_runtime_evidence.md) for the real runtime benchmark and compatibility-failure evidence.

## 1. Preserve calibration explicitly

The production artifact is a `PlattCalibratedClassifier`, not just the base estimator. Exporting only the base estimator would change probabilities used by the frozen policy threshold.

The serving package therefore contains:

```text
support_base
FEATURES → estimator → [P(class0), P(class1)]

support_calibrator
P(class1) → clip → logit → slope/intercept → sigmoid

support_ensemble
support_base → support_calibrator → SUPPORT_PROBABILITY
```

The fitted preprocessing stage remains upstream of Triton and is versioned separately.

## 2. Model repository and provenance

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

`contract.json` records:

- model, policy, schema, platform, and serving versions;
- transformed feature count;
- estimator class and model family;
- batch limits and preferred batch sizes;
- calibration slope/intercept;
- ONNX IR/runtime compatibility metadata;
- SHA-256 hashes for ONNX artifacts and preprocessor;
- CPU-default and GPU-evidence boundaries.

The Triton server image and model-repository artifact are separate so runtime provenance and model provenance are not conflated.

## 3. Five serving gates

A file extension is not deployment evidence. v1.1 separates five gates:

```text
Gate 1  ONNX artifact generated
Gate 2  ONNX checker + local ONNX Runtime parity
Gate 3  Triton repository structural validation
Gate 4  real Triton server loads base/calibrator/ensemble
Gate 5  real HTTP inference preserves native probabilities/policy semantics
```

The real-server gate caught defects that local parity did not:

1. a generated calibrator used ONNX IR v13 while the validated Triton runtime supported at most IR v10;
2. the ensemble was initially missing its required numeric version directory.

The exporter and validator now enforce both constraints.

## 4. Local parity gate

Required CI runs the same observations through:

```text
native calibrated sklearn
        vs
fitted preprocessor
→ base ONNX
→ calibrator ONNX
```

It checks:

- maximum/mean absolute probability error;
- policy-decision mismatches at the frozen threshold;
- finite probability range;
- actual ONNX Runtime providers.

A conversion is rejected if tolerance fails or policy decisions unexpectedly change.

## 5. Real Triton CPU runtime gate

`.github/workflows/triton-cpu-runtime.yml` runs on a standard GitHub-hosted CPU runner:

```text
build validated model/repository
→ pull pinned Triton 25.06 server tag
→ start real server with GPU metrics disabled
→ wait for support_ensemble readiness
→ read runtime-loaded configs
→ issue HTTP batches 1 / 8 / 32 / 64 / 128
→ compare Triton output with native calibrated sklearn
→ capture nv_inference_* metrics
→ write machine-readable evidence
```

A validated reference run passed every batch with maximum absolute probability error below `3.6e-7`.

This proves the CPU serving path. It does **not** prove GPU acceleration.

## 6. Dynamic batching contract

Base and calibrator configs declare:

```text
max batch size            128
preferred batch sizes     8, 32, 64
base queue delay          500 microseconds
calibrator queue delay    250 microseconds
```

Runtime CI verifies Triton actually loaded these settings and executes batch inference at `1/8/32/64/128`.

This does not by itself prove that concurrent requests were coalesced into preferred dynamic batches. Production tuning requires representative concurrent traffic and comparison of queue duration, execution count, request count, p95 latency, and throughput.

## 7. Preprocessing boundary

Current runtime path:

```text
raw validated request
→ fitted sklearn preprocessing
→ FP32 transformed matrix
→ Triton support_ensemble
```

`preprocessor.joblib` is hashed/versioned in the contract. The repository does not claim raw categorical preprocessing runs inside Triton.

## 8. Why the current champion remains CPU_ONLY

The current validated champion is a tree ensemble. Accelerator policy does not treat GPU as a default deployment target.

```text
model_family = tree_ensemble
        ↓
accelerator policy
        ↓
CPU_ONLY
```

A future GPU-compatible model needs:

```text
approved GPU candidate family
+ real GPU runtime evidence
+ probability parity PASS
+ policy parity PASS
+ meaningful throughput improvement
+ acceptable p95 latency
+ useful/safe sustained GPU utilization
→ GPU_ELIGIBLE
```

Without that evidence the result is `GPU_BENCHMARK_REQUIRED` or `GPU_REJECTED`.

## 9. Helm fails closed for GPU

Defaults:

```text
tritonServing.enabled = false
accelerator = cpu
gpuEvidenceApproved = false
modelFamily = tree_ensemble
```

A GPU render requires both explicit evidence approval and a GPU-eligible model family. Required CI proves the current tree model cannot be forced onto an A100 node simply by changing `accelerator=gpu`.

## 10. GPU-aware scaling contract

Scaling decisions combine:

- accelerator eligibility;
- sustained GPU utilization;
- Triton queue time;
- average batch size;
- current/min/max replicas.

```text
high queue + useful GPU pressure
→ SCALE_OUT candidate

low GPU + low queue + healthy batching
→ SCALE_IN candidate

low GPU + low queue + batch ≈ 1
→ HOLD; inspect batching first

accelerator != GPU_ELIGIBLE
→ GPU_PROFILE_DISABLED
```

This is tested decision logic, not evidence of a live GPU autoscaler.

## 11. Observability

Prometheus rules cover/derive:

- Triton request rate;
- queue duration;
- average batch size where inferable;
- inference errors;
- GPU utilization/memory metrics when GPU runtime is actually available.

The incident runbook requires preserving model-repository version, ONNX hashes, batching config, accelerator decision, queue/batch metrics, and capacity evidence before changing serving state.

## 12. Evidence hierarchy

```text
required hosted CI
  ├── ONNX conversion/checker
  ├── local ONNX parity
  ├── real Triton CPU server + HTTP parity
  ├── Triton repository/Helm contracts
  └── current tree GPU fail-closed gate

optional self-hosted GPU workflow
  └── only becomes GPU evidence after an actual run with reviewable artifacts
```

Running the CPU tree model on a GPU-capable host is not proof that the model benefits from GPU acceleration.

## Boundary

This is a synthetic regulated-ML reference platform. v1.1 proves the ONNX/Triton CPU serving path and defines fail-closed accelerator controls. It does not claim live bank traffic, production A100 operation, TensorRT optimization, CUDA performance, real GPU throughput improvement, or live GPU autoscaling without separate runtime evidence.
