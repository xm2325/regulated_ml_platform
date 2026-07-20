# Governed Triton serving path: v1.1

## Result first

The v1.1 serving path converts the validated calibrated model into a versioned Triton model repository while keeping accelerator claims evidence-gated.

```text
validated sklearn champion
        ↓
fitted preprocessing contract
        ↓
base estimator ONNX
        ↓
Platt calibration ONNX
        ↓
Triton ensemble
        ↓
probability + policy parity gate
        ↓
CPU serving evidence
        ↓
accelerator policy
   ┌────┴──────────────┐
   ↓                   ↓
tree / no GPU       GPU-compatible model
benefit evidence       + real GPU evidence
   ↓                   ↓
CPU_ONLY           GPU_ELIGIBLE
```

The validated model release remains `0.6.0`. Platform and service version `1.1.0` add a serving representation and deployment control path; they do not silently change the validated model.

## 1. Why the calibration stage is explicit

The production artifact is a `PlattCalibratedClassifier`, not only a base estimator. Exporting the underlying classifier and ignoring calibration would change the probabilities used by the policy threshold.

v1.1 therefore creates two ONNX models:

```text
support_base
FEATURES
  ↓
base estimator
  ↓
[P(class 0), P(class 1)]

support_calibrator
[P(class 0), P(class 1)]
  ↓
select P(class 1)
  ↓
clip
  ↓
logit
  ↓
slope × logit + intercept
  ↓
sigmoid
  ↓
SUPPORT_PROBABILITY
```

`support_ensemble` connects those stages in Triton. This preserves the probability semantics used by the existing policy layer.

## 2. Artifact chain and provenance

The generated serving package contains:

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
        └── config.pbtxt
```

`contract.json` records:

- model, policy, feature-schema, platform, and serving versions;
- transformed feature count;
- estimator class and model family;
- batching configuration;
- calibration slope/intercept;
- SHA-256 hashes for both ONNX models and the fitted preprocessor;
- CPU default and GPU-evidence boundary.

The model repository can be packaged separately from the Triton server image. The Helm deployment uses an init container to copy the immutable model repository into a shared runtime volume, then mounts it read-only into Triton.

## 3. Required parity gate

A successful ONNX conversion is not sufficient. v1.1 executes the same customer batch through:

```text
native calibrated sklearn model
             vs
fitted preprocessor
  → base ONNX
  → calibration ONNX
```

The gate checks:

- maximum absolute probability error;
- mean absolute probability error;
- policy-decision mismatch count at the frozen threshold;
- finite probability range;
- ONNX Runtime providers used by the validation.

A serving artifact is not release-eligible when probability tolerance fails or any policy decision changes unexpectedly.

## 4. Dynamic batching contract

The base and calibration stages declare bounded dynamic batching:

```text
max batch size            128
preferred batch sizes     8, 32, 64
base max queue delay       500 microseconds
calibrator queue delay     250 microseconds
```

These values are engineering defaults for the reference workload, not universal production settings. Real tuning should use request concurrency, queue time, p95 latency, throughput, and batch-size evidence together.

Increasing queue delay can improve aggregation but can also add latency. Adding replicas can reduce queue pressure but may also reduce batching efficiency by splitting traffic across more model instances.

## 5. Three evidence levels

### Level A — conversion and parity evidence

Runs on required hosted CI:

```text
sklearn artifact
→ ONNX export
→ ONNX checker
→ ONNX Runtime execution
→ probability parity
→ policy-decision parity
```

This proves semantic compatibility of the exported serving representation.

### Level B — CPU execution benchmark

Runs on required hosted CI:

```text
native sklearn CPU
vs
ONNX Runtime CPU
```

The report includes batch-specific p50/p95 latency and throughput. It is explicitly labelled `not_a_triton_runtime_benchmark=true`.

### Level C — real Triton / accelerator evidence

Requires an actual Triton server. GPU claims additionally require an actual NVIDIA runtime.

The repository provides a real HTTP benchmark client and a separate manual self-hosted GPU workflow. A hosted CPU CI success must never be translated into a CUDA, TensorRT, A100, or GPU-throughput claim.

## 6. Why the current reference champion remains CPU_ONLY

The current validated champion is classified as a `tree_ensemble` by the serving contract. The accelerator policy does not treat this family as a default GPU candidate.

```text
model family = tree_ensemble
        ↓
accelerator policy
        ↓
CPU_ONLY
```

This avoids allocating expensive GPU capacity merely because the platform supports GPUs.

For a future GPU-compatible model, `GPU_ELIGIBLE` requires all of the following:

```text
approved GPU candidate family
+
real_gpu runtime evidence
+
probability parity PASS
+
policy-decision parity PASS
+
minimum throughput speedup
+
p95 latency ratio within limit
+
sustained GPU utilization within useful/safe range
```

Without those conditions the decision is `GPU_BENCHMARK_REQUIRED` or `GPU_REJECTED`.

## 7. Helm fails closed for GPU

The optional Triton deployment defaults to:

```text
tritonServing.enabled = false
accelerator            = cpu
gpuEvidenceApproved    = false
modelFamily            = tree_ensemble
```

A GPU render requires both:

1. `gpuEvidenceApproved=true`; and
2. a model family listed as GPU-eligible.

Therefore manually changing only `accelerator=gpu` cannot place the current tree-ensemble reference model on an A100 node.

## 8. GPU-aware scaling uses more than utilization

The reference scaling decision combines:

- sustained GPU utilization;
- Triton queue time per request;
- observed average batch size;
- current/min/max replicas;
- accelerator eligibility.

Examples:

```text
high queue + moderate/high GPU
→ SCALE_OUT candidate

low GPU + low queue + healthy batching
→ SCALE_IN candidate

low GPU + low queue + average batch ≈ 1
→ HOLD; inspect batching first

accelerator decision != GPU_ELIGIBLE
→ GPU_PROFILE_DISABLED
```

Low utilization alone is not a safe scale-in signal because weak request aggregation can create the same symptom.

## 9. Triton/GPU observability

Prometheus recording rules derive:

- average Triton batch size;
- queue time per request;
- request rate;
- GPU utilization;
- GPU memory utilization.

Alerts cover:

- sustained queue pressure;
- high GPU utilization with queue pressure;
- low GPU utilization with weak batching.

The incident runbook requires responders to preserve model-repository version, ONNX hashes, batch configuration, accelerator decision, queue/batch metrics, and GPU capacity evidence before changing serving state.

## 10. CI control path

```text
required hosted CI
  ↓
train validated model
  ↓
export ONNX + Triton repository
  ↓
repository validation
  ↓
probability/policy parity
  ↓
CPU benchmark
  ↓
accelerator decision
  ↓
Triton model-repository image build
  ↓
CPU Triton Helm render
  ↓
prove current tree model GPU render is rejected
  ↓
validate future GPU-compatible contract only
  ↓
container/kind + registry/canary/rollback
  ↓
environment promotion controls
```

A separate `workflow_dispatch` self-hosted GPU workflow is intentionally outside the required hosted badge. Its existence does not imply it has run; GPU runtime claims require its actual artifacts.

## Boundary

This is a synthetic regulated-ML reference platform. v1.1 demonstrates ONNX conversion, Triton repository construction, calibration preservation, parity controls, batching configuration, deployment contracts, accelerator eligibility, and observability design. It does not claim live bank traffic, production A100 operation, TensorRT optimization, CUDA performance, or real GPU autoscaling until those are supported by runtime evidence.
