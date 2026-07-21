# Triton CPU runtime evidence

## Result first

The validated `RandomForestClassifier + Platt calibration` serving package has been executed on a **real NVIDIA Triton Inference Server 25.06** in GitHub-hosted CI using CPU execution.

The required runtime path is:

```text
validated sklearn + Platt champion
        ↓
fitted preprocessing outside Triton
        ↓
support_base ONNX
        ↓
support_calibrator ONNX
        ↓
support_ensemble
        ↓
real Triton 25.06 server
        ↓
HTTP inference: batch 1 / 8 / 32 / 64 / 128
        ↓
native sklearn ↔ Triton probability parity
        ↓
runtime batching-config + Prometheus metric checks
        ↓
PASS
```

This evidence is **CPU-only**. It does not claim CUDA, TensorRT, A100 acceleration, GPU throughput improvement, GPU memory behaviour, or live GPU autoscaling.

## Real hosted-CI benchmark evidence

A validated PR run produced:

| Batch | Triton p50 | Triton p95 | Rows/s at p50 | Max absolute probability error |
|---:|---:|---:|---:|---:|
| 1 | 2.03 ms | 2.78 ms | 493 | `3.82e-8` |
| 8 | 1.18 ms | 1.26 ms | 6,790 | `2.77e-7` |
| 32 | 2.08 ms | 2.16 ms | 15,371 | `3.57e-7` |
| 64 | 2.43 ms | 3.80 ms | 26,376 | `1.36e-7` |
| 128 | 4.00 ms | 4.07 ms | 32,000 | `1.76e-7` |

All tested batch sizes returned `parity_status=PASS` at tolerance `5e-5`.

The machine-readable evidence also verifies:

```text
support_base READY
support_calibrator READY
support_ensemble READY
base max_batch_size >= 128
calibrator max_batch_size >= 128
preferred dynamic batches include 8 / 32 / 64
required runtime batches 1 / 8 / 32 / 64 / 128 executed
all runtime probability parity checks PASS
Triton nv_inference_* metrics present
gpu_runtime_claim = false
```

The benchmark is a small hosted-runner reference measurement, not a capacity plan. Production sizing still requires representative concurrency, request distribution, CPU limits, instance counts, SLO targets, and sustained-load evidence.

## Why real Triton testing was necessary

Local ONNX Runtime parity passed before the serving package was proven deployable in Triton. The real-server test exposed two compatibility defects that static/local checks initially missed.

### Failure 1 — ONNX IR version mismatch

The hand-built Platt calibrator was initially emitted with ONNX IR version `13`.

Triton 25.06's bundled ONNX Runtime rejected it because the validated runtime supported at most IR version `10`:

```text
local ONNX Runtime parity        PASS
support_base in Triton           READY
support_calibrator in Triton     FAIL
reason                           unsupported model IR version
```

The exporter now pins the custom calibrator graph to the validated Triton IR compatibility level, checks it with `onnx.checker`, records the IR versions in the serving contract, and keeps probability/policy parity as a mandatory post-export gate.

This is not an opset downgrade. The graph keeps its declared ONNX opset; the compatibility control concerns the ONNX model IR/container version supported by the target server runtime.

### Failure 2 — ensemble version directory

After fixing ONNX IR compatibility, both ONNX components loaded but the ensemble failed:

```text
support_base        READY
support_calibrator  READY
support_ensemble    FAIL: No model version was found
```

Triton's model-repository contract requires a numeric version directory for the ensemble as well. The exporter now creates:

```text
support_ensemble/
├── config.pbtxt
└── 1/
```

The repository validator now fails if that directory is missing.

These failures are why the release chain distinguishes:

```text
ONNX file created
≠
local ONNX parity passed
≠
Triton repository structurally valid
≠
real Triton server loaded all models
≠
runtime HTTP parity passed
```

Each is a separate gate.

## Runtime evidence workflow

`.github/workflows/triton-cpu-runtime.yml` performs the real CPU proof:

```text
train validated model
→ build Triton repository
→ pull pinned Triton server image tag
→ record resolved image metadata
→ start real server with GPU metrics disabled
→ wait for server and ensemble readiness
→ read runtime-loaded model configs
→ issue HTTP inference for batches 1/8/32/64/128
→ compare Triton output with native calibrated sklearn
→ capture Triton Prometheus metrics
→ validate machine-readable runtime evidence
→ upload logs/configs/benchmark/evidence artifact
```

The workflow preserves container inspect and server logs on failure so model-loading errors remain reviewable.

## Preprocessing boundary

The current Triton ensemble consumes the **transformed FP32 feature matrix**, not raw account/employment categorical fields.

```text
raw validated request
→ deterministic fitted sklearn preprocessor
→ FP32 feature matrix
→ Triton support_ensemble
```

The fitted `preprocessor.joblib` is versioned and hashed in the serving contract. A future feature-store or native preprocessing service can replace this boundary, but this repository does not claim that raw categorical preprocessing currently runs inside Triton.

## GPU boundary

The current validated champion is a tree ensemble and remains:

```text
accelerator decision = CPU_ONLY
```

A future GPU-compatible model must separately provide:

```text
real GPU runtime evidence
+ probability parity PASS
+ policy-decision parity PASS
+ meaningful throughput improvement
+ acceptable p95 latency
+ useful/safe sustained GPU utilization
→ GPU_ELIGIBLE
```

The optional self-hosted GPU workflow is not evidence until it actually runs and produces reviewable artifacts. Merely running this CPU model on a GPU-capable host does not prove GPU acceleration.
