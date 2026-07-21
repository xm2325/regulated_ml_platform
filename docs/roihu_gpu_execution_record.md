# CSC Roihu GH200 execution record

This record captures real CSC Roihu execution performed while preparing the
v1.3 accelerator-qualification path. It is deliberately outcome-neutral: failed
jobs remain visible, a smoke pass is not promoted into formal eligibility, and a
formal rejection is not rewritten as success.

## Claim boundary

- The workload is a deterministic synthetic neural candidate, not the current
  calibrated tree champion and not a customer model.
- Results apply only to the named source commits, artifacts, GH200 jobs, software
  stack, inputs, batch/concurrency profile, and measurement scope.
- `gputest` results are `SMOKE_ONLY`. They cannot establish production capacity,
  cost, semantic correctness, GPU eligibility, or promotion approval.
- The formal result below is `GPU_REJECTED`. It keeps the current accelerator and
  profile unchanged and does not authorize automatic promotion.
- Roihu uses Slurm and Apptainer. These runs do not prove live Kubernetes GPU
  node-pool, operator, autoscaling, service-mesh, or on-call operation.

## Exact source and cloud validation

Formal runtime commit:
`e4d3f321f8f93865a7e863c5db10e2fcb6b861a7`

Immutable source archive SHA-256:
`0f8ca616356efa8a1f47e515b6de090cab9e7529e2dad77fb4664e86be35075a`

All five GitHub Actions workflows for that head completed successfully. They
validate code and contracts but do not themselves execute a GPU workload:

- [Platform build, test, container, and evidence — run 29852931607](https://github.com/xm2325/regulated_ml_platform/actions/runs/29852931607)
- [Triton CPU runtime parity and batching — run 29852931672](https://github.com/xm2325/regulated_ml_platform/actions/runs/29852931672)
- [Triton concurrency, batching, and capacity — run 29852931563](https://github.com/xm2325/regulated_ml_platform/actions/runs/29852931563)
- [Roihu GH200 evidence contract — run 29852931568](https://github.com/xm2325/regulated_ml_platform/actions/runs/29852931568)
- [CodeQL — run 29852931603](https://github.com/xm2325/regulated_ml_platform/actions/runs/29852931603)

## PyTorch, CUDA, and ONNX smoke

Slurm job `304885` ran on `gputest`, node `rg2101`, and completed `0:0` in
28 seconds for exact commit `e4d3f321f8f93865a7e863c5db10e2fcb6b861a7`.
The allocation was one full NVIDIA GH200 with its 72-core Grace CPU comparator.

At batch 256, steady-state model execution with model and input resident on the
measured device produced:

| Path | Throughput (items/s) | p95 latency (ms) | Speedup vs CPU |
| --- | ---: | ---: | ---: |
| Grace CPU FP32 | 193,832.1 | 1.3868 | 1.0000x |
| GH200 CUDA FP32 | 645,525.7 | 0.4086 | 3.3303x |
| GH200 CUDA BF16 | 693,765.9 | 0.3774 | 3.5792x |

Both CUDA precision checks passed within their declared FP32/BF16 tolerances.
The job also passed source/archive validation, offline hash-locked ONNX dependency
installation, ONNX checker, and benchmark/export cross-artifact validation.

Key digests:

- source archive:
  `0f8ca616356efa8a1f47e515b6de090cab9e7529e2dad77fb4664e86be35075a`
- ONNX candidate:
  `c279a9e202735c0781cae99a01f8ce157bd626497532b737080b540f0cca8268`
- export manifest:
  `56698ef7b8c38aa4dfc89f258dc129b99759c95fe3bb58a557131d4ef1d6947d`

The timing excludes model loading and host-to-device transfer. This is a
steady-state smoke comparison, not a universal CPU/GPU cost result.

## TensorRT and Triton smoke

Slurm job `304686` ran on `gputest`, node `rg2101`, and completed `0:0` in
1 minute 52 seconds for commit
`a468e694025c178dc7805ee1a748b9bafd3d5d40`.

The job digest-verified ARM64 NGC 25.06 server and SDK SIFs, built a BF16
TensorRT plan on the target GH200, started Triton 2.59.0 on loopback interfaces,
and ran Perf Analyzer. The batch-256/concurrency-4 point recorded 500,548
inferences/s with p99 request latency 2.087 ms.

Perf Analyzer used random contract-valid input. The job explicitly records
`semantic_parity_against_pytorch_established=false`; therefore this result is
service-performance smoke evidence, not a correctness or capacity approval.

## Formal candidate-profile decision

Slurm job `304890` ran on `gpumedium`, node `rg2122`, and completed `0:0` in
11 minutes for exact commit
`e4d3f321f8f93865a7e863c5db10e2fcb6b861a7`. It passed both source-provenance
preflights, built a target-side FP32 TensorRT plan with TF32 disabled, made
Triton 2.59.0 ready, and completed the required CPU, GPU, and parity stages with
telemetry sampled at 200 ms intervals.

Dependent Roihu-CPU finalizer job `304891` then verified `sacct`, built the
formal evidence envelope, rechecked all nine declared artifact hashes, and
exited `2` because the governed decision was `GPU_REJECTED`. Exit 2 is the
validator's defined rejection result, not a finalizer crash.

Observed formal evidence:

| Measure | CPU | GPU/Triton | Derived or policy result |
| --- | ---: | ---: | --- |
| Duration | 300.0006 s | 300.0011 s | both pass 300 s minimum |
| Successful requests | 323,661 | 892,291 | both pass 1,000 minimum |
| Row throughput | 69,047.54 rows/s | 190,354.74 rows/s | 2.756865x; pass >= 1.50x |
| p95 request latency | 1.1495 ms | 1.6296 ms | ratio 1.417711; fail <= 1.10 |
| HTTP error rate | n/a | 0.000004483 | pass <= 0.001 |
| Sustained GPU utilization | n/a | 22.7203% | fail governed 35%–90% band |
| Peak / total GPU memory | n/a | 683 / 97,871 MiB | 99.3021% headroom; pass |

Parity used 1,024 rows and 524,288 probability outputs from the same fixture.
Maximum absolute probability error was `5.6624e-6`, below the `5e-5` limit;
there were zero policy-decision mismatches and zero parity HTTP errors.

The decision failed exactly two checks:

1. `p95_ratio_slo`: GPU-to-CPU p95 ratio exceeded policy.
2. `sustained_gpu_utilization_range`: sustained utilization was below the
   governed efficiency band.

All other source, Slurm, hardware, software, duration, sample, throughput,
HTTP, parity, memory, artifact, and checksum controls passed. The decision
therefore correctly leaves:

- `gpu_profile_enabled=false`;
- `automatic_promotion_authorized=false`;
- current model accelerator and profile unchanged;
- the synthetic candidate workload unverified for GPU eligibility.

Decision SHA-256:
`4e12703fdcd62a0d1ea5ef3d8dad050884f1059f091fea95a5089e4d4ea392ee`

## Fail-closed learning record

The execution sequence also preserved these non-passing attempts:

| Job | Outcome | Finding and controlled response |
| --- | --- | --- |
| `304431` | failed | expected PyTorch module was unavailable under the first environment assumption |
| `304437` | failed after PyTorch stage | CSC PyTorch module lacked ONNX; added an exact ARM64, hash-locked offline wheelhouse |
| `304615` | failed | `trtexec` was not on container `PATH`; pinned the reviewed absolute executable path |
| `304674` | failed after Triton READY | Perf Analyzer rejected the long batch flag; changed to its supported `-b` flag |
| `304737`, `304764` | failed after target-side build/READY | a Slurm step could not import the workload; replaced environment-dependent lookup with an attested sibling-path import and added an actual-entrypoint preflight |
| `304754` | failed before execution | submission came from the login home rather than project scratch; resubmitted from the required controlled path |
| `304890` + `304891` | runtime pass, governed rejection | full evidence completed; validator rejected the inefficient latency/utilization profile and prevented promotion |

This history is useful evidence of operational judgement: infrastructure
readiness, higher throughput, or a green smoke result is insufficient when the
actual candidate profile misses latency and efficiency policy.
