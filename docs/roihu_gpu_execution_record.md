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

## v1.4 profiling and same-node scaling exercise

The follow-up lab answered a narrower engineering question without changing the
formal decision: how does the rejected synthetic TensorRT plan behave under
Nsight profiling and one, two, and four independent Triton instances on one
Roihu GH200 node?

Exact runtime source:

- commit: `17ce7f9ae7eb104eb7c95c02cf6e5dff560c909f`;
- source archive SHA-256:
  `5edf8c27494a8d3357255eedfa2ede2386a58b25fb653a215aa3374977d56589`;
- parent plan SHA-256:
  `46f414ab796163f2e269e07b802735b0ef853e4fe8260c89bac33eeeb2c13abd`;
- GitHub Roihu contract:
  [run 30009063991](https://github.com/xm2325/regulated_ml_platform/actions/runs/30009063991).

Each `gputest` job used 72 Grace CPU cores per GH200, one loopback-only Triton
server per GPU, batch 64, concurrency 4 per server, and 200 ms GPU telemetry.
All three jobs completed `0:0`:

| GPUs | Job | Elapsed | Total infer/s | Speedup | Efficiency | Worst p95 | Mean GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `318684` | 1:53 | 396,132 | 1.000x | 100.0% | 0.736 ms | 44.57% |
| 2 | `318690` | 0:38 | 873,294 | 2.205x | 110.23% | 0.598 ms | 48.58% |
| 4 | `318694` | 0:41 | 1,608,875 | 4.061x | 101.54% | 0.987 ms | 45.92% |

The apparent super-linear efficiencies are not a production conclusion: these
are short independent smoke points and do not control cache state, placement,
thermal state, or representative traffic. The four-server result also ranged
from 357,815 to 430,602 infer/s per GPU and had the highest worst-server p95,
which is a reason to investigate placement and load balance before any capacity
claim.

The one-GPU Nsight trace retained 232,527 instances of the main TensorRT GEMM
kernel and the same number of split-K instances. The main GEMM accounted for
78.6% of captured CUDA kernel time and split-K for 16.4%. In the CUDA API
summary, event synchronisation represented 36.8% of captured API time, kernel
launches 29.6%, and asynchronous copies 13.1%. Peak allocated GPU memory was
731 MiB out of 97,871 MiB. The practical diagnosis is many small GEMM launches
with meaningful synchronisation/transfer overhead, not a memory-capacity
bottleneck.

Retained receipts:

- aggregate SHA-256:
  `f397fb8ad1b4adc7b6ddceadcda02fe8d58bb3be992e0c426622ebbdbb3ff7ba`;
- `sacct` receipt SHA-256:
  `1d3d8d2e7687b0f405b0c5a208b76892521683b538ce3118f7e3386f4bd4d1a4`;
- one-GPU Nsight report SHA-256:
  `74f6126483587fb95c79b424f35df41e6798e5c7c6145256e0f60dada1e7dc41`.

The aggregate status is `PASS` for this `SMOKE_ONLY` exercise. It explicitly
forbids automatic right-sizing and production-capacity claims, does not
re-establish semantic parity, and cannot alter the parent `GPU_REJECTED`
decision.

## TorchServe compatibility result

The v1.4 follow-up also implemented a bounded, offline TorchServe 0.12.0 GPU
compatibility gate because the target JD names TorchServe. It did not relax the
project's evidence policy: a healthy Java frontend was insufficient without a
successful CUDA worker response.

Preflight job `318733` completed the wheel install and confirmed PyTorch
2.10.0+cu130, CUDA 13, the assigned GH200, TorchServe 0.12.0, and staged
Temurin OpenJDK 25.0.1. Final job `319021` used exact source commit
`79c7385eac9ef60211d49d5dca3b73a04afb5031`, moved both the virtual environment
and temporary model extraction to project scratch, and forced Java's `FORK`
process launcher. TorchServe answered its loopback `/ping`, but its Java process
could not spawn the Python model worker:

```text
Exec failed, error: 107 (Transport endpoint is not connected)
```

The job recorded 28 worker-spawn errors and 28 HTTP 503 prediction responses,
then failed closed after 2:56 as `FAILED 1:0`. No `summary.json` or
`SMOKE_PASS` was emitted. This matters because the archived TorchServe project
documents Java 17, whereas the available Roihu module/staged environment was
Java 25; the experiment does not establish TorchServe inference compatibility.

Retained checksums:

- source archive:
  `d110c0c83cd63996d1a87d9132f391b3dc60af5448efc4b2961850927c24f98c`;
- TorchServe wheel:
  `db127160102d29f390964f758b7ecc5039d3d278fafc85bf9994c273b3ef6954`;
- staged JDK manifest:
  `6a4a39d45a4900a0cb56113d684eb60b8be842383e0bcf31f1184359d4e5c053`;
- job `319021` TorchServe log:
  `dbaced0758f63b87bd18cbb63bdb0baf48e9fc2d3962f3751e9473bbcc1d60d4`;
- job `319021` Slurm output:
  `280a40c6fb7ad2955df23d20aff03478c4499d322a43ce9c11b1db311fcd12fe`.

This is useful hands-on compatibility and diagnosis evidence, but it is not a
TorchServe inference, throughput, security, maintenance, or production claim.
The next legitimate experiment requires Java 17 in a controlled image or a
maintained serving alternative.

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
| `318645` | failed during profiled serving | Nsight exhausted the container's 63 MiB `/tmp`; moved profiler temporary data to project scratch and changed telemetry to an unbuffered append path |
| `318663` | workload and profiling completed, finalisation failed | Nsight 2025.3 appended report identifiers to CSV filenames; changed finalisation to require exactly one matching kernel/API report and kept the failed evidence directory |
| `318684`, `318690`, `318694` | smoke exercise passed | exact corrected source completed the one-, two-, and four-GH200 points; aggregate retained the formal rejection and production claim boundary |
| `318733` | TorchServe preflight passed | offline wheels, staged Java, PyTorch/CUDA and the assigned GH200 were visible; no inference claim was made |
| `318896`, `318976` | TorchServe failed closed | isolated Java visibility and process-execution assumptions before staging a project-local runtime |
| `318999`, `319007` | bounded cancellation | frontend was ready, but worker spawn error 107 persisted across node-local temporary storage and Java's explicit `FORK` launcher |
| `319021` | TorchServe failed closed | project-scratch work/temp paths still produced 28 worker spawn errors and 28 HTTP 503s; unsupported Java 25 compatibility remains unproved |

This history is useful evidence of operational judgement: infrastructure
readiness, higher throughput, or a green smoke result is insufficient when the
actual candidate profile misses latency and efficiency policy.
