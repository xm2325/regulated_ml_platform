# Roihu JD gap lab: profiling and same-node right-sizing

This lab extends the v1.3 formal qualification without changing its result. The
parent formal job `304890` and CPU finalizer `304891` remain `GPU_REJECTED`.
The new path asks a narrower engineering question: why did that small synthetic
candidate under-utilise a GH200, and what happens when the same TensorRT plan is
scheduled across one, two, or four GH200s on one Roihu node?

## Claim boundary

- Every run is `SMOKE_ONLY`.
- The existing model plan is reused only after its SHA-256 is verified.
- Perf Analyzer input is random and contract-valid. The lab does not re-establish
  semantic parity and cannot replace the formal decision.
- A short same-node comparison is not a production SLO, cloud cost model,
  Kubernetes GPU-fleet test, or authorization to allocate more GPUs.
- Results remain tied to the exact source archive, plan, config, SIFs, Slurm
  jobs, GH200 node, batch, concurrency, and software versions.

## What the job exercises

`hpc/roihu/triton_profile_rightsizing.sbatch` supports three declared points:

| Point | Slurm override | Purpose |
| --- | --- | --- |
| 1 GH200 | defaults | baseline; optionally collect a 45-second Nsight Systems trace |
| 2 GH200 | `--gres=gpu:gh200:2 --cpus-per-task=144` | same-node scheduling and scaling |
| 4 GH200 | `--gres=gpu:gh200:4 --cpus-per-task=288` | full-node scheduling and scaling |

Each GPU receives one loopback-only Triton instance and one independent Perf
Analyzer client at batch 64 and concurrency 4. The job samples every allocated
GPU at 200 ms, captures Triton Prometheus metrics, and produces a fail-closed
`summary.json`. The one-GPU profiler additionally exports:

- `nsys-profile.nsys-rep`;
- CUDA kernel summary;
- CUDA API summary;
- CUDA memory-operation summary;
- OS runtime summary.

The job performs no network download and accepts only project-owned files below
`/projappl/<account>` or `/scratch/<account>`.

## Submission contract

Create and transfer an immutable source bundle from a clean, committed branch.
Use the verified v1.3 formal plan and its exact config as immutable parents.
Example paths below reflect the current project layout; every digest must be
calculated from the named file before submission.

```bash
export PROJECT=project_2012997
export USER_ROOT=/scratch/${PROJECT}/${USER}/regulated_ml_platform
export V13=${USER_ROOT}/v1.3.0
export V14=${USER_ROOT}/v1.4.0

export SOURCE_ARCHIVE=/projappl/${PROJECT}/${USER}/regulated_ml_platform/v1.4.0/source/regulated-ml-platform-<commit>.tar.gz
export SOURCE_SHA256=<sha256>
export SOURCE_GIT_COMMIT=<full-commit>
export TRITON_SERVER_SIF=${V13}/containers/tritonserver-25.06-py3-arm64.sif
export TRITON_SERVER_SIF_SHA256=34e1d04b8c4956a3bff13b0a65188c42297d6f2d81afc58dbc32d360cc199137
export TRITON_SDK_SIF=${V13}/containers/tritonserver-25.06-py3-sdk-arm64.sif
export TRITON_SDK_SIF_SHA256=6813718d8b4f50012b6e70012570fb28c17f8c502660027d202c4454a6415772
export MODEL_PLAN=${V13}/evidence/full-qualification-304890/release/model.plan
export MODEL_PLAN_SHA256=<sha256>
export TRITON_CONFIG_PATH=${V13}/evidence/pytorch-gh200-304885/onnx/config.pbtxt
export TRITON_CONFIG_SHA256=eb167a921b080a567921d57304f35463412bd13de187765b24108f9ef0c78698
export EVIDENCE_ROOT=${V14}/evidence
export PARENT_FORMAL_JOB_ID=304890
```

Submit from a private project scratch directory. The entrypoint itself should be
extracted from the verified archive rather than maintained as a second mutable
copy.

```bash
common_exports=ALL,SOURCE_ARCHIVE=${SOURCE_ARCHIVE},SOURCE_SHA256=${SOURCE_SHA256},SOURCE_GIT_COMMIT=${SOURCE_GIT_COMMIT},TRITON_SERVER_SIF=${TRITON_SERVER_SIF},TRITON_SERVER_SIF_SHA256=${TRITON_SERVER_SIF_SHA256},TRITON_SDK_SIF=${TRITON_SDK_SIF},TRITON_SDK_SIF_SHA256=${TRITON_SDK_SIF_SHA256},MODEL_PLAN=${MODEL_PLAN},MODEL_PLAN_SHA256=${MODEL_PLAN_SHA256},TRITON_CONFIG_PATH=${TRITON_CONFIG_PATH},TRITON_CONFIG_SHA256=${TRITON_CONFIG_SHA256},EVIDENCE_ROOT=${EVIDENCE_ROOT},PARENT_FORMAL_JOB_ID=${PARENT_FORMAL_JOB_ID}

sbatch --export=${common_exports},GPU_COUNT=1,ENABLE_NSYS=1 triton_profile_rightsizing.sbatch
sbatch --gres=gpu:gh200:2 --cpus-per-task=144 \
  --export=${common_exports},GPU_COUNT=2,ENABLE_NSYS=0 triton_profile_rightsizing.sbatch
sbatch --gres=gpu:gh200:4 --cpus-per-task=288 \
  --export=${common_exports},GPU_COUNT=4,ENABLE_NSYS=0 triton_profile_rightsizing.sbatch
```

After all three jobs have reached terminal `sacct` state, aggregate only successful
summaries:

```bash
python3 hpc/roihu/summarize_triton_rightsizing.py aggregate \
  --summaries \
    ${EVIDENCE_ROOT}/triton-rightsizing-1gpu-<job>/summary.json \
    ${EVIDENCE_ROOT}/triton-rightsizing-2gpu-<job>/summary.json \
    ${EVIDENCE_ROOT}/triton-rightsizing-4gpu-<job>/summary.json \
  --output ${EVIDENCE_ROOT}/triton-rightsizing-aggregate.json
```

## Executed 1/2/4-GH200 result

The source-bound run for commit
`17ce7f9ae7eb104eb7c95c02cf6e5dff560c909f` used source archive SHA-256
`5edf8c27494a8d3357255eedfa2ede2386a58b25fb653a215aa3374977d56589`.
All three jobs completed `0:0` on `gputest`:

| GPUs | Slurm job | Total infer/s | Speedup | Scaling efficiency | Worst p95 | Mean GPU utilisation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `318684` | 396,132 | 1.000x | 100.0% | 0.736 ms | 44.57% |
| 2 | `318690` | 873,294 | 2.205x | 110.23% | 0.598 ms | 48.58% |
| 4 | `318694` | 1,608,875 | 4.061x | 101.54% | 0.987 ms | 45.92% |

The small apparent super-linear points are short-run observations, not a
capacity or cost claim. Cache state, run-to-run variation, server placement, and
the synthetic request stream were not controlled sufficiently to infer
super-linear production scaling. The four-GPU point also exposed imbalance:
individual server throughput ranged from 357,815 to 430,602 infer/s, and its
worst p95 was higher than the one- and two-GPU points.

The one-GPU Nsight trace showed that 78.6% of captured CUDA kernel time was in
the main TensorRT GEMM kernel and another 16.4% in its split-K kernel. Within
captured CUDA API time, `cudaEventSynchronize` accounted for 36.8%,
`cuLaunchKernelEx` for 29.6%, and `cudaMemcpyAsync` for 13.1%. This is a useful
profiling diagnosis: the workload is dominated by many small GEMM launches plus
synchronisation and transfer overhead, rather than memory capacity. Peak memory
was only 731 MiB of 97,871 MiB on the one-GPU point.

The aggregate evidence passed its source, plan, and run-status checks. It remains
`SMOKE_ONLY`; formal job `304890` stays `GPU_REJECTED`.

Key retained evidence:

- aggregate:
  `/scratch/project_2012997/xiaomei/regulated_ml_platform/v1.4.0/evidence/triton-rightsizing-aggregate-17ce7f9.json`,
  SHA-256 `f397fb8ad1b4adc7b6ddceadcda02fe8d58bb3be992e0c426622ebbdbb3ff7ba`;
- Slurm receipt:
  `/scratch/project_2012997/xiaomei/regulated_ml_platform/v1.4.0/evidence/triton-rightsizing-sacct-17ce7f9.psv`,
  SHA-256 `1d3d8d2e7687b0f405b0c5a208b76892521683b538ce3118f7e3386f4bd4d1a4`;
- Nsight report from job `318684`, SHA-256
  `74f6126483587fb95c79b424f35df41e6798e5c7c6145256e0f60dada1e7dc41`.

## Archived TorchServe compatibility gate

The JD names TorchServe, so the lab also tested whether its final 0.12.0 release
could run a bounded GPU inference smoke on Roihu. This is a compatibility
investigation, not a recommendation to adopt TorchServe: the upstream
[repository is archived](https://github.com/pytorch/serve), it states that
maintenance is limited with no planned security fixes, and the official
[troubleshooting guide requires Java 17](https://docs.pytorch.org/serve/Troubleshooting.html).
Roihu exposed OpenJDK 25 rather than the documented Java version.

The gate still established several facts under an offline, source-bound
contract:

- TorchServe and model-archiver 0.12.0 wheels installed with no network access;
- PyTorch 2.10.0+cu130 detected the assigned GH200;
- a staged Temurin OpenJDK 25.0.1 file manifest verified before execution;
- the TorchServe Java frontend started on loopback and `/ping` returned;
- token authorization was disabled only on loopback, while the management model
  API remained disabled;
- prediction success and CUDA execution were mandatory, so frontend readiness
  alone could not produce `SMOKE_PASS`.

Preflight job `318733` completed successfully. Runtime attempts then narrowed
the incompatibility:

| Job | Outcome | Finding |
| ---: | --- | --- |
| `318896` | fail closed | Java was not visible under the initial module assumption |
| `318976` | fail closed | the module Java path was not executable from the TorchServe process |
| `318999` | bounded cancellation | a staged project Java started the frontend, but the worker failed to spawn from node-local temporary storage |
| `319007` | bounded cancellation | Java's explicit `FORK` launcher did not remove worker spawn error 107 |
| `319021` | `FAILED 1:0` after 2:56 | project-scratch work and temp paths still produced 28 worker spawn error-107 events and 28 HTTP 503 predictions |

Job `319021` used source commit
`79c7385eac9ef60211d49d5dca3b73a04afb5031` and source archive SHA-256
`d110c0c83cd63996d1a87d9132f391b3dc60af5448efc4b2961850927c24f98c`.
The TorchServe wheel SHA-256 was
`db127160102d29f390964f758b7ecc5039d3d278fafc85bf9994c273b3ef6954`;
the staged JDK manifest SHA-256 was
`6a4a39d45a4900a0cb56113d684eb60b8be842383e0bcf31f1184359d4e5c053`.
The retained frontend log SHA-256 is
`dbaced0758f63b87bd18cbb63bdb0baf48e9fc2d3962f3751e9473bbcc1d60d4`,
and the Slurm output SHA-256 is
`280a40c6fb7ad2955df23d20aff03478c4499d322a43ce9c11b1db311fcd12fe`.

The correct claim is therefore: the candidate implemented and ran a
fail-closed TorchServe compatibility gate, proved offline toolchain/frontend
readiness, diagnosed an unsupported-Java worker-launch incompatibility, and did
not claim GPU inference success. A future retry should use a supported Java 17
runtime in a maintained image or replace the archived server; repeating the
same Java 25/Roihu combination is not justified.

## Interview use

The strongest answer is not “more GPUs were faster.” Explain:

1. what Nsight showed about kernel time, launch overhead, memory operations, and
   CPU/runtime waiting;
2. whether total throughput scaled and whether per-GPU efficiency fell;
3. why a small workload can be a poor GPU or multi-GPU candidate;
4. why the short evidence cannot override formal latency/utilisation policy;
5. how the same exercise would differ under Kubernetes node pools, device
   plugins, autoscaling, service mesh, representative traffic, and cost controls.
