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

The JD names TorchServe, so the lab tested whether its final 0.12.0 release
could run a bounded GPU inference smoke on Roihu. This is a compatibility
investigation, not a recommendation to adopt TorchServe: the upstream
[repository is archived](https://github.com/pytorch/serve), it states that
maintenance is limited with no planned security fixes, and the official
[troubleshooting guide requires Java 17](https://docs.pytorch.org/serve/Troubleshooting.html).

The final path used a staged Temurin 17.0.19 ARM64 runtime and an NVIDIA PyTorch
24.02 ARM64 SIF under Apptainer `--cleanenv --nv`. The source archive, SIF,
OpenJDK file manifest, TorchServe wheel, and model-archiver wheel were all
SHA-256 verified before execution. Wheel installation was offline and
job-scoped. TorchServe listened only on loopback, the management API remained
disabled, and prediction, CUDA, GH200 telemetry, and Prometheus metrics were all
mandatory for `SMOKE_PASS`.

The fail-closed sequence first isolated the host boundary and then completed the
container path:

| Job | Outcome | Finding |
| ---: | --- | --- |
| `318896` | fail closed | Java was not visible under the initial module assumption |
| `318976` | fail closed | the module Java path was not executable from the TorchServe process |
| `318999` | bounded cancellation | a staged project Java started the frontend, but the worker failed to spawn from node-local temporary storage |
| `319007` | bounded cancellation | Java's explicit `FORK` launcher did not remove worker spawn error 107 |
| `319021` | `FAILED 1:0` after 2:56 | project-scratch work and temp paths still produced 28 worker spawn error-107 events and 28 HTTP 503 predictions |
| `319828` | `FAILED 1:0` | exact Temurin 17 host retry proved Java 17 alone did not remove the Roihu host worker-launch boundary |
| `320648`, `320655` | fail closed | container entry isolated missing `ensurepip` and Debian `pip --prefix` script-layout assumptions |
| `320679`, `320867` | bounded cancellation | node-local short sockets fixed AF_UNIX length; handler initialization was then corrected |
| `320871` | fail closed after 40 CUDA predictions | inference passed, but default log-mode metrics correctly failed the Prometheus gate |
| `321067` | `COMPLETED 0:0` in 22 seconds | Java 17 frontend, worker, 40 HTTP 200 predictions, CUDA response, GH200 telemetry, and Prometheus metrics passed |

Job `321067` used source commit
`ce6060264764beb07100976ac7608f71dd66cfd3`, source archive SHA-256
`835c930ac5f2b5691fa2915d2019c5230a487ff417f8e6014c1877d291e2d24e`,
TorchServe wheel SHA-256
`db127160102d29f390964f758b7ecc5039d3d278fafc85bf9994c273b3ef6954`,
model-archiver wheel SHA-256
`baaf66065396c3512030b3b2c57cce333edab9fffe9e528352cb4cc291645a78`,
Temurin manifest SHA-256
`63e3e87582d1f25e012f08fb391dcb9ba454b51b19d06051a62ac089d2b0d455`,
and PyTorch SIF SHA-256
`9b224b3d66800174e34f375d4fc17086d66b929f1870136d5ab79ecea2797cb5`.
The `SMOKE_PASS` summary SHA-256 is
`b0a4e0d77eea5dd65dd812c1d504950caf9faac691e4c527a67ff44f1a21bfcd`;
Prometheus evidence SHA-256 is
`b5d6ef4e4ee4aad0075c178ea965d746a956d557538e0fbe86576b08e0639f8b`.

The response reported `device=cuda`, batch 256, output shape `[256,16]`; 13
telemetry samples confirmed an NVIDIA GH200 120GB and peak allocated memory of
797 MiB. These values prove only a synthetic compatibility path. They do not
support performance, production capacity, security approval, maintained-runtime
suitability, or a recommendation to use archived TorchServe instead of Triton.

## Interview use

The strongest answer is not “more GPUs were faster.” Explain:

1. what Nsight showed about kernel time, launch overhead, memory operations, and
   CPU/runtime waiting;
2. whether total throughput scaled and whether per-GPU efficiency fell;
3. why a small workload can be a poor GPU or multi-GPU candidate;
4. why the short evidence cannot override formal latency/utilisation policy;
5. how the same exercise would differ under Kubernetes node pools, device
   plugins, autoscaling, service mesh, representative traffic, and cost controls.
