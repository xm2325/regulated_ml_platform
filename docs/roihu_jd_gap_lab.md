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

## Interview use

The strongest answer is not “more GPUs were faster.” Explain:

1. what Nsight showed about kernel time, launch overhead, memory operations, and
   CPU/runtime waiting;
2. whether total throughput scaled and whether per-GPU efficiency fell;
3. why a small workload can be a poor GPU or multi-GPU candidate;
4. why the short evidence cannot override formal latency/utilisation policy;
5. how the same exercise would differ under Kubernetes node pools, device
   plugins, autoscaling, service mesh, representative traffic, and cost controls.
