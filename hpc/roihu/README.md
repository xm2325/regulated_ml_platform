# Roihu GH200 accelerator-qualification runtime

This pack creates **synthetic accelerator evidence** for the ML/AI
Engineer role's GPU, inference, observability, and controlled-delivery skills. It
does not run customer data or the platform champion model, and it cannot approve a
production promotion.

The target is CSC Roihu-GPU: NVIDIA GH200 (aarch64, CUDA compute capability 9.0).
The PyTorch exercise follows CSC's current single-GPU allocation shape: one GH200
and 72 CPU cores. See CSC's [PyTorch module](https://docs.csc.fi/apps/pytorch/),
[Roihu partitions](https://docs.csc.fi/computing/running/batch-job-partitions/),
[disk areas](https://docs.csc.fi/computing/roihu-disk/), and
[Apptainer GPU guidance](https://docs.csc.fi/computing/containers/overview/).

## Evidence flow

1. Create an immutable source archive whose single top-level directory embeds the
   full Git commit.
2. Stage code and small immutable inputs under `/projappl/project_2012997`; submit
   jobs and write results under `/scratch/project_2012997`.
3. Run the bounded `gputest` PyTorch smoke job. It compares the same deterministic MLP on
   Grace CPU FP32, GH200 CUDA FP32, and GH200 CUDA BF16; then exports those exact
   source weights to ONNX.
4. Optionally run the `gputest` Apptainer smoke job with pre-staged ARM64 Triton server
   and SDK SIFs. It compiles the ONNX model to a target-side TensorRT plan, starts
   Triton on loopback interfaces, and runs Perf Analyzer.
5. For a formal candidate-profile attempt, run the separate `gpumedium` full path,
   then a dependent CPU finalizer that verifies completed `sacct` state and invokes
   the governed evidence validator.
6. Review JSON status, raw logs, checksums, parity, utilization, and claim boundaries
   together. A failed or incomplete check never becomes a passing claim.

## 1. Prepare and stage the source

Run from a clean local or CI checkout at the exact commit to be exercised:

```bash
bundle_dir="/tmp/regulated-ml-roihu-source"
bash scripts/prepare_roihu_source_bundle.sh "${bundle_dir}"
python3 -m json.tool "${bundle_dir}/source-bundle.json"
```

The bundle helper rejects a dirty worktree and adds a full-commit marker. The archive
layout is a contract: `gputest_pytorch.sbatch` rejects archives without
exactly one `regulated_ml_platform-<full-40-character-commit>/` root, unsafe paths,
links, special files, excessive member counts, or excessive expanded size. Record
the SHA-256 from a trusted CI job before transfer. Do not put credentials, customer
data, or tokens in the archive or Slurm export list.

Transfer the archive, manifest, and checksum to a user-private project path on
`roihu-gpu.csc.fi`, for example:

```text
/projappl/project_2012997/<user>/regulated_ml_platform/v1.3.0/source/<archive>
```

The runtime validates the archive digest again on the compute node and executes
only the safely extracted copy in node-local `$TMPDIR`. Extract the `.sbatch` entrypoints
from that verified archive into the user's project scratch submit directory; do not
maintain a second mutable runtime copy.

## 2. Run the PyTorch GH200 exercise

Submit from a project scratch directory. Replace the digest and commit with the
recorded values; do not use abbreviated commits.

```bash
cd /scratch/project_2012997/regulated_ml_platform
sbatch \
  --account=project_2012997 \
  --export=ALL,SOURCE_ARCHIVE=/projappl/project_2012997/regulated_ml_platform/source/regulated_ml_platform-<commit>.tar.gz,SOURCE_SHA256=<64-hex>,SOURCE_GIT_COMMIT=<40-hex>,EVIDENCE_ROOT=/scratch/project_2012997/regulated_ml_platform/evidence \
  /projappl/project_2012997/regulated_ml_platform/runtime/hpc/roihu/gputest_pytorch.sbatch
```

`gputest_pytorch.sbatch` is intentionally capped at 15 minutes and marked
`SMOKE_ONLY`. It loads only
`python-pytorch/2.10`, sets `umask 077`, requests `gpu:gh200:1`, validates GH200 and
compute capability 9.0, samples `nvidia-smi` every second, and writes a unique
`pytorch-gh200-<job-id>/` evidence directory.

Important outputs include:

- `pytorch_benchmark.json`: warmup/repetition settings; p50/p95/p99 latency;
  steady-state throughput; CPU-FP32-to-CUDA parity; GPU peak allocated/reserved
  memory; Slurm, source, PyTorch, CUDA, cuDNN, GH200, and aarch64 provenance.
- `nvidia-smi-timeseries.csv` and `nvidia-smi-snapshot.csv`: timestamped utilization,
  memory, power, temperature, clocks, driver, GPU UUID, and compute capability.
- `onnx/accelerator_qualification.onnx`, `config.pbtxt`, and
  `export_manifest.json`: ONNX intermediate and Triton plan-backend contract for the
  exact benchmark weights.
- `cross_artifact_validation.json`: verifies that benchmark and export use the same
  workload name and weights SHA-256.
- `job_manifest.json`: compact artifact inventory with every file's SHA-256 and the
  job's fail-closed status.

The timing scope is steady-state model execution with model and inputs already on
the measured device. It intentionally excludes model loading and host-to-device
transfer. The CPU comparator is the 72-core NVIDIA Grace allocation paired with the
GH200, not Roihu's AMD CPU partition; describe it precisely and do not generalize it
into a universal CPU/GPU cost claim.

## 3. Optional TensorRT and Triton smoke

Pre-stage **ARM64/GH200-compatible** Triton server and SDK SIFs under the project's
`/scratch` area (recommended for large images) or `/projappl` if quota permits,
outside the batch job. Record both SHA-256 values. GPU containers
built for x86, Puhti, or Mahti are not valid on Roihu-GPU. The template never pulls
an image and uses `apptainer exec --cleanenv --containall --nv`.

From the passing PyTorch job, record SHA-256 values for
`accelerator_qualification.onnx` and `export_manifest.json`. Then submit:

```bash
cd /scratch/project_2012997/regulated_ml_platform
sbatch \
  --account=project_2012997 \
  --export=ALL,TRITON_SERVER_SIF=/projappl/project_2012997/containers/tritonserver-25.06-py3-arm64.sif,TRITON_SERVER_SIF_SHA256=<64-hex>,TRITON_SDK_SIF=/projappl/project_2012997/containers/tritonserver-25.06-py3-sdk-arm64.sif,TRITON_SDK_SIF_SHA256=<64-hex>,TRITON_CONTAINER_RELEASE=25.06,ONNX_PATH=/scratch/project_2012997/regulated_ml_platform/evidence/pytorch-gh200-<job-id>/onnx/accelerator_qualification.onnx,ONNX_SHA256=<64-hex>,EXPORT_MANIFEST_PATH=/scratch/project_2012997/regulated_ml_platform/evidence/pytorch-gh200-<job-id>/onnx/export_manifest.json,EXPORT_MANIFEST_SHA256=<64-hex>,SOURCE_GIT_COMMIT=<40-hex>,EVIDENCE_ROOT=/scratch/project_2012997/regulated_ml_platform/evidence,TRT_PRECISION=bf16 \
  /projappl/project_2012997/regulated_ml_platform/runtime/hpc/roihu/triton_tensorrt_apptainer.sbatch
```

The smoke template uses `gputest` for a bounded 15-minute engine-build and runtime
exercise. It emits `SMOKE_PASS`, never `GPU_ELIGIBLE`.

The job validates all SIF, ONNX, and export-manifest digests; verifies the export
commit and non-production workload class; requires aarch64/GH200/compute capability
9.0; builds `model.plan` with `trtexec`; starts Triton on `127.0.0.1` only; and runs
Perf Analyzer at batch sizes 1, 16, 64, and 256. It preserves raw server, TensorRT,
Perf Analyzer, Triton metrics, SIF inspection, and one-second `nvidia-smi` logs.

`runtime_evidence.json` deliberately has two separate Triton version fields:

- `triton_server_version`: Triton's runtime/API 2.x version from `/v2` metadata.
- `triton_container_release`: the NGC year.month release (for example `25.06`) tied
  to digest-verified SIF metadata.

It also records TensorRT and CUDA versions from raw logs. A TensorRT plan is tied to
the target GPU/software stack and is not claimed portable. Perf Analyzer uses random
contract-valid input, so this stage does not establish semantic parity against
PyTorch; that remains an explicit blocked claim.

## 4. Formal `gpumedium` candidate-profile attempt

The formal path is deliberately separate from both smoke jobs. It fixes TensorRT to
FP32 with TF32 disabled, uses the exact deterministic CPU fixture over binary Triton
HTTP, runs CPU and GPU windows for at least 300 seconds each, requires at least 1000
timed samples per path and 1024 parity rows, compares probability outputs and the
0.5 policy decision, and samples `nvidia-smi` every 200 ms during the GPU window.

Submit from Roihu-GPU with the same immutable inputs used above. This example keeps
large SIFs on scratch and captures the numeric job ID:

```bash
cd /scratch/project_2012997/regulated_ml_platform
qualification_job_id="$(sbatch --parsable \
  --account=project_2012997 \
  --export=ALL,SOURCE_ARCHIVE=/projappl/project_2012997/regulated_ml_platform/source/regulated_ml_platform-<commit>.tar.gz,SOURCE_SHA256=<64-hex>,SOURCE_GIT_COMMIT=<40-hex>,TRITON_SERVER_SIF=/scratch/project_2012997/containers/tritonserver-25.06-py3-arm64.sif,TRITON_SERVER_SIF_SHA256=<64-hex>,TRITON_SDK_SIF=/scratch/project_2012997/containers/tritonserver-25.06-py3-sdk-arm64.sif,TRITON_SDK_SIF_SHA256=<64-hex>,TRITON_CONTAINER_RELEASE=25.06,ONNX_PATH=/scratch/project_2012997/regulated_ml_platform/evidence/pytorch-gh200-<smoke-job-id>/onnx/accelerator_qualification.onnx,ONNX_SHA256=<64-hex>,EXPORT_MANIFEST_PATH=/scratch/project_2012997/regulated_ml_platform/evidence/pytorch-gh200-<smoke-job-id>/onnx/export_manifest.json,EXPORT_MANIFEST_SHA256=<64-hex>,EVIDENCE_ROOT=/scratch/project_2012997/regulated_ml_platform/evidence \
  /projappl/project_2012997/regulated_ml_platform/runtime/hpc/roihu/gpumedium_full_qualification.sbatch)"
printf '%s\n' "${qualification_job_id}"
```

The runtime job cannot truthfully declare itself `COMPLETED` while it is still
running. It therefore emits raw evidence and `AWAITING_COMPLETED_SACCT`, not an
eligibility decision. From Roihu-CPU, submit the lightweight finalizer with an
`afterok` dependency (replace the commit and job ID values in the export list):

```bash
evidence_dir="/scratch/project_2012997/regulated_ml_platform/evidence/full-qualification-${qualification_job_id}"
sbatch \
  --account=project_2012997 \
  --dependency="afterok:${qualification_job_id}" \
  --export=ALL,QUALIFICATION_JOB_ID=${qualification_job_id},QUALIFICATION_EVIDENCE_DIR=${evidence_dir},SOURCE_GIT_COMMIT=<40-hex> \
  /projappl/project_2012997/regulated_ml_platform/runtime/hpc/roihu/finalize_roihu_qualification.sbatch
```

The finalizer queries `sacct`, builds `manifest.json` and `benchmark.json` in
`regulated-ml-platform.roihu-gpu-evidence/v1`, writes `SHA256SUMS`, and invokes the
staged `src.operations.roihu_gpu_evidence` validator to produce `decision.json`.
The validator may still return `GPU_REJECTED`: duration, samples, HTTP errors,
speedup, p95, parity, utilization, memory headroom, software versions, Slurm state,
or any digest can fail policy. Even `GPU_ELIGIBLE` applies only to this isolated
candidate workload and exact artifacts; it does not promote the current champion.

## Interview practice

Use the artifacts to explain, with exact boundaries:

- why warmups, repeated samples, p50/p95/p99, throughput, and one-second GPU
  telemetry answer different operational questions;
- why BF16 needs a separate numerical-parity gate even when it improves throughput;
- how source, SIF, ONNX, config, weights, and engine digests prevent accidental
  artifact substitution;
- why TensorRT compilation occurs on GH200/aarch64 and why plans are not portable;
- how loopback-only serving, private file modes, bounded queues, cleanup traps, and
  no in-job downloads reduce operational risk;
- why evidence can qualify an accelerator path without authorizing production
  capacity, champion-model promotion, or customer-data processing;
- what must come next: champion-model export, reviewed semantic fixtures, production
  SLO/error-budget policy, representative traffic, resilience/canary/rollback tests,
  and independent approval.
