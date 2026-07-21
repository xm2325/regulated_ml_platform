# v1.3 CSC Roihu GH200 GPU 实操与证据手册

> 状态：可执行 runbook 与 fail-closed 证据契约。真实运行结果单独记录在 [CSC Roihu GH200 execution record](roihu_gpu_execution_record.md)；正式候选 profile 的当前结论是 `GPU_REJECTED`，不是生产批准。

本手册把 Lloyds ML/AI Engineer 职位中的 GPU 推理、性能验证、可观测性、受控交付和回滚思维，落实为一条可审计的 CSC Roihu 实操路径。目标硬件是 Roihu-GPU 的 NVIDIA Grace Hopper GH200：GPU 节点为 `aarch64`，每个完整 GH200 配套最多 72 个 Grace CPU 核。这里的练习对象是确定性的合成神经网络候选，不是当前 tree-ensemble champion。

必须始终保留以下边界：

- 当前 tree-ensemble champion 仍为 `CPU_ONLY`；不能把本练习的结果写成当前模型 GPU 加速。
- `gputest` 只做 15 分钟 smoke，不满足正式 `GPU_ELIGIBLE` 门禁；正式分区证据必须来自 `gpumedium` 或政策允许的 `gpularge`。
- `gpumedium` TensorRT/Triton 练习仍是候选 accelerator qualification，不是生产容量、客户流量、自动 promotion 或语义正确性的批准。
- Roihu 使用 Slurm 和 Apptainer，不是 Kubernetes。Helm/KEDA 模板只是把证据映射到另一套生产交付契约，不能声称已部署到 Roihu Kubernetes。
- GitHub Actions 使用 `ubuntu-latest` CPU runner，只验证代码、证据 schema、validator 和不可变 source bundle；绿色 workflow 不执行也不证明真实 GPU。

## 1. 能力目标与职位对应

| 实操 | 可证明的能力 | 不能据此声称 |
| --- | --- | --- |
| 不可变 source bundle、完整 Git SHA、逐文件 SHA-256 | artifact provenance、受控 promotion、可复现交付 | 已部署生产模型 |
| `gputest` 的 PyTorch CPU/CUDA FP32/BF16 对比 | CUDA 设备检查、数值 parity、warmup/重复采样、p50/p95/p99、显存与利用率观测 | 正式 GPU eligibility 或生产容量 |
| `gpumedium` 的 Apptainer `--nv`、TensorRT plan、Triton、Perf Analyzer | GH200/aarch64 target-side build、推理服务健康检查、批量/并发测试、Prometheus 指标 | 随机输入建立了业务语义 parity |
| fail-closed validator | governance、证据完整性、阈值与来源校验、无自动 promotion | 单个 `runtime_evidence=real_gpu` 字符串足够可信 |
| Helm PDB/NetworkPolicy/Quota/ServiceMonitor/PrometheusRule/KEDA | Kubernetes GPU 交付、隔离、HA、监控和有界 autoscaling 设计 | Roihu/Slurm 就是 Kubernetes，或这些资源已在真实集群运行 |

## 2. 账户、24 小时 SSH certificate 与正确登录入口

前置条件是 CSC 账户、已启用 Roihu 服务的项目、已接受 Roihu 使用条款，并将 SSH public key 加入 MyCSC。Roihu 还要求给 public key 签发 SSH certificate；certificate 每 24 小时到期，需要重新签发。

推荐流程：

1. 在 MyCSC 的 Profile 中找到 SSH public key，执行 **Sign and download SSH certificate**。
2. 私钥若是 `~/.ssh/id_ed25519`，certificate 保存为 `~/.ssh/id_ed25519-cert.pub`。
3. 在本机确认 certificate 有效期，再登录 GPU 侧：

```bash
ssh-keygen -L -f ~/.ssh/id_ed25519-cert.pub
export CSC_USER=your_csc_username
ssh -i ~/.ssh/id_ed25519 "${CSC_USER}@roihu-gpu.csc.fi"
```

4. 登录后确认服务、项目和架构：

```bash
csc-projects
csc-workspaces
hostname
uname -m
```

`uname -m` 应为 `aarch64`。GPU 作业和 ARM64 容器准备必须从 `roihu-gpu.csc.fi` 侧完成；不要在 login node 上直接跑 benchmark，计算必须交给 Slurm。

CSC 官方参考：

- [How to get access to Roihu](https://docs.csc.fi/support/faq/how-to-get-roihu-access/)
- [Setting up and signing SSH keys](https://docs.csc.fi/computing/connecting/ssh-keys/)
- [Connecting to CSC systems](https://docs.csc.fi/computing/connecting/)
- [Roihu system description](https://docs.csc.fi/computing/systems-roihu/)

## 3. 存储与数据边界

以下示例使用当前脚本的默认项目 `project_2012997`；若实际账户不同，用真实 `project_<digits>` 替换，并在 `sbatch --account`、`/projappl` 和 `/scratch` 中保持一致。

```bash
export PROJECT=project_2012997
export APP_ROOT="/projappl/${PROJECT}/${USER}/regulated_ml_platform/v1.3.0"
export RUN_ROOT="/scratch/${PROJECT}/${USER}/regulated_ml_platform/v1.3.0"
mkdir -p "${APP_ROOT}/source"
mkdir -p "${RUN_ROOT}/submit" "${RUN_ROOT}/evidence" "${RUN_ROOT}/containers"
```

- `/projappl/${PROJECT}`：存放小型、较长期的 source bundle 和参考配置；配额有限，不在这里写大批运行输出。
- `/scratch/${PROJECT}`：提交 Slurm、保存大型 SIF、ONNX、TensorRT、Triton 和 telemetry 证据；它是临时工作区，当前清理策略会处理长期未访问数据。
- `$TMPDIR`：作业内的 node-local 临时空间。两个 sbatch 脚本都在这里安全解包或构建，退出时清理。
- 不使用客户数据、credentials、token 或敏感数据。这个练习只生成确定性的合成输入。

CSC 官方参考：[Roihu disk areas](https://docs.csc.fi/computing/roihu-disk/)。CSC 不为这些区域自动备份；保留证据前应复制到受控存储。

## 4. 在可信 checkout 生成不可变 source bundle

必须先提交所有预期代码。脚本会拒绝 dirty worktree，并把完整 40 字符 Git commit 写进唯一 archive root：

```bash
git status --porcelain=v1 --untracked-files=all
export BUNDLE_DIR="/tmp/regulated-ml-roihu-source"
bash scripts/prepare_roihu_source_bundle.sh "${BUNDLE_DIR}"
python3 -m json.tool "${BUNDLE_DIR}/source-bundle.json"
```

输出包括：

- `regulated-ml-platform-<40-char-commit>.tar.gz`
- 同名 `.sha256`
- `source-bundle.json`
- archive 内唯一顶层目录 `regulated_ml_platform-<40-char-commit>/`
- 顶层 marker `.regulated-ml-source-commit`

把 bundle 复制到 Roihu 的 `/projappl`。示例从本机执行：

```bash
export CSC_USER=your_csc_username
export PROJECT=project_2012997
ssh "${CSC_USER}@roihu-gpu.csc.fi" \
  "mkdir -p /projappl/${PROJECT}/${CSC_USER}/regulated_ml_platform/v1.3.0/source"
rsync -av "${BUNDLE_DIR}/" \
  "${CSC_USER}@roihu-gpu.csc.fi:/projappl/${PROJECT}/${CSC_USER}/regulated_ml_platform/v1.3.0/source/"
```

回到 Roihu-GPU login node，读取 manifest 并再次验证 archive。不要手工重新输入 SHA：

```bash
export PROJECT=project_2012997
export APP_ROOT="/projappl/${PROJECT}/${USER}/regulated_ml_platform/v1.3.0"
export RUN_ROOT="/scratch/${PROJECT}/${USER}/regulated_ml_platform/v1.3.0"
export SOURCE_MANIFEST="${APP_ROOT}/source/source-bundle.json"
export SOURCE_GIT_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "${SOURCE_MANIFEST}")"
export SOURCE_ARCHIVE_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive_filename"])' "${SOURCE_MANIFEST}")"
export SOURCE_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive_sha256"])' "${SOURCE_MANIFEST}")"
export SOURCE_ARCHIVE="${APP_ROOT}/source/${SOURCE_ARCHIVE_NAME}"
export ARCHIVE_ROOT="regulated_ml_platform-${SOURCE_GIT_COMMIT}"

cd "${APP_ROOT}/source"
sha256sum --check "${SOURCE_ARCHIVE_NAME}.sha256"
tar -tzf "${SOURCE_ARCHIVE}" | sed -n '1,20p'
```

为避免提交一个来源不明的外部脚本，从已验证 archive 提取 sbatch 文件到 `/scratch`：

```bash
mkdir -p "${RUN_ROOT}/submit" "${RUN_ROOT}/evidence"
tar -xOzf "${SOURCE_ARCHIVE}" \
  "${ARCHIVE_ROOT}/hpc/roihu/gputest_pytorch.sbatch" \
  > "${RUN_ROOT}/submit/gputest_pytorch.sbatch"
tar -xOzf "${SOURCE_ARCHIVE}" \
  "${ARCHIVE_ROOT}/hpc/roihu/triton_tensorrt_apptainer.sbatch" \
  > "${RUN_ROOT}/submit/triton_tensorrt_apptainer.sbatch"
tar -xOzf "${SOURCE_ARCHIVE}" \
  "${ARCHIVE_ROOT}/hpc/roihu/gpumedium_full_qualification.sbatch" \
  > "${RUN_ROOT}/submit/gpumedium_full_qualification.sbatch"
tar -xOzf "${SOURCE_ARCHIVE}" \
  "${ARCHIVE_ROOT}/hpc/roihu/finalize_roihu_qualification.sbatch" \
  > "${RUN_ROOT}/submit/finalize_roihu_qualification.sbatch"
chmod 700 "${RUN_ROOT}/submit/"*.sbatch
```

## 5. Stage A：`gputest` PyTorch/CUDA smoke

当前 `hpc/roihu/gputest_pytorch.sbatch` 固定请求：

- `gputest`
- 1 node、1 task
- 1 × `gpu:gh200`
- 72 × CPU cores
- 15 分钟
- CSC `python-pytorch/2.10` module

该 module 实测不包含 `onnx`；`python-data` 会替换而不是补充
`python-pytorch`。因此先在 Slurm 作业之外准备项目锁定的 CPython 3.12/aarch64
wheelhouse。首选输入是绿色 GitHub Actions artifact 中的
`roihu-arm64-wheelhouse/` 目录；无法直接转移该 artifact 时，才在 Roihu-GPU login
node 运行相同 helper。helper、JSON contract 和 hash lock 必须来自刚才验证过的
source archive：

```bash
export WHEEL_CONTRACT_DIR="${RUN_ROOT}/wheel-contract"
mkdir -p "${WHEEL_CONTRACT_DIR}" "${APP_ROOT}/wheelhouse"
for name in prepare_onnx_wheelhouse.sh onnx-wheelhouse.json requirements-onnx.lock; do
  tar -xOzf "${SOURCE_ARCHIVE}" \
    "${ARCHIVE_ROOT}/hpc/roihu/${name}" > "${WHEEL_CONTRACT_DIR}/${name}"
done
chmod 700 "${WHEEL_CONTRACT_DIR}/prepare_onnx_wheelhouse.sh"
"${WHEEL_CONTRACT_DIR}/prepare_onnx_wheelhouse.sh" \
  "${APP_ROOT}/wheelhouse/onnx-1.22.0-protobuf-5.29.6-cp312-aarch64"

export PYTHON_WHEELHOUSE="${APP_ROOT}/wheelhouse/onnx-1.22.0-protobuf-5.29.6-cp312-aarch64"
(cd "${PYTHON_WHEELHOUSE}" && sha256sum --check SHA256SUMS)
```

helper 在登录节点只下载经过安全审计的 `onnx==1.22.0` 与
`protobuf==5.29.6`，验证各自 hash，并用 CSC PyTorch 执行 opset-18 export 与
checker。GitHub Actions 下载、审计并上传相同 ARM64 wheelhouse，但明确不把这个动作
当作 GPU runtime evidence。实际 batch job 要求目录没有额外文件，并只在 node-local
`$TMPDIR` 执行 `--no-index --no-deps --require-hashes` 离线安装；检查范围只覆盖 ONNX
直接依赖和实际 import，避免把 CSC module 中未被本练习加载的其他框架依赖混入结论。

这符合 Roihu 每个完整 GH200 最多配套 72 个 Grace CPU cores 的资源形状。提交命令：

```bash
cd "${RUN_ROOT}/submit"
submission="$(sbatch --parsable \
  --account="${PROJECT}" \
  --export=ALL,SOURCE_ARCHIVE="${SOURCE_ARCHIVE}",SOURCE_SHA256="${SOURCE_SHA256}",SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}",PYTHON_WHEELHOUSE="${PYTHON_WHEELHOUSE}",EVIDENCE_ROOT="${RUN_ROOT}/evidence" \
  gputest_pytorch.sbatch)"
export SMOKE_JOB_ID="${submission%%;*}"
printf 'SMOKE_JOB_ID=%s\n' "${SMOKE_JOB_ID}"
```

观察而不干扰作业：

```bash
squeue -j "${SMOKE_JOB_ID}"
tail -f "${RUN_ROOT}/submit/roihu-pytorch-${SMOKE_JOB_ID}.out"
sacct -j "${SMOKE_JOB_ID}" \
  --format=JobIDRaw,JobName,Cluster,Partition,State,ExitCode,Elapsed,AllocTRES,NodeList
```

作业会再次验证 source SHA、archive root、GH200 名称和 compute capability 9.0，然后在同一个确定性 MLP/weights 上比较：

- Grace CPU FP32
- GH200 CUDA FP32
- GH200 CUDA BF16
- batch size 1、16、64、256
- warmup 5 次、测量 30 次
- p50/p95/p99、吞吐、显存、FP32/BF16 parity
- 每秒 `nvidia-smi` utilization/memory/power/temperature/clock telemetry
- 同一 weights 导出的 ONNX 与 Triton `config.pbtxt`
- `onnx.checker`、两份 wheel SHA-256、CPython ABI、离线安装和 ONNX direct-dependency check

证据目录：

```bash
export PYTORCH_EVIDENCE="${RUN_ROOT}/evidence/pytorch-gh200-${SMOKE_JOB_ID}"
find "${PYTORCH_EVIDENCE}" -maxdepth 3 -type f -print | sort
python3 -m json.tool "${PYTORCH_EVIDENCE}/job_manifest.json" | sed -n '1,120p'
python3 -m json.tool "${PYTORCH_EVIDENCE}/pytorch_benchmark.json" | sed -n '1,160p'
```

`job_manifest.json` 是 compact artifact index：它记录作业状态、source commit/source SHA、Slurm 信息、每个文件大小与 SHA-256，以及禁止 production/champion/capacity/promotion claim 的边界。它适合快速 review，但不能代替原始 telemetry、benchmark、ONNX 或正式 validator 所需的完整 evidence root。

CPU 对照是与 GH200 配套的 72-core NVIDIA Grace allocation，不是 Roihu-CPU 的 AMD Turin，也不是通用成本基准。面试中应明确说明测量范围只包含 steady-state model execution，排除了模型加载和 host-to-device transfer。

## 6. Stage B：`gpumedium` Apptainer、TensorRT 与 Triton qualification

只有 Stage A 成功并确认 digest 后才进入 Stage B。`gputest` 用来发现问题，`gpumedium` 才是正式分区路径；当前 validator policy 会拒绝 `gputest` 作为 eligibility 来源。

先在 Roihu-GPU login node、作业之外把 ARM64/GH200-compatible Triton server 与 SDK SIF 放到该项目的私有 scratch。不要在 batch job 内 pull image 或安装 package。SIF 必须可在 `aarch64` 运行，并保留 SHA-256；x86、A100/Puhti/Mahti image 不应复用。若项目 `/projappl` 配额足够，脚本也接受同项目的 `/projappl` 路径，但两份 25.06 SIF 合计约 18.9 GB，更适合 `/scratch`。

```bash
export APPTAINER_CACHEDIR="${RUN_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${RUN_ROOT}/apptainer-tmp"
mkdir -p "${RUN_ROOT}/containers" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
chmod 700 "${RUN_ROOT}/containers" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

export TRITON_SERVER_SIF="${RUN_ROOT}/containers/tritonserver-25.06-py3-arm64.sif"
export TRITON_SDK_SIF="${RUN_ROOT}/containers/tritonserver-25.06-py3-sdk-arm64.sif"
apptainer pull "${TRITON_SERVER_SIF}" docker://nvcr.io/nvidia/tritonserver:25.06-py3
apptainer pull "${TRITON_SDK_SIF}" docker://nvcr.io/nvidia/tritonserver:25.06-py3-sdk

export TRITON_SERVER_SIF_SHA256="$(sha256sum "${TRITON_SERVER_SIF}" | awk '{print $1}')"
export TRITON_SDK_SIF_SHA256="$(sha256sum "${TRITON_SDK_SIF}" | awk '{print $1}')"
export TRITON_CONTAINER_RELEASE=25.06

apptainer exec --cleanenv --containall "${TRITON_SERVER_SIF}" uname -m
apptainer exec --cleanenv --containall "${TRITON_SDK_SIF}" nvcc --version
apptainer inspect --json "${TRITON_SDK_SIF}" | grep -E 'build-arch|tensorrt.version|deffile.from'

export ONNX_PATH="${PYTORCH_EVIDENCE}/onnx/accelerator_qualification.onnx"
export EXPORT_MANIFEST_PATH="${PYTORCH_EVIDENCE}/onnx/export_manifest.json"
export ONNX_SHA256="$(sha256sum "${ONNX_PATH}" | awk '{print $1}')"
export EXPORT_MANIFEST_SHA256="$(sha256sum "${EXPORT_MANIFEST_PATH}" | awk '{print $1}')"
```

提交 1 × GH200、72 CPU、`gpumedium`、30 分钟作业：

```bash
cd "${RUN_ROOT}/submit"
submission="$(sbatch --parsable \
  --account="${PROJECT}" \
  --partition=gpumedium \
  --export=ALL,TRITON_SERVER_SIF="${TRITON_SERVER_SIF}",TRITON_SERVER_SIF_SHA256="${TRITON_SERVER_SIF_SHA256}",TRITON_SDK_SIF="${TRITON_SDK_SIF}",TRITON_SDK_SIF_SHA256="${TRITON_SDK_SIF_SHA256}",TRITON_CONTAINER_RELEASE="${TRITON_CONTAINER_RELEASE}",ONNX_PATH="${ONNX_PATH}",ONNX_SHA256="${ONNX_SHA256}",EXPORT_MANIFEST_PATH="${EXPORT_MANIFEST_PATH}",EXPORT_MANIFEST_SHA256="${EXPORT_MANIFEST_SHA256}",SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}",EVIDENCE_ROOT="${RUN_ROOT}/evidence",TRT_PRECISION=bf16 \
  triton_tensorrt_apptainer.sbatch)"
export TRITON_JOB_ID="${submission%%;*}"
printf 'TRITON_JOB_ID=%s\n' "${TRITON_JOB_ID}"
```

脚本的关键 runtime contract：

```text
apptainer exec --cleanenv --containall --nv --bind <job-work-dir>:/work <sdk.sif> /usr/src/tensorrt/bin/trtexec ...
apptainer exec --cleanenv --containall --nv --bind <job-work-dir>:/work <server.sif> tritonserver ...
```

实际脚本会：

1. 验证两份 SIF、ONNX 和 export manifest 的 SHA-256。
2. 验证 manifest 的 source commit、合成 qualification 分类、非 production、非 champion 和 ONNX/config digest。
3. 验证 host 与 SIF 都是 `aarch64`，GPU 为 GH200、compute capability 9.0。
4. 在目标 GH200/TensorRT stack 上用 `trtexec` 生成 `model.plan`；plan 不被视为跨 GPU 或 TensorRT 版本可移植。
5. 仅绑定 node-local work directory，使用 `--cleanenv --containall --nv`。
6. Triton HTTP/gRPC/metrics 全部监听 `127.0.0.1`，执行 bounded readiness wait。
7. 对 batch 1、16、64、256 运行 Perf Analyzer，并保存 CSV/raw log。
8. 保存 Triton `/v2` metadata、model config、Prometheus metrics、TensorRT/CUDA 原始版本信息和每秒 `nvidia-smi`。

结果目录：

```bash
export TRITON_EVIDENCE="${RUN_ROOT}/evidence/triton-tensorrt-gh200-${TRITON_JOB_ID}"
sacct -j "${TRITON_JOB_ID}" \
  --format=JobIDRaw,JobName,Cluster,Partition,State,ExitCode,Elapsed,AllocTRES,NodeList
python3 -m json.tool "${TRITON_EVIDENCE}/runtime_evidence.json" | sed -n '1,200p'
python3 -m json.tool "${TRITON_EVIDENCE}/job_manifest.json" | sed -n '1,160p'
```

必须分开描述两个 Triton 版本字段：

- `triton_server_version`：`/v2` metadata 中的 Triton runtime/API 2.x 版本。
- `triton_container_release`：NGC `YY.MM` container release，例如 `25.06`。

Perf Analyzer 当前使用随机但符合 contract 的输入，因此可用于服务性能/稳定性观察，不能建立 PyTorch 到 TensorRT 的业务语义 parity。`runtime_evidence.json` 会明确把该 claim 设为 false。

### Stage C：可选正式 `gpumedium` qualification 与 CPU finalizer

短 Triton 作业通过后，`gpumedium_full_qualification.sbatch` 才执行正式 FP32 路径：禁用 TF32，在同一确定性输入上分别测量 Grace CPU 与 Triton HTTP 各 300 秒/至少 1,000 次，并做 1,024 行 probability 与 0.5 decision parity；GPU 运行期间每 200 ms 采集 telemetry。提交参数复用上述已验证 artifact：

```bash
cd "${RUN_ROOT}/submit"
submission="$(sbatch --parsable \
  --account="${PROJECT}" \
  --export=ALL,SOURCE_ARCHIVE="${SOURCE_ARCHIVE}",SOURCE_SHA256="${SOURCE_SHA256}",SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}",TRITON_SERVER_SIF="${TRITON_SERVER_SIF}",TRITON_SERVER_SIF_SHA256="${TRITON_SERVER_SIF_SHA256}",TRITON_SDK_SIF="${TRITON_SDK_SIF}",TRITON_SDK_SIF_SHA256="${TRITON_SDK_SIF_SHA256}",TRITON_CONTAINER_RELEASE="${TRITON_CONTAINER_RELEASE}",ONNX_PATH="${ONNX_PATH}",ONNX_SHA256="${ONNX_SHA256}",EXPORT_MANIFEST_PATH="${EXPORT_MANIFEST_PATH}",EXPORT_MANIFEST_SHA256="${EXPORT_MANIFEST_SHA256}",EVIDENCE_ROOT="${RUN_ROOT}/evidence" \
  gpumedium_full_qualification.sbatch)"
export QUALIFICATION_JOB_ID="${submission%%;*}"
export QUALIFICATION_EVIDENCE_DIR="${RUN_ROOT}/evidence/full-qualification-${QUALIFICATION_JOB_ID}"
```

运行中的 GPU 作业不能自证最终 Slurm `COMPLETED`。作业成功离开队列后，从 `roihu-cpu.csc.fi` 提交一个 `test` 分区 finalizer，并绑定 `afterok` 依赖：

```bash
sbatch --parsable \
  --account="${PROJECT}" \
  --dependency="afterok:${QUALIFICATION_JOB_ID}" \
  --export=ALL,QUALIFICATION_JOB_ID="${QUALIFICATION_JOB_ID}",QUALIFICATION_EVIDENCE_DIR="${QUALIFICATION_EVIDENCE_DIR}",SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
  "${RUN_ROOT}/submit/finalize_roihu_qualification.sbatch"
```

finalizer 只在 `sacct` 记录 `COMPLETED/0:0` 后生成 `manifest.json`、`benchmark.json`、`SHA256SUMS` 和 `decision.json`，并调用随不可变 source 一起打包的 validator。任何速度、p95、parity、利用率、显存、版本、文件 hash 或 Slurm 条件不满足都会得到 `GPU_REJECTED`；两种结果都不会自动 promotion。

CSC 官方参考：

- [Roihu GPU partitions](https://docs.csc.fi/computing/running/batch-job-partitions/)
- [Roihu example job scripts](https://docs.csc.fi/computing/running/example-job-scripts-roihu/)
- [Apptainer containers and `--nv`](https://docs.csc.fi/computing/containers/overview/)
- [GPU-accelerated machine learning](https://docs.csc.fi/support/tutorials/gpu-ml/)

## 7. Raw evidence、compact evidence 与正式 validator

### 7.1 三层证据不能混用

1. **Raw evidence**：`nvidia-smi`、Slurm accounting、module/software versions、benchmark JSON/CSV、parity、Triton metrics/log、SIF/ONNX/plan bytes。
2. **Compact job evidence**：每个 sbatch 目录的 `job_manifest.json`、`pytorch_benchmark.json`、`cross_artifact_validation.json`、`runtime_evidence.json`。它们便于 review，但不替代 raw bytes。
3. **Formal eligibility decision**：`src.operations.roihu_gpu_evidence` 读取完整 evidence envelope，逐项交叉验证并输出 `decision.json`。仅 `decision=GPU_ELIGIBLE` 且 exit code 0 表示候选 GPU profile 通过；它仍不自动 promotion。

不要把 sbatch 的 `job_manifest.json` 改名后冒充 formal `manifest`。正式 envelope 需要按 `regulated-ml-platform.roihu-gpu-evidence/v1` 组织，并真实包含：

- 完整 source commit；
- `slurm.cluster=roihu`、正式 partition、`COMPLETED`、`0:0`；
- GH200 product、compute capability 9.0、`aarch64`、GPU count/memory/UUID/driver；
- CUDA、TensorRT、Triton server 2.x 与 Triton container `YY.MM`；
- `qualified_workload.kind=neural_accelerator_qualification`、policy 允许的 neural model family、`profile=candidate_gpu_profile`、`is_current_tree_champion=false`；
- 至少 300 秒、CPU/GPU/parity 各至少 1000 samples 的正式 benchmark；
- CPU/GPU throughput 与 p95 的可重算 ratio、HTTP request/error accounting、GPU utilization、memory headroom；
- source archive、candidate model、SIF，以及 Slurm、`nvidia-smi`、software、CPU/GPU benchmark、parity raw artifact；
- 覆盖所有上述文件的 `SHA256SUMS`。

若当前 runtime pack 没有生成某个正式字段或没有达到正式采样时长，正确结果是保留 gap 并让 validator 返回 `GPU_REJECTED`，而不是从期望配置中补造观测值。

在可信 workstation 或受控 runner 中对完整 evidence root 执行：

```bash
export FORMAL_EVIDENCE_ROOT=/path/to/formal/evidence
python -m src.operations.roihu_gpu_evidence \
  --manifest "${FORMAL_EVIDENCE_ROOT}/manifest.json" \
  --benchmark "${FORMAL_EVIDENCE_ROOT}/benchmark.json" \
  --policy config/roihu_gpu_evidence_policy.yaml \
  --checksums "${FORMAL_EVIDENCE_ROOT}/SHA256SUMS" \
  --artifact-root "${FORMAL_EVIDENCE_ROOT}" \
  --output "${FORMAL_EVIDENCE_ROOT}/decision.json"
```

CLI contract：

- exit 0：仅 `GPU_ELIGIBLE`；
- exit 2：`GPU_REJECTED`，仍写出机器可读 JSON 和失败原因；
- 单独写 `runtime_evidence: real_gpu` 永远不能解锁 GPU；
- 通过时 `accelerator_product=nvidia-gh200`、`qualified_workload.verified=true`；
- 无论通过或失败，`automatic_promotion_authorized=false`，当前模型 accelerator/profile 均保持 unchanged。

为 review 生成 compact decision index，同时保留完整 evidence root：

```bash
sha256sum "${FORMAL_EVIDENCE_ROOT}/decision.json" > "${FORMAL_EVIDENCE_ROOT}/decision.json.sha256"
python3 - "${FORMAL_EVIDENCE_ROOT}/decision.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "decision": report.get("decision"),
    "status": report.get("status"),
    "source_commit": report.get("source_commit"),
    "accelerator_product": report.get("accelerator_product"),
    "qualified_workload": report.get("qualified_workload"),
    "deployment_effects": report.get("deployment_effects"),
    "verified_file_count": report.get("artifacts", {}).get("verified_file_count"),
    "reasons": report.get("reasons"),
}, indent=2, sort_keys=True))
PY
```

## 8. 与 Helm/KEDA GPU 交付契约的安全映射

Roihu 的 `clusterProfile=csc-roihu-gh200` 只表示 evidence origin/profile；Kubernetes node selector 仍必须明确为 `nvidia-gh200`，GPU 数量资源必须是 `nvidia.com/gpu`。v1.3 模板会拒绝：

- 仅 `gpuEvidenceApproved=true`、没有 real evidence；
- GH200 evidence 绑定 A100 node selector；
- 非 `nvidia.com/gpu` resource；
- SHA-256、cluster profile 或 accelerator product 未验证；
- KEDA max replicas 大于 GPU ResourceQuota 可覆盖容量；
- production PDB 配置为单副本；
- scale-down stabilization 小于 cooldown。

对已独立审核且返回 `GPU_ELIGIBLE` 的 `decision.json`，可做 **render-only** 演练：

```bash
export DECISION_SHA256="$(sha256sum "${FORMAL_EVIDENCE_ROOT}/decision.json" | awk '{print $1}')"
helm template regulated-ai helm/regulated-ai \
  --set tritonServing.enabled=true \
  --set tritonServing.accelerator=gpu \
  --set tritonServing.modelFamily=neural_network \
  --set tritonServing.gpuEvidenceApproved=true \
  --set tritonServing.gpu.resourceName=nvidia.com/gpu \
  --set tritonServing.gpu.nodeSelector.accelerator=nvidia-gh200 \
  --set tritonServing.gpuProductionEvidence.acceleratorDecision=GPU_ELIGIBLE \
  --set tritonServing.gpuProductionEvidence.runtimeEvidence=real_gpu \
  --set tritonServing.gpuProductionEvidence.acceleratorProduct=nvidia-gh200 \
  --set tritonServing.gpuProductionEvidence.clusterProfile=csc-roihu-gh200 \
  --set-string tritonServing.gpuProductionEvidence.reportSha256="${DECISION_SHA256}" \
  --set tritonServing.replicas=2 \
  --set tritonServing.productionControls.podDisruptionBudget.enabled=true \
  --set tritonServing.productionControls.networkPolicy.enabled=true \
  --set tritonServing.productionControls.gpuResourceQuota.enabled=true \
  --set tritonServing.productionControls.serviceMonitor.enabled=true \
  --set tritonServing.productionControls.prometheusRule.enabled=true \
  --set tritonServing.kedaAutoscaling.enabled=true
```

这个命令只验证 Kubernetes manifest contract。它不会在 Roihu 或任何 Kubernetes/GPU 集群创建资源；真实部署还需要匹配的 candidate model image、GH200 Kubernetes node、Prometheus Operator、KEDA CRD、审查批准、canary 与 rollback 演练。

## 9. GitHub Actions 的职责边界

`.github/workflows/triton-gpu-evidence.yml` 的职责是：

- 在 GitHub-hosted `ubuntu-latest` 上执行 Ruff、shell syntax、targeted pytest；
- 在 pull request 中显式 checkout `pull_request.head.sha`，避免把临时 merge SHA 当成 Roihu source provenance；
- 测试 evidence validator 的通过/拒绝路径；
- 从 clean checkout 生成并复核 immutable source bundle；
- 上传名称包含 `staging-contracts-not-gpu-evidence` 的 source/wheel contract artifact；
- 写出 `executes_gpu_workload=false`、`proves_real_gpu=false` 的 claim boundary。

它不会连接 CSC、续签 SSH certificate、提交 Slurm、运行 `nvidia-smi`、使用 `--gpus`、运行 Apptainer `--nv` 或接触真实 Roihu evidence。真实作业输出应先从 `/scratch` 复制到受控 evidence root，在可信环境中验 hash 和运行 validator；不要因 GitHub workflow 绿色就写“GPU benchmark passed”。

## 10. 面试练习：用事实讲清楚设计与取舍

### 90 秒架构回答

1. **为什么用两阶段？** `gputest` 快速发现架构、module、CUDA、parity 和 ONNX 问题，避免直接消耗正式队列；`gpumedium` 才承担 target-side TensorRT/Triton qualification。
2. **为什么 source/SIF/ONNX 都要 digest？** 防止 benchmark、engine build 和 serving 使用了不同 bytes；full Git SHA 只标识 source，不替代 artifact SHA-256。
3. **为什么 TensorRT 在 GH200 上 build？** plan 依赖 accelerator architecture、TensorRT/CUDA stack 和 shape/precision profile，不应假设跨 A100、x86 或版本可移植。
4. **为什么 Perf Analyzer 不等于 correctness？** random contract-valid input 可测 throughput/latency/service stability，但不能证明模型语义或 policy decision parity。
5. **为什么不自动 promotion？** GPU eligibility 只批准 exact candidate/profile/evidence；当前 champion、生产流量、SLO、canary、rollback 和独立审批仍是单独的门。
6. **如何映射 Kubernetes？** 用 ResourceQuota 控制 GPU 容量、PDB 保可用性、NetworkPolicy 最小权限、ServiceMonitor/PrometheusRule 观测、KEDA 以 queue time 与 GPU utilization 有界伸缩；但 Roihu/Slurm evidence 不是 live Kubernetes evidence。

### 练习时应能现场回答

- 为什么 `gputest` 明明是真 GH200，policy 仍拒绝它作为正式资格证据？
- 72 CPU cores 和 1 GH200 的 allocation 如何影响 CPU/GPU 对比解释？
- BF16 parity threshold、p95、throughput、utilization 和 memory headroom 分别防什么风险？
- KEDA 为什么需要 min/max、fallback、cooldown、scale-down stabilization？
- GH200 evidence 为什么不能授权 A100 node selector？
- raw evidence、compact manifest 和 decision JSON 的职责有什么不同？
- 如果 `nv_gpu_` metrics 缺失、SIF release metadata 不匹配或 Slurm exit 非 `0:0`，为什么必须 fail closed？

### 结果记录模板

在真实运行前不要填写数值。运行后只从已验证 JSON/CSV 抄录：

| 项目 | 证据路径 | 观测值 | 门限/判断 |
| --- | --- | --- | --- |
| Slurm job/partition/state/exit | `sacct` raw + formal manifest | 待运行 | 正式证据必须 `gpumedium/gpularge`, `COMPLETED`, `0:0` |
| GPU product/CC/memory/UUID | `nvidia-smi-snapshot.csv` | 待运行 | GH200, CC 9.0, policy memory floor |
| CPU/GPU throughput | benchmark raw JSON/CSV | 待运行 | 可从 raw 重算 speedup |
| CPU/GPU p95 | benchmark raw JSON/CSV | 待运行 | 可从 raw 重算 ratio |
| FP32/BF16 parity | `pytorch_benchmark.json` / formal parity | 待运行 | 必须符合对应 scope 的 tolerance |
| GPU utilization/headroom | one-second telemetry + benchmark | 待运行 | policy operating band 与最小 headroom |
| Triton health/metrics | readiness、`/v2`、`.prom` | 待运行 | 无错误且存在 inference/GPU observations |
| Artifact integrity | manifests + `SHA256SUMS` | 待运行 | 每个 byte 与声明一致 |
| Final decision | `decision.json` + SHA-256 | 待运行 | `GPU_ELIGIBLE` 仍不自动 promotion |

已完成运行不要回填到上面的复用模板；请阅读独立的
[execution record](roihu_gpu_execution_record.md)。该记录保留 smoke passes、失败尝试、
正式 `304890` 运行和 finalizer `304891` 的 `GPU_REJECTED` 结论及 claim boundary。

## 11. 常见故障与排查顺序

| 现象 | 首先检查 |
| --- | --- |
| SSH 拒绝或要求 password | MyCSC public key、24 小时 certificate 是否过期、certificate 名称是否与私钥匹配 |
| `uname -m` 不是 `aarch64` | 是否错误登录 `roihu-cpu.csc.fi`；GPU job 必须从 GPU login 侧提交 |
| source bundle 脚本退出 65 | worktree 是否有 tracked/untracked 改动，commit 是否为完整 SHA |
| sbatch 立即 fail | `SLURM_JOB_ACCOUNT`、partition、`SLURM_SUBMIT_DIR`、`/projappl`/`/scratch` 路径是否一致 |
| GH200/CC 检查失败 | 是否申请 `--gres=gpu:gh200:1`，是否误用 interactive slice 或其他 GPU |
| SIF architecture/release 失败 | 是否为 ARM64 image，`apptainer inspect` 是否含预期 NGC `YY.MM`，SHA 是否来自同一文件 |
| TensorRT plan 不生成 | 查看 `trtexec-build-and-validation.log`、shape/profile/precision、TensorRT/CUDA 兼容性 |
| Triton 不 ready | 查看 server log、loopback port collision、plan/config、model repository |
| `nv_gpu_` metrics 缺失 | GPU metrics 是否开启、实际是否有 inference、容器是否正确注入 `--nv` |
| validator exit 2 | 读取 `decision.json.reasons`；不要绕过 missing field、threshold、partition 或 digest 失败 |
| GitHub workflow 绿色但没有 GPU 数值 | 正常：workflow 只验证 contract，不运行 GPU |
