# GPU serving contract with NVIDIA Triton

## Scope

`gitops/gpu/` defines the Kubernetes runtime contract for a GPU-backed credit-risk model, and `serving/triton/models/credit-risk/config.pbtxt` defines its Triton scheduling contract. The workload targets a dedicated Google Kubernetes Engine A100 node pool and exposes Triton's HTTP, gRPC, and Prometheus interfaces only inside the cluster.

This is a serving contract, not a checked-in model binary. The delivery pipeline must build an immutable `regulated_ml_platform-triton` image containing the approved model repository at `/models`:

```text
/models/
  credit-risk/
    config.pbtxt
    1/
      model.onnx
```

The image digest, ONNX checksum, feature schema, training lineage, validation report, and model approval must refer to the same released artifact. No environment overlay currently includes the GPU resources. CI renders the standalone contract, but Argo CD cannot reconcile it until a later reviewed change provides the image/model release path and explicitly activates it in an environment.

## Triton model contract

The approved model name is `credit-risk`, version `1`, using the ONNX Runtime backend. It accepts a batch of 12 `TYPE_FP32` features named `features` and returns one `TYPE_FP32` value named `support_probability`. Changing names, order, dimensions, type, preprocessing, or output semantics is a breaking interface change and requires contract tests plus renewed validation.

Dynamic batching combines compatible requests to improve A100 utilisation:

- maximum batch size: 32;
- preferred batches: 4, 8, 16, then 32;
- maximum queue delay: 5,000 microseconds;
- response ordering preserved;
- one model instance on GPU 0 per pod.

The 5 ms queue budget is a starting point, not a universal optimum. Tune it with representative arrival patterns and record throughput, p50/p95/p99 latency, queue duration, GPU utilisation, memory, errors, and accuracy parity. A larger batch that violates the 250 ms end-to-end p95 gate must not be promoted.

Response caching is enabled only because the model is deterministic for identical tensors. Do not send customer identifiers in cache keys or enable caching for stochastic/stateful models. Reassess memory and privacy behaviour if preprocessing moves into Triton.

## GKE A100 placement

The Deployment requires both:

```yaml
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-tesla-a100
  cloud.google.com/gke-nodepool: a100-inference
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

The node pool must have compatible NVIDIA drivers/device plugin, tested autoscaling bounds, and the `nvidia.com/gpu` taint. The pod requests and limits exactly one `nvidia.com/gpu`; it also reserves 4 CPU/16 GiB and caps at 8 CPU/32 GiB. The reference Deployment uses `maxSurge: 0` and `maxUnavailable: 1`, allowing two replicas to roll within a two-GPU quota at the cost of temporarily reduced capacity. A real production design must prove that trade-off against its availability objective and reserve failure-domain headroom.

Two replicas and topology spreading reduce a single-host failure, but `ScheduleAnyway` permits recovery in a constrained cluster. The PodDisruptionBudget retains at least one available replica during voluntary disruption. Confirm the node pool spans the required zones for the actual resilience target.

## Runtime safety

The pod runs as UID/GID 1000, drops all Linux capabilities, uses the RuntimeDefault seccomp profile, forbids privilege escalation, disables the Kubernetes API token, and has a read-only root filesystem. Writable `emptyDir` mounts are limited to `/tmp`, `/.cache`, and a 2 GiB memory-backed `/dev/shm`.

The NetworkPolicy permits inference calls only from the API component in the same namespace, metrics only from the `monitoring` namespace, and DNS egress only to `kube-system`. If a service mesh, external model repository, or licence endpoint is introduced, add the narrow destination explicitly and review the data path; do not replace the policy with unrestricted egress.

The image should run as the same non-root UID, include no shell/package manager in the final stage where practical, and be signed and scanned. Production manifests should pin the built image by digest. Never download a mutable model at pod startup: doing so separates runtime behaviour from the reviewed image and makes rollback non-deterministic.

## Health and shutdown

Triton's endpoints are used as follows:

| Probe | Endpoint | Meaning |
|---|---|---|
| Startup | `/v2/health/ready` | allows up to five minutes for CUDA context and model load |
| Readiness | `/v2/health/ready` | removes a pod that cannot serve the explicitly loaded model |
| Liveness | `/v2/health/live` | restarts a wedged Triton process |

`--strict-readiness=true`, `--exit-on-error=true`, explicit model control, and `--load-model=credit-risk` prevent an apparently healthy server from accepting traffic without the approved model. A 60-second termination period and 20-second pre-stop delay allow endpoint removal and in-flight requests to drain.

Readiness is necessary but not sufficient. Before promotion, send a versioned golden tensor through HTTP or gRPC and verify schema, finite output, probability bounds, and tolerance against the approved CPU reference.

## Build and release checks

For every candidate image:

1. Validate the `config.pbtxt` parser and directory layout.
2. Verify the ONNX checksum and generate an SBOM/provenance statement.
3. Run CPU-versus-GPU golden-set parity at the agreed numeric tolerance.
4. Test maximum batch, malformed shape/type, concurrency, timeout, and cancellation behaviour.
5. Benchmark cold start and sustained traffic on the target A100 class.
6. Scan and sign the final image, then add it to a reviewed environment overlay by immutable digest; extend the allow-listed promotion helper and its tests in that same change.
7. Use the Argo Rollout and canary gate; do not activate the standalone contract as a separate ungoverned GPU release path.

The API should bound payload size and deadline and should translate Triton errors without leaking tensors or customer data. Retries must be capped and jittered; retrying overload can amplify an incident.

## Metrics and capacity signals

Triton exposes Prometheus metrics on port 8002. The provisioned dashboard and alerts use:

- `nv_inference_request_success` and `nv_inference_request_failure` for throughput/error rate;
- `nv_inference_queue_duration_us` for batch queue pressure;
- DCGM `DCGM_FI_DEV_GPU_UTIL` for device utilisation;
- API error/latency, drift, and fairness metrics for the end-to-end release decision.

Add GPU memory, OOM/restart, throttling, pending-pod, batch-size, compute duration, and model-load metrics in the production monitoring platform. Correlate by namespace, model, model version, pod, rollout hash, and image digest, without customer-level labels.

## Operational response

For Triton failures or queue latency, first protect the customer path: stop/abort the canary and verify that stable capacity is healthy. Then determine whether the constraint is request errors, batching, CPU preprocessing, GPU compute, memory, model load, or node capacity. Retain the dashboard window, pod events, rollout/analysis objects, Triton logs, image digest, and node/device-plugin state.

High GPU utilisation alone is not a rollback reason when latency and error budgets are healthy; sustained utilisation above 90% is a capacity warning. Conversely, low utilisation with high queue latency often indicates CPU preprocessing, concurrency, or scheduling problems rather than a need for more GPUs.

Do not mutate batching or resource values directly in production. Change the contract in Git, repeat parity/load tests in preproduction, and promote a new reviewed revision. If model behaviour or feature ordering is implicated, involve the model owner and Model Risk before retrying.
