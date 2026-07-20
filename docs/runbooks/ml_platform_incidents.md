# ML platform incident runbook

This runbook is for operational response in the synthetic reference platform. Preserve request IDs, model provenance, release identity, metrics, and registry state before changing production state.

## Error budget burn

1. Confirm the 5xx increase is server-side and exclude expected 4xx validation failures.
2. Check `/ready`, `/runtime/model`, pod restarts, dependency health, and the active release identity.
3. Freeze environment promotion and model promotion.
4. Compare the incident start time with the last application, policy, or model transition.
5. Roll back the last approved release only when evidence links the regression to that release.
6. Record the consumed error budget and corrective action before reopening promotion.

## Latency SLO breach

1. Check saturation before scaling: CPU, memory, request concurrency, dependency latency, and model-serving latency.
2. Compare champion and challenger latency when a canary is active.
3. Stop challenger traffic when its latency ratio exceeds the configured safety limit.
4. Scale only after identifying a capacity constraint; do not hide a dependency or model regression with replicas.
5. Preserve p50/p95/p99 evidence for the incident review.

## Runtime degraded

1. Inspect `/runtime/model` and identify whether the process is using cached registry state or the packaged local fallback.
2. Check MLflow connectivity, artifact-store availability, checksum validation, schema compatibility, and promotion-gate status.
3. Do not promote a new model while the runtime is degraded.
4. Restore registry access or intentionally roll back to the last verified champion.

## Registry reload failure

1. Inspect structured logs using the reload timestamp and registry version.
2. Check required files, SHA-256 provenance, feature-schema version, promotion gate, deserialization, and smoke prediction.
3. Keep the current verified predictor active; never replace it with a partially validated bundle.
4. Repair or remove the invalid registry alias before retrying.

## Canary stopped

1. Leave challenger serving disabled.
2. Capture `/canary/status`, `/canary/evaluate`, Prometheus metrics, and the champion/challenger registry versions.
3. Identify which online limit failed before changing any threshold.
4. Do not widen a safety threshold only to make a release pass.
5. Fix or replace the challenger, register a new immutable version, and start a new evidence window.

## Canary challenger failure

1. Confirm whether challenger prediction exceptions or timeouts caused champion fallback.
2. Compare model bundle, runtime dependency, feature schema, and resource saturation with the champion.
3. Keep fallback-to-champion active and stop the canary when the configured error ceiling is breached.
4. Test the corrected challenger offline before opening another canary.

## Canary decision disagreement

1. Segment disagreements by action, score range, customer cohort, and policy gate.
2. Distinguish expected model-score changes from unexpected policy or schema changes.
3. Review high-impact action changes before promotion.
4. Require a new offline evaluation when disagreement is unexplained or concentrated in protected or vulnerable cohorts.

## Triton queue pressure

1. Confirm the queue signal with `nv_inference_queue_duration_us`, request rate, execution count, and inference count rather than scaling from a single latency metric.
2. Calculate observed average batch size from inference count divided by execution count and compare it with the configured preferred batch sizes.
3. If average batches are small, inspect request concurrency and `max_queue_delay_microseconds` before adding replicas; scaling out can reduce batching efficiency further.
4. If batches are healthy and queue time still grows, inspect CPU/GPU saturation, model instance count, and downstream feature-preparation latency.
5. Preserve the Triton model repository contract, active model version, batch configuration, and benchmark evidence before changing capacity.

## Triton GPU headroom low

1. Confirm both sustained GPU utilization and queue pressure are elevated; high GPU utilization alone is not a scale-out signal.
2. Check GPU memory utilization, OOM/restart events, batch size, request rate, and model instance count.
3. Verify the active model has an approved `GPU_ELIGIBLE` accelerator decision backed by real GPU runtime evidence.
4. Scale only within the approved replica boundary and preserve the pre/post throughput, p95 latency, queue time, and GPU utilization evidence.
5. If the release introduced the regression, stop promotion and use the approved rollback path rather than masking a model regression with capacity.

## Triton GPU underutilized

1. Check average batch size and request concurrency before scaling in; low utilization with batch size near one can indicate weak aggregation rather than excess capacity.
2. Confirm queue time is low and the observation is sustained through the configured scale-in window.
3. Compare current traffic with the benchmark workload used to justify GPU capacity.
4. Scale in only after batching is healthy and minimum availability/replica constraints remain satisfied.
5. Reconsider whether the model should remain on GPU when sustained utilization is below the accelerator-policy floor and CPU evidence meets the SLO.

## Incident review evidence

Every incident review should include:

- incident timeline and detection source;
- affected release ID, image digest, git commit, model registry version, policy version, and feature schema;
- SLI/SLO impact and error-budget consumption;
- Triton model repository version, ONNX artifact hashes, batch configuration, and accelerator decision when model serving is involved;
- GPU utilization, GPU memory, queue time, average batch size, request rate, and replica count when accelerator capacity is involved;
- customer-impact boundary using pseudonymous identifiers only;
- immediate mitigation and rollback decision;
- root cause and contributing conditions;
- corrective controls, tests, and monitoring changes.
