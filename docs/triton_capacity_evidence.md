# Triton concurrency and capacity evidence: v1.2

## Result first

v1.2 answers a different question from v1.1.

v1.1 proved that the calibrated serving package could be loaded by a real Triton server and preserve probability semantics over HTTP. v1.2 measures what happens when multiple requests arrive together and turns those measurements into a bounded reference capacity decision.

```text
real Triton CPU server
        ↓
concurrent request bursts
        ↓
client p50 / p95 / p99 latency
        +
request and row throughput
        +
probability parity
        +
Prometheus inference_count / exec_count
        ↓
observed scheduler average batch size
        ↓
NVIDIA Triton Perf Analyzer cross-check
        ↓
SLO-passing capacity envelope
        ↓
safety headroom
        ↓
bounded reference replica recommendation
```

The CI artifact is the source of truth for measured values. Numbers are copied into the README only after the workflow is green.

## 1. What dynamic batching evidence means

A `dynamic_batching` block in `config.pbtxt` proves configuration intent, not runtime aggregation.

v1.2 reads Triton Prometheus counters before and after each concurrency scenario and derives:

```text
observed average batch size
= delta(nv_inference_count)
  / delta(nv_inference_exec_count)
```

For request batch size 1, a value materially above 1 means multiple inference rows were executed per backend execution on average during that scenario.

The benchmark exercises concurrency levels `1 / 4 / 8 / 16 / 32` using synchronized bursts so the scheduler receives genuinely overlapping requests.

## 2. Semantic safety remains a gate

Every concurrent request is still compared with the native calibrated sklearn model.

The concurrency test records:

```text
HTTP failure rate
maximum absolute probability error
p50 latency
p95 latency
p99 latency
requests per second
rows per second
base-model average batch size
calibrator average batch size
queue time per inference
```

A fast scenario is not accepted when probability parity or HTTP correctness fails.

## 3. Independent load-tool cross-check

The workflow also runs NVIDIA Triton Perf Analyzer from the matching `25.06-py3-sdk` image against the same live `support_ensemble` server.

Perf Analyzer is kept separate from the custom parity benchmark:

```text
custom concurrent HTTP benchmark
→ semantic parity + scheduler metric deltas

Perf Analyzer
→ independent concurrency / latency / throughput measurement
```

Agreement in trend is more useful than pretending two different measurement tools should return identical point estimates.

## 4. Capacity decision

`config/triton_capacity_policy.yaml` defines the reference SLO and safety rules.

A scenario is SLO-passing only when all configured conditions pass:

```text
probability parity
p95 latency
p99 latency
HTTP error rate
maximum probability error
```

The highest-throughput SLO-passing point becomes the measured single-instance reference capacity.

```text
safe reference capacity per replica
= measured SLO-passing rows/s
  × safety_headroom_fraction
```

The reference replica count is then:

```text
ceil(reference_target_rows_per_second
     / safe_reference_rows_per_second_per_replica)
```

This calculation is deliberately blocked or downgraded when batching is ineffective, too few SLO-passing scenarios exist, or the target exceeds the configured replica boundary.

## 5. Why this is not a production capacity promise

The hosted-CI experiment is short and uses synthetic data, a shared GitHub runner, one CPU Triton instance, and a controlled request pattern.

It does not model all of the following:

```text
production traffic arrival distribution
real customer mix
CPU requests/limits and noisy neighbours
network topology and TLS/service-mesh overhead
multi-replica load balancing
pod startup and failure recovery
long-duration saturation
regional failover
production cost
```

Therefore the generated replica number is named a **reference recommendation**. Production sizing requires the same evidence process to be rerun in the target environment.

## 6. Reproduce

After generating the validated model and Triton repository:

```bash
make data features train triton
make triton-capacity
```

The workflow writes machine-readable and human-readable evidence under:

```text
reports/triton_capacity/
├── concurrency_benchmark.json
├── perf_analyzer.csv
├── perf_analyzer.log
├── triton_metrics_after_concurrency.prom
├── capacity_plan.json
├── capacity_plan.md
├── triton_server.log
├── triton_container_inspect.json
├── triton_server_image_inspect.json
└── triton_sdk_image_inspect.json
```

## Boundary

The current validated champion remains a calibrated tree ensemble and remains `CPU_ONLY`. v1.2 does not claim CUDA, TensorRT, A100 speedup, GPU memory behaviour, live production autoscaling, or production capacity.
