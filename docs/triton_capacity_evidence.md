# Triton concurrency and capacity evidence: v1.2

## Result first

v1.1 proved that the calibrated serving package could be loaded by a real Triton server and preserve probability semantics over HTTP. v1.2 answers the next operational questions: whether concurrent requests are actually aggregated by Triton's scheduler, where latency/throughput tradeoffs appear, and how to turn server measurements into a bounded reference replica decision.

```text
real Triton CPU server
        ↓
custom synchronized concurrent HTTP
        ↓
probability parity + HTTP correctness
        +
Prometheus inference_count / exec_count
        ↓
observed runtime dynamic batching
        ↓
NVIDIA Triton Perf Analyzer
        ↓
server latency / throughput sweep
        ↓
SLO-passing server capacity point
        ↓
70% safety headroom
        ↓
bounded reference replica recommendation
```

The two benchmark paths have different roles. The custom Python client protects model semantics and proves runtime batching. NVIDIA Triton Perf Analyzer is the server-capacity source. The CI artifact is the source of truth for each run.

## 1. Runtime batching is measured, not inferred from config

A `dynamic_batching` block in `config.pbtxt` proves configuration intent, not actual request aggregation.

v1.2 reads Triton Prometheus counters before and after each concurrency scenario and derives:

```text
observed average batch size
= delta(nv_inference_count)
  / delta(nv_inference_exec_count)
```

For request batch size 1, a value above 1 means multiple inference rows were executed per backend execution on average.

The semantic benchmark exercises concurrency `1 / 4 / 8 / 16 / 32` using synchronized bursts so the server receives overlapping requests. A passing hosted-CI reference run observed a maximum average backend batch size of about `2.82`; at concurrency 4, 48 inference rows were processed in 17 backend executions.

## 2. Semantic safety remains a hard gate

Every custom concurrent HTTP response is compared with the native calibrated sklearn model.

The benchmark records:

```text
HTTP failure rate
maximum absolute probability error
p50 / p95 / p99 latency
requests and rows per second
base-model average batch size
calibrator average batch size
queue time per inference
```

A passing reference run completed all requests with zero HTTP failures and maximum absolute probability error below `5.5e-7`, far inside the configured `5e-5` tolerance.

This path is deliberately **not** the server-capacity source. Its Python thread scheduling, JSON serialization, HTTP client stack, and synchronized-burst harness add client-side overhead that is useful for end-to-end observation but should not be mistaken for Triton backend capacity.

## 3. Perf Analyzer is the server-capacity source

The workflow runs NVIDIA Triton Perf Analyzer from the matching `25.06-py3-sdk` image against the same live `support_ensemble` server.

```text
custom concurrent HTTP
→ semantic parity + HTTP correctness + scheduler metric deltas

NVIDIA Triton Perf Analyzer
→ optimized Triton client path + concurrency/latency/throughput measurement
```

Perf Analyzer CSV latency values are normalized from microseconds to milliseconds before policy evaluation.

A passing hosted-CI reference run measured:

| Concurrency | Inferences/s | p95 | p99 |
|---:|---:|---:|---:|
| 1 | 807 | 1.279 ms | 1.317 ms |
| 4 | 2,801 | 1.545 ms | 1.637 ms |
| 7 | 4,634 | 1.721 ms | 1.836 ms |
| 10 | 7,554 | 1.796 ms | 2.070 ms |
| 13 | 8,846 | 1.995 ms | 2.566 ms |
| 16 | 9,425 | 2.391 ms | 3.296 ms |

These values are reference measurements from a shared hosted runner, not production capacity guarantees.

## 4. Capacity decision

`config/triton_capacity_policy.yaml` defines the reference SLO and safety rules.

The custom path must first pass:

```text
probability parity
HTTP correctness
runtime dynamic batching evidence
```

Perf Analyzer points are then filtered by the configured server latency objectives:

```text
p95 <= configured p95 SLO
p99 <= configured p99 SLO
throughput > 0
```

The highest-throughput SLO-passing Perf Analyzer point becomes the measured single-instance server reference capacity.

```text
safe reference capacity per replica
= Perf Analyzer SLO-passing inferences/s
  × safety_headroom_fraction
```

The reference replica count is:

```text
ceil(reference_target_rows_per_second
     / safe_reference_rows_per_second_per_replica)
```

The decision fails closed when semantic correctness fails, batching is ineffective, Perf Analyzer evidence is absent, too few server points pass the SLO, or the target exceeds the configured replica boundary.

## 5. Why the custom client and Perf Analyzer should not match exactly

The custom benchmark intentionally performs native-model parity checks, JSON request construction, Python scheduling, and synchronized bursts. Perf Analyzer uses Triton-focused client optimizations and is designed for server performance measurement.

Therefore:

```text
custom client throughput
≠ Triton server capacity
```

The custom path answers **"did the real serving path stay correct and batch requests?"** Perf Analyzer answers **"what latency/throughput envelope did this Triton server expose under the benchmark configuration?"**

## 6. Capacity evidence report

The CI workflow generates `capacity_report.html` with three separate evidence views:

```text
p95 latency vs concurrency
throughput vs concurrency
observed average backend batch size vs concurrency
```

It also records the selected capacity source, safety headroom, reference target, and replica recommendation.

## 7. Why this is not a production capacity promise

The hosted-CI experiment is short and uses synthetic data, a shared GitHub runner, one CPU Triton instance, and a controlled request pattern. It does not model:

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

The generated replica number is therefore a **reference recommendation**. Production sizing requires the same evidence process to be rerun in the target environment.

## 8. Reproduce

```bash
make data features train triton
make triton-capacity
```

The workflow writes:

```text
reports/triton_capacity/
├── concurrency_benchmark.json
├── perf_analyzer.csv
├── perf_analyzer.log
├── triton_metrics_after_concurrency.prom
├── capacity_plan.json
├── capacity_plan.md
├── capacity_report.html
├── triton_server.log
├── triton_container_inspect.json
├── triton_server_image_inspect.json
└── triton_sdk_image_inspect.json
```

## Boundary

The current validated champion remains a calibrated tree ensemble and remains `CPU_ONLY`. v1.2 does not claim CUDA, TensorRT, A100 speedup, GPU memory behaviour, live production autoscaling, or production capacity.
