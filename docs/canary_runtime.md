# Canary runtime design

## Purpose

The canary runtime adds a controlled online transition between the MLflow `champion` and `challenger` aliases. It does not replace offline model validation, the release promotion gate, or incident rollback. It adds an online safety stage after a challenger has already passed the offline release checks.

```text
validated challenger
        ↓
MLflow challenger alias
        ↓
verified challenger bundle load
        ↓
stable customer-level canary assignment
        ↓
score champion + challenger on the same request
        ↓
serve challenger only to its assigned cohort
        ↓
collect online safety evidence
        ↓
WAIT ── insufficient evidence
STOP ── configured safety limit failed
PASS ── configured safety limits passed
        ↓
optional controlled promotion
        ↓
MLflow challenger → champion
        ↓
incident rollback can restore the former champion
```

The validated model release remains separate from the MLflow registry version. A registry version is an immutable registry entry; it is not a model semantic version.

## Activation contract

A challenger is not allowed into canary traffic merely because the `challenger` alias exists. `CanaryController` downloads the same release bundle used by the registry serving runtime and checks:

1. `model.joblib`, `metadata.json`, `model_metrics.json`, and `promotion_gate.json` exist;
2. every required file matches the SHA-256 evidence generated during download;
3. `feature_schema_version` matches the running API contract exactly;
4. the offline promotion gate is `PASS` and `eligible_for_controlled_release`;
5. the model can be deserialized by the running service.

If these checks fail, the challenger is not placed into canary traffic. The champion remains the served model.

## Stable canary cohort

Traffic assignment is deterministic rather than random on every request.

```text
assignment key = canary seed + challenger registry version + pseudonymous customer_id
        ↓
SHA-256
        ↓
0..9999 bucket
        ↓
compare with configured traffic percentage
```

The same pseudonymous customer stays in the same arm for the lifetime of one challenger registry version, even when request IDs change. A new challenger registry version changes the assignment key, so a later canary can form a new cohort rather than repeatedly exposing exactly the same customers.

`customer_id` is expected to be a pseudonymous identifier supplied by the calling system. The canary controller does not need names, email addresses, or other direct identifiers for routing.

## Request path

When canary state is `warming` or `healthy`, the API scores both models on the same validated request.

```text
request
  ├── champion prediction ─────────────────────┐
  └── challenger prediction ───────────────────┤
                                               ↓
                                     comparison evidence
                                               ↓
                              deterministic assignment
                                  ├── champion response
                                  └── challenger response
```

Only requests assigned to the challenger cohort return the challenger decision. If challenger scoring fails on an assigned request, the response falls back to the champion and records `served_model_role=champion_fallback`.

Prediction provenance includes:

- `served_model_role`;
- `canary_assignment`;
- `registry_model_version` for the model that produced the served result;
- `comparison_champion_registry_version`;
- `comparison_challenger_registry_version`;
- model release, policy, feature-schema, decision, and audit identifiers.

## Online safety evidence

The controller keeps a bounded in-process evidence window and computes:

| Metric | Purpose |
|---|---|
| total comparisons | prevents decisions from being made on too little evidence |
| challenger-served requests | confirms that real canary traffic was exposed |
| action disagreement rate | detects changes in downstream policy action |
| p95 absolute probability delta | detects large score shifts even when the final action is unchanged |
| challenger error rate | detects runtime failures in the challenger path |
| champion/challenger mean latency ratio | detects operational regression |
| manual-review-rate increase | detects an unexpected increase in human review burden |

Promotion requires both minimum evidence counts and all configured safety limits to pass.

These metrics are safety and behavioural-equivalence checks. They are not a substitute for delayed outcome labels, causal evaluation, business-value measurement, fairness review, or statistical evidence that the challenger is better than the champion.

## State machine

```text
disabled
   │
   └── enabled + registry runtime
              ↓
waiting_for_challenger
              ↓ verified challenger
           warming
          /       \
 insufficient    enough evidence
 evidence            ↓
   WAIT         online limits
                /          \
             fail          pass
              ↓              ↓
           stopped        healthy
                             ↓ optional auto promotion
                          promoted
                             ↓ external incident rollback
                          rolled_back
```

### `waiting_for_challenger`

No verified challenger is available. All served traffic remains on the champion.

### `warming`

A verified challenger is loaded and canary traffic is active, but minimum evidence has not yet been collected.

### `healthy`

Minimum evidence is available and all configured online safety limits pass. With automatic promotion disabled, the system remains in this state until an external approval process changes the registry alias.

### `stopped`

A configured online limit failed. New requests are no longer assigned to the challenger. The current champion remains served. The challenger version remains identifiable for diagnosis.

### `promoted`

The controlled registry transition moved the challenger to the `champion` alias and the runtime reloaded that champion.

### `rolled_back`

The API detects that the active champion registry version no longer matches the version promoted by the completed canary session. This reports an external incident rollback instead of leaving the operational state misleadingly marked `promoted`.

## Automatic promotion

`CANARY_AUTO_PROMOTE_ENABLED` defaults to `false` in normal configuration and in the Helm chart.

The CI integration stack enables it only to prove the complete transition:

```text
champion A + challenger B
        ↓
real API canary traffic
        ↓
online safety gate PASS
        ↓
automatic controlled promotion
        ↓
B becomes champion
        ↓
incident rollback
        ↓
A becomes champion again
```

Promotion reuses the existing registry lifecycle function and passes the expected challenger registry version. The alias transition therefore remains guarded by the existing offline promotion evidence and version checks.

## Incident rollback

Canary promotion does not replace the existing registry rollback path. A post-promotion incident can still run the normal rollback operation. The registry runtime poller detects the champion alias change and atomically reloads the former champion without restarting the API process.

The canary status layer then reports `rolled_back` when the active champion no longer matches the version that the canary session promoted.

## Prometheus metrics

The API exports canary-specific runtime metrics including:

```text
regulated_ai_canary_enabled
regulated_ai_canary_state
regulated_ai_canary_traffic_percent
regulated_ai_canary_challenger_served
regulated_ai_canary_action_disagreement_rate
regulated_ai_canary_probability_delta_p95
regulated_ai_canary_challenger_error_rate
regulated_ai_canary_latency_ratio
regulated_ai_canary_assignments_total
```

`GET /canary/status` returns the current state, registry provenance, evidence-window metrics, configured limits, and last transition. It intentionally does not expose raw exception text.

## Configuration

Key environment variables:

```text
CANARY_ENABLED=false
CANARY_TRAFFIC_PERCENT=5
CANARY_ASSIGNMENT_SEED=regulated-ai-canary-v1
CANARY_MIN_REQUESTS=200
CANARY_MIN_CHALLENGER_REQUESTS=10
CANARY_WINDOW_SIZE=1000
CANARY_MAX_ACTION_DISAGREEMENT_RATE=0.05
CANARY_MAX_PROBABILITY_DELTA_P95=0.15
CANARY_MAX_CHALLENGER_ERROR_RATE=0.01
CANARY_MAX_LATENCY_RATIO=2.0
CANARY_MAX_MANUAL_REVIEW_RATE_INCREASE=0.10
CANARY_AUTO_PROMOTE_ENABLED=false
CANARY_EVALUATION_INTERVAL_SECONDS=15
CANARY_REFRESH_INTERVAL_SECONDS=30
```

The Helm chart exposes the same controls under `registryRuntime.canary` and keeps canary mode disabled by default.

## Failure behaviour

| Failure | Serving behaviour |
|---|---|
| challenger alias missing | champion only; state waits for challenger |
| challenger bundle checksum fails | candidate rejected; champion unchanged |
| feature schema mismatch | candidate rejected; champion unchanged |
| offline promotion gate not PASS | candidate rejected; champion unchanged |
| challenger scoring error | assigned request falls back to champion |
| online safety limit fails | canary moves to `stopped`; challenger receives no new served traffic |
| promotion fails | champion unchanged; promotion failure is logged |
| registry unavailable after champion is loaded | registry runtime keeps its verified last-known-good behaviour |
| post-promotion incident rollback | former champion is restored by registry hot reload; status reports `rolled_back` |

## CI proof and what it means

The integration test starts real PostgreSQL, MinIO, MLflow, and FastAPI containers. It creates champion registry version A and challenger registry version B, sends deterministic pseudonymous customers through `/predict`, verifies that both arms receive traffic, waits for the online gate, verifies controlled promotion, then executes an incident rollback and verifies that the same API container restores A.

For this lifecycle test, A and B intentionally contain the same validated model artifact. Therefore zero action disagreement and zero probability delta are expected. The test proves routing, provenance, evidence collection, state transitions, alias mutation, hot reload, and rollback. It does **not** claim that a new model is better than the champion.

A separate unit test injects a deliberately divergent challenger and verifies that the online gate returns `STOP` and subsequent requests remain on the champion.

## Multi-replica production boundary

The reference controller stores its evidence window in process memory. This is suitable for the single-API integration proof and for demonstrating the control logic, but it is not a distributed consensus system.

For a multi-replica production deployment, automatic promotion should be driven by one elected controller or by an external deployment controller using centrally aggregated metrics. Otherwise different API replicas can observe different local evidence windows. The Helm chart therefore keeps `autoPromoteEnabled=false` by default.

A production implementation should also add authenticated control-plane APIs, durable canary-session state, centralized metric aggregation, approval records where required, delayed-label evaluation, alerting, and explicit SLO-based rollback policies.
