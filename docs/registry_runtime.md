# Registry-driven serving runtime

## Decision first

The API can run in two explicit modes:

- `MODEL_SOURCE=local`: load the packaged validated model and do not contact MLflow.
- `MODEL_SOURCE=registry`: treat the MLflow `champion` alias as the desired serving version, but switch only after the downloaded release bundle passes local verification.

Changing an MLflow alias alone does not immediately replace the active Python predictor. The runtime first verifies the candidate and then performs an atomic in-memory reference swap.

## Control plane and serving data plane

```text
MLflow registry control plane
  champion / challenger / rollback aliases
          │
          │ poll champion alias
          ▼
ModelRuntimeManager
  download to staging directory
  verify required files
  verify SHA-256
  verify feature schema
  verify promotion gate
  deserialize model
  run smoke prediction
          │
          ├── fail → delete staging → keep current predictor
          │
          └── pass
                ↓
        verified local cache
                ↓
        atomic predictor swap
                ↓
           FastAPI requests
```

The registry is the source of desired version state. The verified local cache is the source of last-known-good recovery state.

## Required release bundle

Every served registry version must provide:

```text
model.joblib
metadata.json
model_metrics.json
promotion_gate.json
registry_provenance.json
```

`registry_provenance.json` records:

- registered model name;
- alias;
- immutable registry version;
- MLflow run ID;
- source model URI;
- synchronized file paths;
- SHA-256 for each required release file.

The runtime creates checksum evidence immediately after a registry sync and verifies it again before model activation.

## Candidate activation checks

A registry candidate is rejected unless all checks pass:

```text
required files present
AND checksum evidence matches
AND feature_schema_version matches service contract
AND promotion gate status == PASS
AND release_recommendation == eligible_for_controlled_release
AND model deserialization succeeds
AND smoke probability is finite and within [0, 1]
```

Only then is the active predictor pointer replaced.

## Atomic hot reload

The API keeps one active `ModelPredictor` reference protected by an `RLock`.

A reload builds and verifies the replacement object outside the final swap section. After validation:

```text
old predictor remains active
        │
        ├── build candidate
        ├── validate candidate
        ├── publish verified cache
        ▼
lock
  predictor = candidate
  update runtime provenance
unlock
        ▼
new requests use candidate
```

Requests already holding the old Python object can finish. New requests obtain the new predictor after the pointer swap.

No partially downloaded model can become active.

## Runtime states

| State | Meaning | Ready |
|---|---|---|
| `ready_local` | packaged local model is active by configuration | yes |
| `ready_registry` | current registry champion was downloaded and verified | yes |
| `ready_registry_cached` | process restarted using a previously verified cached registry bundle | yes |
| `degraded_local_fallback` | registry mode requested, but packaged local model is active because no verified registry model is available | yes unless strict startup is enabled |
| `degraded_registry_cached` | registry is unavailable or reload failed, but the last verified registry model remains active | yes |

Readiness is separated from degradation. A service can remain available with a verified fallback while exposing that it is not serving the currently requested registry state.

## Strict startup

`REGISTRY_STRICT_STARTUP=true` is intended for deployments where serving the packaged fallback is not acceptable.

Startup succeeds when:

- a valid cached registry bundle is available; or
- the current champion can be downloaded and verified.

Startup fails when neither condition is true.

With strict startup disabled, the packaged local model can remain available and the runtime reports `degraded_local_fallback`.

## Last-known-good cache

Verified registry bundles are stored under:

```text
<REGISTRY_CACHE_DIR>/
  current.json
  releases/
    <registry_version>/
      model.joblib
      metadata.json
      model_metrics.json
      promotion_gate.json
      registry_provenance.json
```

`current.json` is replaced atomically after a candidate has passed validation.

The cache is not a general artifact store. It contains only serving bundles that passed local activation checks.

## Non-root writable storage

The container runs as UID/GID `10001` and keeps a read-only application filesystem in Kubernetes.

Registry cache writes use:

```text
/var/lib/regulated-ai/registry-cache
```

The Docker image creates this directory with ownership `10001:10001`.

The Helm registry mode mounts an `emptyDir` at the same path and uses:

```text
runAsUser: 10001
runAsGroup: 10001
fsGroup: 10001
fsGroupChangePolicy: OnRootMismatch
```

This avoids running the API as root only to make the cache writable.

## Polling and rollback

The background poller checks the configured alias at `REGISTRY_RELOAD_INTERVAL_SECONDS`.

When the alias still points to the active registry version:

```text
reload_attempts += 1
last_reload_status = unchanged
active predictor does not change
```

When the alias changes:

```text
sync candidate
→ validate
→ cache
→ atomic swap
→ reload_successes += 1
```

When registry rollback changes `champion` back to an earlier immutable registry version, the same poller follows the alias and loads that earlier version using the same validation path.

Rollback is therefore not a special unverified model-loading path.

## Failure handling

| Failure | Runtime action |
|---|---|
| MLflow unavailable | keep current verified predictor |
| artifact download failure | delete staging; keep current predictor |
| checksum mismatch | reject candidate; keep current predictor |
| feature schema mismatch | reject candidate; keep current predictor |
| non-PASS promotion gate | reject candidate; keep current predictor |
| model cannot deserialize | reject candidate; keep current predictor |
| invalid smoke prediction | reject candidate; keep current predictor |
| cache write failure | keep current predictor and record reload failure |

`reload_failures` is recorded by the reload operation itself, so startup, background reload, and explicit reload calls use the same accounting path.

## API provenance

`GET /version` reports:

```text
platform_version
service_version
model_release_version
policy_version
feature_schema_version
model_source
runtime_state
registry_model_name
registry_alias
registry_model_version
registry_run_id
last_reload_status
```

`POST /predict` also returns the active model source and registry version.

`GET /runtime/model` reports reload counters and active runtime state but does not return the stored internal error text.

## Prometheus metrics

The API exports:

```text
regulated_ai_model_ready
regulated_ai_registry_model_active
regulated_ai_runtime_degraded
regulated_ai_model_reload_attempts
regulated_ai_model_reload_successes
regulated_ai_model_reload_failures
```

These metrics are separate from request latency, HTTP status, prediction, action, and review-route metrics.

## CI proof

The registry integration job uses real containers for PostgreSQL, MinIO, MLflow, and the API.

A successful test performs:

```text
register A
→ promote A
→ register B
→ promote B
→ rollback B to A
→ start API from champion A
→ register C
→ promote C
→ API hot-reloads C without restart
→ call /predict and verify C provenance
→ rollback C to A
→ same API hot-reloads A without restart
→ verify final aliases and synchronized champion bundle
```

The verified v0.8 CI run observed:

```text
initial registry version: 1
hot-reloaded registry version: 3
rolled-back registry version: 1
reload failures: 0
```

The integer registry versions are created by the temporary CI registry. The validated model release stored inside those registry versions remains `0.6.0`.

## Production boundary

The local stack intentionally uses demonstration credentials and plain HTTP.

A real deployment still needs controls such as:

- authenticated MLflow access;
- TLS;
- secret management;
- PostgreSQL backup and recovery;
- object-store retention and access policy;
- registry availability design;
- network isolation;
- audit retention;
- controlled emergency rollback authority.

The Helm chart only provides an opt-in registry-runtime client path. It does not claim that the demonstration PostgreSQL, MinIO, and MLflow Compose stack is a production registry deployment.
