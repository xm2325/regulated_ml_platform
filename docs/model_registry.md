# MLflow model registry

The platform adds a runnable MLflow registry backed by PostgreSQL for metadata and MinIO for model artifacts.

## Lifecycle

```text
local release artifacts
        │
        ▼
register → challenger
        │
        ├── gate is not PASS → promotion blocked
        │
        └── gate is PASS
                │
                ▼
previous champion → rollback
challenger        → champion
                │
                ▼
incident rollback
rollback          → champion
failed champion   → challenger
```

Registration and promotion are separate operations. Training does not silently change the served model. The promotion command checks `reports/promotion_gate.json` and stops unless both conditions hold:

- `status == PASS`;
- `release_recommendation == eligible_for_controlled_release`.

## Start the stack

```bash
docker compose -f docker-compose.registry.yml up -d --build postgres minio minio-init mlflow
curl --fail http://localhost:5000/health
```

The local interfaces are:

| Service | Address | Purpose |
|---|---|---|
| MLflow | `http://localhost:5000` | experiments, runs, model versions, aliases, tags |
| MinIO S3 API | `http://localhost:9000` | model and release artifacts |
| MinIO console | `http://localhost:9001` | local object-store inspection |
| PostgreSQL | internal only | runs, experiments, model registry metadata |

The credentials in `docker-compose.registry.yml` are local demonstration credentials and must not be reused outside an isolated development machine.

## Register a challenger

```bash
docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli \
  register --model-version 0.6.0
```

This operation logs metrics and release files, logs the scikit-learn model flavor, creates a model version, adds audit tags, and points `challenger` to the new version.

## Controlled promotion

```bash
docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli \
  promote --gate reports/promotion_gate.json
```

When a champion already exists, promotion moves that version to `rollback` before changing `champion`.

## Rollback

```bash
docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli \
  rollback --reason "calibration drift exceeded the approved limit"
```

The safe version under `rollback` becomes champion. The failed champion becomes challenger for investigation.

## Inspect and verify

```bash
docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli status

docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli \
  verify --alias champion --request examples/review_request.json
```

`verify` downloads the raw versioned `model.joblib` from the MLflow artifact proxy and runs one probability prediction. This checks that the alias resolves, the MinIO artifact can be retrieved, and the serialized model can be loaded.

## Sync for serving

```bash
docker compose -f docker-compose.registry.yml --profile registry run --rm registry-cli \
  sync --alias champion --output-dir registry-sync
```

The sync operation downloads the model, metadata, metrics, and promotion gate, writes them through temporary files, then atomically replaces the local release files. It also writes `registry_provenance.json` with the model version, run ID, alias, and source URI.

## Production boundary

The Compose stack is a local integration environment. A production setup still needs secret management, TLS, authentication, database backup and restore, object retention, access policies, high availability, audit-log export, and network isolation.
