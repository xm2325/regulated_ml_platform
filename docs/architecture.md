# Architecture

## System conclusion

The platform treats model fitting, probability calibration, policy logic, human review, audit evidence, and deployment as separate units. Each unit has its own version or artifact and can be tested or rolled back without treating the full decision as one opaque model output.

## Request path

```text
Client
  │  HTTPS + X-Request-ID
  ▼
FastAPI contract validation
  │
  ▼
Feature builder ───────────── feature_schema_version
  │
  ▼
Calibrated champion model ─── model_version
  │ probability
  ▼
Decision policy ───────────── policy_version
  │ action + policy reasons
  ▼
Review router
  ├── auto_serve
  └── manual_review
  │
  ▼
Decision response + redacted audit event + Prometheus metrics
```

## Training and release path

```text
dated synthetic source
  → schema and privacy checks
  → point-in-time feature build
  → train window
  → model-selection window
  → calibration window
  → policy-validation window
  → frozen calibrated model + threshold
  → out-of-time test window
  → segment, drift, explanation, and uncertainty reports
  → promotion gate
  → deployment validation
  → SBOM and artifact manifest
  → release approval pack
  → MLflow registration as challenger
  → controlled promotion gate
  → champion / rollback aliases
```

The chronological windows prevent later observations from entering fitting or selection stages. The final test dates are later than every date used for fitting, calibration, or threshold choice.

## CI/CD path

```text
evidence job
  ├── model-artifacts
  ├── release-evidence
  └── generated-site
          │
          ├── build image once
          │     → local container smoke test
          │     → kind + Helm deployment test
          │     → GHCR push on main
          │
          ├── registry integration
          │     → PostgreSQL metadata backend
          │     → MinIO artifact store
          │     → MLflow tracking and registry server
          │     → register A → promote A
          │     → register B → promote B
          │     → rollback → verify → sync
          │
          └── GitHub Pages deployment
```

Training and evidence generation run once per commit. Docker, the registry integration test, and Pages consume artifacts from that run rather than retraining independently. The registry test proves that model aliases and MinIO artifacts can be used to promote, roll back, download, and score a release.

## Versioned units

| Unit | Version field | Rollback unit |
|---|---|---|
| API service | `service_version` | container image |
| Calibrated champion | `model_version` | `model.joblib` + metadata |
| Feature schema | `feature_schema_version` | feature code and contract |
| Decision policy | `policy_version` | policy module and YAML |
| Model contract | `contract_version` | JSON contract |
| Deployment | image tag and manifest commit | Kubernetes revision |
| Registry release | MLflow model version + run ID | `champion`, `challenger`, and `rollback` aliases |

## Production boundary

The local JSONL audit sink is for development only. A real deployment should send redacted audit events to an external append-only store with access control, retention rules, and monitored delivery failures.
