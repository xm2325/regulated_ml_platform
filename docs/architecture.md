# Platform architecture and evidence boundaries

## Objective

The platform demonstrates how a model probability becomes a versioned service decision whose source, model, policy, deployment and release evidence can be reviewed and rolled back together. The architecture favours explicit contracts and fail-closed gates over invisible automation.

## Trust boundaries

| Boundary | Authoritative object | Control |
|---|---|---|
| Source | reviewed Git commit | tests, security checks and `VERSION` consistency |
| Model | immutable artifact plus checksum | chronological evaluation, calibration and promotion gate |
| Registry | immutable MLflow model version | challenger/champion/rollback aliases and controlled transition |
| Container | tested image digest | build once, test once, publish the same image |
| Environment | GitOps desired state | reviewed digest change and Argo CD reconciliation |
| Traffic | stable and canary Services | Argo Rollouts weights and Prometheus analysis |
| Decision | model + schema + policy + review route | versioned response and audit identifiers |
| Release evidence | one successful workflow run and source SHA | cross-artifact validation before GHCR or Pages publication |

Tracked documents and reports are a committed evidence snapshot for orientation. A CI-generated release artifact from one successful Platform run is the publication authority.

## Planes

### 1. Evidence plane

The Platform workflow creates dated synthetic data, point-in-time features, candidate models, a Platt calibrator and an out-of-time evidence set. Candidate selection, calibration, policy-threshold selection and final evaluation use separate chronological windows. Governance generators add data-quality, privacy, drift, segment, explainability, reproducibility, deployment, load, incident and approval evidence.

### 2. Registry plane

MLflow stores runs and immutable model versions. PostgreSQL provides tracking and registry metadata; MinIO stores artifacts. Registration creates a challenger. Promotion requires a passing gate, moves the previous champion to rollback and then assigns champion to the approved challenger. Rollback preserves the incident reason and the failed version as challenger. Serving synchronisation copies model, metadata, metrics, gate and provenance as one release unit.

### 3. Serving and decision plane

FastAPI validates the request and constructs features. The calibrated model returns a probability; a separate deterministic policy maps the probability and safety conditions to an action; another component selects automatic service or manual review. Every response carries model, policy and feature-schema versions plus decision and audit identifiers. Prometheus exposes request, latency, decision and readiness signals.

### 4. Build and release plane

The required Platform workflow builds evidence once and reuses it. The same container image is smoke-tested directly and inside kind with Helm. The release workflow runs only after a successful main-branch Platform run, validates repository/event/branch/workflow/SHA, downloads model, evidence, site and exact image artifacts, verifies cross-file consistency, then publishes GHCR and Pages independently of the engineering gate.

### 5. GitOps and progressive-delivery plane

Kustomize overlays define dev, preprod and prod namespaces and quotas. Argo CD automatically reconciles non-production desired state; the production Application uses manual synchronisation. A promotion helper changes only an approved immutable image digest. Argo Rollouts directs 5%, 25%, 50% and then 100% of traffic to a candidate. Prometheus analyses request volume, availability, errors, p95 latency, drift and fairness. Missing mandatory evidence is a failure. An aborted rollout leaves the stable Service on the previous ReplicaSet.

The Python canary gate evaluates an exported observation window against the same operational intent and produces retained PASS or ROLLBACK JSON evidence. Its comparison against the stable baseline catches regressions that remain under an absolute ceiling.

### 6. GPU serving plane

The Triton configuration defines an ONNX contract, dynamic batching and one GPU instance per pod. Kubernetes configuration targets a tainted GKE A100 node pool, requests one `nvidia.com/gpu`, spreads replicas, limits writable storage, disables the API token, drops capabilities and applies probes and NetworkPolicy. Triton and DCGM signals feed Prometheus and Grafana.

The checked-in files are a deployment and scheduling contract, not a checked-in model binary or proof of GPU execution. A production build must add the approved ONNX model, pin the final image by digest, run CPU/GPU parity tests and benchmark representative traffic on the actual accelerator.

## Version and release unit

`VERSION` is the platform/service source of truth. The release unit contains:

- source commit and image digest;
- model checksum, version and registry lineage;
- policy and feature-schema versions;
- OpenAPI and decision contract;
- rendered environment manifest;
- SBOM and dependency/security results;
- promotion, canary and approval evidence.

`scripts/check_release_consistency.py` rejects source declarations that disagree with `VERSION` and rejects downloaded artifacts whose version, commit, policy, schema, training event or PASS state does not identify one release.

## Failure and rollback

Operational failure stops environment promotion. Argo Rollouts aborts a failing canary and preserves the stable route. MLflow aliases restore an already approved model. The declared GitOps image/config change is reverted so live and desired state agree. The incident record retains the failing/stable digests, model/policy/schema versions, request/audit identifiers, AnalysisRuns, alerts, metrics and corrective evidence.

Rolling back only the model while keeping an incompatible policy or schema is not allowed.

## Security and governance boundary

The reference controls include non-root containers, read-only filesystems, explicit resources, probes, service-account minimisation, NetworkPolicy, quotas, SBOM, dependency audit, CodeQL, privacy checks, model cards, reproducibility manifests and controlled promotion. Real use still requires organisation-specific identity, secrets, encryption, service mesh, retention, high availability, threat modelling, data protection, model-risk and accountable business approval.
