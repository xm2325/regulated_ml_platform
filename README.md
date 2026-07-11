# Regulated AI MLOps Platform

[![Platform](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/platform.yml)
[![Progressive delivery](https://github.com/xm2325/regulated_ml_platform/actions/workflows/gitops.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/gitops.yml)
[![CodeQL](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/xm2325/regulated_ml_platform/actions/workflows/codeql.yml)

**A production-style reference platform for turning an ML model into a controlled, observable, reviewable and reversible service.**

Platform/service release: `0.8.0`

## Result first

The platform separates model evaluation, decision policy, human review, registry state, deployment state and release evidence. Version `0.8.0` adds the controls that sit between a tested container and a governed multi-environment release:

| Capability | Implemented evidence |
|---|---|
| GitOps environments | Argo CD definitions for dev, preprod and prod plus Kustomize overlays and ResourceQuotas |
| Progressive delivery | Argo Rollouts stages at 5%, 25%, 50% and 100% with Prometheus analysis and automatic abort |
| Fail-closed promotion | Deterministic Python canary gate covering traffic, reliability, drift, fairness and stable-baseline regression |
| GPU serving contract | Inactive NVIDIA Triton ONNX/A100 contract with dynamic batching, probes and hardened networking; CI renders it but no environment deploys it |
| Operational visibility | Prometheus discovery and alert rules plus a provisioned Grafana dashboard for rollout, model and GPU signals |
| Release integrity | One `VERSION`, cross-surface consistency checks, source-run validation and same-SHA artifact verification |

These controls are executable configuration and tested contracts. They are not a claim that this demonstration has operated real bank traffic, a live GKE/A100 estate, Harness or Dynatrace.

## Verified benchmark baseline

The latest publicly inspectable model benchmark before the `0.8.0` infrastructure upgrade is the successful [Platform run 29098202613](https://github.com/xm2325/regulated_ml_platform/actions/runs/29098202613) at commit [`a672e081e6b8331c546eb4e5a742c1a6cd356903`](https://github.com/xm2325/regulated_ml_platform/commit/a672e081e6b8331c546eb4e5a742c1a6cd356903). It used a dated synthetic cohort and model version `0.6.0`.

| Out-of-time synthetic result | Value |
|---|---:|
| AUC | 0.7884 |
| AUC bootstrap 95% interval | 0.7470-0.8226 |
| Brier score | 0.1795 |
| Expected calibration error | 0.0748 |
| Policy precision | 0.8755 |
| Policy recall | 0.7098 |
| API load-test p95 | 50.8 ms |
| Promotion and release gates | PASS |

The `0.8.0` release must create a new same-SHA evidence set before publication. The release workflow rejects a model version, source commit, policy, schema, gate or site that does not match the validated source run.

## Evidence truth model

The repository deliberately distinguishes three things:

1. Source contracts are the Python, policy, workflow and deployment files reviewed in Git.
2. A committed evidence snapshot under `docs/`, `reports/` or `site/` is useful for orientation but may describe an earlier validated release.
3. A CI-generated release artifact from one successful Platform run is authoritative for publication. It contains the matching model artifacts, metrics, approval evidence and generated site for one commit.

`scripts/check_release_consistency.py` validates both the source declarations and the downloaded artifact set. Account-level GHCR or Pages failures remain separate from required engineering validation.

## Evaluation design

```text
2025-01-01                                                     2025-12-31
|                                                                     |
+--------- train ---------+ selection + calibration + policy + OOT test
|       3,000 rows        | 500 rows |  500 rows  |500 rows| 500 rows
|                         |          |            |        |
fit preprocessing         choose     fit Platt    freeze   report only
and candidate models      champion   scaling      threshold
```

The final window is not used for candidate selection, probability calibration or policy-threshold selection. The dataset, target, drift and consumer actions are synthetic.

## Controlled decision path

```text
validated request
      |
      v
feature construction ---------------- feature_schema_version
      |
      v
calibrated probability --------------- model_version
      |
      v
versioned policy + safety gates ------ policy_version
      |
      +---- auto_serve
      +---- manual_review
      |
      v
decision_id + audit_event_id + metrics + redacted log
```

The model does not directly issue the final action. `src/serving/policy.py` maps probability and safety conditions to an action; `src/serving/review_workflow.py` independently decides whether human review is required.

## Build, registry and release flow

```text
Platform workflow
  +-- dated data, model selection, calibration and OOT evidence
  +-- tests, lint, security, dependency audit and SBOM
  +-- exact Docker image tested locally and in kind/Helm
  +-- PostgreSQL + MinIO + MLflow lifecycle integration
  +-- model-artifacts + release-evidence + generated-site
                 |
                 v
Release workflow validates run, repository, event, branch, SHA and artifacts
  +-- publish the exact tested image to GHCR
  +-- publish the generated evidence site to GitHub Pages
                 |
                 v
GitOps promotion changes an immutable image digest by reviewed pull request
  +-- Argo CD reconciles desired state
  +-- Argo Rollouts advances 5% -> 25% -> 50% -> 100%
  +-- missing or failed analysis aborts to the stable ReplicaSet
```

The normal GitHub-hosted runner validates the GitOps manifests and the Triton deployment contract without pretending to have an A100. Real CUDA/TensorRT parity, latency, throughput and cost evidence requires a controlled GPU runner or cloud environment.

## Repository map

| Layer | Key paths |
|---|---|
| Data and temporal evaluation | `src/data/`, `src/features/`, `src/models/` |
| Registry lifecycle | `src/registry/`, `docker-compose.registry.yml` |
| Decision service | `src/serving/`, `config/policy.yaml` |
| Governance and evidence | `src/governance/`, `config/promotion_gate.yaml` |
| Canary decision | `src/operations/canary_gate.py`, `config/canary_gate.yaml` |
| GitOps and rollouts | `gitops/argocd/`, `gitops/base/`, `gitops/environments/` |
| GPU/Triton contract | `gitops/gpu/`, `serving/triton/` |
| Observability | `monitoring/` |
| Cloud validation | `.github/workflows/platform.yml`, `.github/workflows/gitops.yml`, `.github/workflows/release.yml` |

## Reproduce locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

make evidence
make lint
make security
make release-consistency
```

Validate the new delivery controls:

```bash
pytest -q tests/test_progressive_delivery.py tests/test_gitops_image_update.py
python -m src.operations.canary_gate \
  --metrics examples/canary_metrics_pass.json \
  --config config/canary_gate.yaml \
  --output reports/canary_gate.json
kubectl kustomize gitops/environments/preprod
kubectl kustomize gitops/gpu
```

Read [`docs/progressive_delivery.md`](docs/progressive_delivery.md), [`docs/gpu_serving.md`](docs/gpu_serving.md), [`docs/model_registry.md`](docs/model_registry.md), and [`docs/architecture.md`](docs/architecture.md) for the control boundaries and operational rationale.

## Boundary

This repository is an engineering demonstration using synthetic data. It is not financial advice, a validated banking model, evidence of production operation, or a substitute for security, data-protection, model-risk and accountable business approval.
