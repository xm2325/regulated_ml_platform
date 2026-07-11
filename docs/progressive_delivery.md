# Progressive delivery and automatic rollback

## Purpose

This control plane limits the blast radius of a model/API release. Argo Rollouts moves traffic through 5%, 25%, 50%, and 100% stages, while Prometheus checks operational and model-risk signals at each analysis point. A failed or unavailable mandatory signal aborts the rollout and leaves the stable Service on the last healthy ReplicaSet.

The design provides two related controls:

1. `gitops/base/analysis-template.yaml` makes the in-cluster, real-time decision that can stop traffic promotion.
2. `src/operations/canary_gate.py` makes a deterministic `PASS` or `ROLLBACK` decision from an exported observation window, suitable for retention in a release evidence pack.

Neither control replaces model-risk approval. A `PASS` only authorises the next bounded delivery stage.

## Repository map

| Path | Responsibility |
|---|---|
| `gitops/base/` | Reusable Rollout, stable/canary Services, and Prometheus `AnalysisTemplate` |
| `gitops/environments/dev/` | One-replica development overlay and small ResourceQuota |
| `gitops/environments/preprod/` | Two-replica dress-rehearsal overlay and intermediate quota |
| `gitops/environments/prod/` | Four-replica production overlay and production quota |
| `gitops/gpu/` | Inactive Triton/A100 serving contract, rendered in CI but not reconciled by an environment |
| `gitops/argocd/` | Constrained Argo CD project, automated nonproduction ApplicationSet, and manual production Application |
| `config/canary_gate.yaml` | Versioned threshold policy |
| `examples/canary_metrics_*.json` | Auditable pass and rollback input contracts |
| `monitoring/` | Prometheus discovery/rules and provisioned Grafana dashboard |

Each active environment owns a namespace labelled with the Kubernetes `restricted` Pod Security standard. CPU, memory, pod, Service, PVC, and reserved GPU quotas bound future capacity, but the GPU contract is not included in any environment overlay until its image and approved ONNX model have a complete build, parity-test, signing, and publication path.

## Delivery sequence

1. CI builds and tests an immutable image. Production promotion should identify it by digest even when a human-readable tag is also retained.
2. CI updates the image reference in the appropriate GitOps overlay. Argo CD or another reconciler applies the desired state; operators should not patch production Deployments directly.
3. Argo Rollouts creates a canary ReplicaSet and directs 5% of traffic through `regulated-ai-mlops-canary`.
4. After the initial pause, an `AnalysisRun` queries the canary Service. The same analysis repeats after the 25% and 50% stages.
5. All metrics must remain inside policy. Only then does the rollout advance to 100% and mark the new ReplicaSet stable.
6. The exported observation window is evaluated by the Python gate and retained with the image digest, source commit, model version, policy version, reviewer, and `AnalysisRun` name.

The production sequence intentionally takes longer than a rolling Deployment. That time buys three explicit observation windows before all customers see the release.

## Decision contract

The default policy is fail-closed:

| Signal | Promotion requirement | Why it matters |
|---|---:|---|
| Request volume | at least 1,000 requests in exported evidence | avoids decisions from an unrepresentative sample |
| Availability | at least 99.5% | customer-facing reliability |
| Error rate | no more than 1% | functional regression guard |
| p95 latency | no more than 250 ms | tail-latency customer impact |
| Population Stability Index | no more than 0.20 | material input/population shift |
| Fairness gap | no more than 0.10 | protected-group outcome disparity |
| Error-rate increase over stable | no more than 0.5 percentage points | relative regression guard |
| p95 increase over stable | no more than 20% | catches regressions below the absolute ceiling |

Prometheus analysis also requires at least 100 requests in each live five-minute query window. A missing request, drift, or fairness series is a failed analysis rather than an implicit zero. Threshold changes require a reviewed Git change to `config/canary_gate.yaml` and a matching change to the `AnalysisTemplate`; never relax a threshold during an active incident.

The evidence input is JSON:

```json
{
  "release_id": "credit-risk-2026-07-11.1",
  "observed_at": "2026-07-11T09:30:00Z",
  "window_seconds": 600,
  "sample_count": 12500,
  "canary": {
    "availability": 0.9992,
    "error_rate": 0.0038,
    "p95_latency_ms": 171.0,
    "drift_psi": 0.07,
    "fairness_gap": 0.042
  },
  "baseline": {"error_rate": 0.0031, "p95_latency_ms": 160.0}
}
```

Evaluate and retain it with:

```bash
python -m src.operations.canary_gate \
  --metrics examples/canary_metrics_pass.json \
  --config config/canary_gate.yaml \
  --output reports/canary_gate.json
```

Exit code `0` means `PASS`; exit code `2` means `ROLLBACK`. Missing, non-finite, out-of-range, or otherwise invalid mandatory metrics also produce `ROLLBACK`, with `input_errors` recorded in the output.

## Metric label contract

The FastAPI service exports `regulated_ai_http_requests_total` and `regulated_ai_http_request_duration_seconds_bucket`. Prometheus endpoint discovery adds `namespace`, `service`, and `rollout_hash` target labels. The Argo queries select the canary Service name rather than combining stable and canary pods.

The online monitoring publisher must expose the following gauges for the same `namespace` and canary `service` labels:

- `model_drift_psi{model_version=...}`: maximum PSI across monitored features for the observation window.
- `model_fairness_gap{model_version=...,protected_attribute=...}`: maximum absolute outcome gap across governed groups.

Publishing a stale value is not acceptable. Exporters should include the observation time, and a production deployment should add a freshness rule before promotion if metrics can persist beyond one analysis window.

## GitOps reconciliation and environment separation

Argo CD is the in-cluster reconciler. Bootstrap the constrained project first, then its applications:

```bash
kubectl apply -f gitops/argocd/project.yaml
kubectl apply -f gitops/argocd/nonprod-applicationset.yaml
kubectl apply -f gitops/argocd/prod-application.yaml
```

The `ApplicationSet` creates `dev` and `preprod` Applications from a fixed list. Both automatically prune removed resources and self-heal drift, so Git remains their source of truth. The production `Application` points only at `gitops/environments/prod` and deliberately has no `automated` sync policy. A production change therefore needs an approved Git merge followed by a separately authorised manual sync. Branch protection and Argo CD RBAC must ensure the same person cannot unilaterally merge and sync.

The `AppProject` restricts sources to this GitHub repository and destinations to `regulated-ml-*` namespaces. It permits only Namespace as a cluster-scoped resource and warns on orphans. Argo CD itself must run in a separately administered namespace; application workloads receive no Argo credentials.

Image promotion is digest-controlled. Once a cloud build has pushed and verified an image, update the desired overlay using the allow-listed helper:

```bash
python scripts/update_gitops_image.py \
  --environment preprod \
  --component api \
  --digest sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

python scripts/update_gitops_image.py \
  --environment prod \
  --component api \
  --digest sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210
```

The helper accepts only the published `api` component, one of the three known environment paths, and a lowercase 64-hex SHA-256 digest. It replaces only an image already declared in that overlay and removes any mutable `newTag`. CI should commit this change through the normal reviewed pull-request path; it should not apply workload manifests directly to the cluster. Development and preproduction start from the checked-in `0.8.0` bootstrap tag. Production starts from an intentionally non-pullable all-zero digest, so even a manual Argo sync fails closed until a reviewed promotion replaces it with the tested GHCR digest.

The Triton resources are deliberately absent from dev, preproduction, and production overlays. CI renders `gitops/gpu/` as a standalone contract, but activation is blocked until a separate change adds a real Triton image build, the approved `/models/credit-risk/1/model.onnx`, CPU/GPU parity and load evidence, signing, scanning, and digest promotion.

## Render and validate

Render overlays before merge:

```bash
kubectl kustomize gitops/environments/dev > /tmp/dev.yaml
kubectl kustomize gitops/environments/preprod > /tmp/preprod.yaml
kubectl kustomize gitops/environments/prod > /tmp/prod.yaml
kubectl kustomize gitops/gpu > /tmp/gpu-contract.yaml
kubectl apply --dry-run=server -f /tmp/preprod.yaml
```

Required cluster components are the Argo Rollouts CRDs/controller, Argo CD with ApplicationSet support, Prometheus with permission to discover pods/endpoints, and a metrics publisher for drift and fairness. The application ServiceAccount is part of the base. Separate Applications reconcile each environment path so a development change cannot accidentally target production.

Watch a release with:

```bash
kubectl argo rollouts get rollout regulated-ai-mlops -n regulated-ml-prod --watch
kubectl get analysisrun -n regulated-ml-prod
```

## Automatic rollback

An `AnalysisRun` failure reaches `failureLimit` and causes Argo Rollouts to abort the release. The stable Service continues selecting the previous ReplicaSet. `abortScaleDownDelaySeconds` keeps the failed canary briefly available for inspection, then removes its capacity; `progressDeadlineAbort: true` also aborts a rollout that cannot become ready within 15 minutes. The rollback window retains three revisions for a rapid, controlled return.

During an automatic rollback:

1. Confirm the stable endpoint meets the availability and latency SLOs.
2. Capture the Rollout, ReplicaSet, AnalysisRun, alert, dashboard window, image digest, and relevant logs before cleanup.
3. Stop downstream environment promotion and open an incident/change record.
4. Revert the GitOps image/config change so declared state agrees with the live stable version.
5. Diagnose against stable using the same traffic window; do not retry until the failed check has an evidenced remediation.
6. Repeat preproduction before a new production attempt. A manual `promote --full` is a break-glass action requiring incident commander and model-risk approval.

If the analysis system itself is unavailable, the release remains stopped. Service restoration has priority over bypassing a missing control.

## Evidence and ownership

The release evidence pack should contain the rendered production manifest, source revision, immutable image digest and SBOM, model/version lineage, signed approvals, policy files, canary gate JSON, all AnalysisRuns, and links to the corresponding Prometheus/Grafana time range. Retention and access controls should follow the organisation's model governance and change-management policy.

The ML platform team owns Rollout mechanics and observability. The model owner owns drift interpretation. Model Risk owns fairness acceptance and threshold exceptions. SRE owns incident command and restoration of the stable customer path.
