# Deployment validation

## Result: **PASS**

The checked Kubernetes deployment includes the controls below.

| Control | Result |
|---|---|
| non-root user | PASS |
| fixed user and group ID | PASS |
| read-only root filesystem | PASS |
| privilege escalation disabled | PASS |
| Linux capabilities dropped | PASS |
| RuntimeDefault seccomp profile | PASS |
| service-account token disabled | PASS |
| startup probe | PASS |
| readiness probe | PASS |
| liveness probe | PASS |
| CPU and memory requests | PASS |
| CPU and memory limits | PASS |
| rolling update with zero unavailable pods | PASS |
| HorizontalPodAutoscaler | PASS |
| PodDisruptionBudget | PASS |
| NetworkPolicy | PASS |

The manifest is a reference deployment. A real environment still needs image signing, admission policy, secret management, cluster-specific ingress, external audit storage, and platform-owner approval.
