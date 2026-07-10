# GitHub Actions release troubleshooting

## Result first

The `Platform build, test, container, and evidence` workflow is the required engineering gate. It tests the model evidence, Python code, dependency locks, Docker image, kind cluster, Helm chart, API endpoints, and MLflow registry lifecycle.

Publication is intentionally separated into `Publish tested image and evidence site`. A Pages or GHCR account-setting problem therefore does not make the engineering validation badge red.

## Workflow relationship

```text
Platform workflow on main
  ├── generated-site artifact
  ├── model/release evidence artifacts
  └── exact Docker image tested locally and in kind
            │
            ▼
Release workflow
  ├── verify and push the same image to GHCR
  └── deploy the generated site to GitHub Pages
```

The release workflow does not retrain the model and does not rebuild the container.

## GitHub Pages failure

A custom Pages workflow requires the repository publishing source to be set to **GitHub Actions**.

1. Open repository **Settings**.
2. Open **Pages** under **Code and automation**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Re-run the failed `deploy-pages` job or manually run the Release workflow with the successful Platform run ID and source commit SHA.

The workflow needs `pages: write` and `id-token: write`. These are set on the `deploy-pages` job. The normal `GITHUB_TOKEN` cannot enable Pages for a repository that has never been enabled, so this one repository setting must be made by an administrator.

## GHCR failure

The GHCR job uses `GITHUB_TOKEN` and requests `packages: write`.

If the first push is denied:

1. Open **Settings → Actions → General**.
2. Confirm workflows are allowed to request write permissions.
3. Check the package page under the account's **Packages** section.
4. If a package named `regulated_ml_platform` already exists but was created outside this repository, open its package settings and grant this repository Actions access, or connect the package to this repository.

The runtime Dockerfile includes `org.opencontainers.image.source`, which helps associate the container with this repository.

## Artifact failure

The Release workflow must use artifacts from a successful Platform run on `main`:

- `tested-container-image`
- `generated-site`

Artifacts are retained for three and fourteen days respectively. A manual release must therefore use a recent successful run. The Release workflow verifies the image SHA-256 checksum before loading and publishing it.

## What to inspect

For a red run, identify the failed job first:

| Job | Meaning |
|---|---|
| `evidence` | Data, model, governance, test, lint, or dependency problem |
| `container-and-kind` | Docker runtime, Kubernetes, Helm, or API smoke-test problem |
| `registry-integration` | PostgreSQL, MinIO, MLflow, alias, promotion, rollback, or registry download problem |
| `publish-ghcr` | Package authentication, package ownership, or account permission problem |
| `deploy-pages` | Pages source, Pages permission, or environment configuration problem |

Do not add `continue-on-error` to required validation jobs. Fix the root cause and keep the failure history.
