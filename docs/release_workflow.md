# Release workflow contract

The release workflow consumes artifacts only from a successful Platform workflow run produced by a `push` to `main` in this repository.

It validates the source run ID, event type, branch, repository, conclusion and commit SHA before using any artifact. The Docker image is not rebuilt. The exact image already tested as a container and inside kind is exported by the Platform workflow, protected with a SHA-256 checksum, downloaded by the Release workflow and then pushed to GHCR.

GitHub Pages and GHCR are account-level publication targets. Their failures are reported separately from required model, code, Docker, Kubernetes and registry checks.
