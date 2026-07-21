#!/usr/bin/env bash
set -euo pipefail

umask 077

if [[ $# -ne 1 ]]; then
  printf 'usage: %s OUTPUT_DIRECTORY\n' "$0" >&2
  exit 64
fi

output_dir=$1
repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
  printf 'the command must run inside a Git worktree\n' >&2
  exit 64
}
cd "$repo_root"

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  printf 'refusing to package a dirty worktree; commit or remove every change first\n' >&2
  exit 65
fi

source_commit=$(git rev-parse HEAD)
if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'expected a full lowercase Git SHA-1, got %s\n' "$source_commit" >&2
  exit 65
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd -P)
archive_name="regulated-ml-platform-${source_commit}.tar.gz"
archive_path="${output_dir}/${archive_name}"
manifest_path="${output_dir}/source-bundle.json"
checksum_path="${archive_path}.sha256"
archive_root="regulated_ml_platform-${source_commit}"

git archive --format=tar \
  --prefix="${archive_root}/" \
  --add-virtual-file="${archive_root}/.regulated-ml-source-commit:${source_commit}" \
  "$source_commit" | gzip -n >"$archive_path"

while IFS= read -r member; do
  if [[ -z "$member" || "$member" == /* || "$member" == ../* || "$member" == */../* ]]; then
    printf 'unsafe archive member: %s\n' "$member" >&2
    exit 65
  fi
done < <(tar -tzf "$archive_path")

marker=$(tar -xOzf "$archive_path" "${archive_root}/.regulated-ml-source-commit")
if [[ "$marker" != "$source_commit" ]]; then
  printf 'source marker does not match the packaged commit\n' >&2
  exit 65
fi

if command -v sha256sum >/dev/null 2>&1; then
  archive_sha256=$(sha256sum "$archive_path" | awk '{print $1}')
else
  archive_sha256=$(shasum -a 256 "$archive_path" | awk '{print $1}')
fi
if [[ ! "$archive_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'failed to calculate a SHA-256 digest\n' >&2
  exit 70
fi

printf '%s  %s\n' "$archive_sha256" "$archive_name" >"$checksum_path"
printf '{\n  "schema_version": "regulated-ml.roihu-source-bundle.v1",\n  "source_commit": "%s",\n  "archive_root": "%s",\n  "archive_filename": "%s",\n  "archive_sha256": "%s"\n}\n' \
  "$source_commit" "$archive_root" "$archive_name" "$archive_sha256" >"$manifest_path"

printf 'source_commit=%s\n' "$source_commit"
printf 'archive=%s\n' "$archive_path"
printf 'archive_sha256=%s\n' "$archive_sha256"
printf 'manifest=%s\n' "$manifest_path"
