#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$(mktemp -t regulated-ml-platform.XXXXXX.tar.xz)"
trap 'rm -f "$ARCHIVE"' EXIT

cd "$ROOT"
cat \
  bundles/platform_source_v0.5.0.tar.xz.b64.part001 \
  bundles/platform_source_v0.5.0.tar.xz.b64.part002 \
  | base64 --decode > "$ARCHIVE"
cat \
  bundles/source_binary_003.bin \
  bundles/source_binary_004.bin \
  bundles/source_binary_005_*.bin \
  >> "$ARCHIVE"
echo "0785e71b6a733c20be97adee4d75348d7fb5b88aefc531f717516057019756bf  $ARCHIVE" | sha256sum --check --strict
tar -xJf "$ARCHIVE" -C "$ROOT"
printf '%s\n' "Source restored and checksum verified. Run: make all"
