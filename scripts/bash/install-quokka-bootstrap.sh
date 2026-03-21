#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-$HOME/.local/bin/quokka}"

mkdir -p "$(dirname "$DEST")"
install -m 0755 "$ROOT/scripts/bash/_quokka-launcher.sh" "$DEST"

echo "Installed bootstrapper at $DEST"
echo "Ensure $(dirname "$DEST") is on PATH."
