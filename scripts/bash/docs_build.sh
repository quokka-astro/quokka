#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Build the HTML documentation using MkDocs"
bash "$SCRIPT_DIR/docs_mkdocs.sh" build "$@"
