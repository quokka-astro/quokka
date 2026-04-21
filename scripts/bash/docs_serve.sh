#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Starting the MkDocs development server"
bash "$SCRIPT_DIR/docs_mkdocs.sh" serve "$@"
