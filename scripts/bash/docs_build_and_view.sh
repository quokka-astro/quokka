#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

echo "Build the HTML documentation using MkDocs"
bash "$SCRIPT_DIR/docs_mkdocs.sh" build "$@"

cd "$REPO_ROOT/docs/site"

# Find an available port with smart port selection
if [ -n "${PORT:-}" ]; then
    # User specified a port via environment variable
    if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Error: Specified port $PORT is already in use"
        exit 1
    fi
else
    # Try common development ports first
    for candidate_port in 3000 5000 8000 8080; do
        if ! lsof -Pi :$candidate_port -sTCP:LISTEN -t >/dev/null 2>&1; then
            PORT=$candidate_port
            break
        fi
    done

    # If no common ports available, find next available port starting from 8080
    if [ -z "${PORT:-}" ]; then
        PORT=8080
        MAX_ATTEMPTS=100
        attempts=0

        while lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; do
            PORT=$((PORT + 1))
            attempts=$((attempts + 1))
            if [ $attempts -ge $MAX_ATTEMPTS ]; then
                echo "Error: Could not find an available port after $MAX_ATTEMPTS attempts"
                exit 1
            fi
        done
    fi
fi

echo "Serving documentation on http://localhost:$PORT"
python3 -m http.server "$PORT"
