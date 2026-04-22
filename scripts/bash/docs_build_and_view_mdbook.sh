#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

echo "Serving the HTML documentation using mdBook"

if ! command -v mdbook-bib >/dev/null 2>&1; then
    echo "Error: mdbook-bib is required for citation rendering."
    echo "Install it with: cargo install mdbook-bib --version 0.5.2 --locked"
    exit 1
fi

# Find an available port with smart port selection
if [ -n "$PORT" ]; then
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Error: Specified port $PORT is already in use"
        exit 1
    fi
else
    for candidate_port in 3000 5000 8000 8080; do
        if ! lsof -Pi :$candidate_port -sTCP:LISTEN -t >/dev/null 2>&1; then
            PORT=$candidate_port
            break
        fi
    done

    if [ -z "$PORT" ]; then
        PORT=8080
        MAX_ATTEMPTS=100
        attempts=0

        while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
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
mdbook serve docs --hostname localhost --port "$PORT"
