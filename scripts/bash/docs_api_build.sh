#!/bin/bash
set -euo pipefail

echo "Build the HTML API documentation using Doxygen"

if ! command -v doxygen >/dev/null 2>&1; then
    echo "Error: Doxygen is required to build the API documentation."
    echo "Install Doxygen, then rerun this script."
    exit 1
fi

doxygen docs/Doxyfile

if [ ! -f docs/site/api/index.html ]; then
    echo "Error: Doxygen did not produce docs/site/api/index.html"
    exit 1
fi
