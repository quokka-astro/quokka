#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

echo "Build the HTML documentation using mdBook"

if ! command -v mdbook >/dev/null 2>&1 || ! command -v mdbook-bib >/dev/null 2>&1 || ! command -v mdbook-plantuml >/dev/null 2>&1; then
    echo "Error: mdBook tooling is required to build the docs."
    echo "Install it with: ./scripts/bash/install_mdbook.sh"
    exit 1
fi

mdbook clean docs
mdbook build docs
python3 scripts/check_mdbook_summary.py
