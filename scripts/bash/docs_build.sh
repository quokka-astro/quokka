#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

echo "Build the HTML documentation using mdBook"

if ! command -v mdbook-bib >/dev/null 2>&1; then
    echo "Error: mdbook-bib is required for citation rendering."
    echo "Install it with: cargo install mdbook-bib --version 0.5.2 --locked"
    exit 1
fi

mdbook clean docs
mdbook build docs
python3 scripts/check_mdbook_summary.py
