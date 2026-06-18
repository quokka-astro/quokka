#!/bin/bash
set -euo pipefail

readonly MDBOOK_VERSION="0.5.3"
readonly MDBOOK_BIB_VERSION="0.5.2"
readonly MDBOOK_PLANTUML_VERSION="2.0.0"

echo "Install mdBook tooling used by the CI docs jobs"

if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo is required to install mdBook tooling."
    echo "Install the Rust toolchain, then rerun this script."
    exit 1
fi

cargo install mdbook --version "${MDBOOK_VERSION}"
cargo install mdbook-bib --version "${MDBOOK_BIB_VERSION}"
cargo install mdbook-plantuml --version "${MDBOOK_PLANTUML_VERSION}"
