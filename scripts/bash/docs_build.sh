#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

echo "Build the HTML documentation using mdBook"
"$(dirname "$0")/docs_build_mdbook.sh"
