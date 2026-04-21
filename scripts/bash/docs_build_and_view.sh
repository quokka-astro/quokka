#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

echo "Serve the HTML documentation using mdBook"
"$(dirname "$0")/docs_build_and_view_mdbook.sh"
