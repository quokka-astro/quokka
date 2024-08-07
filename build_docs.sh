#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

# Doxygen
echo "Build the HTML documentation using MkDocs"
cd docs2
mkdocs build
cd ../
