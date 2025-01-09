#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

echo "Build the HTML documentation using MkDocs"
cd docs
mkdocs build
cd ../
