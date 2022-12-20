#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

# Doxygen
echo "Build the Doxygen documentation"
cd docs
doxygen &> doxygen.out
cd ../

# sphinx
cd docs

echo "Build the Sphinx documentation for Quokka."
#make PYTHON="python3" LATEXMKOPTS="-interaction=nonstopmode" latexpdf
make PYTHON="python3" html &> make_source_html.out
cd ../
