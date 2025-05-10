#!/bin/bash

BUILD_DIR="$1"
COMPARETO="${2:-changed}"  # Default to HEAD if not specified

# get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get current branch name
CURRENT_BRANCH=$(git branch --show-current)

# run clang-tidy on the diff
if [ "$COMPARETO" = "changed" ]; then
    git diff -U0 HEAD | python "$SCRIPT_DIR/tidy.py" -p1 -path "$BUILD_DIR"
elif [ "$COMPARETO" = "previous" ]; then
    git diff -U0 HEAD^ | python "$SCRIPT_DIR/tidy.py" -p1 -path "$BUILD_DIR"
elif [ "$COMPARETO" = "origin" ]; then
    git diff -U0 "origin/$CURRENT_BRANCH" | python "$SCRIPT_DIR/tidy.py" -p1 -path "$BUILD_DIR"
else
    echo "Invalid COMPARETO argument. Use 'changed', 'previous' or 'origin'"
    exit 1
fi
