#!/bin/bash

# Script: tidy2.sh
# Purpose: Run clang-tidy on C++ source files that have been modified in git
# Usage: ./tidy2.sh <build_directory> [target]
# Arguments:
#   build_directory: Path to the build directory containing compile_commands.json
#   target: Optional argument to specify which files to check (default: "changed")
#           - "changed": Files modified in current working directory
#           - "previous": Files modified in the last commit
#           - "origin": Files different from the remote branch

# Store the build directory path from first argument
BUILD_DIR="$1"
# Set target type with default value "changed" if not specified
target="${2:-changed}"

# Get the name of the current git branch
CURRENT_BRANCH=$(git branch --show-current)

# Initialize files variable
files=""
# Determine which files to process based on the target argument
if [ "$target" = "changed" ]; then
    # Get files modified in working directory
    files=$(git diff --name-only HEAD)
elif [ "$target" = "previous" ]; then
    # Get files modified in the last commit
    files=$(git diff --name-only HEAD^)
elif [ "$target" = "origin" ]; then
    # Get files that differ from the remote branch
    files=$(git diff --name-only "origin/$CURRENT_BRANCH")
else
    echo "Invalid target argument. Use 'changed', 'previous' or 'origin'"
    exit 1
fi

# Display the list of files that will be processed
echo "Will process the following files with clang-tidy:"
for file in $files; do
    # Only process C++ source and header files
    if [[ "$file" == *.cpp || "$file" == *.hpp ]]; then
        echo "$file"
    fi
done

echo

# Process each file with clang-tidy
for file in $files; do
    # Only process C++ source and header files
    if [[ "$file" == *.cpp || "$file" == *.hpp ]]; then
        echo "Processing $file..."
        # Run clang-tidy on the file using the specified build directory
        # -p flag points to the build directory containing compile_commands.json
        clang-tidy "$file" -p "$BUILD_DIR"
    fi
done
