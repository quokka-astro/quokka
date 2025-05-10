#!/bin/bash

BUILD_DIR="$1"
target="${2:-changed}"  # Default to changed if not specified

# get current branch name
CURRENT_BRANCH=$(git branch --show-current)

files=""
if [ "$target" = "changed" ]; then
    files=$(git diff --name-only HEAD)
elif [ "$target" = "previous" ]; then
    files=$(git diff --name-only HEAD^)
elif [ "$target" = "origin" ]; then
    files=$(git diff --name-only "origin/$CURRENT_BRANCH")
else
    echo "Invalid target argument. Use 'changed', 'previous' or 'origin'"
    exit 1
fi

# run clang-tidy on the diff
echo "Will process the following files with clang-tidy:"
for file in $files; do
		if [[ "$file" == *.cpp || "$file" == *.hpp ]]; then
				echo "$file"
		fi
done

echo

for file in $files; do
		if [[ "$file" == *.cpp || "$file" == *.hpp ]]; then
				echo "Processing $file..."
				clang-tidy "$file" -p "$BUILD_DIR"
		fi
done
