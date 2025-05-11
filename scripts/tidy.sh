#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 <build_directory> [target]"
    echo
    echo "Arguments:"
    echo "  build_directory   Path to the build directory containing compile_commands.json"
    echo "  target            Optional argument to specify which files to check (default: 'changed')"
    echo "                    - 'changed': Files modified in current working directory"
    echo "                    - 'previous': Files modified in the last commit"
    echo "                    - 'origin': Files different from the remote branch"
    echo "                    - 'dev': Files different from the development branch"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    exit 0
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Error: Missing required argument 'build_directory'"
    echo "Use -h or --help for usage information"
    exit 1
fi

# check if the first argument is a valid directory
if [ ! -d "$1" ]; then
    echo "Invalid build directory. Please provide a valid directory path."
    exit 1
fi

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
elif [ "$target" = "dev" ]; then
    # Get files that differ from the development branch
    files=$(git diff --name-only development)
else
    echo "Invalid target argument. Use 'changed', 'previous', 'origin' or 'dev'"
    exit 1
fi

# Display the list of files that will be processed
echo "Will process the following files with clang-tidy:"
files_select=""
for file in $files; do
    # Only process C++ source and header files
    if [[ "$file" == *.cpp || "$file" == *.hpp ]]; then
        echo "$file"
        files_select="$files_select $file"
    fi
done

echo

clang-tidy $files_select -p "$BUILD_DIR"
