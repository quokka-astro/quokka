#!/bin/bash

set -e

# Function to display help message
show_help() {
    echo "Usage: $0 <build_directory> [target]"
    echo
    echo "Arguments:"
    echo "  build_directory   Path to the build directory containing compile_commands.json. clang-tidy will use this to find the compile commands for the files."
    echo "  target            Optional argument to specify which files to check (default: 'changed')"
    echo "                    - 'changed': Files modified in current working directory"
    echo "                    - 'previous': Files modified in the last commit"
    echo "                    - 'origin': Files different from the remote branch"
    echo "                    - 'dev': Files different from the development branch"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  --fix            Apply clang-tidy fix-it hints"
    exit 0
}

# Initialize variables
fix_flag=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --fix)
            fix_flag="-fix"
            shift
            ;;
        *)
            if [ -z "$BUILD_DIR" ]; then
                BUILD_DIR="$1"
            elif [ -z "$target" ]; then
                target="$1"
            else
                echo "Error: Unexpected argument '$1'"
                echo "Use -h or --help for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if build directory is provided
if [ -z "$BUILD_DIR" ]; then
    echo "Error: Missing required argument 'build_directory'"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Set target type with default value "changed" if not specified
target="${target:-changed}"

# check if the first argument is a valid directory
if [ ! -d "$BUILD_DIR" ]; then
    echo "Invalid build directory. Please provide a valid directory path."
    exit 1
fi

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
files_select=()
for file in $files; do
    # Only process C++ source and header files
    if [[ "$file" == *.cpp || "$file" == *.hpp ]]; then
        echo "$file"
        files_select+=("$file")
    fi
done

echo

# Only run clang-tidy if there are files to process
if [ ${#files_select[@]} -gt 0 ]; then
    clang-tidy --extra-arg="-isysroot$(xcrun --show-sdk-path)" "${files_select[@]}" -p "$BUILD_DIR" $fix_flag
fi
