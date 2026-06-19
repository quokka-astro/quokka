#!/bin/bash

set -e

# Function to display help message
show_help() {
	echo "Usage: $0 [OPTIONS]"
	echo
	echo "Run pre-commit checks on all files in the repository."
	echo
	echo "Options:"
	echo "  -h, --help        Show this help message"
	exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-h|--help)
			show_help
			;;
		*)
			echo "Unknown option: $1"
			show_help
			;;
	esac
done

# Resolve pre-commit command: prefer 'uv run pre-commit', fall back to 'pre-commit' directly
if command -v uv &>/dev/null; then
	PRECOMMIT="uv run pre-commit"
elif command -v pre-commit &>/dev/null; then
	PRECOMMIT="pre-commit"
else
	echo "pre-commit not found. Installing via pip install --user..."
	pip install --user pre-commit
	# After install, prefer 'pre-commit' directly (uv may still be unavailable)
	PRECOMMIT="pre-commit"
fi

echo "Using: $PRECOMMIT"

$PRECOMMIT install
$PRECOMMIT run --all-files || true
$PRECOMMIT uninstall

# Commit any changes made by pre-commit hooks (e.g. clang-format)
if [[ -n $(git status --porcelain) ]]; then
	git add -A
	git commit -m "style: apply pre-commit fixes

Co-Authored-By: Claude <noreply@anthropic.com>"
fi
