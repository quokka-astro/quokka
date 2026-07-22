#!/bin/bash

set -e

if [[ ! -d ".github" ]] || [[ ! -d ".ci" ]] || [[ ! -d "src" ]]; then
	echo "Error: run this from the Quokka repository root (missing .github, .ci, or src)"
	exit 1
fi

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

# Abort if working tree is dirty or has untracked files, to avoid accidentally staging unrelated work
if [[ -n $(git status --porcelain) ]]; then
	echo "Error: working tree is not clean. Commit or stash your changes before running this script."
	exit 1
fi

# Ensure pre-commit has a writable cache directory
if [[ ! -w "${PRE_COMMIT_HOME:-$HOME/.cache/pre-commit}" ]]; then
	export PRE_COMMIT_HOME="${TMPDIR:-/tmp}/pre-commit-cache"
	mkdir -p "$PRE_COMMIT_HOME"
fi

# Resolve pre-commit command: prefer 'uv run --with pre-commit', fall back to 'pre-commit' directly
if command -v uv &>/dev/null; then
	PRECOMMIT=(uv run --with pre-commit pre-commit)
elif command -v pre-commit &>/dev/null; then
	PRECOMMIT=(pre-commit)
else
	echo "pre-commit not found. Installing via pip install --user..."
	pip install --user pre-commit
	[[ -d "$HOME/.local/bin" ]] && export PATH="$HOME/.local/bin:$PATH"
	PRECOMMIT=(pre-commit)
fi

echo "Using: ${PRECOMMIT[*]}"

STATUS=0
"${PRECOMMIT[@]}" run --all-files || STATUS=$?

# Commit any changes made by pre-commit hooks (e.g. clang-format), then rerun
# the hooks to check for remaining (non-mutating) failures such as invalid
# YAML or a failed checker.
if [[ -n $(git status --porcelain) ]]; then
	git add -A
	git commit -m "style: apply pre-commit fixes"

	STATUS=0
	"${PRECOMMIT[@]}" run --all-files || STATUS=$?
fi

exit "$STATUS"
