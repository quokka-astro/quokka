#!/usr/bin/env bash
# format.sh - Run pre-commit hooks on all files in the Quokka repository
#
# This script verifies the repository, installs pre-commit if needed,
# runs formatting checks, and cleans up hooks afterward.
#
# Usage: ./scripts/bash/format.sh

set -e

# Check if this is the Quokka repository
if [[ ! -d ".github" ]] || [[ ! -d ".ci" ]] || [[ ! -d "src" ]]; then
    echo "Error: Not in Quokka repository root (missing .github, .ci, or src directories)"
    exit 1
fi

# Check if pre-commit is installed, offer to install if missing
if ! command -v pre-commit &> /dev/null; then
    echo "pre-commit is not installed."
    
    # Determine which package manager to use
    if command -v uv &> /dev/null; then
        PKG_MGR="uv pip"
    else
        PKG_MGR="pip"
    fi
    
    # Install in virtual environment or with --user flag
    if [[ -z "$VIRTUAL_ENV" ]]; then
        read -p "Install via '$PKG_MGR install --user pre-commit'? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $PKG_MGR install --user pre-commit
            # Add to PATH if needed
            [[ -d "$HOME/.local/bin" ]] && export PATH="$HOME/.local/bin:$PATH"
        else
            echo "Aborted: pre-commit is required. Install it manually and try again."
            exit 1
        fi
    else
        read -p "Install via '$PKG_MGR install pre-commit'? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $PKG_MGR install pre-commit
        else
            echo "Aborted: pre-commit is required. Install it manually and try again."
            exit 1
        fi
    fi
fi

# Run pre-commit
pre-commit install
pre-commit run --all-files

# Always uninstall hooks when done
pre-commit uninstall

echo "Uninstalled pre-commit hooks"

echo "Done!"