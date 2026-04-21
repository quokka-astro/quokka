#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DOCS_DIR="$REPO_ROOT/docs"
VENV_DIR="$DOCS_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
MKDOCS_BIN="$VENV_DIR/bin/mkdocs"
STAMP_FILE="$VENV_DIR/.docs-requirements.sha256"
REQUIREMENTS_FILE="$DOCS_DIR/requirements.txt"
CACHE_DIR="$DOCS_DIR/.cache"

mkdir -p "$CACHE_DIR/pip" "$VENV_DIR/.pycache"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export PYTHONPYCACHEPREFIX="$VENV_DIR/.pycache"
export XDG_CACHE_HOME="$CACHE_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
	echo "Creating Python virtual environment in $VENV_DIR"
	python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
	echo "Bootstrapping pip in $VENV_DIR"
	"$PYTHON_BIN" -m ensurepip --upgrade
fi

requirements_hash=$(shasum -a 256 "$REQUIREMENTS_FILE" | awk '{print $1}')
install_requirements=false

if [ ! -x "$MKDOCS_BIN" ]; then
	install_requirements=true
elif [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE")" != "$requirements_hash" ]; then
	install_requirements=true
elif ! "$PYTHON_BIN" -c "import mermaid2" >/dev/null 2>&1; then
	install_requirements=true
fi

if [ "$install_requirements" = true ]; then
	echo "Installing documentation dependencies from $REQUIREMENTS_FILE"
	"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
	printf '%s\n' "$requirements_hash" >"$STAMP_FILE"
fi

cd "$DOCS_DIR"
exec "$MKDOCS_BIN" "$@"
