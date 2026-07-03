#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/bin"

install_script() {
	local src="$1"
	local name="$2"

	if command -v "$name" &>/dev/null; then
		echo "$name is already in PATH ($(command -v "$name")), skipping."
		return
	fi

	mkdir -p "$INSTALL_DIR"
	cp "$src" "$INSTALL_DIR/$name"
	chmod +x "$INSTALL_DIR/$name"
	echo "Installed $name -> $INSTALL_DIR/$name"

	if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
		echo "WARNING: $INSTALL_DIR is not in your PATH."
		echo "  Add this to your shell rc file:"
		echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
	fi
}

install_script "$SCRIPT_DIR/quokka" "quokka"
install_script "$SCRIPT_DIR/pre-commit.sh" "pre-commit.sh"
