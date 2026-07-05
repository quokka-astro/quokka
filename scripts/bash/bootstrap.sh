#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/bin"

install_script() {
	local src="$1"
	local name="$2"
	local target="$INSTALL_DIR/$name"
	local resolved=""

	if resolved="$(command -v "$name" 2>/dev/null)"; then
		if [[ "$resolved" != "$target" ]]; then
			echo "WARNING: $name already resolves to $resolved; updating $target anyway."
		fi
	fi

	mkdir -p "$INSTALL_DIR"
	cp "$src" "$target"
	chmod +x "$target"
	echo "Installed $name -> $target"

	if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
		echo "WARNING: $INSTALL_DIR is not in your PATH."
		echo "  Add this to your shell rc file:"
		echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
	fi
}

install_script "$SCRIPT_DIR/quokka" "quokka"
install_script "$SCRIPT_DIR/quokka-pre-commit.sh" "quokka-pre-commit.sh"
