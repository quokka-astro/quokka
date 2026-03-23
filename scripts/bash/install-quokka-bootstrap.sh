#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-$HOME/.local/bin/quokka}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to install the Quokka CLI bootstrapper." >&2
  exit 10
fi

pip_supports_pyproject_editable() {
  python3 - <<'PY'
import re
import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True, text=True)
match = re.search(r"pip (\d+)\.(\d+)", result.stdout)
if match is None:
    sys.exit(1)
version = (int(match.group(1)), int(match.group(2)))
sys.exit(0 if version >= (21, 3) else 1)
PY
}

project_runtime_dependencies() {
  python3 - "$ROOT/pyproject.toml" <<'PY'
import ast
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
if project_match is None:
    raise SystemExit("Unable to locate [project] in pyproject.toml.")
deps_match = re.search(r"(?ms)^dependencies\s*=\s*(\[[^\]]*\])", project_match.group(1))
if deps_match is None:
    raise SystemExit("Unable to locate project.dependencies in pyproject.toml.")
dependencies = ast.literal_eval(deps_match.group(1))
for dependency in dependencies:
    print(dependency)
PY
}

install_runtime_dependencies() {
  local dep
  local deps=()
  while IFS= read -r dep; do
    deps+=("$dep")
  done < <(project_runtime_dependencies)
  if ((${#deps[@]} == 0)); then
    return 0
  fi
  "${pip_install[@]}" "${deps[@]}"
}

pip_install=(python3 -m pip install)
if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  INSTALL_SCOPE="environment"
else
  pip_install+=(--user)
  INSTALL_SCOPE="user site"
fi

INSTALL_MODE="dependencies-only"
INSTALL_NOTE=""
if pip_supports_pyproject_editable; then
  if "${pip_install[@]}" -e "$ROOT"; then
    INSTALL_MODE="editable"
  else
    INSTALL_NOTE="Editable install failed; installed launcher dependencies only. The launcher still uses the worktree sources directly."
    install_runtime_dependencies
  fi
else
  INSTALL_NOTE="Installed launcher dependencies only because the active pip does not support pyproject editable installs."
  install_runtime_dependencies
fi

mkdir -p "$(dirname "$DEST")"
install -m 0755 "$ROOT/scripts/bash/_quokka-launcher.sh" "$DEST"

echo "Installed bootstrapper at $DEST"
if [[ "$INSTALL_MODE" == "editable" ]]; then
  echo "Installed the Python package and dependencies into the $INSTALL_SCOPE (editable install)."
else
  echo "Installed the launcher runtime dependencies into the $INSTALL_SCOPE."
fi
if [[ -n "$INSTALL_NOTE" ]]; then
  echo "$INSTALL_NOTE"
fi
echo "Ensure $(dirname "$DEST") is on PATH."
