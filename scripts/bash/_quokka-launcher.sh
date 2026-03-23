#!/usr/bin/env bash

set -euo pipefail

resolve_from_cwd() {
  local dir="$PWD"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/quokka.toml" && -f "$dir/src/quokka/__main__.py" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

WORKTREE=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -C|--worktree)
      if [[ $# -lt 2 ]]; then
        echo "quokka: missing value for $1" >&2
        exit 10
      fi
      WORKTREE="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$WORKTREE" ]]; then
  TARGET_ROOT="$(cd "$WORKTREE" && pwd)"
elif [[ -n "${QUOKKA_WORKTREE_ROOT:-}" ]]; then
  TARGET_ROOT="$(cd "$QUOKKA_WORKTREE_ROOT" && pwd)"
else
  TARGET_ROOT="$(resolve_from_cwd)" || {
    echo "quokka: unable to resolve a worktree; use -C /path/to/worktree or activate one first." >&2
    exit 10
  }
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "quokka: python3 not found on PATH." >&2
  exit 10
fi

export PYTHONPATH="$TARGET_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("typer") is not None else 1)
PY
then
  echo "quokka: Python dependency 'typer' is not installed for python3." >&2
  echo "Run: $TARGET_ROOT/scripts/bash/install-quokka-bootstrap.sh" >&2
  exit 10
fi

if ((${#ARGS[@]})); then
  exec python3 -m quokka "${ARGS[@]}"
fi

exec python3 -m quokka
