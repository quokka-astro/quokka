#!/usr/bin/env bash

quokka__return_or_exit() {
  local code="$1"
  return "$code" 2>/dev/null || exit "$code"
}

quokka__script_source() {
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    printf '%s\n' "${BASH_SOURCE[0]}"
    return 0
  fi
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    eval 'printf "%s\n" "${(%):-%x}"'
    return 0
  fi
  printf '%s\n' "$0"
}

quokka__is_sourced() {
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    [[ "${BASH_SOURCE[0]}" != "$0" ]]
    return
  fi
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    eval '[[ "${(%):-%x}" != "$0" ]]'
    return
  fi
  return 1
}

quokka__unset_function() {
  unset -f "$1" 2>/dev/null || true
  unfunction "$1" 2>/dev/null || true
}

quokka__activate() {
  local root=""
  local profile="${1:-}"
  local env_lines=""
  local exit_code=0

  root="$(cd "$(dirname "$(quokka__script_source)")/../.." && pwd)" || {
    echo "Unable to resolve the Quokka worktree root." >&2
    return 10
  }

  if ! command -v quokka >/dev/null 2>&1; then
    echo "quokka bootstrapper is not installed on PATH." >&2
    echo "Install it with: $root/scripts/bash/install-quokka-bootstrap.sh" >&2
    return 10
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to run the Quokka CLI." >&2
    return 10
  fi

  if [[ ! -f "$root/src/quokka/__main__.py" ]]; then
    echo "worktree-local quokka package is missing: $root/src/quokka/__main__.py" >&2
    return 10
  fi

  if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("typer") is not None else 1)
PY
  then
    echo "python3 is missing the Quokka CLI Python dependencies." >&2
    echo "Install them with: $root/scripts/bash/install-quokka-bootstrap.sh" >&2
    return 10
  fi

  # Re-resolve the activation environment without inheriting a stale runtime dir
  # from a previous activation.
  local -a cli_env=(env -u QUOKKA_ACTIVE -u QUOKKA_WORKTREE_ROOT -u QUOKKA_WORKTREE_ID -u QUOKKA_PROFILE -u QUOKKA_RUNTIME_DIR -u QUOKKA_PROMPT_PREFIX PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}")

  if [[ -n "$profile" ]]; then
    env_lines="$("${cli_env[@]}" python3 -m quokka -C "$root" _activate-env --profile "$profile")"
    exit_code=$?
  else
    env_lines="$("${cli_env[@]}" python3 -m quokka -C "$root" _activate-env)"
    exit_code=$?
  fi
  if [[ "$exit_code" -ne 0 ]]; then
    return "$exit_code"
  fi
  if [[ -z "$env_lines" ]]; then
    echo "Quokka CLI returned an empty activation environment." >&2
    return 10
  fi

  if typeset -f quokka_deactivate >/dev/null 2>&1; then
    quokka_deactivate
  fi

  export QUOKKA_OLD_PS1="${PS1-}"
  export QUOKKA_OLD_PATH="${PATH-}"
  eval "$env_lines" || {
    echo "Unable to apply the Quokka activation environment." >&2
    return 10
  }

  quokka_deactivate() {
    if [[ -n "${QUOKKA_OLD_PS1:-}" ]]; then
      PS1="$QUOKKA_OLD_PS1"
      export PS1
    fi
    if [[ -n "${QUOKKA_OLD_PATH:-}" ]]; then
      PATH="$QUOKKA_OLD_PATH"
      export PATH
    fi
    unset QUOKKA_ACTIVE QUOKKA_WORKTREE_ROOT QUOKKA_WORKTREE_ID QUOKKA_PROFILE QUOKKA_RUNTIME_DIR QUOKKA_PROMPT_PREFIX QUOKKA_OLD_PS1 QUOKKA_OLD_PATH
    quokka__unset_function quokka_deactivate
  }

  PS1="${QUOKKA_PROMPT_PREFIX} ${PS1-}"
  export PS1
}

if ! quokka__is_sourced; then
  echo "source this script instead of executing it" >&2
  quokka__return_or_exit 10
fi

quokka__activate "$@"
quokka__return_or_exit "$?"
