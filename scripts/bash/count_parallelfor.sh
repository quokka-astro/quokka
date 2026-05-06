#!/usr/bin/env bash

set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  count_parallelfor.sh [--root <path>] [--output <file>] [--include-problems]

Count ParallelFor call sites in first-party source and emit a CSV listing each
call site as filename,line.

Options:
  --root <path>        Repository root to scan (default: git root, otherwise cwd)
  --output <file>      Write CSV to this file instead of stdout
  --include-problems   Include src/problems/ in the scan
  -h, --help           Show this help

The scan always excludes build/ and extern/.
EOF
}

die() {
	echo "Error: $*" >&2
	exit 1
}

ROOT=""
OUTPUT=""
INCLUDE_PROBLEMS=0

while [ "$#" -gt 0 ]; do
	case "$1" in
	--root)
		[ "$#" -ge 2 ] || die "missing value for --root"
		ROOT="$2"
		shift 2
		;;
	--output)
		[ "$#" -ge 2 ] || die "missing value for --output"
		OUTPUT="$2"
		shift 2
		;;
	--include-problems)
		INCLUDE_PROBLEMS=1
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		die "unknown argument '$1'"
		;;
	esac
done

if [ -z "$ROOT" ]; then
	ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

[ -d "$ROOT" ] || die "root path '$ROOT' is not a directory"
command -v rg >/dev/null 2>&1 || die "ripgrep (rg) is required"

TMP_CSV="$(mktemp "$ROOT/.parallelfor_callsites.XXXXXX")"
trap 'rm -f "$TMP_CSV"' EXIT

RG_ARGS=(
	-n
	--glob '!build/**'
	--glob '!extern/**'
)

if [ "$INCLUDE_PROBLEMS" -eq 0 ]; then
	RG_ARGS+=(--glob '!src/problems/**')
fi

{
	echo "filename,line"
	(
		cd "$ROOT"
		rg "${RG_ARGS[@]}" '\bParallelFor\s*\(' src || true
	) | awk -F: '{ gsub(/"/, "\"\"", $1); print "\"" $1 "\"," $2 }'
} >"$TMP_CSV"

COUNT="$(tail -n +2 "$TMP_CSV" | wc -l | tr -d '[:space:]')"

if [ -n "$OUTPUT" ]; then
	cp "$TMP_CSV" "$OUTPUT"
else
	cat "$TMP_CSV"
fi

echo "ParallelFor call sites: $COUNT" >&2
