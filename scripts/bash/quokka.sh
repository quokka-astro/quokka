#!/bin/bash

set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  quokka config <preset> [--root <path>]
  quokka build <preset> <problem> [--root <path>]
  quokka run <preset> <problem> [--input <file>] [--fpe] [--root <path>]
  quokka list <preset> [--root <path>]
  quokka target <preset> [--root <path>]

Presets:
  1d
  3d
  1d-debug
  3d-debug
EOF
}

die() {
	echo "Error: $*" >&2
	exit 1
}

require_arg() {
	local option="$1"
	local value="${2:-}"
	[ -n "$value" ] || die "missing value for ${option}"
}

parse_preset() {
	local preset="$1"

	case "$preset" in
	1d)
		DIM=1
		BUILD_TYPE=Release
		BUILD_NAME=1d
		;;
	3d)
		DIM=3
		BUILD_TYPE=Release
		BUILD_NAME=3d
		;;
	1d-debug)
		DIM=1
		BUILD_TYPE=Debug
		BUILD_NAME=1d-debug
		;;
	3d-debug)
		DIM=3
		BUILD_TYPE=Debug
		BUILD_NAME=3d-debug
		;;
	*)
		die "unsupported preset '${preset}'"
		;;
	esac

	BUILD_DIR="${ROOT}/build/${BUILD_NAME}"
}

resolve_root() {
	local root="$1"

	if ! ROOT="$(cd "$root" && pwd)"; then
		die "cannot access root '${root}'"
	fi
}

configure_build() {
	mkdir -p "$BUILD_DIR"
	cd "$BUILD_DIR"

	if [ -f CMakeCache.txt ]; then
		rm -rf ./*
	fi

	cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DAMReX_SPACEDIM="${DIM}"
}

build_problem() {
	local problem="$1"

	cmake --build "$BUILD_DIR" --target "$problem"
}

run_problem() {
	local problem="$1"
	local input_file="$2"
	shift 2

	local exe="${BUILD_DIR}/src/problems/${problem}/${problem}"
	[ -x "$exe" ] || die "executable not found: ${exe}"
	[ -f "$input_file" ] || die "input file not found: ${input_file}"

	"$exe" "$input_file" "$@"
}

list_problems() {
	find "${ROOT}/src/problems" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
}

show_targets() {
	cmake --build "$BUILD_DIR" --target help
}

COMMAND="${1:-}"
[ -n "$COMMAND" ] || {
	usage
	exit 1
}
shift

PRESET=""
ROOT="."
PROBLEM=""
INPUT_FILE=""
ENABLE_FPE=0

while [ "$#" -gt 0 ]; do
	case "$1" in
	--root)
		require_arg "$1" "${2:-}"
		ROOT="$2"
		shift 2
		;;
	--input)
		require_arg "$1" "${2:-}"
		INPUT_FILE="$2"
		shift 2
		;;
	--fpe)
		ENABLE_FPE=1
		shift
		;;
	-h|--help)
		usage
		exit 0
		;;
	-*)
		die "unknown option '$1'"
		;;
	*)
		if [ -z "$PRESET" ]; then
			PRESET="$1"
		elif [ -z "$PROBLEM" ]; then
			PROBLEM="$1"
		else
			die "unexpected argument '$1'"
		fi
		shift
		;;
	esac
done

[ -n "$PRESET" ] || die "missing preset"
resolve_root "$ROOT"
parse_preset "$PRESET"

case "$COMMAND" in
config)
	[ -z "$PROBLEM" ] || die "config does not take a problem name"
	[ -z "$INPUT_FILE" ] || die "config does not accept --input"
	[ "$ENABLE_FPE" -eq 0 ] || die "config does not accept --fpe"
	configure_build
	;;
build)
	[ -n "$PROBLEM" ] || die "missing problem name"
	[ -z "$INPUT_FILE" ] || die "build does not accept --input"
	[ "$ENABLE_FPE" -eq 0 ] || die "build does not accept --fpe"
	build_problem "$PROBLEM"
	;;
run)
	[ -n "$PROBLEM" ] || die "missing problem name"
	if [ -z "$INPUT_FILE" ]; then
		INPUT_FILE="${ROOT}/inputs/${PROBLEM}.toml"
	elif [[ "$INPUT_FILE" != /* ]]; then
		INPUT_FILE="${ROOT}/${INPUT_FILE}"
	fi

	RUN_ARGS=()
	if [ "$ENABLE_FPE" -eq 1 ]; then
		RUN_ARGS+=(
			amrex.fpe_trap_invalid=1
			amrex.fpe_trap_overflow=1
			amrex.fpe_trap_zero=1
		)
	fi

	run_problem "$PROBLEM" "$INPUT_FILE" "${RUN_ARGS[@]}"
	;;
list)
	[ -z "$PROBLEM" ] || die "list does not take a problem name"
	[ -z "$INPUT_FILE" ] || die "list does not accept --input"
	[ "$ENABLE_FPE" -eq 0 ] || die "list does not accept --fpe"
	list_problems
	;;
target)
	[ -z "$PROBLEM" ] || die "target does not take a problem name"
	[ -z "$INPUT_FILE" ] || die "target does not accept --input"
	[ "$ENABLE_FPE" -eq 0 ] || die "target does not accept --fpe"
	show_targets
	;;
*)
	die "unknown command '${COMMAND}'"
	;;
esac
