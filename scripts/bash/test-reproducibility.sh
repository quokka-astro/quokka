#!/bin/bash
# Test bit-for-bit reproducibility by running a problem three times and comparing outputs.
# Prerequisites: source your environment and build the executable before running.
#
# Usage: test-reproducibility.sh <problem_name> [build_path] [--fcompare <path>] [--maxstep <N>]
#
# Example:
#   test-reproducibility.sh SphericalCollapse build/gpu-3d

set -e

if [[ ! -d "extern" || ! -d "src" ]]; then
    echo "Error: must be run from the Quokka repository root (extern/ and src/ not found)"
    exit 1
fi

usage() {
    echo "Usage: $0 <problem_name> [build_path] [--fcompare <path>] [--maxstep <N>]"
    echo "  problem_name:  e.g. SphericalCollapse"
    echo "  build_path:    e.g. build/gpu-3d (default: build/gpu-3d)"
    echo "  --fcompare:    path to fcompare executable (default: auto-detected from extern/)"
    echo "  --maxstep:     override max_timesteps from input file"
    exit 1
}

PROBLEM=""
BUILD_PATH="build/gpu-3d"
FCOMPARE_ARG=""
MAXSTEP_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --fcompare)
        [[ -n "${2:-}" ]] || { echo "Error: --fcompare requires a path"; exit 1; }
        FCOMPARE_ARG="$2"
        shift 2
        ;;
    --maxstep)
        [[ -n "${2:-}" ]] || { echo "Error: --maxstep requires a value"; exit 1; }
        MAXSTEP_ARG="$2"
        shift 2
        ;;
    -h|--help) usage ;;
    -*)
        echo "Error: unknown option '$1'"; exit 1 ;;
    *)
        if [[ -z "$PROBLEM" ]]; then
            PROBLEM="$1"
        elif [[ "$BUILD_PATH" == "build/gpu-3d" ]]; then
            BUILD_PATH="$1"
        else
            echo "Error: unexpected argument '$1'"; exit 1
        fi
        shift
        ;;
    esac
done

[[ -n "$PROBLEM" ]] || usage

ROOT="$(pwd)"
EXE="${ROOT}/${BUILD_PATH}/src/problems/${PROBLEM}/${PROBLEM}"
if [[ -n "$FCOMPARE_ARG" ]]; then
    FCOMPARE="$FCOMPARE_ARG"
else
    FCOMPARE="$(find "${ROOT}/extern/amrex/Tools/Plotfile" -name 'fcompare*' -executable 2>/dev/null | head -1)"
fi

if [[ -f "${ROOT}/inputs/${PROBLEM}.toml" ]]; then
    INPUT="${ROOT}/inputs/${PROBLEM}.toml"
elif [[ -f "${ROOT}/inputs/${PROBLEM}.in" ]]; then
    INPUT="${ROOT}/inputs/${PROBLEM}.in"
else
    echo "Error: input file not found for ${PROBLEM} in inputs/"
    exit 1
fi

[[ -x "$EXE" ]]      || { echo "Error: executable not found or not executable: $EXE"; exit 1; }
[[ -x "$FCOMPARE" ]] || { echo "Error: fcompare not found or not executable: $FCOMPARE"; exit 1; }

if [[ -n "$MAXSTEP_ARG" ]]; then
    max_timesteps="$MAXSTEP_ARG"
else
    max_timesteps=$(awk '/^[[:space:]]*max_timesteps[[:space:]]*=/{print $3}' "$INPUT")
    [[ -n "$max_timesteps" ]] || { echo "Error: could not read max_timesteps from $INPUT"; exit 1; }
fi
echo "max_timesteps: ${max_timesteps}"

plt="plt$(printf "%07d" "${max_timesteps}")"
flag=$(date +%Y%m%d%H%M%S)
echo "flag: ${flag}"

cd "${ROOT}/tests"

run() {
    local n="$1"
    echo "--- Run ${n} ---"
    "$EXE" "$INPUT" ${MAXSTEP_ARG:+max_timesteps="${max_timesteps}"} plotfile_interval="${max_timesteps}" >> "log.${flag}.r${n}.log"
    [[ -d "$plt" ]] || { echo "Error: plotfile ${plt} not found after run ${n}"; exit 1; }
    mv "$plt" "${plt}.${flag}.r${n}"
    echo "Output: ${plt}.${flag}.r${n}"
}

run 1
run 2
run 3

r1="${plt}.${flag}.r1"
r2="${plt}.${flag}.r2"
r3="${plt}.${flag}.r3"

compare() {
    local a="$1" b="$2" label="$3"
    echo ""
    echo "=== Comparing ${label} ==="
    if diff -rq "$a" "$b" > /dev/null 2>&1; then
        echo "PASS: ${label} are identical"
    else
        echo "FAIL: ${label} differ"
        "$FCOMPARE" "$a" "$b" || true
    fi
}

compare "$r1" "$r2" "run1 vs run2"
compare "$r2" "$r3" "run2 vs run3"

echo ""
if diff -rq "${r1}" "${r2}" > /dev/null 2>&1 && \
   diff -rq "${r2}" "${r3}" > /dev/null 2>&1; then
    echo "All three runs are identical — reproducibility confirmed."
    exit 0
else
    echo "Reproducibility check FAILED."
    exit 1
fi
