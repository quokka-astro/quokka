#!/bin/bash

set -eu

export PATH="/opt/singularity-ce/4.3.0/bin:$PATH"

#######################################
# Wait for GPU to be free.
# Run on the HOST before launching Singularity so that GPU memory
# allocated by the container runtime itself does not cause false alarms.
#######################################
wait_for_gpu() {
    local check_interval=60
    local max_intervals=240
    local waited=0

    echo "=========================================="
    echo "Checking for conflicting jobs on the GPU"
    echo "=========================================="

    local intervals=0
    while [ "$intervals" -lt "$max_intervals" ]; do
        local gpu_procs=""
        gpu_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
            | grep -v "^0," || true)

        # Calculate total GPU memory usage
        local total_memory=0
        if [ -n "$gpu_procs" ]; then
            total_memory=$(echo "$gpu_procs" | awk -F',' '{sum += $2} END {print int(sum)}')
        fi

        # Consider GPU free if memory usage is under 6000 MB
        if [ "$total_memory" -le 6000 ]; then
            echo "✓ GPU is free (${total_memory} MB used), proceeding..."
            echo ""
            return
        fi

        local count
        count=$(echo "$gpu_procs" | grep -c '[0-9]' || true)
        local now
        now=$(date +%Y%m%d\ %H%M%S)
        echo "Waiting: GPU has ${count} active CUDA process(es) using ${total_memory} MB (threshold: 6000 MB). Rechecking in ${check_interval}s (${waited}s elapsed) [${now}]..."
        sleep "$check_interval"
        waited=$((waited + check_interval))
        intervals=$((intervals + 1))
    done

    echo "Timed out waiting for GPU after ${waited}s. Aborting."
    echo ""
    exit 1
}

sif="quokka-linux-amd64-cuda-azp-agent-cached.sif"
if [ ! -f "$sif" ]; then
  singularity pull "$sif" docker://ghcr.io/quokka-astro/quokka-linux-amd64-cuda-azp-agent:development
fi

#cd azp-agent-avatargpu

TARGET=/priv/avatar/cche/azp-agent-in-docker-cuda/azp-agent-avatargpu/regression-tests

#######################################
# Parse webTopDir from the regression ini file.
# Prints the path to stdout; prints warnings to stderr.
# Returns 1 if the path cannot be determined.
#######################################
parse_web_dir() {
    local ini_file="$1"
    if [ ! -f "$ini_file" ]; then
        echo "WARNING: Ini file not found: $ini_file" >&2
        return 1
    fi
    local web_dir
    web_dir=$(grep -E "^webTopDir[[:space:]]*=" "$ini_file" \
        | sed -E 's/^webTopDir[[:space:]]*=[[:space:]]*//' \
        | tr -d '\r')
    if [ -z "$web_dir" ]; then
        echo "WARNING: Could not parse webTopDir from $ini_file" >&2
        return 1
    fi
    echo "$web_dir"
}

#######################################
# Push committed results to the remote.
# Runs on the HOST so that SSH credentials are always available and
# results are pushed even when the container timed out or was killed.
# Arguments:
#   $1 - path to the web/git directory
#######################################
push_results() {
    local web_dir="$1"

    echo "=========================================="
    echo "Pushing results to GitHub Pages (host)"
    echo "=========================================="

    # Validate that web_dir is within the expected base directory.
    # The path is parsed from quokka-tests.ini which is part of the source tree;
    # an attacker-controlled PR could point it at a directory containing a
    # malicious .git/hooks/ or .git/config on the shared volume.
    local expected_base
    expected_base="$(realpath "$TARGET")"
    local resolved_dir
    resolved_dir="$(realpath "$web_dir" 2>/dev/null)" || {
        echo "WARNING: Cannot resolve $web_dir, skipping push"
        return 0
    }
    if [[ "$resolved_dir" != "$expected_base"* ]]; then
        echo "WARNING: web_dir '$resolved_dir' is outside expected base '$expected_base', skipping push"
        return 0
    fi

    if [ ! -d "$web_dir/.git" ]; then
        echo "WARNING: $web_dir is not a git repository, skipping push"
        return 0
    fi

    cd "$web_dir" || { echo "WARNING: Cannot access $web_dir, skipping push"; return 0; }

    set +e
    # Disable hooks and override sshCommand to prevent execution of anything
    # written into .git/ by the container (e.g. a malicious pre-push hook or
    # core.sshCommand in .git/config planted via a PR that changes webTopDir).
    git -c core.hooksPath=/dev/null \
        -c core.sshCommand=ssh \
        push origin HEAD 2>&1
    local push_exit=$?
    set -e

    if [ $push_exit -ne 0 ]; then
        echo ""
        echo "WARNING: Push failed with exit code $push_exit"
        echo "         Results are committed locally but not yet on the remote."
    else
        echo ""
        echo "✓ Successfully pushed results to GitHub Pages"
    fi
    echo ""
}

#######################################
# Parse arguments
#######################################
MAKEBENCH=0
MAKEBENCH_TITLE="update$(date +%Y%m%d)"
MAKEBENCH_BRANCH=""
TESTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --title)
            MAKEBENCH_TITLE="$2"
            shift 2
            ;;
        --br)
            MAKEBENCH_BRANCH="$2"
            shift 2
            ;;
        --makebench)
            MAKEBENCH=1
            shift
            ;;
        --tests)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TESTS+=("$1")
                shift
            done
            if [ ${#TESTS[@]} -eq 0 ]; then
                echo "ERROR: --tests requires at least one test target"
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resolve web directory from the local ini file once, before launching the container.
LOCAL_INI_FILE="${TARGET}/quokka/regression/quokka-tests.ini"
WEB_DIR=$(parse_web_dir "$LOCAL_INI_FILE") || WEB_DIR=""

# Check GPU occupancy on the host before launching the container.
# This avoids false alarms from GPU memory allocated by Singularity's --nv init.
wait_for_gpu

# Download quokka-tests.ini from the requested branch, or use the local copy.
INI_FILE="$LOCAL_INI_FILE"
SOURCE_BRANCH_ARGS=()
if [ -n "$MAKEBENCH_BRANCH" ]; then
    INI_FILE="/tmp/tmp-quokka-tests.ini"
    wget "https://raw.githubusercontent.com/quokka-astro/quokka/refs/heads/${MAKEBENCH_BRANCH}/regression/quokka-tests.ini" -O "$INI_FILE"
    SOURCE_BRANCH_ARGS=(--source_branch "$MAKEBENCH_BRANCH")
fi

# Build test-selection args (shared by both modes).
TEST_ARGS=()
if [ ${#TESTS[@]} -eq 1 ]; then
    TEST_ARGS=(--single_test "${TESTS[0]}")
elif [ ${#TESTS[@]} -gt 1 ]; then
    TEST_ARGS=(--tests "${TESTS[*]}")
fi

# 4 hours timeout.  Capture the exit code instead of exiting immediately so
# that we can always attempt to push committed results afterwards.
set +e
if [ "$MAKEBENCH" -eq 1 ]; then
    # Benchmark creation mode: call regtest.py directly with --make_benchmarks.
    REGTEST="${TARGET}/regression_testing/regtest.py"
    timeout 14400 singularity exec --nv \
        --bind $TARGET:$TARGET \
        --pwd $TARGET \
        $sif python3 "$REGTEST" \
        --make_benchmarks "$MAKEBENCH_TITLE" \
        "${TEST_ARGS[@]}" \
        "${SOURCE_BRANCH_ARGS[@]}" \
        "$INI_FILE"
else
    # Regression test mode.
    RUN_TEST_ARGS=()
    if [ ${#TESTS[@]} -gt 0 ]; then
        RUN_TEST_ARGS=(--tests "${TESTS[@]}")
    fi
    RUN_SOURCE_BRANCH_ARGS=()
    if [ -n "$MAKEBENCH_BRANCH" ]; then
        RUN_SOURCE_BRANCH_ARGS=(--source-branch "$MAKEBENCH_BRANCH")
    fi
    timeout 14400 singularity exec --nv \
        --bind $TARGET:$TARGET \
        --pwd $TARGET \
        $sif \
        bash quokka2/scripts/bash/run-regression-tests.sh --ini-file "$INI_FILE" \
        --ccache-dir ${TARGET}/ccache --source-dir ${TARGET}/quokka \
        --skip-gpu-wait \
        "${RUN_SOURCE_BRANCH_ARGS[@]}" \
        "${RUN_TEST_ARGS[@]}"
fi
container_rc=$?
set -e

if [ $container_rc -eq 124 ]; then
    echo "ERROR: singularity exec timed out after 4 hours."
fi

# Push whatever was committed inside the container (log, status.json, html).
# This runs on the host where SSH credentials are available and is reached
# even when the container timed out, was OOM-killed, or exited with an error.
# Benchmark creation does not commit results to the web repo, so skip the push.
if [ "$MAKEBENCH" -eq 0 ]; then
    if [ -n "$WEB_DIR" ]; then
        push_results "$WEB_DIR"
    else
        echo "WARNING: WEB_DIR unknown, skipping git push"
    fi
fi

exit $container_rc

