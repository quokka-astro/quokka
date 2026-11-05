#!/bin/bash
#
# run-canary-regression-tests.sh
#
# Build and run the Canary-based nightly regression suite in a dedicated work
# tree. This is intended as a small host-side driver for shared GPU machines.
#

set -euo pipefail

DEFAULT_SOURCE_DIR="$PWD"
DEFAULT_BUILD_DIR="build"
DEFAULT_SUITE_DIR="regression/canary/tests"
DEFAULT_WORK_DIR="regression/canary/work"
DEFAULT_WORKERS=1
DEFAULT_GPU_COUNT=1
DEFAULT_BUILD_JOBS=8
DEFAULT_SESSION_TIMEOUT="4h"

SOURCE_DIR="$DEFAULT_SOURCE_DIR"
BUILD_DIR="$DEFAULT_BUILD_DIR"
SUITE_DIR="$DEFAULT_SUITE_DIR"
WORK_DIR="$DEFAULT_WORK_DIR"
WORKERS="$DEFAULT_WORKERS"
GPU_COUNT="$DEFAULT_GPU_COUNT"
BUILD_JOBS="$DEFAULT_BUILD_JOBS"
SESSION_TIMEOUT="$DEFAULT_SESSION_TIMEOUT"
SKIP_BUILD=0
SKIP_GPU_WAIT=0

readonly NIGHTLY_TARGETS=(
	"HydroBlast3D"
	"ShockCloud"
	"RandomBlast"
	"RadhydroShell"
	"MHDBlast"
	"AlfvenWaveLinear"
	"Turbulence"
)

usage() {
	cat <<EOF
Usage: $0 [OPTIONS]

Run the Canary-based Quokka nightly regression suite.

Options:
  --source-dir PATH      Quokka source tree (default: $DEFAULT_SOURCE_DIR)
  --build-dir PATH       Quokka build tree, relative to source if not absolute (default: $DEFAULT_BUILD_DIR)
  --suite-dir PATH       Canary test directory, relative to source if not absolute (default: $DEFAULT_SUITE_DIR)
  --work-dir PATH        Canary work tree, relative to source if not absolute (default: $DEFAULT_WORK_DIR)
  --workers N            Maximum Canary workers (default: $DEFAULT_WORKERS)
  --gpu-count N          GPU slots exposed to Canary (default: $DEFAULT_GPU_COUNT)
  --build-jobs N         Parallel jobs for cmake/make builds (default: $DEFAULT_BUILD_JOBS)
  --session-timeout T    Canary session timeout, e.g. 4h (default: $DEFAULT_SESSION_TIMEOUT)
  --skip-build           Skip Quokka target/tool builds
  --skip-gpu-wait        Skip the GPU occupancy check
  --help                 Show this help text
EOF
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		--source-dir)
			SOURCE_DIR="$2"
			shift 2
			;;
		--build-dir)
			BUILD_DIR="$2"
			shift 2
			;;
		--suite-dir)
			SUITE_DIR="$2"
			shift 2
			;;
		--work-dir)
			WORK_DIR="$2"
			shift 2
			;;
		--workers)
			WORKERS="$2"
			shift 2
			;;
		--gpu-count)
			GPU_COUNT="$2"
			shift 2
			;;
		--build-jobs)
			BUILD_JOBS="$2"
			shift 2
			;;
		--session-timeout)
			SESSION_TIMEOUT="$2"
			shift 2
			;;
		--skip-build)
			SKIP_BUILD=1
			shift
			;;
		--skip-gpu-wait)
			SKIP_GPU_WAIT=1
			shift
			;;
		--help)
			usage
			exit 0
			;;
		*)
			echo "ERROR: Unknown option: $1" >&2
			usage
			exit 1
			;;
		esac
	done
}

resolve_path() {
	local base_dir="$1"
	local candidate="$2"
	if [[ "$candidate" = /* ]]; then
		echo "$candidate"
	else
		echo "$base_dir/$candidate"
	fi
}

wait_for_gpu() {
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo "nvidia-smi not found; skipping GPU occupancy check"
		return
	fi

	local check_interval=60
	local waited=0

	echo "=========================================="
	echo "Checking for conflicting jobs on the GPU"
	echo "=========================================="

	while true; do
		local gpu_procs=""
		gpu_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
			| grep -v "^0," || true)

		if [[ -z "$gpu_procs" ]]; then
			echo "✓ GPU is free, proceeding..."
			echo ""
			return
		fi

		local count
		count=$(echo "$gpu_procs" | grep -c '[0-9]' || true)
		local now
		now=$(date -u +"%Y-%m-%dT%H:%M:%S")
		echo "Waiting: GPU has ${count} active CUDA process(es). Rechecking in ${check_interval}s (${waited}s elapsed) [${now}]..."
		sleep "$check_interval"
		waited=$((waited + check_interval))
	done
}

build_quokka_targets() {
	echo "=========================================="
	echo "Building Quokka nightly targets"
	echo "=========================================="
	cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" --target "${NIGHTLY_TARGETS[@]}"
	echo ""
}

build_compare_tools() {
	local make_cmd="${MAKE:-make}"
	local plotfile_dir="$SOURCE_DIR/extern/amrex/Tools/Plotfile"
	local post_dir="$SOURCE_DIR/extern/amrex/Tools/Postprocessing/C_Src"

	echo "=========================================="
	echo "Building AMReX comparison tools"
	echo "=========================================="

	"$make_cmd" -C "$plotfile_dir" clean >/dev/null 2>&1 || true
	for tool in fcompare fsnapshot; do
		echo "Building $tool"
		"$make_cmd" -C "$plotfile_dir" "programs=$tool" DEBUG=FALSE USE_MPI=FALSE USE_OMP=FALSE
	done

	"$make_cmd" -C "$post_dir" clean >/dev/null 2>&1 || true
	echo "Building particle_compare"
	"$make_cmd" -C "$post_dir" EBASE=particle_compare DEBUG=FALSE USE_MPI=FALSE USE_OMP=FALSE
	echo ""
}

write_canary_config() {
	local config_file="$1"
	cat >"$config_file" <<EOF
workspace:
  view: TestResults
environment:
  set:
    CUDA_VISIBLE_DEVICES: "%(gpu_ids)s"
resource_pool:
  gpus: $GPU_COUNT
EOF
}

write_status_file() {
	local status_file="$1"
	local run_rc="$2"
	local log_file="$3"
	local branch commit timestamp
	branch=$(git -C "$SOURCE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
	commit=$(git -C "$SOURCE_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
	timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")

	if command -v jq >/dev/null 2>&1; then
		jq -n \
			--arg timestamp "$timestamp" \
			--arg branch "$branch" \
			--arg commit "$commit" \
			--arg source_dir "$SOURCE_DIR" \
			--arg build_dir "$BUILD_DIR" \
			--arg suite_dir "$SUITE_DIR" \
			--arg work_dir "$WORK_DIR" \
			--arg log_file "$log_file" \
			--argjson exit_code "$run_rc" \
			'{
				timestamp: $timestamp,
				branch: $branch,
				commit: $commit,
				source_dir: $source_dir,
				build_dir: $build_dir,
				suite_dir: $suite_dir,
				work_dir: $work_dir,
				exit_code: $exit_code,
				status: (if $exit_code == 0 then "SUCCESS" else "FAILED" end),
				log_file: $log_file
			}' >"$status_file"
	else
		cat >"$status_file" <<EOF
{
  "timestamp": "$timestamp",
  "branch": "$branch",
  "commit": "$commit",
  "source_dir": "$SOURCE_DIR",
  "build_dir": "$BUILD_DIR",
  "suite_dir": "$SUITE_DIR",
  "work_dir": "$WORK_DIR",
  "exit_code": $run_rc,
  "status": "$( [[ "$run_rc" -eq 0 ]] && echo SUCCESS || echo FAILED )",
  "log_file": "$log_file"
}
EOF
	fi
}

main() {
	parse_args "$@"

	if ! command -v canary >/dev/null 2>&1; then
		echo "ERROR: canary is not on PATH" >&2
		exit 1
	fi

	SOURCE_DIR=$(cd "$SOURCE_DIR" && pwd)
	BUILD_DIR=$(resolve_path "$SOURCE_DIR" "$BUILD_DIR")
	SUITE_DIR=$(resolve_path "$SOURCE_DIR" "$SUITE_DIR")
	WORK_DIR=$(resolve_path "$SOURCE_DIR" "$WORK_DIR")

	mkdir -p "$WORK_DIR"
	WORK_DIR=$(cd "$WORK_DIR" && pwd)

	if [[ ! -d "$BUILD_DIR" ]]; then
		echo "ERROR: build directory not found: $BUILD_DIR" >&2
		exit 1
	fi
	if [[ ! -d "$SUITE_DIR" ]]; then
		echo "ERROR: Canary suite directory not found: $SUITE_DIR" >&2
		exit 1
	fi

	echo "=========================================="
	echo "Quokka Canary Nightly Runner"
	echo "=========================================="
	echo "Source:        $SOURCE_DIR"
	echo "Build:         $BUILD_DIR"
	echo "Suite:         $SUITE_DIR"
	echo "Work tree:     $WORK_DIR"
	echo "Workers:       $WORKERS"
	echo "GPU slots:     $GPU_COUNT"
	echo "Session limit: $SESSION_TIMEOUT"
	echo ""

	if [[ "$SKIP_GPU_WAIT" -eq 0 ]] && [[ "$GPU_COUNT" -gt 0 ]]; then
		wait_for_gpu
	fi

	if [[ "$SKIP_BUILD" -eq 0 ]]; then
		build_quokka_targets
		build_compare_tools
	fi

	local config_file="$WORK_DIR/canary-nightly.yaml"
	write_canary_config "$config_file"

	export QUOKKA_CANARY_BUILD_DIR="$BUILD_DIR"
	export PYTHONUNBUFFERED=1

	local log_file="$WORK_DIR/canary-run.log"
	local status_file="$WORK_DIR/canary-status.json"

	cd "$WORK_DIR"

	echo "=========================================="
	echo "Running Canary nightly suite"
	echo "=========================================="
	set +e
	canary -f "$config_file" run -w --workers="$WORKERS" --timeout "session=$SESSION_TIMEOUT" "$SUITE_DIR" \
		2>&1 | tee "$log_file"
	local run_rc=${PIPESTATUS[0]}
	set -e
	echo ""

	if [[ -d "$WORK_DIR/TestResults" ]]; then
		echo "=========================================="
		echo "Generating Canary reports"
		echo "=========================================="
		cd "$WORK_DIR/TestResults"
		canary report html create --dest . || true
		canary report json create -o canary.json || true
		canary report junit create -o junit.xml || true
		canary status -rA > canary-status.txt || true
		cd "$WORK_DIR"
	fi

	write_status_file "$status_file" "$run_rc" "$log_file"

	echo "=========================================="
	echo "Summary"
	echo "=========================================="
	echo "Exit code:    $run_rc"
	echo "Log file:     $log_file"
	echo "Status file:  $status_file"
	if [[ -d "$WORK_DIR/TestResults" ]]; then
		echo "Results view: $WORK_DIR/TestResults"
	fi
	echo ""

	exit "$run_rc"
}

main "$@"
