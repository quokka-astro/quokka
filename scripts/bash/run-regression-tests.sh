#!/bin/bash
#
# run-regression-tests.sh
#
# Standalone script to run Quokka regression tests and publish results to GitHub Pages.
# Designed to run inside Docker containers independently of Azure Pipelines.
#
# Usage:
#   ./run-regression-tests.sh [OPTIONS]
#
# Options:
#   --ini-file PATH       Path to regression ini file (default: regression/quokka-tests.ini)
#   --ccache-dir PATH     Ccache directory (default: $HOME/cache/ccache)
#   --source-dir PATH     Quokka source directory (default: $PWD)
#   --help                Show this help message
#
# Environment variables (overridden by CLI args):
#   REGRESSION_INI_FILE
#   CCACHE_DIR
#   REGRESSION_SOURCE_DIR
#
# Note: webTopDir is automatically parsed from the ini file.
#

set -uo pipefail

# Default values
DEFAULT_INI_FILE="regression/quokka-tests.ini"
DEFAULT_CCACHE_DIR="$HOME/cache/ccache"
DEFAULT_SOURCE_DIR="$PWD"

# Initialize from environment variables or defaults
INI_FILE="${REGRESSION_INI_FILE:-$DEFAULT_INI_FILE}"
CCACHE_DIR="${CCACHE_DIR:-$DEFAULT_CCACHE_DIR}"
SOURCE_DIR="${REGRESSION_SOURCE_DIR:-$DEFAULT_SOURCE_DIR}"

# WEB_DIR will be parsed from ini file
WEB_DIR=""

#######################################
# Print usage information
#######################################
usage() {
	cat <<EOF
Usage: $0 [OPTIONS]

Standalone script to run Quokka regression tests and publish results to GitHub Pages.

Options:
  --ini-file PATH       Path to regression ini file (default: $DEFAULT_INI_FILE)
  --ccache-dir PATH     Ccache directory (default: $DEFAULT_CCACHE_DIR)
  --source-dir PATH     Quokka source directory (default: $DEFAULT_SOURCE_DIR)
  --help                Show this help message

Environment variables (overridden by CLI args):
  REGRESSION_INI_FILE
  CCACHE_DIR
  REGRESSION_SOURCE_DIR

Note: webTopDir is automatically parsed from the ini file.

Examples:
  # Run with defaults
  ./run-regression-tests.sh

  # Override ini file
  ./run-regression-tests.sh --ini-file custom-tests.ini

  # Use environment variables
  export REGRESSION_INI_FILE=custom-tests.ini
  ./run-regression-tests.sh
EOF
}

#######################################
# Parse command-line arguments
#######################################
parse_args() {
	while [[ $# -gt 0 ]]; do
		case $1 in
		--ini-file)
			INI_FILE="$2"
			shift 2
			;;
		--ccache-dir)
			CCACHE_DIR="$2"
			shift 2
			;;
		--source-dir)
			SOURCE_DIR="$2"
			shift 2
			;;
		--help)
			usage
			exit 0
			;;
		*)
			echo "ERROR: Unknown option: $1"
			usage
			exit 1
			;;
		esac
	done
}

#######################################
# Parse webTopDir from ini file
#######################################
parse_web_dir() {
	if [ ! -f "$INI_FILE" ]; then
		echo "ERROR: Ini file not found: $INI_FILE"
		exit 1
	fi

	# Extract webTopDir from ini file
	# Format: webTopDir  = /path/to/web
	WEB_DIR=$(grep -E "^webTopDir\s*=" "$INI_FILE" | sed -E 's/^webTopDir\s*=\s*//' | tr -d '\r')

	if [ -z "$WEB_DIR" ]; then
		echo "ERROR: Could not parse webTopDir from ini file: $INI_FILE"
		exit 1
	fi

	echo "✓ Parsed webTopDir from ini file: $WEB_DIR"
}

#######################################
# Setup environment and ccache
#######################################
setup_environment() {
	echo "=========================================="
	echo "Setting up environment"
	echo "=========================================="

	# Verify ccache is available
	if ! command -v ccache &>/dev/null; then
		echo "ERROR: ccache not found but is required"
		exit 1
	fi
	echo "✓ ccache found: $(command -v ccache)"

	# Create ccache directory if needed
	mkdir -p "$CCACHE_DIR"
	echo "✓ ccache directory: $CCACHE_DIR"

	# Export ccache environment variables
	export CCACHE_DIR="$CCACHE_DIR"
	export CCACHE_BASEDIR="$SOURCE_DIR"
	export CCACHE_NOHASHDIR=1

	# Set CMake launcher args for regtest.py
	export CMAKE_C_COMPILER_LAUNCHER=ccache
	export CMAKE_CXX_COMPILER_LAUNCHER=ccache
	export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

	echo "✓ ccache environment configured"

	# Change to source directory
	cd "$SOURCE_DIR" || {
		echo "ERROR: Cannot access source directory: $SOURCE_DIR"
		exit 1
	}
	echo "✓ Source directory: $SOURCE_DIR"

	# Verify web directory parent exists (regtest.py will create the actual web dir)
	local web_parent
	web_parent=$(dirname "$WEB_DIR")
	if [ ! -d "$web_parent" ]; then
		echo "WARNING: Web directory parent does not exist: $web_parent"
		echo "         regtest.py may fail to create output directory"
	fi

	# Print configuration summary
	echo ""
	echo "Configuration:"
	echo "  Source:  $SOURCE_DIR"
	echo "  Ini:     $INI_FILE"
	echo "  Web:     $WEB_DIR"
	echo "  Ccache:  $CCACHE_DIR"
	echo ""
}

#######################################
# Run regression tests and capture output
# Returns: exit code from regtest.py
#######################################
run_regression_tests() {
	echo "=========================================="
	echo "Running regression tests"
	echo "=========================================="

	local exit_code=0
	local log_file="$WEB_DIR/regression-run.log"

	# Create web directory if it doesn't exist
	mkdir -p "$WEB_DIR"

	echo "Command: ./extern/regression_testing/regtest.py --clean_testdir $INI_FILE"
	echo "Log file: $log_file"
	echo "Output will be written to log file (this may take a while)..."
	echo ""

	# Run regtest.py, capturing both stdout and stderr to log file
	# Don't exit on failure - we want to always publish results
	set +e
	./extern/regression_testing/regtest.py --clean_testdir "$INI_FILE" \
		> "$log_file" 2>&1
	exit_code=$?
	set -e

	echo ""
	echo "Regression tests completed with exit code: $exit_code"
	echo ""

	return $exit_code
}

#######################################
# Detect failure mode from log file
# Arguments:
#   $1 - exit code from regtest.py
# Returns: status string
#######################################
detect_failure_mode() {
	local exit_code=$1
	local log_file="$WEB_DIR/regression-run.log"

	# Success case
	if [ "$exit_code" -eq 0 ]; then
		echo "SUCCESS"
		return
	fi

	# Check log file for specific error patterns
	if [ ! -f "$log_file" ]; then
		echo "UNKNOWN_FAILURE"
		return
	fi

	# Check for disk quota issues
	if grep -qi "disk quota exceeded\|no space left on device" "$log_file"; then
		echo "DISK_QUOTA"
		return
	fi

	# Check for out of memory
	if grep -qi "out of memory\|cannot allocate memory\|killed.*memory\|oom" "$log_file"; then
		echo "OUT_OF_MEMORY"
		return
	fi

	# Check for compilation errors
	if grep -qi "compilation failed\|error:.*\\.cpp\|error:.*\\.hpp\|CMake Error\|ninja: build stopped" "$log_file"; then
		echo "COMPILATION_ERROR"
		return
	fi

	# Check for crashes
	if grep -qi "segmentation fault\|core dumped\|signal [0-9]" "$log_file"; then
		echo "CRASH"
		return
	fi

	# Check for timeouts
	if grep -qi "timeout\|timed out" "$log_file"; then
		echo "TIMEOUT"
		return
	fi

	# Unknown failure
	echo "UNKNOWN_FAILURE"
}

#######################################
# Create status file with test results
# Arguments:
#   $1 - status string
#   $2 - exit code
#######################################
create_status_file() {
	local status=$1
	local exit_code=$2
	local log_file="$WEB_DIR/regression-run.log"
	local status_file="$WEB_DIR/status.json"

	echo "=========================================="
	echo "Creating status file"
	echo "=========================================="

	# Extract error context if failed
	local error_details=""
	if [ "$status" != "SUCCESS" ] && [ -f "$log_file" ]; then
		# Get last 20 lines containing "error" or "fail" keywords
		error_details=$(grep -i "error\|fail\|fatal" "$log_file" | tail -20 | head -c 500 || true)
	fi

	# Get git info
	local branch
	local commit
	branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
	commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

	# Get timestamp
	local timestamp
	timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

	# Write JSON (using jq for proper escaping if available, otherwise printf)
	if command -v jq &>/dev/null; then
		jq -n \
			--arg timestamp "$timestamp" \
			--arg status "$status" \
			--arg exit_code "$exit_code" \
			--arg hostname "$(hostname)" \
			--arg branch "$branch" \
			--arg commit "$commit" \
			--arg error_details "$error_details" \
			--arg log_file "regression-run.log" \
			'{
				timestamp: $timestamp,
				status: $status,
				exit_code: ($exit_code | tonumber),
				hostname: $hostname,
				branch: $branch,
				commit: $commit,
				error_details: $error_details,
				log_file: $log_file
			}' >"$status_file"
	else
		# Fallback without jq - simple JSON escaping
		error_details=$(printf '%s' "$error_details" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | paste -sd '\\n' -)
		cat >"$status_file" <<EOF
{
  "timestamp": "$timestamp",
  "status": "$status",
  "exit_code": $exit_code,
  "hostname": "$(hostname)",
  "branch": "$branch",
  "commit": "$commit",
  "error_details": "$error_details",
  "log_file": "regression-run.log"
}
EOF
	fi

	echo "✓ Status file created: $status_file"
	echo "  Status: $status"
	echo "  Exit code: $exit_code"
	echo "  Branch: $branch"
	echo "  Commit: $commit"
	echo ""
}

#######################################
# Publish results to GitHub Pages
# Arguments:
#   $1 - status string
#######################################
publish_results() {
	local status=$1

	echo "=========================================="
	echo "Publishing results to GitHub Pages"
	echo "=========================================="

	# Change to web directory
	cd "$WEB_DIR" || {
		echo "ERROR: Cannot access web directory $WEB_DIR"
		return 1
	}

	# Check if this is a git repository
	if [ ! -d .git ]; then
		echo "ERROR: $WEB_DIR is not a git repository"
		echo "       Cannot publish results without git repository"
		return 1
	fi

	# Check git status
	echo "Git status:"
	git status --short

	# Stage all changes
	echo ""
	echo "Staging changes..."
	git add .

	# Check if there are changes to commit
	if git diff --staged --quiet; then
		echo "No changes to commit, skipping push"
		return 0
	fi

	# Show what will be committed
	echo ""
	echo "Changes to commit:"
	git diff --staged --stat

	# Create commit message based on status
	local commit_msg="Updated regression results - $status [skip ci]"

	# Commit with bot identity
	echo ""
	echo "Committing changes..."
	git -c user.name="Quokka CI Bot" \
		-c user.email="quokka-ci-bot@quokka.dev" \
		commit --no-verify -m "$commit_msg"

	# Push to remote (origin HEAD = current branch)
	echo ""
	echo "Pushing to remote..."
	set +e
	git push origin HEAD 2>&1
	local push_exit=$?
	set -e

	if [ $push_exit -ne 0 ]; then
		echo ""
		echo "WARNING: Push failed with exit code $push_exit"
		echo "         Continuing anyway (results are committed locally)"
		return 0
	fi

	echo ""
	echo "✓ Successfully published results to GitHub Pages"
	return 0
}

#######################################
# Wait for GPU to be free before starting tests.
#
# Checks nvidia-smi --query-compute-apps, which queries the GPU driver
# directly and sees CUDA processes from ALL Docker containers on the host.
# Only non-trivial compute processes (pid > 0) are counted to avoid
# false alarms from transient or monitoring processes.
#######################################
wait_for_gpu() {
	local check_interval=60
	local waited=0

	echo "=========================================="
	echo "Checking for conflicting jobs on the GPU"
	echo "=========================================="

	while true; do
		# Check for active CUDA processes via GPU driver (cross-container)
		local gpu_procs=""
		gpu_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
			| grep -v "^0," || true)

		if [ -z "$gpu_procs" ]; then
			if [ "$waited" -gt 0 ]; then
				echo "✓ GPU is free, proceeding..."
			else
				echo "✓ GPU is free, proceeding..."
			fi
			echo ""
			return
		fi

		local count
		count=$(echo "$gpu_procs" | grep -c '[0-9]' || true)
		local now
		now=$(date -u +%H:%M:%SZ)
		echo "Waiting: GPU has ${count} active CUDA process(es). Rechecking in ${check_interval}s (${waited}s elapsed) [${now}]..."
		sleep "$check_interval"
		waited=$((waited + check_interval))
	done
}

#######################################
# Main function - orchestrate the workflow
#######################################
main() {
	echo "=========================================="
	echo "Quokka Regression Test Runner"
	echo "=========================================="
	echo ""

	# Parse command-line arguments
	parse_args "$@"

	# Parse web directory from ini file
	parse_web_dir

	# Setup environment
	setup_environment

	# Wait for GPU to be free before starting
	wait_for_gpu

	# Run regression tests (capture exit code)
	local test_exit_code=0
	set +e
	run_regression_tests
	test_exit_code=$?
	set -e

	# Detect failure mode
	local status
	status=$(detect_failure_mode $test_exit_code)

	# Create status file
	create_status_file "$status" "$test_exit_code"

	# Publish results (always, even on failure)
	set +e
	publish_results "$status"
	local publish_exit=$?
	set -e

	# Print final summary
	echo "=========================================="
	echo "Summary"
	echo "=========================================="
	echo "Test status:    $status"
	echo "Test exit code: $test_exit_code"
	echo "Publish result: $([ $publish_exit -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
	echo ""

	if [ "$status" = "SUCCESS" ]; then
		echo "✓ All regression tests passed!"
		exit 0
	else
		echo "✗ Regression tests failed with status: $status"
		echo "  Check $WEB_DIR/regression-run.log for details"
		exit $test_exit_code
	fi
}

# Run main function
main "$@"
