#!/bin/bash

set -e

# Function to display help message
show_help() {
	cat << EOF
Usage: $0 [OPTIONS] [TARGETS]

Run CUDA tests in a Docker container for Quokka.

The most common usage is a simple "$0" command without any option, or "$0 'Problem1 Problem2'" to build specific problems.

This script automatically:
  1. Detects the Quokka repository root
  2. Selects the appropriate CUDA container image for your platform
  3. Pulls or builds the container image if needed
  4. Creates/starts a container with the repository mounted
  5. Builds Quokka with CUDA support
  6. Optionally runs tests

Options:
  -h, --help              Show this help message
  --reuse-build          Reuse existing build directory without asking
  --log                  Save build output to a log file with timestamp
  --dim DIM              Space dimension (2 or 3, default: 3)
  --build-dir DIR        Build directory name (default: build/container-cuda-\$DIM)
  --jobs N               Number of parallel build jobs (default: 8)
  --run-tests            Run ctest after building

Examples:
  $0                                    # Build with default settings (3D, CUDA)
  $0 --dim 2                            # Build 2D version
  $0 --reuse-build                      # Automatically use existing build
  $0 --log                              # Build and save logs
  $0 --run-tests                        # Build and run tests
  $0 --jobs 16                          # Use 16 parallel build jobs

TARGETS:
  Optional CMake/Ninja targets to build (e.g., specific test executables).
  If not specified, builds all targets.

EOF
	exit 0
}

# Initialize variables
REUSE_BUILD=false
USE_LOG=false
DIM=3
BUILD_DIR=""
JOBS=8
RUN_TESTS=false
TARGETS=""

# Detect repository root
# Use git to find the worktree root (works for both regular repos and git worktrees)
REPO_ROOT="$(git -c core.hooksPath=/dev/null rev-parse --show-toplevel 2>/dev/null)"

if [[ -z "$REPO_ROOT" ]]; then
	echo "Error: Could not find Quokka repository root (.git directory)"
	echo "Please run this script from within the Quokka repository"
	exit 1
fi

# Detect platform architecture
ARCH=""
if [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "aarch64" ]]; then
	ARCH="arm64"
elif [[ "$(uname -m)" == "x86_64" ]] || [[ "$(uname -m)" == "amd64" ]]; then
	ARCH="amd64"
else
	echo "Warning: Unknown architecture $(uname -m), defaulting to amd64"
	ARCH="amd64"
fi

# Detect OS
OS_TYPE=""
if [[ "$(uname)" == "Darwin" ]]; then
	OS_TYPE="macos"
elif [[ "$(uname)" == "Linux" ]]; then
	OS_TYPE="linux"
else
	echo "Warning: Unknown OS $(uname), defaulting to linux"
	OS_TYPE="linux"
fi

# Set image name based on platform
if [[ "$ARCH" == "arm64" ]]; then
	IMAGE_NAME="ghcr.io/quokka-astro/quokka-arm64-cuda:development"
else
	IMAGE_NAME="ghcr.io/quokka-astro/quokka-linux-amd64-cuda:development"
fi

# Set container name (include worktree name so each worktree gets its own container with the correct mount)
WORKTREE_NAME="$(basename "$REPO_ROOT" | tr -cs 'a-zA-Z0-9_.-' '_')"
CONTAINER_NAME="quokka-${ARCH}-cuda-${WORKTREE_NAME}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-h|--help)
			show_help
			;;
		--reuse-build)
			REUSE_BUILD=true
			shift
			;;
		--log)
			USE_LOG=true
			shift
			;;
		--dim)
			DIM="$2"
			if [[ ! "$DIM" =~ ^[23]$ ]]; then
				echo "Error: Dimension must be 2 or 3"
				exit 1
			fi
			shift 2
			;;
		--build-dir)
			BUILD_DIR="$2"
			shift 2
			;;
		--jobs)
			JOBS="$2"
			shift 2
			;;
		--run-tests)
			RUN_TESTS=true
			shift
			;;
		*)
			break
			;;
	esac
done

# Everything remaining is a target
TARGETS=("$@")

# Set default build directory if not specified
if [[ -z "$BUILD_DIR" ]]; then
	BUILD_DIR="build/container-cuda-${DIM}D"
fi

BUILD_DIR_FULL="$REPO_ROOT/$BUILD_DIR"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
	echo "Error: Docker is not installed or not in PATH"
	echo "Please install Docker to use this script"
	exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
	echo "Error: Docker daemon is not running"
	echo "Please start Docker and try again"
	exit 1
fi

# Handle Docker image - pull if not present
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}$"; then
	echo "Image $IMAGE_NAME not found locally. Attempting to pull..."
	if ! docker pull "$IMAGE_NAME"; then
		echo "Error: Failed to pull image $IMAGE_NAME"
		echo "You may need to:"
		echo "  1. Check your internet connection"
		echo "  2. Verify you have access to ghcr.io/quokka-astro"
		exit 1
	fi
fi

# Handle Docker container
if docker ps -a -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
	# Container exists
	if docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
		echo "Container $CONTAINER_NAME is already running."
	else
		echo "Starting existing container $CONTAINER_NAME..."
		docker start "$CONTAINER_NAME"
	fi
else
	# Container doesn't exist, create it
	echo "Creating container $CONTAINER_NAME from image $IMAGE_NAME..."
	
	# Build docker run command with GPU support if running tests
	if [[ "$RUN_TESTS" == true ]]; then
		docker run -d --name "$CONTAINER_NAME" --gpus all \
			-v "$REPO_ROOT:/home/ubuntu/workspace" \
			"$IMAGE_NAME" tail -f /dev/null
	else
		docker run -d --name "$CONTAINER_NAME" \
			-v "$REPO_ROOT:/home/ubuntu/workspace" \
			"$IMAGE_NAME" tail -f /dev/null
	fi
fi

# Handle build directory
IS_RECONFIG=false
if [[ -d "$BUILD_DIR_FULL" ]]; then
	if [[ "$REUSE_BUILD" == true ]]; then
		# Automatically use existing build
		echo "Using existing build directory: $BUILD_DIR_FULL"
	else
		# Ask user what to do
		read -p "Build directory $BUILD_DIR_FULL exists. Continue with existing build (y) or remove and rebuild (r)? [y/r] " response
		if [[ "$response" =~ ^[Rr]$ ]]; then
			echo "Removing directory $BUILD_DIR_FULL..."
			rm -rf "$BUILD_DIR_FULL"
			IS_RECONFIG=true
			echo "Creating build directory: $BUILD_DIR_FULL"
			mkdir -p "$BUILD_DIR_FULL"
		elif [[ "$response" =~ ^[Yy]$ ]]; then
			echo "Will build upon existing build..."
		else
			echo "Aborting..."
			exit 0
		fi
	fi
else
	IS_RECONFIG=true
	echo "Creating build directory: $BUILD_DIR_FULL"
	mkdir -p "$BUILD_DIR_FULL"
fi

# Prepare build command
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-Wno-deprecated-gpu-targets' -DAMReX_GPU_BACKEND=CUDA -DAMReX_MPI=OFF -DAMReX_SPACEDIM=$DIM -G Ninja"

CMAKE_CMD="cd /home/ubuntu/workspace/$BUILD_DIR"

if [[ "$IS_RECONFIG" == true ]]; then
	CMAKE_CMD="mkdir -p /home/ubuntu/workspace/$BUILD_DIR && cd /home/ubuntu/workspace/$BUILD_DIR && cmake /home/ubuntu/workspace $CMAKE_ARGS"
else
	CMAKE_CMD="cd /home/ubuntu/workspace/$BUILD_DIR"
fi

if [[ -n "$TARGETS" ]]; then
	NINJA_CMD="ninja -j$JOBS $TARGETS"
else
	NINJA_CMD="ninja -j$JOBS"
fi

CMAKE_CMD="$CMAKE_CMD && $NINJA_CMD"

# Execute build
if [[ "$USE_LOG" == true ]]; then
	LOG_DIR_REL="build/container-cuda-logs/$(date +'%Y%m%d-%H%M%S')-cuda-${DIM}D"
	LOG_DIR="$REPO_ROOT/$LOG_DIR_REL"
	mkdir -p "$LOG_DIR"

	# Save git info
	cd "$REPO_ROOT"
	git status >> "$LOG_DIR/git.log" 2>&1 || true
	git log -3 >> "$LOG_DIR/git.log" 2>&1 || true

	echo "Building (output saved to $LOG_DIR/build.log)..."
	docker exec "$CONTAINER_NAME" bash -c "$CMAKE_CMD" &> "$LOG_DIR/build.log"
	BUILD_EXIT_CODE=${PIPESTATUS[0]}

	if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
		echo "Build completed successfully. Logs saved to: $LOG_DIR"
	else
		echo "Build failed with exit code $BUILD_EXIT_CODE. Check logs at: $LOG_DIR/build.log"
		exit $BUILD_EXIT_CODE
	fi
else
	# Output to terminal
	echo "Building..."
	docker exec "$CONTAINER_NAME" bash -c "$CMAKE_CMD"
	BUILD_EXIT_CODE=$?

	if [[ $BUILD_EXIT_CODE -ne 0 ]]; then
		echo "Build failed with exit code $BUILD_EXIT_CODE"
		exit $BUILD_EXIT_CODE
	fi
fi

# Run tests if requested
if [[ "$RUN_TESTS" == true ]]; then
	echo "Running tests..."
	TEST_CMD="cd /home/ubuntu/workspace/$BUILD_DIR && ctest --output-on-failure"
	if [[ "$USE_LOG" == true ]]; then
		docker exec "$CONTAINER_NAME" bash -c "$TEST_CMD" >> "$LOG_DIR/test.log" 2>&1
		TEST_EXIT_CODE=${PIPESTATUS[0]}
	else
		docker exec "$CONTAINER_NAME" bash -c "$TEST_CMD"
		TEST_EXIT_CODE=$?
	fi

	if [[ $TEST_EXIT_CODE -ne 0 ]]; then
		echo "Tests failed with exit code $TEST_EXIT_CODE"
		exit $TEST_EXIT_CODE
	fi
fi

echo "Done!"

