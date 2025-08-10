#!/bin/bash
# ABOUTME: Test script to detect GPU race conditions by comparing runs with different CUDA_LAUNCH_BLOCKING settings
# ABOUTME: Runs the same test twice and uses fcompare to verify outputs match

set -e

# Function to display usage
usage() {
    echo "Usage: $0 -b <binary> -i <input_file> -n <max_timesteps> [-h]"
    echo ""
    echo "Options:"
    echo "  -b <binary>         Path to the test binary"
    echo "  -i <input_file>     Path to the input file"
    echo "  -n <max_timesteps>  Maximum number of timesteps to run"
    echo "  -h                  Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -b ./build/src/FieldLoop/test_field_loop -i inputs/field_loop.in -n 10"
    exit 1
}

# Parse command line arguments
while getopts "b:i:n:h" opt; do
    case ${opt} in
        b)
            BINARY="${OPTARG}"
            ;;
        i)
            INPUT_FILE="${OPTARG}"
            ;;
        n)
            MAX_TIMESTEPS="${OPTARG}"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            usage
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            usage
            ;;
    esac
done

# Check if all required arguments are provided
if [ -z "${BINARY}" ] || [ -z "${INPUT_FILE}" ] || [ -z "${MAX_TIMESTEPS}" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Store the original working directory
ORIG_DIR=$(pwd)

# Convert relative paths to absolute paths based on current working directory
if [[ "${BINARY}" != /* ]]; then
    BINARY="${ORIG_DIR}/${BINARY}"
fi

if [[ "${INPUT_FILE}" != /* ]]; then
    INPUT_FILE="${ORIG_DIR}/${INPUT_FILE}"
fi

# Check if binary exists
if [ ! -f "${BINARY}" ]; then
    echo "Error: Binary '${BINARY}' not found"
    exit 1
fi

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: Input file '${INPUT_FILE}' not found"
    exit 1
fi

# Check if fcompare exists, build if necessary
# Look for fcompare relative to original working directory
FCOMPARE_DIR="${ORIG_DIR}/extern/amrex/Tools/Plotfile"
FCOMPARE=$(find "${FCOMPARE_DIR}" -name "fcompare.*.ex" 2>/dev/null | head -1)

if [ -z "${FCOMPARE}" ] || [ ! -f "${FCOMPARE}" ]; then
    echo "fcompare tool not found, building it..."
    cd "${FCOMPARE_DIR}"
    make -j$(nproc) programs=fcompare
    cd "${ORIG_DIR}"
    
    # Find the built binary
    FCOMPARE=$(find "${FCOMPARE_DIR}" -name "fcompare.*.ex" 2>/dev/null | head -1)
    
    if [ -z "${FCOMPARE}" ] || [ ! -f "${FCOMPARE}" ]; then
        echo "Error: Failed to build fcompare tool"
        exit 1
    fi
    echo "fcompare built successfully at: ${FCOMPARE}"
fi

# Create temporary directory for test outputs
TEMP_DIR=$(mktemp -d -t gpu_race_test.XXXXXX)
echo "Using temporary directory: ${TEMP_DIR}"

# Function to clean up on exit
cleanup() {
    echo "Cleaning up temporary directory..."
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# Copy input file to temp directory
cp "${INPUT_FILE}" "${TEMP_DIR}/input.in"

# Run with CUDA_LAUNCH_BLOCKING=1
echo ""
echo "=========================================="
echo "Running with CUDA_LAUNCH_BLOCKING=1..."
echo "=========================================="
cd "${TEMP_DIR}"
mkdir run_blocking
cd run_blocking
echo "Running: CUDA_LAUNCH_BLOCKING=1 ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} plotfile_interval=${MAX_TIMESTEPS} ..."
CUDA_LAUNCH_BLOCKING=1 "${BINARY}" ../input.in \
    max_timesteps=${MAX_TIMESTEPS} \
    plotfile_interval=${MAX_TIMESTEPS} \
    checkpoint_interval=-1 \
    ascent_interval=-1 \
    projection_interval=-1 \
    statistics_interval=-1 \
    slice_interval=-1 \
    amr.plot_file=plt_blocking
echo "Blocking run completed, checking for plotfiles..."

# Check if run completed successfully
PLOTFILE_BLOCKING=$(find . -maxdepth 1 -name "plt_blocking*" -type d | head -1)
if [ -z "${PLOTFILE_BLOCKING}" ]; then
    echo "Error: No plotfile generated for blocking run"
    echo "Contents of run directory:"
    ls -la
    exit 1
fi

# Get just the directory name (remove ./ prefix)
PLOTFILE_BLOCKING=$(basename "${PLOTFILE_BLOCKING}")

# Run with CUDA_LAUNCH_BLOCKING=0
echo ""
echo "=========================================="
echo "Running with CUDA_LAUNCH_BLOCKING=0..."
echo "=========================================="
cd "${TEMP_DIR}"
mkdir run_nonblocking
cd run_nonblocking
echo "Running: CUDA_LAUNCH_BLOCKING=0 ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} plotfile_interval=${MAX_TIMESTEPS} ..."
CUDA_LAUNCH_BLOCKING=0 "${BINARY}" ../input.in \
    max_timesteps=${MAX_TIMESTEPS} \
    plotfile_interval=${MAX_TIMESTEPS} \
    checkpoint_interval=-1 \
    ascent_interval=-1 \
    projection_interval=-1 \
    statistics_interval=-1 \
    slice_interval=-1 \
    amr.plot_file=plt_nonblocking
echo "Non-blocking run completed, checking for plotfiles..."

# Check if run completed successfully
PLOTFILE_NONBLOCKING=$(find . -maxdepth 1 -name "plt_nonblocking*" -type d | head -1)
if [ -z "${PLOTFILE_NONBLOCKING}" ]; then
    echo "Error: No plotfile generated for non-blocking run"
    echo "Contents of run directory:"
    ls -la
    exit 1
fi

# Get just the directory name (remove ./ prefix)
PLOTFILE_NONBLOCKING=$(basename "${PLOTFILE_NONBLOCKING}")

# Compare the plotfiles
echo ""
echo "=========================================="
echo "Comparing plotfiles with fcompare..."
echo "=========================================="
cd "${TEMP_DIR}"

# Get absolute paths
PLOT_BLOCKING="${TEMP_DIR}/run_blocking/${PLOTFILE_BLOCKING}"
PLOT_NONBLOCKING="${TEMP_DIR}/run_nonblocking/${PLOTFILE_NONBLOCKING}"

# Run fcompare and capture output
FCOMPARE_OUTPUT=$(mktemp)
"${FCOMPARE}" --abs_tol 0.0 --rel_tol 0.0 "${PLOT_BLOCKING}" "${PLOT_NONBLOCKING}" > "${FCOMPARE_OUTPUT}" 2>&1

# Check fcompare exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS: Plotfiles are identical - No race condition detected"
    echo ""
    cat "${FCOMPARE_OUTPUT}"
    exit 0
else
    echo ""
    echo "✗ FAILURE: Plotfiles differ - RACE CONDITION DETECTED!"
    echo ""
    echo "fcompare output:"
    echo "----------------"
    cat "${FCOMPARE_OUTPUT}"
    echo ""
    echo "This indicates a GPU race condition in the code."
    echo "The results depend on kernel execution order."
    echo ""
    echo "Plotfiles saved in:"
    echo "  Blocking:     ${PLOT_BLOCKING}"
    echo "  Non-blocking: ${PLOT_NONBLOCKING}"
    echo ""
    echo "To preserve these files, copy them before this script exits:"
    echo "  cp -r ${TEMP_DIR} ./gpu_race_results"
    echo ""
    echo "Press Enter to clean up and exit, or Ctrl+C to keep files..."
    read -r
    exit 1
fi