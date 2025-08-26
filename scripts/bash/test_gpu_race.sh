#!/bin/bash
# ABOUTME: Test script to detect GPU race conditions by comparing runs with different CUDA_LAUNCH_BLOCKING settings
# ABOUTME: Runs the same test twice and uses fcompare to verify outputs match

set -e

# Function to display usage
usage() {
    echo "Usage: $0 -b <binary> -i <input_file> -n <max_timesteps> [-s] [-r] [-c] [-h]"
    echo ""
    echo "Options:"
    echo "  -b <binary>         Path to the test binary"
    echo "  -i <input_file>     Path to the input file"
    echo "  -n <max_timesteps>  Maximum number of timesteps to run"
    echo "  -s                  Use single GPU stream (amrex.max_gpu_streams=1)"
    echo "  -r                  Test reproducibility (run twice with CUDA_LAUNCH_BLOCKING=1)"
    echo "  -c                  Run under compute-sanitizer to detect memory errors and race conditions"
    echo "  -h                  Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -b ./build/src/FieldLoop/test_field_loop -i inputs/field_loop.in -n 10"
    echo "  $0 -b ./build/src/FieldLoop/test_field_loop -i inputs/field_loop.in -n 10 -s"
    echo "  $0 -b ./build/src/FieldLoop/test_field_loop -i inputs/field_loop.in -n 10 -r"
    echo "  $0 -b ./build/src/FieldLoop/test_field_loop -i inputs/field_loop.in -n 10 -c"
    exit 1
}

# Initialize variables
SINGLE_STREAM=false
REPRODUCIBILITY_TEST=false
COMPUTE_SANITIZER=false

# Parse command line arguments
while getopts "b:i:n:srch" opt; do
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
        s)
            SINGLE_STREAM=true
            ;;
        r)
            REPRODUCIBILITY_TEST=true
            ;;
        c)
            COMPUTE_SANITIZER=true
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

# Function to clean up on exit (disabled - leaving files for analysis)
cleanup() {
    echo "Temporary directory preserved at: ${TEMP_DIR}"
    echo "Use 'rm -rf ${TEMP_DIR}' to clean up when done analyzing."
}
trap cleanup EXIT

# Copy input file to temp directory
cp "${INPUT_FILE}" "${TEMP_DIR}/input.in"

# Prepare additional arguments
ADDITIONAL_ARGS=""
if [ "${SINGLE_STREAM}" = true ]; then
    ADDITIONAL_ARGS="amrex.max_gpu_streams=1"
fi

<<<<<<< HEAD
# Check if running reproducibility test mode
if [ "${REPRODUCIBILITY_TEST}" = true ]; then
    # Run reproducibility test (two runs with CUDA_LAUNCH_BLOCKING=1)
    echo ""
    echo "=========================================="
    echo "REPRODUCIBILITY TEST MODE"
    echo "Running twice with CUDA_LAUNCH_BLOCKING=1"
    echo "=========================================="
    
    # First run with CUDA_LAUNCH_BLOCKING=1
    echo ""
    echo "=========================================="
    echo "First run with CUDA_LAUNCH_BLOCKING=1..."
    if [ "${SINGLE_STREAM}" = true ]; then
        echo "(with amrex.max_gpu_streams=1)"
    fi
    echo "=========================================="
    cd "${TEMP_DIR}"
    mkdir run1
    cd run1
    echo "Running: CUDA_LAUNCH_BLOCKING=1 ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} plotfile_interval=${MAX_TIMESTEPS} ${ADDITIONAL_ARGS}..."
    CUDA_LAUNCH_BLOCKING=1 "${BINARY}" ../input.in \
        max_timesteps=${MAX_TIMESTEPS} \
        plotfile_interval=${MAX_TIMESTEPS} \
        checkpoint_interval=-1 \
        ascent_interval=-1 \
        projection_interval=-1 \
        statistics_interval=-1 \
        slice_interval=-1 \
        plotfile_prefix=plt_run1 \
        ${ADDITIONAL_ARGS}
    echo "First run completed, checking for plotfiles..."
    
    # Check if run completed successfully - find the FINAL (highest numbered) plotfile
    PLOTFILE_RUN1=$(find . -maxdepth 1 -name "plt_run1*" -type d | sort | tail -1)
    if [ -z "${PLOTFILE_RUN1}" ]; then
        echo "Error: No plotfile generated for first run"
        echo "Contents of run directory:"
        ls -la
        exit 1
    fi
    
    # Get just the directory name (remove ./ prefix)
    PLOTFILE_RUN1=$(basename "${PLOTFILE_RUN1}")
    echo "Found first run plotfile: ${PLOTFILE_RUN1}"
    
    # Second run with CUDA_LAUNCH_BLOCKING=1
    echo ""
    echo "=========================================="
    echo "Second run with CUDA_LAUNCH_BLOCKING=1..."
    if [ "${SINGLE_STREAM}" = true ]; then
        echo "(with amrex.max_gpu_streams=1)"
    fi
    echo "=========================================="
    cd "${TEMP_DIR}"
    mkdir run2
    cd run2
    echo "Running: CUDA_LAUNCH_BLOCKING=1 ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} plotfile_interval=${MAX_TIMESTEPS} ${ADDITIONAL_ARGS}..."
    CUDA_LAUNCH_BLOCKING=1 "${BINARY}" ../input.in \
        max_timesteps=${MAX_TIMESTEPS} \
        plotfile_interval=${MAX_TIMESTEPS} \
        checkpoint_interval=-1 \
        ascent_interval=-1 \
        projection_interval=-1 \
        statistics_interval=-1 \
        slice_interval=-1 \
        plotfile_prefix=plt_run2 \
        ${ADDITIONAL_ARGS}
    RUN2_EXIT_CODE=$?
    
    if [ ${RUN2_EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "✗ REPRODUCIBILITY FAILURE!"
        echo ""
        echo "The second run crashed (exit code: ${RUN2_EXIT_CODE})"
        echo "while the first run completed successfully."
        echo ""
        echo "This indicates non-deterministic behavior even with CUDA_LAUNCH_BLOCKING=1."
        echo ""
        echo "Contents of second run directory:"
        ls -la
        echo ""
        echo "Temporary directory preserved for analysis: ${TEMP_DIR}"
        echo "- First run results:  ${TEMP_DIR}/run1/"
        echo "- Second run results: ${TEMP_DIR}/run2/"
        exit 1
    fi
    
    echo "Second run completed, checking for plotfiles..."
    
    # Check if run completed successfully - find the FINAL (highest numbered) plotfile  
    PLOTFILE_RUN2=$(find . -maxdepth 1 -name "plt_run2*" -type d | sort | tail -1)
    if [ -z "${PLOTFILE_RUN2}" ]; then
        echo "Error: No plotfile generated for second run"
        echo "Contents of run directory:"
        ls -la
        exit 1
    fi
    
    # Get just the directory name (remove ./ prefix)
    PLOTFILE_RUN2=$(basename "${PLOTFILE_RUN2}")
    echo "Found second run plotfile: ${PLOTFILE_RUN2}"
    
    # Compare the plotfiles
    echo ""
    echo "=========================================="
    echo "Comparing plotfiles with fcompare..."
    echo "=========================================="
    cd "${TEMP_DIR}"
    
    # Get absolute paths
    PLOT_RUN1="${TEMP_DIR}/run1/${PLOTFILE_RUN1}"
    PLOT_RUN2="${TEMP_DIR}/run2/${PLOTFILE_RUN2}"
    
    # Run fcompare and capture output
    FCOMPARE_OUTPUT=$(mktemp)
    echo "Running fcompare command:"
    echo "${FCOMPARE}" --abs_tol 0.0 --rel_tol 0.0 "${PLOT_RUN1}" "${PLOT_RUN2}"
    echo ""
    
    # Check if fcompare binary exists
    if [ ! -f "${FCOMPARE}" ]; then
        echo "Error: fcompare binary not found at ${FCOMPARE}"
        exit 1
    fi
    
    # Check if plotfile directories exist
    if [ ! -d "${PLOT_RUN1}" ]; then
        echo "Error: First run plotfile directory not found: ${PLOT_RUN1}"
        exit 1
    fi
    
    if [ ! -d "${PLOT_RUN2}" ]; then
        echo "Error: Second run plotfile directory not found: ${PLOT_RUN2}"
        exit 1
    fi
    
    echo "Running fcompare..."
    set +e  # Don't exit on error so we can capture crashed output
    "${FCOMPARE}" --abs_tol 0.0 --rel_tol 0.0 "${PLOT_RUN1}" "${PLOT_RUN2}" 2>&1 | tee "${FCOMPARE_OUTPUT}"
    FCOMPARE_EXIT_CODE=${PIPESTATUS[0]}
    set -e
    
    echo ""
    echo "fcompare completed with exit code: ${FCOMPARE_EXIT_CODE}"
    
    # Show captured output if there was any
    if [ -s "${FCOMPARE_OUTPUT}" ]; then
        echo ""
        echo "fcompare output:"
        echo "----------------"
        cat "${FCOMPARE_OUTPUT}"
        echo "----------------"
        echo ""
    else
        echo ""
        echo "No output captured from fcompare (process may have been killed by signal)"
        echo ""
    fi
    
    # Check fcompare exit code
    if [ ${FCOMPARE_EXIT_CODE} -eq 0 ]; then
        echo "✓ SUCCESS: Runs are bitwise identical - Code is reproducible with CUDA_LAUNCH_BLOCKING=1"
        exit 0
    else
        echo "✗ FAILURE: Runs differ - NON-REPRODUCIBLE BEHAVIOR DETECTED!"
        echo ""
        echo "The two runs with CUDA_LAUNCH_BLOCKING=1 produced different results."
        echo "This indicates non-deterministic behavior that persists even when"
        echo "kernels are forced to execute synchronously."
        echo ""
        echo "Possible causes:"
        echo "- Use of uninitialized memory"
        echo "- Non-associative floating-point operations with varying order"
        echo "- Random number generation without fixed seed"
        echo "- Time-dependent operations"
        echo ""
        echo "Temporary directory preserved for analysis: ${TEMP_DIR}"
        echo "- First run results:  ${TEMP_DIR}/run1/"
        echo "- Second run results: ${TEMP_DIR}/run2/"
        echo ""
        exit 1
    fi
    
    # Exit after reproducibility test - don't run the race condition test
    exit 0
fi

# Check if running compute-sanitizer mode
if [ "${COMPUTE_SANITIZER}" = true ]; then
    # Run with compute-sanitizer for race condition and memory error detection
    echo ""
    echo "=========================================="
    echo "COMPUTE-SANITIZER MODE"
    echo "Running with compute-sanitizer --tool racecheck"
    echo "=========================================="
    
    # Check if compute-sanitizer is available
    if ! command -v compute-sanitizer &> /dev/null; then
        echo "Error: compute-sanitizer not found in PATH"
        echo "Please ensure CUDA toolkit is installed and compute-sanitizer is in your PATH"
        exit 1
    fi
    
    cd "${TEMP_DIR}"
    mkdir run_sanitizer
    cd run_sanitizer
    
    echo ""
    echo "Running race condition check..."
    echo "Command: compute-sanitizer --tool racecheck ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} ${ADDITIONAL_ARGS}"
    echo ""
    
    # Run with racecheck tool
    set +e  # Don't exit on error so we can capture output
    compute-sanitizer --tool racecheck "${BINARY}" ../input.in \
        max_timesteps=${MAX_TIMESTEPS} \
        plotfile_interval=-1 \
        checkpoint_interval=-1 \
        ascent_interval=-1 \
        projection_interval=-1 \
        statistics_interval=-1 \
        slice_interval=-1 \
        ${ADDITIONAL_ARGS} 2>&1 | tee racecheck_output.txt
    RACECHECK_EXIT_CODE=${PIPESTATUS[0]}
    set -e
    
    echo ""
    echo "=========================================="
    echo "Running memory error check..."
    echo "Command: compute-sanitizer --tool memcheck ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} ${ADDITIONAL_ARGS}"
    echo ""
    
    # Run with memcheck tool
    set +e  # Don't exit on error so we can capture output
    compute-sanitizer --tool memcheck "${BINARY}" ../input.in \
        max_timesteps=${MAX_TIMESTEPS} \
        plotfile_interval=-1 \
        checkpoint_interval=-1 \
        ascent_interval=-1 \
        projection_interval=-1 \
        statistics_interval=-1 \
        slice_interval=-1 \
        ${ADDITIONAL_ARGS} 2>&1 | tee memcheck_output.txt
    MEMCHECK_EXIT_CODE=${PIPESTATUS[0]}
    set -e
    
    echo ""
    echo "=========================================="
    echo "COMPUTE-SANITIZER RESULTS"
    echo "=========================================="
    
    # Check for race conditions by looking for actual hazards in the summary
    if grep -q "RACECHECK SUMMARY:" racecheck_output.txt; then
        # Extract the number of hazards from the summary line
        HAZARD_COUNT=$(grep "RACECHECK SUMMARY:" racecheck_output.txt | grep -oE "[0-9]+ hazards" | grep -oE "[0-9]+" || echo "0")
        if [ "${HAZARD_COUNT}" -gt 0 ]; then
            echo "✗ RACE CONDITIONS DETECTED: ${HAZARD_COUNT} hazards found"
            echo ""
            echo "Race condition details saved in: ${TEMP_DIR}/run_sanitizer/racecheck_output.txt"
            RACE_FOUND=true
        else
            echo "✓ No race conditions detected by racecheck"
            RACE_FOUND=false
        fi
    else
        # If no summary line found, check for any error indicators
        if grep -qE "ERROR|hazard|Hazard" racecheck_output.txt; then
            echo "✗ RACE CONDITIONS DETECTED!"
            echo ""
            echo "Race condition details saved in: ${TEMP_DIR}/run_sanitizer/racecheck_output.txt"
            RACE_FOUND=true
        else
            echo "✓ No race conditions detected by racecheck"
            RACE_FOUND=false
        fi
    fi
    
    # Check for memory errors
    if grep -q "ERROR SUMMARY" memcheck_output.txt; then
        ERROR_COUNT=$(grep "ERROR SUMMARY" memcheck_output.txt | grep -oE "[0-9]+ errors" | grep -oE "[0-9]+")
        if [ "${ERROR_COUNT}" -gt 0 ]; then
            echo "✗ MEMORY ERRORS DETECTED: ${ERROR_COUNT} errors found"
            echo ""
            echo "Memory error details saved in: ${TEMP_DIR}/run_sanitizer/memcheck_output.txt"
            MEM_ERROR_FOUND=true
        else
            echo "✓ No memory errors detected by memcheck"
            MEM_ERROR_FOUND=false
        fi
    else
        echo "✓ No memory errors detected by memcheck"
        MEM_ERROR_FOUND=false
    fi
    
    echo ""
    echo "Compute-sanitizer output saved in:"
    echo "- Race check: ${TEMP_DIR}/run_sanitizer/racecheck_output.txt"
    echo "- Memory check: ${TEMP_DIR}/run_sanitizer/memcheck_output.txt"
    echo ""
    
    # Exit with error if any issues found
    if [ "${RACE_FOUND}" = true ] || [ "${MEM_ERROR_FOUND}" = true ]; then
        echo "Issues detected by compute-sanitizer. Please review the output files for details."
        exit 1
    else
        echo "No issues detected by compute-sanitizer."
        exit 0
    fi
fi

# Run with CUDA_LAUNCH_BLOCKING=1
echo ""
echo "=========================================="
echo "Running with CUDA_LAUNCH_BLOCKING=1..."
if [ "${SINGLE_STREAM}" = true ]; then
    echo "(with amrex.max_gpu_streams=1)"
fi
echo "=========================================="
cd "${TEMP_DIR}"
mkdir run_blocking
cd run_blocking
echo "Running: CUDA_LAUNCH_BLOCKING=1 ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} plotfile_interval=${MAX_TIMESTEPS} ${ADDITIONAL_ARGS}..."
CUDA_LAUNCH_BLOCKING=1 "${BINARY}" ../input.in \
    max_timesteps=${MAX_TIMESTEPS} \
    plotfile_interval=${MAX_TIMESTEPS} \
    checkpoint_interval=-1 \
    ascent_interval=-1 \
    projection_interval=-1 \
    statistics_interval=-1 \
    slice_interval=-1 \
    plotfile_prefix=plt_blocking \
    ${ADDITIONAL_ARGS}
echo "Blocking run completed, checking for plotfiles..."

# Check if run completed successfully - find the FINAL (highest numbered) plotfile
PLOTFILE_BLOCKING=$(find . -maxdepth 1 -name "plt_blocking*" -type d | sort | tail -1)
if [ -z "${PLOTFILE_BLOCKING}" ]; then
    echo "Error: No plotfile generated for blocking run"
    echo "Contents of run directory:"
    ls -la
    exit 1
fi

# Get just the directory name (remove ./ prefix)
PLOTFILE_BLOCKING=$(basename "${PLOTFILE_BLOCKING}")
echo "Found blocking plotfile: ${PLOTFILE_BLOCKING}"

# Run with CUDA_LAUNCH_BLOCKING=0
echo ""
echo "=========================================="
echo "Running with CUDA_LAUNCH_BLOCKING=0..."
if [ "${SINGLE_STREAM}" = true ]; then
    echo "(with amrex.max_gpu_streams=1)"
fi
echo "=========================================="
cd "${TEMP_DIR}"
mkdir run_nonblocking
cd run_nonblocking
echo "Running: CUDA_LAUNCH_BLOCKING=0 ${BINARY} ../input.in max_timesteps=${MAX_TIMESTEPS} plotfile_interval=${MAX_TIMESTEPS} ${ADDITIONAL_ARGS}..."
CUDA_LAUNCH_BLOCKING=0 "${BINARY}" ../input.in \
    max_timesteps=${MAX_TIMESTEPS} \
    plotfile_interval=${MAX_TIMESTEPS} \
    checkpoint_interval=-1 \
    ascent_interval=-1 \
    projection_interval=-1 \
    statistics_interval=-1 \
    slice_interval=-1 \
    plotfile_prefix=plt_nonblocking \
    ${ADDITIONAL_ARGS}
NONBLOCKING_EXIT_CODE=$?

if [ ${NONBLOCKING_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "✗ RACE CONDITION DETECTED!"
    echo ""
    echo "The non-blocking run crashed (exit code: ${NONBLOCKING_EXIT_CODE})"
    echo "while the blocking run completed successfully."
    echo ""
    echo "This indicates a GPU race condition causing non-deterministic behavior."
    echo "The race condition causes instabilities that crash the simulation"
    echo "when kernels execute in different orders."
    echo ""
    echo "Contents of non-blocking run directory:"
    ls -la
    echo ""
    echo "Temporary directory preserved for analysis: ${TEMP_DIR}"
    echo "- Blocking run results:     ${TEMP_DIR}/run_blocking/"
    echo "- Non-blocking run results: ${TEMP_DIR}/run_nonblocking/"
    exit 1
fi

echo "Non-blocking run completed, checking for plotfiles..."

# Check if run completed successfully - find the FINAL (highest numbered) plotfile  
PLOTFILE_NONBLOCKING=$(find . -maxdepth 1 -name "plt_nonblocking*" -type d | sort | tail -1)
if [ -z "${PLOTFILE_NONBLOCKING}" ]; then
    echo "Error: No plotfile generated for non-blocking run"
    echo "Contents of run directory:"
    ls -la
    exit 1
fi

# Get just the directory name (remove ./ prefix)
PLOTFILE_NONBLOCKING=$(basename "${PLOTFILE_NONBLOCKING}")
echo "Found non-blocking plotfile: ${PLOTFILE_NONBLOCKING}"

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
echo "Running fcompare command:"
echo "${FCOMPARE}" --abs_tol 0.0 --rel_tol 0.0 "${PLOT_BLOCKING}" "${PLOT_NONBLOCKING}"
echo ""

# Check if fcompare binary exists
if [ ! -f "${FCOMPARE}" ]; then
    echo "Error: fcompare binary not found at ${FCOMPARE}"
    exit 1
fi

# Check if plotfile directories exist
if [ ! -d "${PLOT_BLOCKING}" ]; then
    echo "Error: Blocking plotfile directory not found: ${PLOT_BLOCKING}"
    exit 1
fi

if [ ! -d "${PLOT_NONBLOCKING}" ]; then
    echo "Error: Non-blocking plotfile directory not found: ${PLOT_NONBLOCKING}"
    exit 1
fi

echo "Running fcompare..."
set +e  # Don't exit on error so we can capture crashed output
"${FCOMPARE}" --abs_tol 0.0 --rel_tol 0.0 "${PLOT_BLOCKING}" "${PLOT_NONBLOCKING}" 2>&1 | tee "${FCOMPARE_OUTPUT}"
FCOMPARE_EXIT_CODE=${PIPESTATUS[0]}
set -e

echo ""
echo "fcompare completed with exit code: ${FCOMPARE_EXIT_CODE}"

# Show captured output if there was any
if [ -s "${FCOMPARE_OUTPUT}" ]; then
    echo ""
    echo "fcompare output:"
    echo "----------------"
    cat "${FCOMPARE_OUTPUT}"
    echo "----------------"
    echo ""
else
    echo ""
    echo "No output captured from fcompare (process may have been killed by signal)"
    echo ""
fi

# Check fcompare exit code
if [ ${FCOMPARE_EXIT_CODE} -eq 0 ]; then
    echo "✓ SUCCESS: Plotfiles are identical - No race condition detected"
    exit 0
else
    echo "✗ FAILURE: Plotfiles differ - RACE CONDITION DETECTED!"
    echo ""
    echo "This indicates a GPU race condition in the code."
    echo "The results depend on kernel execution order."
    echo ""
    echo "Temporary directory preserved for analysis: ${TEMP_DIR}"
    echo "- Blocking run results:     ${TEMP_DIR}/run_blocking/"
    echo "- Non-blocking run results: ${TEMP_DIR}/run_nonblocking/"
    echo ""
    exit 1
fi
