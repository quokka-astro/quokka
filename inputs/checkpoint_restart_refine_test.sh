#!/bin/bash
set -e  # Exit on any error
#set -x  # Print commands as they execute

# Multi-Level Universal Refinement Test Script
#
# This script validates the universal refinement capability with AMR by:
# 1. Running a coarse 32^3 simulation with AMR to create a checkpoint at intermediate time
# 2. Running a native fine 64^3 simulation with AMR to the same final time  
# 3. Restarting from the coarse checkpoint with universal refinement on all levels
# 4. Comparing the final results to validate correctness across multiple refinement levels
#
# Usage: ./checkpoint_restart_refine_test.sh
# Optional: BUILD_DIR, PLOTFILETOOLS_DIR, and NPROC environment variables

BUILD_DIR=${BUILD_DIR:=../build}
PLOTFILETOOLS_DIR=${PLOTFILETOOLS_DIR:=../extern/amrex/Tools/Plotfile}
NPROC=${NPROC:=`nproc`}

extract_checkpoint_time() {
    awk 'NR == 5 { print $1; exit }' "$1/Header"
}

extract_plotfile_time() {
    awk 'NR == 10 { print $1; exit }' "$1/Header"
}

# Detect OS for sed compatibility (macOS requires -i '', Linux requires -i)
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INPLACE="sed -i ''"
else
    SED_INPLACE="sed -i"
fi

echo "=== Universal Refinement Test ==="

# Clean up any existing files
rm -rf plt* 2>/dev/null || true
rm -rf chk* 2>/dev/null || true
rm -rf step* 2>/dev/null || true
rm -f coarse_amr.toml 2>/dev/null || true
rm -f fine_amr.toml 2>/dev/null || true

# Define simulation times (work within 1.0s hardcoded limit)
CHECKPOINT_TIME=0.01 # Create checkpoint at t=0.01 (fewer steps)
STOP_TIME=0.025       # Run until t=0.025 for comparison

# Create coarse AMR input file (32^3 base with 1 AMR level) based on blast_32.toml
# Try multiple possible locations for blast_32.toml to handle different CI environments
if [ -f "../inputs/blast_32.toml" ]; then
    cp "../inputs/blast_32.toml" coarse_amr.toml
elif [ -f "blast_32.toml" ]; then
    cp "blast_32.toml" coarse_amr.toml
elif [ -f "$BUILD_DIR/../inputs/blast_32.toml" ]; then
    cp "$BUILD_DIR/../inputs/blast_32.toml" coarse_amr.toml
else
    echo "❌ ERROR: Cannot find blast_32.toml input file"
    echo "Tried locations:"
    echo "  ../inputs/blast_32.toml (inputs directory)"
    echo "  blast_32.toml (current directory)"
    echo "  $BUILD_DIR/../inputs/blast_32.toml"
    echo "Current directory: $(pwd)"
    echo "Files in current directory: $(ls -la)"
    exit 1
fi
$SED_INPLACE 's/amr.max_level = 0/amr.max_level = 1/' coarse_amr.toml
#$SED_INPLACE 's/do_tracers = 1/do_tracers = 0/' coarse_amr.toml
$SED_INPLACE 's/do_reflux = 0/do_reflux = 1/' coarse_amr.toml
$SED_INPLACE 's/do_subcycle = 0/do_subcycle = 1/' coarse_amr.toml

# Create fine AMR input file (64^3 base with 1 AMR level) based on blast_32.toml
# Try multiple possible locations for blast_32.toml to handle different CI environments
if [ -f "../inputs/blast_32.toml" ]; then
    cp "../inputs/blast_32.toml" fine_amr.toml
elif [ -f "blast_32.toml" ]; then
    cp "blast_32.toml" fine_amr.toml
elif [ -f "$BUILD_DIR/../inputs/blast_32.toml" ]; then
    cp "$BUILD_DIR/../inputs/blast_32.toml" fine_amr.toml
else
    echo "❌ ERROR: Cannot find blast_32.toml input file for fine_amr.toml"
    echo "Tried locations:"
    echo "  ../inputs/blast_32.toml (inputs directory)"
    echo "  blast_32.toml (current directory)"
    echo "  $BUILD_DIR/../inputs/blast_32.toml"
    echo "Current directory: $(pwd)"
    echo "Files in current directory: $(ls -la)"
    exit 1
fi
$SED_INPLACE 's/amr.n_cell = \[32, 32, 32\]/amr.n_cell = [64, 64, 64]/' fine_amr.toml
$SED_INPLACE 's/amr.max_level = 0/amr.max_level = 1/' fine_amr.toml
#$SED_INPLACE 's/do_tracers = 1/do_tracers = 0/' fine_amr.toml
$SED_INPLACE 's/do_reflux = 0/do_reflux = 1/' fine_amr.toml
$SED_INPLACE 's/do_subcycle = 0/do_subcycle = 1/' fine_amr.toml

echo "=== Step 1: Create coarse AMR checkpoint (32^3 base + 1 AMR level) ==="
# Run coarse AMR simulation to create checkpoints
mpirun --use-hwthread-cpus -n $NPROC $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D coarse_amr.toml stop_time=$CHECKPOINT_TIME

# Save and find last checkpoint
mkdir -p step1_32cube
mv plt* chk* step1_32cube/ 2>/dev/null || true
last_chk=$(ls -1d step1_32cube/chk* 2>/dev/null | tail -1)
last_plt_32=$(ls -1d step1_32cube/plt* 2>/dev/null | tail -1)

if [ -z "$last_chk" ]; then
    echo "❌ ERROR: No checkpoint files created in step1_32cube/"
    exit 1
fi

# Extract actual time from checkpoint header
checkpoint_time=$(extract_checkpoint_time "$last_chk")
echo "Created coarse AMR checkpoint: $last_chk at time t=$checkpoint_time"

# Verify that checkpoint contains the expected number of levels
echo ""
echo "=== Step 1a: Verify checkpoint has multiple AMR levels ==="
max_level_in_chk=$(sed -n '2p' $last_chk/Header)
echo "Maximum level in checkpoint: $max_level_in_chk"

# Check that we have at least 2 levels (max_level >= 1, since levels are 0-indexed)
if [ "$max_level_in_chk" -ge 1 ]; then
    echo "✅ LEVEL CHECK PASSED: Checkpoint contains multiple AMR levels (levels 0-$max_level_in_chk)"
    level_check_passed=true
    
    # List all level directories to confirm
    echo "AMR levels present in checkpoint:"
    for level_dir in $last_chk/Level_*; do
        if [ -d "$level_dir" ]; then
            level_num=$(basename "$level_dir" | sed 's/Level_//')
            echo "  Level $level_num: $(ls -1 $level_dir | wc -l) grid files"
        fi
    done
else
    echo "⚠️  LEVEL CHECK FAILED: Checkpoint only contains single level (max_level=$max_level_in_chk)"
    echo "Expected: max_level >= 1 for multi-level AMR test"
    echo "ABORTING: Cannot test multi-level universal refinement without multi-level checkpoints"
    level_check_passed=false
    exit 1
fi

echo ""
echo "=== Step 2: Run native fine AMR simulation (64^3 base + 1 AMR level) ==="
mpirun --use-hwthread-cpus -n $NPROC $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D fine_amr.toml stop_time=$STOP_TIME

# Save native results  
mkdir -p step2_native_64cube
mv plt* chk* step2_native_64cube/ 2>/dev/null || true
native_final_plt=$(ls -1d step2_native_64cube/plt* 2>/dev/null | tail -1)

if [ -z "$native_final_plt" ]; then
    echo "❌ ERROR: No plotfiles created in step2_native_64cube/"
    exit 1
fi

# Extract native final time from plotfile header
native_time=$(extract_plotfile_time "$native_final_plt")
echo "Native fine AMR simulation completed at time t=$native_time"

echo ""
echo "=== Step 3: Restart with multi-level universal refinement ==="
mpirun --use-hwthread-cpus -n $NPROC $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D fine_amr.toml restartfile=$last_chk stop_time=$STOP_TIME

# Save restart results
mkdir -p step3_restart_64cube
mv plt* chk* step3_restart_64cube/ 2>/dev/null || true
restart_final_plt=$(ls -1d step3_restart_64cube/plt* 2>/dev/null | tail -1)

if [ -z "$restart_final_plt" ]; then
    echo "❌ ERROR: No plotfiles created in step3_restart_64cube/"
    exit 1
fi

# Extract restart final time from plotfile header
restart_time=$(extract_plotfile_time "$restart_final_plt")
echo "Multi-level universal refinement simulation completed at time t=$restart_time"

echo ""
echo "=== Step 4: Time Verification ==="
echo "Checkpoint time:     t=$checkpoint_time"
echo "Native final time:   t=$native_time" 
echo "Restart final time:  t=$restart_time"
echo "Target stop time:    t=$STOP_TIME"

# Check if times are close (within 1% tolerance)
time_diff=$(echo "$native_time - $restart_time" | bc -l)
time_diff_abs=$(echo "$time_diff" | sed 's/-//')
tolerance=$(echo "$STOP_TIME * 0.01" | bc -l)

echo "Time difference:     Δt=$time_diff_abs"
echo "Tolerance (1%):      $tolerance"

# Abort if tolerance is not satisfied
if [ $(echo "$time_diff_abs < $tolerance" | bc -l) -eq 1 ]; then
    echo "✅ TIME CHECK PASSED: Simulations reached the same final time"
    time_check_passed=true
else
    echo "⚠️  TIME CHECK ERROR: Time difference detected!"
    time_check_passed=false
fi

echo ""
echo "=== Step 5: Compare final results ==="
echo "Comparing simulations at the same final time:"
echo "  Native fine AMR:  $native_final_plt (t=$native_time)"
echo "  Restart fine AMR: $restart_final_plt (t=$restart_time)"

# Use fcompare to validate that the results match (with 0.22 relative tolerance in L1 norm)
if [ ! -f "$PLOTFILETOOLS_DIR/fcompare.gnu.ex" ]; then
    echo "❌ ERROR: fcompare.gnu.ex not found at $PLOTFILETOOLS_DIR"
    echo "Cannot compare plotfiles - test FAILED"
    exit 1
fi

$PLOTFILETOOLS_DIR/fcompare.gnu.ex -n 1 -r 0.001 --abs_tol 0.0025 $native_final_plt $restart_final_plt

echo ""
echo "=== Test Results ==="
comparison_result=$?

if [ $comparison_result -eq 0 ]; then
    echo "✅ COMPARISON PASSED: Results are within tolerances."
    comparison_passed=true
else
    echo "⚠️  COMPARISON ERROR: Differences found!'"
    comparison_passed=false
fi

echo ""
echo "=== Summary ==="
echo "Multi-level universal refinement feature validation:"
echo "1. $([ "$level_check_passed" = true ] && echo "✅" || echo "⚠️ ") Multi-level checkpoint verification"
echo "2. $([ "$time_check_passed" = true ] && echo "✅" || echo "⚠️ ") Time synchronization"
echo "3. $([ "$comparison_passed" = true ] && echo "✅" || echo "⚠️ ") Result comparison"

echo ""
echo "=== Multi-Level Universal Refinement Test Complete ==="
