#!/bin/bash
set -e  # Exit on any error
#set -x  # Print commands as they execute

# Universal Refinement Test Script
#
# This script validates the universal refinement capability by:
# 1. Running a 32^3 simulation to create a checkpoint at intermediate time
# 2. Running a native 64^3 simulation to the same final time  
# 3. Restarting from the 32^3 checkpoint with 64^3 grid (universal refinement)
# 4. Comparing the final results to validate correctness
#
# Usage: ./universal_refine_time_comparison.sh
# Requires: BUILD_DIR and PLOTFILETOOLS_DIR environment variables

BUILD_DIR=${BUILD_DIR:=../build}
PLOTFILETOOLS_DIR=${PLOTFILETOOLS_DIR:=../extern/amrex/Tools/Plotfile}
NPROC=${NPROC:=`nproc`}

echo "=== Universal Refinement Test ==="

# Clean up any existing files
rm -rf plt* chk* step*

# Define simulation times (work within 1.0s hardcoded limit)
CHECKPOINT_TIME=0.01 # Create checkpoint at t=0.01 (fewer steps)
STOP_TIME=0.05       # Run until t=0.05 for comparison

echo "=== Step 1: Create 32^3 checkpoint (run briefly, stop after checkpoint creation) ==="
# Run 32^3 simulation to create checkpoints
mpirun --use-hwthread-cpus -n $NPROC $BUILD_DIR/src/problems/HydroBlast3D/test_hydro3d_blast blast_32.in stop_time=$CHECKPOINT_TIME

# Save and find last checkpoint
mkdir -p step1_32cube
mv plt* chk* step1_32cube/
last_chk=$(ls -1d step1_32cube/chk* | tail -1)
last_plt_32=$(ls -1d step1_32cube/plt* | tail -1)

# Extract actual time from checkpoint (time is on line 4 of Header)
checkpoint_time=$(sed -n '4p' $last_chk/Header)
echo "Created 32^3 checkpoint: $last_chk at time t=$checkpoint_time"

echo ""
echo "=== Step 2: Run native 64^3 simulation ==="
mpirun --use-hwthread-cpus -n $NPROC $BUILD_DIR/src/problems/HydroBlast3D/test_hydro3d_blast blast_64.in stop_time=$STOP_TIME

# Save native results  
mkdir -p step2_native_64cube
mv plt* chk* step2_native_64cube/
native_final_plt=$(ls -1d step2_native_64cube/plt* | tail -1)

# Extract native final time (time is on line 13 of plotfile Header)
native_time=$(sed -n '13p' $native_final_plt/Header)
echo "Native 64^3 simulation completed at time t=$native_time"

echo ""
echo "=== Step 3: Restart with universal refinement ==="
mpirun --use-hwthread-cpus -n $NPROC $BUILD_DIR/src/problems/HydroBlast3D/test_hydro3d_blast blast_64.in restartfile=$last_chk stop_time=$STOP_TIME

# Save restart results
mkdir -p step3_restart_64cube
mv plt* chk* step3_restart_64cube/
restart_final_plt=$(ls -1d step3_restart_64cube/plt* | tail -1)

# Extract restart final time (time is on line 13 of plotfile Header)
restart_time=$(sed -n '13p' $restart_final_plt/Header)
echo "Universal refinement simulation completed at time t=$restart_time"

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
echo "  Native 64^3:  $native_final_plt (t=$native_time)"
echo "  Restart 64^3: $restart_final_plt (t=$restart_time)"

# Use fcompare to validate that the results match (with 0.22 relative tolerance in L1 norm)
$PLOTFILETOOLS_DIR/fcompare.gnu.ex -n 1 -r 0.22 $native_final_plt $restart_final_plt

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
echo "Universal refinement feature validation:"
echo "1. $([ "$time_check_passed" = true ] && echo "✅" || echo "⚠️ ") Time synchronization"
echo "2. $([ "$comparison_passed" = true ] && echo "✅" || echo "⚠️ ") Result comparison"

echo ""
echo "=== Universal Refinement Test Complete ==="
