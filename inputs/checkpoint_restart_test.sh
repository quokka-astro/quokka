#!/bin/sh
set -x

# set number of binary output files per level
# (NOTE: AMReX *never* outputs more binary files than the number of MPI ranks,
#  so a non-trivial test requires running with NPROC > NFILES.)
NFILES=2
NPROC=4

# run to generate checkpoint
mpirun --use-hwthread-cpus -np $NPROC $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D ../inputs/blast_32.toml max_walltime=0:00:10 plotfile_interval=100 checkpoint_interval=100 amr.plot_nfiles=$NFILES amr.checkpoint_nfiles=$NFILES

# [amr.plot_nfiles test] verify that the last plotfile contains two binary files per level
plotfile=`ls -1drt plt* | head -1`
nfiles_plt_actual=`ls -1 $plotfile/Level_0/Cell_D_* | wc -l | tr -d ' '`
if [ "$nfiles_plt_actual" = "$NFILES" ]; then
    echo "amr.plot_nfiles working as expected."
else
    echo "TEST FAILED: Wrong number of binary cell data files in plotfiles!"
    exit 1
fi

# [amr.checkpoint_nfiles test] verify that the last checkpoint contains two binary files per level
chkfile=`ls -1drt chk* | head -1`
nfiles_chk_actual=`ls -1 $chkfile/Level_0/Cell_D_* | wc -l | tr -d ' '`
if [ "$nfiles_chk_actual" = "$NFILES" ]; then
    echo "amr.checkpoint_nfiles working as expected."
else
    echo "TEST FAILED: Wrong number of binary cell data files in checkpoints!"
    exit 1
fi

# restart from checkpoint
mpirun --use-hwthread-cpus -np $NPROC $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D ../inputs/blast_32.toml restartfile=last_chk max_timesteps=1 plotfile_interval=100 checkpoint_interval=100

# verify that the original run and restart produce the same final plotfile
old_plotfile=`ls -1drt plt*.old.* | head -1`
plotfile=${old_plotfile%.old.*}
$PLOTFILETOOLS_DIR/fcompare.gnu.ex $plotfile $old_plotfile

# [default amr.plot_nfiles test] verify that -1 means one binary file per MPI rank
nfiles_restart_plt_actual=`ls -1 $plotfile/Level_0/Cell_D_* | wc -l | tr -d ' '`
if [ "$nfiles_restart_plt_actual" = "$NPROC" ]; then
    echo "default amr.plot_nfiles working as expected."
else
    echo "TEST FAILED: Default number of binary cell data files in plotfiles should match MPI ranks!"
    exit 1
fi

# [default amr.checkpoint_nfiles test] verify that -1 means one binary file per MPI rank
chkfile=last_chk
nfiles_restart_chk_actual=`ls -1 $chkfile/Level_0/Cell_D_* | wc -l | tr -d ' '`
if [ "$nfiles_restart_chk_actual" = "$NPROC" ]; then
    echo "default amr.checkpoint_nfiles working as expected."
else
    echo "TEST FAILED: Default number of binary cell data files in checkpoints should match MPI ranks!"
    exit 1
fi

# [ghost-cell corruption test] verify that checkpoint files do not contain stale
# ghost-cell data that could corrupt valid cells after restart with a different
# MPI decomposition.  (See https://github.com/quokka-astro/quokka/issues/1868.)
#
# Strategy:
#   1. Clean up output from the first phase.
#   2. Run a fresh simulation that writes a checkpoint WITHOUT any prior
#      plotfile (so ghost cells are never sanitized by fillBoundaryConditions).
#   3. Save a plotfile written at the same step as the checkpoint.
#   4. Restart from that checkpoint with a DIFFERENT number of MPI ranks
#      (triggering re-boxing / load-balancing) and take one step.
#   5. Compare the restarted plotfile against the saved one.
#
# If stale checkpoint ghost cells leak into valid cells after re-boxing,
# fcompare will report differences.

# Clean up from the first phase
rm -rf plt* chk* last_chk 2>/dev/null || true

NPROC_ORIG=4
NPROC_RESTART=2

# Run to generate a checkpoint (with NO prior plotfile sanitization)
mpirun --use-hwthread-cpus -np $NPROC_ORIG $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D ../inputs/blast_32.toml max_walltime=0:00:10 plotfile_interval=0 checkpoint_interval=100 amr.checkpoint_nfiles=$NFILES

# Save the checkpoint name
chkfile_ghost=`ls -1drt chk* | head -1`
if [ -z "$chkfile_ghost" ]; then
    echo "TEST FAILED: No checkpoint produced for ghost-cell test!"
    exit 1
fi

# Write a reference plotfile at the checkpoint time for later comparison
mpirun --use-hwthread-cpus -np $NPROC_ORIG $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D ../inputs/blast_32.toml restartfile=$chkfile_ghost max_timesteps=0 plotfile_interval=1 checkpoint_interval=0
ref_plotfile=`ls -1drt plt* | head -1`
if [ -z "$ref_plotfile" ]; then
    echo "TEST FAILED: No reference plotfile produced for ghost-cell test!"
    exit 1
fi
mv "$ref_plotfile" ref_plotfile

# Restart from checkpoint with a DIFFERENT MPI rank count (no advance).
# This tests that the initial restarted state is correct; if stale
# checkpoint ghost cells leak into valid cells after re-boxing,
# the first plotfile written on restart will differ from the reference.
mpirun --use-hwthread-cpus -np $NPROC_RESTART $BUILD_DIR/src/problems/HydroBlast3D/HydroBlast3D ../inputs/blast_32.toml restartfile=$chkfile_ghost max_timesteps=0 plotfile_interval=1 checkpoint_interval=0
restart_plotfile=`ls -1drt plt* | head -1`
if [ -z "$restart_plotfile" ]; then
    echo "TEST FAILED: No restart plotfile produced for ghost-cell test!"
    exit 1
fi

# The plotfile written immediately after restart (step 0) must match
# the reference. If stale ghost cells from the checkpoint leaked into
# valid cells after re-boxing, fcompare will report a mismatch.
$PLOTFILETOOLS_DIR/fcompare.gnu.ex -a -n 1 -r 0.001 --abs_tol 0.0025 ref_plotfile "$restart_plotfile"
