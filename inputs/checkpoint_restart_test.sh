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
