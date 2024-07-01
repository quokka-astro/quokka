#!/bin/sh
set -x

$BUILD_DIR/src/HydroBlast3D/test_hydro3d_blast blast_32.in max_walltime=0:00:10 plotfile_interval=100 checkpoint_interval=100
$BUILD_DIR/src/HydroBlast3D/test_hydro3d_blast blast_32.in restartfile=last_chk max_timesteps=1 plotfile_interval=100 checkpoint_interval=100

old_plotfile=`ls -1drt plt*.old.* | head -1`
plotfile=${old_plotfile%.old.*}

$PLOTFILETOOLS_DIR/fcompare.gnu.ex $plotfile $old_plotfile
