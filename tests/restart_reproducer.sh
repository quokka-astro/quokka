#!/bin/sh
set -x

../build/src/ShockCloud/shock_cloud ShockCloud_32.in max_walltime=0:00:30
../build/src/ShockCloud/shock_cloud ShockCloud_32.in restartfile=last_chk max_timesteps=1

old_plotfile=`ls -1drt plt*.old.* | head -1`
plotfile=${old_plotfile%.old.*}

fcompare.gnu.ex $plotfile $old_plotfile
