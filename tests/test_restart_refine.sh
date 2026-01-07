#!/bin/bash
set -x
rm -rf chk* last_chk
mpirun -np 8 ../build/src/problems/DiskGalaxy/DiskGalaxy DiskGalaxy.in max_timesteps=1
mpirun -np 8 ../build/src/problems/DiskGalaxy/DiskGalaxy DiskGalaxy_restart.in max_timesteps=2 restartfile=last_chk
