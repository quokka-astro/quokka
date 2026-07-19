#!/bin/bash
set -x
rm -rf chk* last_chk
mpirun -np 8 ../build/src/problems/DiskGalaxy/DiskGalaxy ../inputs/DiskGalaxy.toml max_timesteps=1
mpirun -np 8 ../build/src/problems/DiskGalaxy/DiskGalaxy ../inputs/DiskGalaxy.toml max_timesteps=1 restartfile=last_chk
