#!/bin/bash

set -e

cd /software/projects/pawsey0807/chongchong/setonix/quokka/build/gpu-3d
ninja -j12 SphericalCollapse
cd /software/projects/pawsey0807/chongchong/setonix/quokka/tests
flag=$SLURM_JOBID
echo "flag: ${flag}"
../build/gpu-3d/src/problems/SphericalCollapse/SphericalCollapse ../inputs/SphericalCollapseAMR_regression.in > /dev/null
mv chk00020 chk00020.${flag} 
../build/gpu-3d/src/problems/SphericalCollapse/SphericalCollapse ../inputs/SphericalCollapseAMR_regression.in > /dev/null

diff -q chk00020/Level_0/Cell_H chk00020.${flag}/Level_0/Cell_H && echo "test passed"

