#!/bin/bash

set -e

QK=/software/projects/pawsey0807/chongchong/setonix/quokka/
BD=build/gpu-3d

source ${QK}/scripts/hpc_profiles/setonix-gpu.profile

cd ${QK}/${BD} 
[[ -e CMakeCache.txt ]] && rm -rf *
cmake -B . -S ../.. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=HIP -DAMReX_SPACEDIM=3 -DHDF5_ROOT=${PAWSEY_HDF5_HOME} -G Ninja

ninja -j16 SN
