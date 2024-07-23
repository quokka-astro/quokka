#!/bin/bash

#source /opt/cray/pe/cpe/23.09/restore_lmod_system_defaults.sh
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load cray-mpich
module load cce/16.0.1
module load rocm/5.7.3

module load cray-hdf5
module load cray-python/3.9.13.1
module load cmake/3.27.7

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which cc)
export CXX=$(which CC)
export FC=$(which ftn)
export CFLAGS="-I${ROCM_PATH}/include"
export CXXFLAGS="-I${ROCM_PATH}/include -mllvm -amdgpu-function-calls=true"

