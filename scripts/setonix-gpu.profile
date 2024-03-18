#!/bin/bash

source /opt/cray/pe/cpe/23.03/restore_lmod_system_defaults.sh

module load cmake/3.24.3
module load craype-accel-amd-gfx90a
module load rocm/5.2.3
module load cray-mpich
module load cce/15.0.1
module load cray-hdf5
module load cray-python/3.9.13.1

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# allow CMake to find Ascent
export Ascent_DIR=/software/projects/pawsey0807/bwibking/ascent_06082023/install/ascent-develop/lib/cmake/ascent/

# compiler environment hints
export CC=$(which cc)
export CXX=$(which CC)
export FC=$(which ftn)
export CFLAGS="-I${ROCM_PATH}/include"
export CXXFLAGS="-I${ROCM_PATH}/include -fno-cray"

