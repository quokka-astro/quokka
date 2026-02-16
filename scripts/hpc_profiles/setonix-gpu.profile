#!/bin/bash

source /opt/cray/pe/cpe/25.03/restore_lmod_system_defaults.sh

module load cpe/25.03
module load pawseyenv/2025.08

module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a

module load rocm/6.4.1 # MUST use 6.3+ to avoid compiler bugs
module load cray-mpich
module load cce/19.0.0

# hdf5
module load hdf5/1.14.5-parallel-api-v112

# adios2 (optional)
#module load adios2/2.10.2-hdf5

# python
module load cray-python/3.11.7

# cmake
module load cmake/3.30.5

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which hipcc)
export CXX=$(which hipcc)
export FC=$(which ftn)

# these flags are REQUIRED
export CFLAGS="-I${MPICH_DIR}/include --gcc-install-dir=/usr/lib64/gcc/x86_64-suse-linux/14"
export CXXFLAGS="-I${MPICH_DIR}/include --gcc-install-dir=/usr/lib64/gcc/x86_64-suse-linux/14"
export LDFLAGS="-L${MPICH_DIR}/lib -lmpi \
  ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem \
  ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
