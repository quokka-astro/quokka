#!/bin/bash

source /opt/cray/pe/cpe/22.09/restore_lmod_system_defaults.sh

module load cmake/3.21.4
module load craype-accel-amd-gfx90a

. /software/projects/pawsey0807/bwibking/spack/share/spack/setup-env.sh
spack load hip@5.3.0
spack load rocrand@5.3.0
spack load rocprim@5.3.0

module load cray-mpich
module load cce/14.0.3
module load cray-hdf5
module load cray-python/3.9.13.1

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which hipcc)
export CXX=$(which hipcc)
export FC=$(which ftn)
export CFLAGS="-I${ROCM_PATH}/include -I${MPICH_DIR}/include -I${HDF5_DIR}/include"
export CXXFLAGS="-I${ROCM_PATH}/include -I${MPICH_DIR}/include -I${HDF5_DIR}/include"
export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64 -L${MPICH_DIR}/lib -lmpi -L${HDF5_DIR}/lib -lhdf5 ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi_gtl_hsa"
