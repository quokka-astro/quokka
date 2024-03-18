#!/bin/bash

source /opt/cray/pe/cpe/23.03/restore_lmod_system_defaults.sh

module load cmake/3.24.3
module load cray-mpich
module load cce/15.0.1
module load cray-hdf5
module load cray-python/3.9.13.1
module load rocm/5.4.3

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=0

# compiler environment hints
export CC=$(which cc)
export CXX="$(which CC)"
export FC=$(which ftn)
export CXXFLAGS="$CXXFLAGS -fno-cray"
