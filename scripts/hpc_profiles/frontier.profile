#!/bin/bash
export proj="ast236"

## modules
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load Core/26.05
module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load cce/21.0.2
module load rocm/10.0.0
module load cray-mpich/9.1.0
module load cray-hdf5/1.14.3.9
module load cray-python/3.12.12
module load cmake/4.1.5

# emacs (optional)
module load emacs

## aliases
alias getNode="salloc -A $proj -J quokka -t 01:00:00 -p batch -N 1"
#   usage: runNode <command>
alias runNode="srun -A $proj -J quokka -t 00:30:00 -p batch -N 1"
alias snodes="sinfo -O PartitionName:12,StateComplete:50,Nodes:10,Reason:90 -S +P,+E,+t"
alias savail="sinfo -O PartitionName:12,Nodes:10,StateComplete:50 -S +P,+E,+t -t alloc,idle"

## environment variables

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a
# compilers
export CC=$(which hipcc)
export CXX=$(which hipcc)

# These flags are required when using hipcc directly instead of the Cray
# compiler wrappers. In particular, link the GPU Transport Layer (GTL) needed
# by GPU-aware MPICH on gfx90a.
export CFLAGS="-I${MPICH_DIR}/include"
export CXXFLAGS="-I${MPICH_DIR}/include"
export LDFLAGS="-L${MPICH_DIR}/lib -lmpi \
  ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem \
  ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
