#!/bin/bash
# please set your project account
export proj="ast236"  # change me!

source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh

module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a

module load cce/21.0.0
module load rocm/7.2.0 # MUST use rocm/6.3.1 or newer
module load cray-mpich/9.1.0

module load cray-hdf5/1.14.3.9
module load cray-python/3.12.12
module load cmake/4.1.0

# adios2 (optional)
module load adios2/2.11.0-mpi

# emacs (optional)
module load emacs

# alias to request an interactive batch node for one hour
alias getNode="salloc -A $proj -J quokka -t 01:00:00 -p batch -N 1"
# alias to run a command on a batch node for up to 30min
#   usage: runNode <command>
alias runNode="srun -A $proj -J quokka -t 00:30:00 -p batch -N 1"

alias snodes="sinfo -O PartitionName:12,StateComplete:50,Nodes:10,Reason:90 -S +P,+E,+t"
alias savail="sinfo -O PartitionName:12,Nodes:10,StateComplete:50 -S +P,+E,+t -t alloc,idle"

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which hipcc)
export CXX=$(which hipcc)
export FC=$(which ftn)

# these flags are REQUIRED
export CFLAGS="-I${MPICH_DIR}/include"
export CXXFLAGS="-I${MPICH_DIR}/include"
export LDFLAGS="-L${MPICH_DIR}/lib -lmpi \
  ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem \
  ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
