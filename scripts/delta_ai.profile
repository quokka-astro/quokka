module load nvidia
module unload cray-mpich
export MODULEPATH="/opt/cray/pe/lmod/modulefiles/comnet/nvidia/23.11/ofi/1.0:$MODULEPATH"
module load cray-mpich
module load craype-accel-nvidia90 

# hdf5
module load cray-hdf5

# cmake
module load cmake/3.30.2

# python
module load cray-python/3.11.7

# emacs (optional)
module load emacs/29.3

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia90

# optimize CUDA compilation for Grace-Hopper
export AMREX_CUDA_ARCH=9.0

# compiler environment hints
export CC=cc
export CXX=CC
export FC=ftn
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=CC
