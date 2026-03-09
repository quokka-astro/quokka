module load cpe/25.09

module load PrgEnv-gnu
module load gcc-native/14
module load cray-mpich/9.0.1
module load craype-accel-nvidia90 
module load cudatoolkit/25.5_12.9

# hdf5
module load cray-hdf5/1.14.3.7

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

# an alias to request an interactive batch node for one hour
#   for parallel execution, start on the batch node: srun <command>
alias getNode="salloc -N 1 --ntasks-per-node=4 -t 1:00:00 --gpu-bind=none -c 32 -G 4 -A $proj"
