# please set your project account
export proj=" " # FILL THIS IN

# required dependencies
module load cmake/3.20.2
module load gcc/9.3.0 # newer versions conflict with HDF5 modules
module load cuda/11.7.1
module load hdf5/1.12.2

# optional: faster re-builds
module load ccache
module load ninja

# often unstable at runtime with dependencies
module unload darshan-runtime

# optional: Ascent in situ support
export Ascent_DIR=/sw/summit/ums/ums010/ascent/0.8.0_warpx/summit/cuda/gnu/ascent-install/

# optional: for Python bindings or libEnsemble
module load python/3.8.10
module load freetype/2.10.4     # matplotlib

# set tuning vars for maximizing network bandwidth
# https://docs.olcf.ornl.gov/systems/summit_user_guide.html#spectrum-mpi-tunings-needed-for-maximum-bandwidth
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"

# an alias to request an interactive batch node for two hours
#   for paralle execution, start on the batch node: jsrun <command>
alias getNode="bsub -q debug -P $proj -W 2:00 -nnodes 1 -Is /bin/bash"

# an alias to run a command on a batch node
#   usage: runNode <command>
alias runNode="bsub -q debug -P $proj -W 2:00 -nnodes 1 -I"

# make output group-readable by default
umask 0027

# optimize CUDA compilation for V100
export AMREX_CUDA_ARCH=7.0

# compiler environment hints
export CC=$(which gcc)
export CXX=$(which g++)
export FC=$(which gfortran)
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$(which g++)
