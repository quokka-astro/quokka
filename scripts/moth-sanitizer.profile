module load mpi/openmpi-x86_64
module load rocm/5.7.0

# set CMake envs
export CMAKE_GENERATOR="Unix Makefiles"

## for GPU ASAN on MI210 GPUs:
export AMREX_AMD_ARCH=gfx90a:xnack+
export HSA_XNACK=1
export LD_LIBRARY_PATH=/opt/rocm-5.7.0/llvm/lib/clang/17.0.0/lib/linux:$LD_LIBRARY_PATH

# compiler environment hints
export CC=$(which hipcc)
export CXX=$(which hipcc)
export CFLAGS="-I${ROCM_PATH}/include"
export CXXFLAGS="-fsanitize=address -shared-libsan -g -I${ROCM_PATH}/include"
export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
