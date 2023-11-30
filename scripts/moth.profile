module load mpi/openmpi-x86_64
module load rocm/5.7.0

# optimize ROCm/HIP compilation for MI210
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which hipcc)
export CXX=$(which hipcc)
export CFLAGS="-I${ROCM_PATH}/include"
export CXXFLAGS="-I${ROCM_PATH}/include"
export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
