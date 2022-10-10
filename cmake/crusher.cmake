# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

# (these modules are required)
# module load cpe/22.08 cmake/3.23.2 craype-accel-amd-gfx90a rocm/5.2.0 cray-mpich cce/14.0.2 cray-hdf5

# (this must be set to use GPU-aware MPI)
# export MPICH_GPU_SUPPORT_ENABLED=1

set(CMAKE_C_COMPILER "cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "CC" CACHE PATH "")
set(AMReX_GPU_BACKEND HIP CACHE STRING "")
set(AMReX_AMD_ARCH gfx90a CACHE STRING "") # MI250X
set(AMREX_GPUS_PER_NODE 8 CACHE STRING "")
option(QUOKKA_PYTHON OFF)
