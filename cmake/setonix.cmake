# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "CC" CACHE PATH "")
set(AMReX_GPU_BACKEND HIP CACHE STRING "")
set(AMReX_AMD_ARCH gfx90a CACHE STRING "") # MI250X
set(AMREX_GPUS_PER_NODE 8 CACHE STRING "")
set(AMReX_ASCENT OFF BOOL STRING "")

option(QUOKKA_PYTHON ON)
set(AMReX_SPACEDIM 3 CACHE STRING "")