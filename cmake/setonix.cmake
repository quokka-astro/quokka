# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(AMReX_GPU_BACKEND HIP CACHE STRING "")
set(AMReX_AMD_ARCH gfx90a CACHE STRING "") # MI250X
set(AMReX_ASCENT OFF BOOL STRING "")

option(QUOKKA_PYTHON ON)
set(AMReX_SPACEDIM 3 CACHE STRING "")