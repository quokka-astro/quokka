## host configuration
## run with `cmake -C host_config.cmake ..` from inside build directory
##
## for ascent support:
##   Ascent_DIR=../../ascent/install cmake -C ../cmake/avatar_gcc.cmake .. -G Ninja

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(AMReX_DIFFERENT_COMPILER OFF CACHE BOOL "")
set(AMReX_GPU_BACKEND NONE CACHE STRING "")
