# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
set(AMReX_DIFFERENT_COMPILER OFF CACHE BOOL "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
