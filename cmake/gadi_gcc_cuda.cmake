# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
# see https://github.com/BenWibking/quokka/wiki/Building-Ascent
set(AMReX_ASCENT ON CACHE BOOL "")
set(AMReX_CONDUIT ON CACHE BOOL "")
option(QUOKKA_PYTHON OFF)
