# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "")
set(AMReX_ASCENT OFF CACHE BOOL "" FORCE)
set(AMReX_CONDUIT OFF CACHE BOOL "" FORCE)
set(ENV{Ascent_DIR} "/projects/cvz/bwibking/ascent/install/ascent-develop")
option(QUOKKA_PYTHON OFF)
