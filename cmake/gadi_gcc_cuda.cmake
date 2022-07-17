# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
set(AMReX_ASCENT ON CACHE BOOL "" FORCE)
set(AMReX_CONDUIT ON CACHE BOOL "" FORCE)
option(QUOKKA_PYTHON OFF)
set(ENV{Ascent_DIR} "/g/data/jh2/bw0729/ascent_build/0.9.0-dev/install/ascent-develop")
