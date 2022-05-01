# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
set(AMReX_DIFFERENT_COMPILER ON CACHE BOOL "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
#set(ENV{Ascent_DIR} "/g/data/jh2/bw0729/ascent_build/ascent/install")
set(VTKH_DIR "/g/data/jh2/bw0729/ascent_build/vtk-h/install" CACHE PATH "" FORCE)
set(VTKM_DIR "/g/data/jh2/bw0729/ascent_build/vtk-m/install" CACHE PATH "" FORCE)
option(QUOKKA_PYTHON OFF)
