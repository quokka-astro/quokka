## host configuration
## run with `cmake -C host_config.cmake ..` from inside build directory
##
## for ascent support:
##   Ascent_DIR=../../ascent/install cmake -C ../cmake/avatar_gcc_cuda.cmake .. -G Ninja

set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "")
set(AMReX_DIFFERENT_COMPILER OFF CACHE BOOL "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
#set(ENV{Ascent_DIR} "/avatar/bwibking/ascent/install")
set(VTKH_DIR "/avatar/bwibking/vtk-h/install" CACHE PATH "" FORCE)
set(VTKM_DIR "/avatar/bwibking/vtk-m/install" CACHE PATH "" FORCE)
