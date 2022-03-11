# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "nvc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "nvc++" CACHE PATH "")
set(AMReX_DIFFERENT_COMPILER ON CACHE BOOL "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
#set(ENV{Ascent_DIR} "/avatar/bwibking/ascent/install")
set(VTKH_DIR "/avatar/bwibking/vtk-h/install" CACHE PATH "" FORCE)
set(VTKM_DIR "/avatar/bwibking/vtk-m/install" CACHE PATH "" FORCE)