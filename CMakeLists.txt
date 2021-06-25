cmake_minimum_required(VERSION 3.16)
project(QuokkaCode VERSION 1.0
  DESCRIPTION "Radiation hydrodynamics with structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/extern/amrex/Tools/CMake)
include(CTest)

set(AMReX_SPACEDIM 3 CACHE STRING "" FORCE)
set(AMReX_MPI ON CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "" FORCE)
#set(AMReX_ASSERTIONS ON CACHE BOOL "" FORCE)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  enable_language(CUDA)
  set(AMReX_CUDA_FASTMATH OFF CACHE BOOL "" FORCE)
  # fused MAD causes directional asymmetry
  # (unfortunately, disabling reduces GPU performance by ~30%)
  set(CMAKE_CUDA_FLAGS "--fmad=false" CACHE STRING "" FORCE)
  # Include the AMReX-provided CUDA setup module -- OBSOLETE with CMake >= 3.20
  if(CMAKE_VERSION VERSION_LESS 3.20)
    include(AMReX_SetupCUDA)
  endif(CMAKE_VERSION VERSION_LESS 3.20)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrex ${QuokkaCode_BINARY_DIR}/amrex)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/fmt ${QuokkaCode_BINARY_DIR}/fmt)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)
