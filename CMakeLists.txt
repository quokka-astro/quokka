cmake_minimum_required(VERSION 3.16)
project(QuokkaCode VERSION 1.0
  DESCRIPTION "Radiation hydrodynamics with structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")
#set(CMAKE_C_COMPILER_LAUNCHER "ccache")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/extern/amrex/Tools/CMake)
include(CTest)

set(AMReX_SPACEDIM 1 CACHE STRING "")
set(AMReX_MPI ON CACHE BOOL "" FORCE)
set(AMReX_HDF5 OFF CACHE BOOL "" FORCE)
#set(AMReX_HDF5_ASYNC ON CACHE BOOL "" FORCE)
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_PROBINIT OFF CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "" FORCE)
set(AMReX_SUNDIALS OFF CACHE BOOL "" FORCE)
set(AMReX_ASSERTIONS OFF CACHE BOOL "" FORCE)
set(AMReX_FPE OFF CACHE BOOL "" FORCE)
set(AMReX_TINY_PROFILE ON CACHE BOOL "" FORCE)

option(QUOKKA_PYTHON "Compile with Python support (on/off)" ON)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  enable_language(CUDA)
  set(ENABLE_CUDA ON CACHE BOOL "" FORCE) # for SUNDIALS superbuild
  set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
  set(AMReX_CUDA_FASTMATH OFF CACHE BOOL "" FORCE)
  # fused MAD causes directional asymmetry
  set(CMAKE_CUDA_FLAGS "--fmad=false" CACHE STRING "" FORCE)
  if(CMAKE_VERSION VERSION_LESS 3.20)
    include(AMReX_SetupCUDA)
  endif(CMAKE_VERSION VERSION_LESS 3.20)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # abort! Intel Compiler Classic cannot compile Quokka!
  message(FATAL_ERROR "You have configured CMake to use the 'classic' Intel compilers (icc/icpc), which are the old Intel compilers. They cannot compile Quokka correctly! You must use the new LLVM-based Intel compilers (icx/icpx) instead by adding the following CMake command-line options: -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

#set(EXAMPLES_ENABLE_C OFF CACHE BOOL "" FORCE)
#set_property(GLOBAL PROPERTY ALLOW_DUPLICATE_CUSTOM_TARGETS ON) # sundials superbuild requires this
#add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/sundials ${QuokkaCode_BINARY_DIR}/sundials EXCLUDE_FROM_ALL)

add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrex ${QuokkaCode_BINARY_DIR}/amrex)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/fmt ${QuokkaCode_BINARY_DIR}/fmt)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)
