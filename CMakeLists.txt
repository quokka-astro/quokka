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
set(AMReX_MPI_THREAD_MULTIPLE ON CACHE BOOL "" FORCE)
set(AMREX_ALLOW_USER_OMP ON CACHE BOOL "" FORCE)
set(AMReX_HDF5 OFF CACHE BOOL "" FORCE) # do not use
set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "" FORCE)
set(AMReX_CONDUIT OFF CACHE BOOL "") # optional
set(AMReX_ASCENT OFF CACHE BOOL "") # optional
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_PROBINIT OFF CACHE BOOL "" FORCE)
set(AMReX_ASSERTIONS OFF CACHE BOOL "" FORCE)
set(AMReX_FPE OFF CACHE BOOL "" FORCE)
set(AMReX_TINY_PROFILE ON CACHE BOOL "" FORCE)

option(QUOKKA_PYTHON "Compile with Python support (on/off)" ON)
option(DISABLE_FMAD "Disable fused multiply-add instructions on GPU (on/off)" ON)
option(ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  enable_language(CUDA)
  
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.2)
    message(FATAL_ERROR "For best performance, you must use CUDA 11.2 or higher to compile Quokka. It is strongly recommended to use a newer version of CUDA.")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION MATCHES 11.6)
    message(FATAL_ERROR "You must use a CUDA version 11.2-11.5 to compile Quokka. CUDA 11.6 has compiler bugs that cause Quokka to crash.")
  endif()

  set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
  
  if(CMAKE_VERSION VERSION_LESS 3.20)
    include(AMReX_SetupCUDA)
  endif(CMAKE_VERSION VERSION_LESS 3.20)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # abort! Intel Compiler Classic cannot compile Quokka!
  message(FATAL_ERROR "You have configured CMake to use the 'classic' Intel compilers (icc/icpc), which are the old Intel compilers. They cannot compile Quokka correctly! You must use the new LLVM-based Intel compilers (icx/icpx) instead by adding the following CMake command-line options: -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

# add OpenMP (needed for async communication thread)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-fopenmp>)
    endif()
endif()

add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrex ${QuokkaCode_BINARY_DIR}/amrex)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/fmt ${QuokkaCode_BINARY_DIR}/fmt)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)
