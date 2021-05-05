cmake_minimum_required(VERSION 3.14)
project(QuokkaCode VERSION 1.0
  DESCRIPTION "Radiation hydrodynamics with structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/extern/amrex/Tools/CMake)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

include(CTest)

set(AMReX_SPACEDIM 1 CACHE STRING "" FORCE)
set(AMReX_MPI OFF CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "" FORCE)
set(AMReX_GPU_BACKEND CUDA)
enable_language(CUDA)
include(AMReX_SetupCUDA)

add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrex ${QuokkaCode_BINARY_DIR}/amrex)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/fmt ${QuokkaCode_BINARY_DIR}/fmt)

add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)

find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")