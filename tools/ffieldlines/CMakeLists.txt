cmake_minimum_required(VERSION 3.20)

# Standalone build: this project is not part of the Quokka problem build.
#   cmake -S tools/ffieldlines -B build/ffieldlines -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/ffieldlines
#   ctest --test-dir build/ffieldlines --output-on-failure
project(ffieldlines LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(FFIELDLINES_AMREX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../extern/amrex"
    CACHE PATH "AMReX source tree (defaults to Quokka's submodule)")
if(NOT EXISTS "${FFIELDLINES_AMREX_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "AMReX not found at ${FFIELDLINES_AMREX_SOURCE_DIR}; run `git submodule update --init extern/amrex`")
endif()

set(AMReX_SPACEDIM 3 CACHE STRING "" FORCE)
set(AMReX_PRECISION DOUBLE CACHE STRING "" FORCE)
set(AMReX_PARTICLES ON CACHE BOOL "" FORCE)
set(AMReX_PARTICLES_PRECISION DOUBLE CACHE STRING "" FORCE)
set(AMReX_FORTRAN OFF CACHE BOOL "" FORCE)
set(AMReX_FORTRAN_INTERFACES OFF CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "" FORCE)
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_PROBINIT OFF CACHE BOOL "" FORCE)
set(AMReX_HDF5 OFF CACHE BOOL "" FORCE)
set(AMReX_MPI ON CACHE BOOL "Build with MPI")
set(AMReX_OMP OFF CACHE BOOL "Build with OpenMP")
set(AMReX_GPU_BACKEND NONE CACHE STRING "AMReX GPU backend (v1 targets CPU)")
add_subdirectory(${FFIELDLINES_AMREX_SOURCE_DIR} ${CMAKE_BINARY_DIR}/amrex)

add_library(ffieldlines_core STATIC
  Assemble.cpp
  FieldData.cpp
  FieldLineContainer.cpp
  Options.cpp
  Seeds.cpp
  VtkWriter.cpp)
target_include_directories(ffieldlines_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ffieldlines_core PUBLIC AMReX::amrex)

add_executable(ffieldlines main.cpp)
target_link_libraries(ffieldlines PRIVATE ffieldlines_core)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
