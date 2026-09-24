cmake_minimum_required(VERSION 3.22)
project(PlotfileToOpenPMD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)

# Point to the locally built ADIOS2 from scripts/bash/build_adios2_with_compressors.sh
if(NOT DEFINED ADIOS2_DIR OR ADIOS2_DIR STREQUAL "ADIOS2_DIR-NOTFOUND")
  set(ADIOS2_DIR "${REPO_ROOT}/scripts/bash/local-adios2/lib/cmake/adios2" CACHE PATH "Path to ADIOS2 CMake config" FORCE)
endif()
list(PREPEND CMAKE_PREFIX_PATH "${ADIOS2_DIR}")
if(NOT (EXISTS "${ADIOS2_DIR}/ADIOS2Config.cmake" OR EXISTS "${ADIOS2_DIR}/adios2-config.cmake"))
  message(FATAL_ERROR "ADIOS2 CMake config not found in ADIOS2_DIR='${ADIOS2_DIR}'. Build ADIOS2 via scripts/bash/build_adios2_with_compressors.sh or set ADIOS2_DIR accordingly.")
endif()

# Build bundled AMReX submodule
set(AMReX_SPACEDIM 3 CACHE STRING "Dimensionality" FORCE)
set(AMReX_BUILD_TUTORIALS OFF CACHE BOOL "" FORCE)
set(AMReX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AMReX_BUILD_MPI ON CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "" FORCE)
add_subdirectory("${REPO_ROOT}/extern/amrex" "${CMAKE_BINARY_DIR}/amrex")

# Build bundled openPMD-api submodule with ADIOS2 only
set(openPMD_USE_ADIOS2 ON CACHE BOOL "Enable ADIOS2 backend" FORCE)
set(openPMD_USE_HDF5 OFF CACHE BOOL "Disable HDF5 backend" FORCE)
set(openPMD_USE_PYTHON OFF CACHE BOOL "" FORCE)
set(openPMD_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(openPMD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(openPMD_BUILD_CLI_TOOLS OFF CACHE BOOL "" FORCE)
set(openPMD_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${REPO_ROOT}/extern/openPMD-api" "${CMAKE_BINARY_DIR}/openPMD-api")

add_executable(PlotfileToOpenPMD PlotfileToOpenPMD.cpp)
target_link_libraries(PlotfileToOpenPMD PRIVATE AMReX::amrex openPMD::openPMD)
