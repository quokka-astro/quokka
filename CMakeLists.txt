cmake_minimum_required(VERSION 3.1)
project(QuokkaCode VERSION 1.0
  DESCRIPTION "Radiation hydrodynamics with structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CTest)

set(AMReX_SPACEDIM 1)
set(AMReX_MPI OFF)
set(AMReX_LINEAR_SOLVERS OFF)

add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)

find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")