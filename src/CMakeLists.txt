cmake_minimum_required(VERSION 3.1)
project(TwoMomentRad VERSION 1.0
  DESCRIPTION "Radiation transport library for structured AMR"
  LANGUAGES CXX)

add_subdirectory(${TwoMomentRad_SOURCE_DIR}/kokkos ${TwoMomentRad_BINARY_DIR}/kokkos)
find_package(PythonLibs REQUIRED)

add_executable(driver.exe hydro_system.cpp radiation_system.cpp test_radiation_marshak.cpp hyperbolic_system.cpp test_advection.cpp test_radiation_matter_coupling.cpp linear_advection.cpp test_hydro_shocktube.cpp test_radiation_streaming.cpp main.cpp test_hydro_wave.cpp fmt/include/fmt/format.h)

add_definitions(-DFMT_HEADER_ONLY)
set_target_properties(driver.exe PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(driver.exe ${PYTHON_LIBRARY})
target_link_libraries(driver.exe kokkos)

find_program(CLANG_TIDY_EXE
  NAMES "clang-tidy")
set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")

