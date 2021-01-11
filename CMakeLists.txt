cmake_minimum_required(VERSION 3.1)
project(QuokkaCode VERSION 1.0
  DESCRIPTION "Radiation hydrodynamics with structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CTest)

add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)

find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE} -checks='*,-fuchsia-*,-misc-non-private-member-variables-in-classes,-cppcoreguidelines-non-private-member-variables-in-classes,-google-runtime-references,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers")