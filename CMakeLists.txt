cmake_minimum_required(VERSION 3.1)
project(TwoMomentRad VERSION 1.0
  DESCRIPTION "Radiation transport library for structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${TwoMomentRad_SOURCE_DIR}/src ${TwoMomentRad_BINARY_DIR}/src)

include(CTest)

find_program(CLANG_TIDY_EXE
  NAMES "clang-tidy")
set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks='*,-fuchsia-*,-misc-non-private-member-variables-in-classes,-cppcoreguidelines-non-private-member-variables-in-classes,-google-runtime-references,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers")