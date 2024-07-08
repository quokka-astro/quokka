#ifndef MAIN_HPP_ // NOLINT
#define MAIN_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file main.hpp
///

#if defined(__INTEL_COMPILER)
#error                                                                                                                                                         \
    "Quokka cannot be compiled with Intel Compiler Classic! Use the newer LLVM-based Intel compilers (icx/icpx) instead by adding the following CMake command-line options: -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
#endif // __INTEL_COMPILER

// function declarations
auto main(int argc, char **argv) -> int;
auto problem_main() -> int; // defined in problem generator

#endif // MAIN_HPP_
