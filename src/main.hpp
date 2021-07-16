#ifndef MAIN_HPP_ // NOLINT
#define MAIN_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file main.hpp
///

// function declarations
auto main(int argc, char **argv) -> int;
auto problem_main() -> int; // defined in problem generator

#endif // MAIN_HPP_
