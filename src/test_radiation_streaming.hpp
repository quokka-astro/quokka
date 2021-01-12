#ifndef TEST_RADIATION_STREAMING_HPP_ // NOLINT
#define TEST_RADIATION_STREAMING_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_streaming.hpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

// external headers
#include "matplotlibcpp.h"
#include <fmt/format.h>

// internal headers

#include "radiation_system.hpp"

// function definitions
auto main(int argc, char** argv) -> int;
auto testproblem_radiation_streaming() -> int;

#endif // TEST_RADIATION_STREAMING_HPP_
