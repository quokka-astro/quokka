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
#include "fmt/include/fmt/format.h"
#include "matplotlibcpp.h"

// internal headers
#include "athena_arrays.hpp"
#include "radiation_system.hpp"

// function definitions
void testproblem_radiation_streaming();

#endif // TEST_RADIATION_STREAMING_HPP_