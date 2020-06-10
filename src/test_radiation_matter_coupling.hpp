#ifndef TEST_RADIATION_MATTER_COUPLING_HPP_ // NOLINT
#define TEST_RADIATION_MATTER_COUPLING_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_matter_coupling.hpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

// external headers
#include <fmt/format.h>
#include "matplotlibcpp.h"

// internal headers
#include "athena_arrays.hpp"
#include "radiation_system.hpp"

extern "C" {
	#include "interpolate.h"
}

// function definitions
auto testproblem_radiation_matter_coupling() -> int;

#endif // TEST_RADIATION_MATTER_COUPLING_HPP_
