#ifndef TEST_RADHYDRO_SHOCK_HPP_ // NOLINT
#define TEST_RADHYDRO_SHOCK_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radhydro_shock.hpp
/// \brief Defines a test problem for a radiative shock.
///

// external headers
#include "matplotlibcpp.h"
#include <fmt/format.h>
#include <fstream>

// internal headers

#include "hydro_system.hpp"
#include "radiation_system.hpp"

extern "C" {
#include "interpolate.h"
}

// function definitions
auto testproblem_radhydro_shock() -> int;

#endif // TEST_RADHYDRO_SHOCK_HPP_
