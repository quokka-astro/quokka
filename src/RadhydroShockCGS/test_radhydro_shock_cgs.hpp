#ifndef TEST_RADHYDRO_SHOCK_CGS_HPP_ // NOLINT
#define TEST_RADHYDRO_SHOCK_CGS_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radhydro_shock.hpp
/// \brief Defines a test problem for a radiative shock.
///

// external headers
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif
#include <fmt/format.h>
#include <fstream>

// internal headers
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "interpolate.hpp"
#include "radiation_system.hpp"

// function definitions
auto problem_main() -> int;

#endif // TEST_RADHYDRO_SHOCK_CGS_HPP_
