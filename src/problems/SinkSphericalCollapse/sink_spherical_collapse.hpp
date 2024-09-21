#ifndef TEST_SINK_SPHERICAL_COLLAPSE_HPP_ // NOLINT
#define TEST_SINK_SPHERICAL_COLLAPSE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file sink_spherical_collapse.hpp
/// \brief Defines a test problem for a pressureless spherical collapse with sink particle formation
///

// external headers
#include <fstream>

// internal headers
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"

// function definitions
auto problem_main() -> int;

#endif // TEST_SINK_SPHERICAL_COLLAPSE_HPP_
