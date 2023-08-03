#ifndef TEST_SPHERICAL_COLLAPSE_HPP_ // NOLINT
#define TEST_SPHERICAL_COLLAPSE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file spherical_collapse.hpp
/// \brief Defines a test problem for a pressureless spherical collapse
///

// external headers
#include <fstream>

// internal headers
#include "hydro_system.hpp"
#include "interpolate.hpp"

// function definitions
auto testproblem_hydro_sedov() -> int;

#endif // TEST_SPHERICAL_COLLAPSE_HPP_
