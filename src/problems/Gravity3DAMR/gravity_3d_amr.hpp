#ifndef TEST_GRAVITY_3D_AMR_HPP_ // NOLINT
#define TEST_GRAVITY_3D_AMR_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file gravity_3d_amr.hpp
/// \brief Defines a test problem for a binary orbit with AMR
///

// external headers
#include <fstream>

// internal headers
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"

// function definitions
auto problem_main() -> int;

#endif // TEST_GRAVITY_3D_AMR_HPP_
