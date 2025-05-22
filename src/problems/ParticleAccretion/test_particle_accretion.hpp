#ifndef TEST_PARTICLE_ACCRETION_HPP_ // NOLINT
#define TEST_PARTICLE_ACCRETION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_particle_accretion.hpp
/// \brief Defines a test problem for Bondi-Hoyle accretion.
///

// external headers
#include <fstream>

// internal headers
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"

// function definitions
auto problem_main() -> int;

#endif // TEST_PARTICLE_ACCRETION_HPP_
