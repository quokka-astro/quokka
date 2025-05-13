#ifndef PARTICLE_SINK_ACCRETION_HPP_ // NOLINT
#define PARTICLE_SINK_ACCRETION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file particle_sink_accretion.hpp
/// \brief Defines a test problem for Bondi-Hoyle accretion.
///

// external headers
#include <fstream>

// internal headers
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"

// function definitions
auto problem_main() -> int;

#endif // PARTICLE_SINK_ACCRETION_HPP_
