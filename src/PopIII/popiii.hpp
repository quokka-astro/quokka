#ifndef POPIII_HPP_ // NOLINT
#define POPIII_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file star_cluster.hpp
/// \brief Defines a test problem for Pop III star formation.
/// Author: Piyush Sharda (Leiden University, 2023)
///

// external headers
#include <fstream>

// internal headers
#include "hydro_system.hpp"
#include "interpolate.hpp"

// function definitions
auto problem_main() -> int;

#endif // POPIII_HPP_
