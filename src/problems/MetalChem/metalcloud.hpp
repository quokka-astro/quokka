#ifndef METALCLOUD_HPP_ // NOLINT
#define METALCLOUD_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file star_cluster.hpp
/// \brief Defines a test problem for Pop I/II star formation.
/// Author: Piyush Sharda (Leiden University, 2024)
///

// external headers
#include <fstream>

// internal headers
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"

// function definitions
auto problem_main() -> int;

#endif // METALCLOUD_HPP_
