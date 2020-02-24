#ifndef TEST_HYDRO_SHOCKTUBE_HPP_ // NOLINT
#define TEST_HYDRO_SHOCKTUBE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.hpp
/// \brief Defines a test problem for a shock tube.
///

// external headers
#include "fmt/include/fmt/format.h"
#include "matplotlibcpp.h"

// internal headers
#include "athena_arrays.hpp"
#include "hydro_system.hpp"

// function definitions
void testproblem_hydro_shocktube();

#endif // TEST_HYDRO_SHOCKTUBE_HPP_