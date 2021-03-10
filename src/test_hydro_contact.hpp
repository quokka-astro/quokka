#ifndef TEST_HYDRO_LEBLANC_HPP_ // NOLINT
#define TEST_HYDRO_LEBLANC_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.hpp
/// \brief Defines a test problem for a shock tube.
///

// external headers
#include "matplotlibcpp.h"
#include <fmt/format.h>
#include <fstream>

// internal headers

#include "hydro_system.hpp"
extern "C" {
    #include "interpolate.h"
}

// function definitions
auto testproblem_hydro_contact() -> int;

#endif // TEST_HYDRO_LEBLANC_HPP_
