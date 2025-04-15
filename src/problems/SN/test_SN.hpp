#ifndef TEST_SN_HPP_ // NOLINT
#define TEST_SN_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_SN.hpp
/// \brief Defines a test problem for supernova feedback.
///

// external headers
#include <fstream>

// internal headers
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"

// function definitions
auto testproblem_SN() -> int;

#endif // TEST_SN_HPP_
