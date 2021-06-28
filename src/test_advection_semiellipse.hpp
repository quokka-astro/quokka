#ifndef TEST_ADVECTION_HPP_ // NOLINT
#define TEST_ADVECTION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

// external headers
#include "matplotlibcpp.h"
#include <fmt/format.h>

// internal headers

#include "linear_advection.hpp"
#include "AdvectionSimulation.hpp"

// function definitions
auto testproblem_advection() -> int;

#endif // TEST_ADVECTION_HPP_
