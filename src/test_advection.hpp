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
#include "fmt/include/fmt/format.h"
#include "matplotlibcpp.h"

// internal headers
#include "athena_arrays.hpp"
#include "linear_advection.hpp"

// function definitions
void write_density(LinearAdvectionSystem &advection_system);
void testproblem_advection();

#endif // TEST_ADVECTION_HPP_