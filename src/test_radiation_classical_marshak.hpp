#ifndef TEST_RADIATION_MARSHAK_HPP_ // NOLINT
#define TEST_RADIATION_MARSHAK_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.hpp
/// \brief Defines a test problem for radiation in the static diffusion regime.
///

// external headers
#include <fmt/format.h>
#include "matplotlibcpp.h"

// internal headers
#include "athena_arrays.hpp"
#include "radiation_system.hpp"

// function definitions
void testproblem_radiation_classical_marshak();

#endif // TEST_RADIATION_MARSHAK_HPP_
