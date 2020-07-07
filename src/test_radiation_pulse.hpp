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
#include "matplotlibcpp.h"
#include <fmt/format.h>
#include <fstream>

// internal headers
#include "athena_arrays.hpp"
#include "radiation_system.hpp"

extern "C" {
#include "interpolate.h"
}

// function definitions
auto testproblem_radiation_pulse() -> int;

#endif // TEST_RADIATION_MARSHAK_HPP_
