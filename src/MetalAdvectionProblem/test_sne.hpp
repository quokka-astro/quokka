#ifndef TEST_HYDRO3D_BLAST_HPP_ // NOLINT
#define TEST_HYDRO3D_BLAST_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.hpp
/// \brief Defines a test problem for a 3D explosion.
///

// external headers
#include <fstream>
#include <fmt/format.h>

// internal headers
#include "hydro_system.hpp"
extern "C" {
    #include "interpolate.h"
}

// function definitions
auto testproblem_hydro_sedov() -> int;

#endif // TEST_HYDRO3D_BLAST_HPP_
