#ifndef TEST_ODE_HPP_ // NOLINT
#define TEST_ODE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_ode.hpp
/// \brief Defines a test problem for ODE integration.
///

// external headers
#include <memory>
#include <string>
#include <vector>

#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif
#include <fmt/format.h>

// internal headers
#include "CloudyCooling.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "rk4.hpp"
#include "valarray.hpp"

// types

struct ODETest {};

constexpr double seconds_in_year = 3.154e7;

// function definitions

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
user_rhs(Real t, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int;

#endif // TEST_ODE_HPP_
