#ifndef TEST_POISSON_HPP_ // NOLINT
#define TEST_POISSON_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_poisson.hpp
/// \brief Defines a test problem for a 3D explosion.
///

// external headers
#include <fstream>

// internal headers
#include "hydro_system.hpp"

void computeReferenceSolution(
    amrex::MultiFab &ref,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi);

#endif // TEST_POISSON_HPP_
