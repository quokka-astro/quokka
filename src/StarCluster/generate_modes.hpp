#ifndef GENERATE_MODES_HPP_ // NOLINT
#define GENERATE_MODES_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file generate_modes.hpp
/// \brief Sample a Gaussian random field.
///

#include "AMReX.H"
#include "AMReX_TableData.H"

auto generateRandomModes(int kmin, int kmax, int alpha_PL, int seed) -> amrex::TableData<amrex::Real, 4>;

void projectModes(int kmin, int kmax, amrex::TableData<amrex::Real, 4> &dvx, amrex::TableData<amrex::Real, 4> &dvy, amrex::TableData<amrex::Real, 4> &dvz);

auto computeRms(int kmin, int kmax, amrex::TableData<amrex::Real, 4> &dvx, amrex::TableData<amrex::Real, 4> &dvy, amrex::TableData<amrex::Real, 4> &dvz)
    -> amrex::Real;

#endif // GENERATE_MODES_HPP_
