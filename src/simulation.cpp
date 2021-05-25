//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file simulation.cpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation.

#include "simulation.hpp"

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto clamp(double v, double lo, double hi) -> double
{
	return (v < lo) ? lo : (hi < v) ? hi : v;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CheckNaN(amrex::FArrayBox const &arr, amrex::Box const &indexRange, const int ncomp)
{
	// need to rewrite for GPU
}