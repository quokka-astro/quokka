//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hydro_system.cpp
/// \brief Implements methods for solving the (1d) Euler equations.
///

#include "hydro_system.hpp"

template <>
HydroSystem<AthenaArray<double>>::HydroSystem(HydroSystemArgs args)
    : HyperbolicSystem{args.nx, args.lx, args.cflNumber, 3}, gamma_(args.gamma)
{
	assert((gamma_ > 1.0)); // NOLINT

	density_.InitWithShallowSlice(consVar_, 2, density_index, 0);
	x1Momentum_.InitWithShallowSlice(consVar_, 2, x1Momentum_index, 0);
	energy_.InitWithShallowSlice(consVar_, 2, energy_index, 0);

	primDensity_.InitWithShallowSlice(primVar_, 2, primDensity_index, 0);
	x1Velocity_.InitWithShallowSlice(primVar_, 2, x1Velocity_index, 0);
	pressure_.InitWithShallowSlice(primVar_, 2, pressure_index, 0);
}