//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.cpp
/// \brief Implements methods for solving the (1d) radiation moment equations.
///

#include "radiation_system.hpp"

template <>
RadSystem<AthenaArray<double>>::RadSystem(RadSystemArgs args)
    : HyperbolicSystem{args.nx, args.lx, args.cflNumber, 4}
{
	radEnergy_.InitWithShallowSlice(consVar_, 2, radEnergy_index, 0);
	x1RadFlux_.InitWithShallowSlice(consVar_, 2, x1RadFlux_index, 0);

	gasEnergy_.InitWithShallowSlice(consVar_, 2, gasEnergy_index, 0);
	staticGasDensity_.InitWithShallowSlice(consVar_, 2, gasDensity_index,
					       0);

	radEnergySource_.NewAthenaArray(args.nx + 2 * nghost_);
}
