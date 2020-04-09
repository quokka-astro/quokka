//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file linear_advection.cpp
/// \brief Implements methods for solving a scalar linear advection equation.
///

#include "linear_advection.hpp"

template <>
LinearAdvectionSystem<AthenaArray<double>>::LinearAdvectionSystem(
    const LinearAdvectionArgs args)
    : HyperbolicSystem{args.nx, args.lx, args.cflNumber, args.nvars},
      advectionVx_(args.vx)
{
	assert(advectionVx_ != 0.0); // NOLINT

	enum varIndex { density_index = 0 };
	density_.InitWithShallowSlice(consVar_, 2, density_index, 0);
}
