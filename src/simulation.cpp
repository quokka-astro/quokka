//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file simulation.cpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation.

#include "simulation.hpp"
#include <limits>

using amrex::Real;

std::tuple<Real, Real, Real> ComputeLBStatistics (const amrex::DistributionMapping& dm,
                                        	 const amrex::Vector<Real>& cost)
{
    const int nprocs = amrex::ParallelDescriptor::NProcs();
    amrex::Vector<amrex::Vector<Real>> rankToCosts(nprocs);

    // Count the number of grids belonging to each rank
    amrex::Vector<int> cnt(nprocs);
    for (int i=0; i<dm.size(); ++i)
    {
        ++cnt[dm[i]];
    }

    for (int i=0; i<rankToCosts.size(); ++i)
    {
        rankToCosts[i].reserve(cnt[i]);
    }

    for (int i=0; i<cost.size(); ++i)
    {
        rankToCosts[dm[i]].push_back(cost[i]);
    }

    Real maxCost = -1.0;
	Real minCost = std::numeric_limits<Real>::infinity();

    // This will store mapping from (proc) --> (sum of cost) for each proc
    amrex::Vector<Real> rankToCost(nprocs);
    for (int i=0; i<nprocs; ++i)
    {
        const Real rwSum = std::accumulate(rankToCosts[i].begin(),
                                           rankToCosts[i].end(), 0.0);
        rankToCost[i] = rwSum;
        maxCost = std::max(maxCost, rwSum);
		minCost = std::min(minCost, rwSum);
    }

    // Write `efficiency` (number between 0 and 1), the mean cost per processor
    // (normalized to the max cost)
    Real avgCost = (std::accumulate(rankToCost.begin(),
                                    rankToCost.end(), 0.0) / nprocs);

	return std::make_tuple(minCost, avgCost, maxCost);
}