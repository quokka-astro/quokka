//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Chemistry.cpp
/// \brief Implements methods for primordial chemistry.
///

#include "chemistry/Chemistry.hpp"
#include "chemistry/rosenbrock/Rosenbrock.hpp"

namespace quokka::chemistry
{

AMREX_GPU_DEVICE auto chemburner(IntegratorState<PrimordialChemNetwork::variable_count> &state, const Real dt, PrimordialChemNetwork const &network,
				 IntegratorOptions const &options) -> bool
{
	const auto diagnostics = rosenbrock::integrate_with_retry(network, state, dt, options);
	return diagnostics.succeeded();
}

} // namespace quokka::chemistry
