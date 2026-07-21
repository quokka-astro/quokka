//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Photochemistry.cpp
/// \brief Implements methods for photoionization chemistry.
///

#include "radiation/photochemistry.hpp"
#include "chemistry/rosenbrock/Rosenbrock.hpp"

namespace quokka::photochemistry
{

AMREX_GPU_DEVICE auto photochem_burner(quokka::chemistry::IntegratorState<quokka::chemistry::PhotoionizationNetwork::variable_count> &state, const Real dt,
				       quokka::chemistry::PhotoionizationNetwork const &network, quokka::chemistry::IntegratorOptions const &options)
    -> quokka::chemistry::IntegratorDiagnostics
{
	return quokka::chemistry::rosenbrock::integrate_with_retry(network, state, dt, options);
}
} // namespace quokka::photochemistry
