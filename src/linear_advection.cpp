//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file linear_advection.cpp
/// \brief Implements methods for solving a scalar linear advection equation.
///

#include "linear_advection.hpp"

LinearAdvectionSystem::LinearAdvectionSystem(const int nx)
{
	this->density.NewAthenaArray(nx);
}

void LinearAdvectionSystem::AdvanceTimestep() {}

void LinearAdvectionSystem::AddSourceTerms(AthenaArray<double> *source_terms) {}

void LinearAdvectionSystem::FillGhostZones() {}

void LinearAdvectionSystem::ConservedToPrimitive() {}

void LinearAdvectionSystem::ComputeTimestep() {}

void LinearAdvectionSystem::ReconstructStates() {}

void LinearAdvectionSystem::ComputeFluxes() {}

void LinearAdvectionSystem::AddFluxes() {}
