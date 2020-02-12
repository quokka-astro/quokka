//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file linear_advection.cpp
/// \brief Implements methods for solving a scalar linear advection equation.
///

#include "linear_advection.hpp"

LinearAdvectionSystem::LinearAdvectionSystem(const int nx, const double vx,
					     const double Lx)
{
	assert(vx != 0.0); // NOLINT
	assert(Lx > 0.0);  // NOLINT
	assert(nx > 2);	   // NOLINT

	this->nx = nx;
	this->advection_vx = vx;
	this->Lx = Lx;
	this->dx = Lx / (static_cast<double>(nx));
	this->dt = 0.;

	this->density.NewAthenaArray(nx + 2 * (this->nghost));
	this->interface_density.NewAthenaArray(nx + 2 * (this->nghost));
	this->flux_density.NewAthenaArray(nx + 2 * (this->nghost));
}

void LinearAdvectionSystem::AdvanceTimestep()
{
	FillGhostZones();
	ComputeTimestep();
	ReconstructStates();
	ComputeFluxes();
	AddFluxes();
	//	AddSourceTerms();
}

auto LinearAdvectionSystem::Nx() -> int { return this->nx; }

auto LinearAdvectionSystem::NumGhostZones() -> int { return this->nghost; }

void LinearAdvectionSystem::FillGhostZones()
{
	// FIXME: assume periodic boundary conditions

	// x1 right side boundary
	for (int i = nghost + nx; i < nghost + nx + nghost; ++i) {
		this->density(i) = this->density(i - nx);
	}

	// x1 left side boundary
	for (int i = 0; i < nghost; ++i) {
		this->density(i) = this->density(i + nx);
	}
}

void LinearAdvectionSystem::ConservedToPrimitive() {}

void LinearAdvectionSystem::ComputeTimestep()
{
	this->dt = (this->CFL_number) * (this->dx) / (this->advection_vx);
}

void LinearAdvectionSystem::ReconstructStates()
{
	// Use upwind (donor-cell) reconstruction

	if (this->advection_vx < 0.0) {
		// upwind direction for cell i is cell (i+1)

		for (int i = nghost; i < nx + nghost; i++) {
			this->interface_density(i) = this->density(i + 1);
		}

	} else {
		// upwind direction for cell i is cell (i-1)

		for (int i = nghost; i < nx + nghost; ++i) {
			this->interface_density(i) = this->density(i - 1);
		}
	}
}

void LinearAdvectionSystem::ComputeFluxes()
{
	for (int i = nghost; i < nx + nghost; ++i) {
		this->flux_density(i) =
		    this->interface_density(i) * this->advection_vx;
	}
}

void LinearAdvectionSystem::AddFluxes()
{
	for (int i = nghost; i < nx + nghost; ++i) {
		this->density(i) += (this->dt) * this->flux_density(i);
	}
}

void LinearAdvectionSystem::AddSourceTerms(AthenaArray<double> *source_terms) {}
