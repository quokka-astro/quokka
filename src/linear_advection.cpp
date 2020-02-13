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
	this->density_xleft.NewAthenaArray(nx + 2 * (this->nghost));
	this->density_xright.NewAthenaArray(nx + 2 * (this->nghost));
	this->density_flux.NewAthenaArray(nx + 2 * (this->nghost));
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
	// In general, this step will require MPI communication, and interaction
	// with the main AMR code.

	// FIXME: currently we assume periodic boundary conditions.

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

	for (int i = nghost; i < nx + nghost; i++) {

		if (this->advection_vx < 0.0) { // upwind switch

			// upwind direction for cell i is cell (i+1)
			this->density_xleft(i) = this->density(i);
			this->density_xright(i) = this->density(i + 1);

		} else {
			// upwind direction for cell i is cell (i-1)
			this->density_xleft(i) = this->density(i - 1);
			this->density_xright(i) = this->density(i);
		}
	}
}

void LinearAdvectionSystem::ComputeFluxes()
{

	for (int i = nghost; i < nx + nghost; ++i) {

		this->density_flux(i) =
		    -1.0 * this->advection_vx *
		    (this->density_xright(i) - this->density_xleft(i)) / dx;
	}
}

void LinearAdvectionSystem::AddFluxes()
{
	for (int i = nghost; i < nx + nghost; ++i) {
		this->density(i) += (this->dt) * this->density_flux(i);
	}
}

auto LinearAdvectionSystem::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost; i < nx + nghost; ++i) {
		mass += this->density(i) * this->dx;
	}

	return mass;
}

void LinearAdvectionSystem::AddSourceTerms(AthenaArray<double> &source_terms) {}
