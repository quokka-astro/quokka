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
    : HyperbolicSystem{nx, Lx}, advection_vx_(vx)
{
	assert(advection_vx_ != 0.0); // NOLINT
	assert(Lx_ > 0.0);	      // NOLINT
	assert(nx_ > 2);	      // NOLINT
	assert(nghost_ > 1);	      // NOLINT

	const int dim1 = nx + 2 * nghost_;

	density_.NewAthenaArray(dim1);
	density_xleft_.NewAthenaArray(dim1);
	density_xright_.NewAthenaArray(dim1);
	density_flux_.NewAthenaArray(dim1);
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

auto LinearAdvectionSystem::Nx() -> int { return nx_; }

auto LinearAdvectionSystem::NumGhostZones() -> int { return nghost_; }

void LinearAdvectionSystem::SetCFLNumber(double CFL_number)
{
	assert((CFL_number > 0.0) && (CFL_number <= 1.0)); // NOLINT
	CFL_number_ = CFL_number;
}

void LinearAdvectionSystem::FillGhostZones()
{
	// In general, this step will require MPI communication, and interaction
	// with the main AMR code.

	// FIXME: currently we assume periodic boundary conditions.

	// x1 right side boundary
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		density_(i) = density_(i - nx_);
	}

	// x1 left side boundary
	for (int i = 0; i < nghost_; ++i) {
		density_(i) = density_(i + nx_);
	}
}

void LinearAdvectionSystem::ConservedToPrimitive() {}

void LinearAdvectionSystem::ComputeTimestep()
{
	dt_ = CFL_number_ * (dx_ / advection_vx_);
}

void LinearAdvectionSystem::ReconstructStates()
{
	// Use upwind (donor-cell) reconstruction

	for (int i = nghost_; i < nx_ + nghost_; i++) {

		if (advection_vx_ < 0.0) { // upwind switch

			// upwind direction for cell i is cell (i+1)
			density_xleft_(i) = density_(i);
			density_xright_(i) = density_(i + 1);

		} else {
			// upwind direction for cell i is cell (i-1)
			density_xleft_(i) = density_(i - 1);
			density_xright_(i) = density_(i);
		}
	}
}

void LinearAdvectionSystem::ComputeFluxes()
{
	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		density_flux_(i) = -1.0 * advection_vx_ *
				   (density_xright_(i) - density_xleft_(i)) /
				   dx_;
	}
}

void LinearAdvectionSystem::AddFluxes()
{
	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		density_(i) += dt_ * density_flux_(i);
	}
}

auto LinearAdvectionSystem::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		mass += density_(i) * dx_;
	}

	return mass;
}

void LinearAdvectionSystem::AddSourceTerms(AthenaArray<double> &source_terms) {}
