//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file linear_advection.cpp
/// \brief Implements methods for solving a scalar linear advection equation.
///

#include "linear_advection.hpp"

// We must *define* static member variables here, outside of the class
// *declaration*, even though the definitions are trivial.
const LinearAdvectionSystem::Nx::argument LinearAdvectionSystem::nx;
const LinearAdvectionSystem::Lx::argument LinearAdvectionSystem::lx;
const LinearAdvectionSystem::Vx::argument LinearAdvectionSystem::vx;
const LinearAdvectionSystem::CFLType::argument LinearAdvectionSystem::CFL;

LinearAdvectionSystem::LinearAdvectionSystem(Nx const &nx, Lx const &lx,
					     Vx const &vx, CFLType const &CFL)
    : HyperbolicSystem{nx.get(), lx.get(), CFL.get()}, advection_vx_(vx.get())
{
	assert(advection_vx_ != 0.0);			     // NOLINT
	assert(Lx_ > 0.0);				     // NOLINT
	assert(nx_ > 2);				     // NOLINT
	assert(nghost_ > 1);				     // NOLINT
	assert((CFL_number_ > 0.0) && (CFL_number_ <= 1.0)); // NOLINT

	const int dim1 = nx_ + 2 * nghost_;

	density_.NewAthenaArray(dim1);
	density_xleft_.NewAthenaArray(dim1);
	density_xright_.NewAthenaArray(dim1);
	density_xinterface_.NewAthenaArray(dim1);
	density_flux_.NewAthenaArray(dim1);
}

void LinearAdvectionSystem::AdvanceTimestep()
{
	FillGhostZones();
	ComputeTimestep();
	ReconstructStatesPLM(HyperbolicSystem::minmod);
	DoRiemannSolve();
	ComputeFluxes();
	AddFluxes();
	//	AddSourceTerms();
}

auto LinearAdvectionSystem::GetNx() -> int { return nx_; }

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

void LinearAdvectionSystem::ReconstructStatesConstant()
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; i++) {

		// Use piecewise-constant reconstruction
		// (This converges at first order in spatial resolution.)

		density_xleft_(i) = density_(i - 1);
		density_xright_(i) = density_(i);
	}
}

template <typename F>
void LinearAdvectionSystem::ReconstructStatesPLM(F &&limiter)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; i++) {

		// Use piecewise-linear reconstruction
		// (This converges at second order in spatial resolution.)

		const auto lslope = limiter(density_(i) - density_(i - 1),
					    density_(i - 1) - density_(i - 2));

		const auto rslope = limiter(density_(i + 1) - density_(i),
					    density_(i) - density_(i - 1));

		density_xleft_(i) = density_(i - 1) + 0.25 * lslope; // NOLINT
		density_xright_(i) = density_(i) + 0.25 * rslope;    // NOLINT
	}
}

void LinearAdvectionSystem::DoRiemannSolve()
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; i++) {

		// For advection, simply choose upwind side of the interface.

		if (advection_vx_ < 0.0) { // upwind switch
			// upwind direction is the right-side of the interface
			density_xinterface_(i) = density_xright_(i);

		} else {
			// upwind direction is the left-side of the interface
			density_xinterface_(i) = density_xleft_(i);
		}
	}
}

void LinearAdvectionSystem::ComputeFluxes()
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the left-side
	// interface.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; i++) {
		density_flux_(i) = advection_vx_ * density_xinterface_(i);
	}
}

void LinearAdvectionSystem::AddFluxes()
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		density_(i) += -1.0 * (dt_ / dx_) *
			       (density_flux_(i + 1) - density_flux_(i));
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
