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
const LinearAdvectionSystem::NxType::argument LinearAdvectionSystem::Nx;
const LinearAdvectionSystem::LxType::argument LinearAdvectionSystem::Lx;
const LinearAdvectionSystem::VxType::argument LinearAdvectionSystem::Vx;
const LinearAdvectionSystem::CFLType::argument LinearAdvectionSystem::CFL;

LinearAdvectionSystem::LinearAdvectionSystem(NxType const &nx, LxType const &lx,
					     VxType const &vx,
					     CFLType const &cflNumber)
    : HyperbolicSystem{nx.get(), lx.get(), cflNumber.get()},
      advectionVx_(vx.get())
{
	assert(advectionVx_ != 0.0);			   // NOLINT
	assert(lx_ > 0.0);				   // NOLINT
	assert(nx_ > 2);				   // NOLINT
	assert(nghost_ > 1);				   // NOLINT
	assert((cflNumber_ > 0.0) && (cflNumber_ <= 1.0)); // NOLINT

	const int dim1 = nx_ + 2 * nghost_;

	density_.NewAthenaArray(dim1);
	densityXLeft_.NewAthenaArray(dim1);
	densityXRight_.NewAthenaArray(dim1);
	densityXInterface_.NewAthenaArray(dim1);
	densityXFlux_.NewAthenaArray(dim1);
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

auto LinearAdvectionSystem::nx() -> int { return nx_; }

auto LinearAdvectionSystem::nghost() -> int { return nghost_; }

void LinearAdvectionSystem::set_cflNumber(double cflNumber)
{
	assert((cflNumber > 0.0) && (cflNumber <= 1.0)); // NOLINT
	cflNumber_ = cflNumber;
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
	dt_ = cflNumber_ * (dx_ / advectionVx_);
}

void LinearAdvectionSystem::ReconstructStatesConstant()
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; ++i) {

		// Use piecewise-constant reconstruction
		// (This converges at first order in spatial resolution.)

		densityXLeft_(i) = density_(i - 1);
		densityXRight_(i) = density_(i);
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

	for (int i = nghost_; i < (nx_ + 1) + nghost_; ++i) {

		// Use piecewise-linear reconstruction
		// (This converges at second order in spatial resolution.)

		const auto lslope = limiter(density_(i) - density_(i - 1),
					    density_(i - 1) - density_(i - 2));

		const auto rslope = limiter(density_(i + 1) - density_(i),
					    density_(i) - density_(i - 1));

		densityXLeft_(i) = density_(i - 1) + 0.25 * lslope; // NOLINT
		densityXRight_(i) = density_(i) + 0.25 * rslope;    // NOLINT
	}
}

void LinearAdvectionSystem::ReconstructStatesPPM()
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; ++i) {
		// TODO(ben): implement PPM reconstruction following Collela &
		// Woodward (1984)
	}
}

// TODO(ben): combine this function with LinearAdvectionSystem::ComputeFluxes()
//
void LinearAdvectionSystem::DoRiemannSolve()
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; ++i) {

		// For advection, simply choose upwind side of the interface.

		if (advectionVx_ < 0.0) { // upwind switch
			// upwind direction is the right-side of the interface
			densityXInterface_(i) = densityXRight_(i);

		} else {
			// upwind direction is the left-side of the interface
			densityXInterface_(i) = densityXLeft_(i);
		}
	}
}

void LinearAdvectionSystem::ComputeFluxes()
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the left-side
	// interface.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = nghost_; i < (nx_ + 1) + nghost_; ++i) {
		densityXFlux_(i) = advectionVx_ * densityXInterface_(i);
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
			       (densityXFlux_(i + 1) - densityXFlux_(i));
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
