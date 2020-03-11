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
const LinearAdvectionSystem::NvarsType::argument LinearAdvectionSystem::Nvars;

LinearAdvectionSystem::LinearAdvectionSystem(NxType const &nx, LxType const &lx,
					     VxType const &vx,
					     CFLType const &cflNumber,
					     NvarsType const &nvars)
    : HyperbolicSystem{nx.get(), lx.get(), cflNumber.get(), nvars.get()},
      advectionVx_(vx.get())
{
	assert(advectionVx_ != 0.0); // NOLINT

	enum varIndex { density_index = 0 };
	density_.InitWithShallowSlice(consVar_, 2, density_index, 0);
}

auto LinearAdvectionSystem::density(const int i) -> double
{
	return density_(i);
}

auto LinearAdvectionSystem::set_density(const int i) -> double &
{
	return density_(i);
}

auto LinearAdvectionSystem::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		mass += density_(i) * dx_;
	}

	return mass;
}

void LinearAdvectionSystem::ComputeTimestep(const double dt_max)
{
	dt_ = std::min(cflNumber_ * (dx_ / advectionVx_), dt_max);
}

void LinearAdvectionSystem::ConservedToPrimitive(
    AthenaArray<double> &cons, const std::pair<int, int> range)
{
	for (int n = range.first; n < range.second; ++n) {
		for (int i = 0; i < dim1_; ++i) {
			primVar_(n, i) = cons(n, i);
		}
	}
}

// TODO(ben): add flux limiter for positivity preservation.
void LinearAdvectionSystem::ComputeFluxes(const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on
	// the left edge of each zone, i.e.
	// xinterface_(i) is the solution to the Riemann
	// problem at the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces
	// for nx zones.

	for (int n = 0; n < nvars_; ++n) {
		for (int i = range.first; i < (range.second + 1); ++i) {

			// For advection, simply choose
			// upwind side of the interface.

			if (advectionVx_ < 0.0) { // upwind switch
				// upwind direction is
				// the right-side of the
				// interface
				x1Flux_(n, i) =
				    advectionVx_ * x1RightState_(n, i);

			} else {
				// upwind direction is
				// the left-side of the
				// interface
				x1Flux_(n, i) =
				    advectionVx_ * x1LeftState_(n, i);
			}
		}
	}
}

void LinearAdvectionSystem::AddSourceTerms(AthenaArray<double> &U,
					   std::pair<int, int> range)
{
	// TODO(ben): to be implemented
}

void LinearAdvectionSystem::AddFluxesSDC(AthenaArray<double> &U_new,
					 AthenaArray<double> &U_0)
{
	// TODO(ben): to be implemented
}