//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hydro_system.cpp
/// \brief Implements methods for solving the (1d) Euler equations.
///

#include "hydro_system.hpp"

// We must *define* static member variables here, outside of the class
// *declaration*, even though the definitions are trivial.
const HydroSystem::NxType::argument HydroSystem::Nx;
const HydroSystem::LxType::argument HydroSystem::Lx;
const HydroSystem::CFLType::argument HydroSystem::CFL;
const HydroSystem::NvarsType::argument HydroSystem::Nvars;

HydroSystem::HydroSystem(NxType const &nx, LxType const &lx,
			 CFLType const &cflNumber, NvarsType const &nvars)
    : HyperbolicSystem{nx.get(), lx.get(), cflNumber.get(), nvars.get()}
{
	assert(lx_ > 0.0);				   // NOLINT
	assert(nx_ > 2);				   // NOLINT
	assert(nghost_ > 1);				   // NOLINT
	assert((cflNumber_ > 0.0) && (cflNumber_ <= 1.0)); // NOLINT

	enum varIndex {
		density_index = 0,
		x1Momentum_index = 1,
		energy_index = 2
	};
	density_.InitWithShallowSlice(consVar_, 2, density_index, 0);
	x1Momentum_.InitWithShallowSlice(consVar_, 2, x1Momentum_index, 0);
	energy_.InitWithShallowSlice(consVar_, 2, energy_index, 0);
}

auto HydroSystem::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		mass += density_(i) * dx_;
	}

	return mass;
}

void HydroSystem::ConservedToPrimitive(AthenaArray<double> &cons) {}

void HydroSystem::ComputeTimestep()
{
	//	dt_ = cflNumber_ * (dx_ / advectionVx_);
}

// TODO(ben): add flux limiter for positivity preservation.
void HydroSystem::ComputeFluxes(const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = range.first; i < (range.second + 1); ++i) {
		// TODO(ben): write Riemann solver.
	}
}

#if 0
void HydroSystem::FlattenShocks(AthenaArray<double> &q,
				const std::pair<int, int> range)
{
	// N.B.: shock flattening doesn't appear to really do much. Perhaps a
	// bug?

	for (int i = range.first; i < range.second; ++i) {

		const double a_minus = densityXRight_(i);   // a_L,i in C&W
		const double a_plus = densityXLeft_(i + 1); // a_R,i in C&W
		const double a = q(i);			    // a_i in C&W

		auto flatten_f = [q](int j) {
			const double a1 = 0.75;
			const double a2 = 10.;
			const double eps = 0.33;
			double f = 0.;

			const double shock_ratio =
			    (q(j + 1) - q(j - 1)) / (q(j + 2) - q(j - 2));
			const double qa = (q(j + 1) - q(j - 1)) /
					  std::min(q(j + 1), q(j - 1));
			if ((qa > eps) && ((q(j - 1) - q(j + 1)) > 0.)) {
				f = 1.0 - std::max(0., (shock_ratio - a1) * a2);
			}

			return f;
		};

		const double f_i = flatten_f(i);
		double f_s;
		if (q(i + 1) - q(i - 1) < 0.) {
			f_s = flatten_f(i + 1);
		} else {
			f_s = flatten_f(i - 1);
		}

		const double f = std::max(f_i, f_s);

		const double new_a_minus = a * f + a_minus * (1.0 - f);
		const double new_a_plus = a * f + a_plus * (1.0 - f);

		densityXRight_(i) = new_a_minus;
		densityXLeft_(i + 1) = new_a_plus;
	}
}
#endif

void HydroSystem::AddSourceTerms(AthenaArray<double> &source_terms) {}
