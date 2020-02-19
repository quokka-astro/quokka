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

	density_.NewAthenaArray(dim1_);
	densityPrediction_.NewAthenaArray(dim1_);
	densityXLeft_.NewAthenaArray(dim1_);
	densityXRight_.NewAthenaArray(dim1_);
	densityXFlux_.NewAthenaArray(dim1_);
}

void LinearAdvectionSystem::AdvanceTimestep()
{
	// Initialize data
	FillGhostZones();
	ComputeTimestep();

	// Predictor step
	const int hlo = (-3) + nghost_;
	const int hhi = (nx_ + 3) + nghost_;

	ReconstructStatesConstant(hlo, hhi);
	// ReconstructStatesPLM(HyperbolicSystem::minmod, hlo, hhi);
	ComputeFluxes(hlo, hhi);
	PredictHalfStep(hlo, hhi);

	// Clear temporary arrays
	densityXLeft_.ZeroClear();
	densityXRight_.ZeroClear();
	densityXFlux_.ZeroClear();

	// Corrector step
	ReconstructStatesPPM(densityPrediction_, (-1) + nghost_,
			     (nx_ + 1) + nghost_);
	// TODO(ben): Add shock flattening following C&W
	// FlattenShocks();
	ComputeFluxes(nghost_, nx_ + nghost_);
	AddFluxes();

	// Adjust our clock
	time_ += dt_;
}

auto LinearAdvectionSystem::time() -> double { return time_; }

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

void LinearAdvectionSystem::ReconstructStatesConstant(const int lo,
						      const int hi)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = lo; i < (hi + 1); ++i) {

		// Use piecewise-constant reconstruction
		// (This converges at first order in spatial resolution.)

		densityXLeft_(i) = density_(i - 1);
		densityXRight_(i) = density_(i);
	}
}

template <typename F>
void LinearAdvectionSystem::ReconstructStatesPLM(F &&limiter, const int lo,
						 const int hi)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = lo; i < (hi + 1); ++i) {

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

void LinearAdvectionSystem::ReconstructStatesPPM(AthenaArray<double> &q,
						 const int lo, const int hi)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = lo; i < (hi + 1); ++i) {
		// PPM reconstruction following Colella & Woodward (1984), with
		// some modifications following Mignone (2014), as implemented
		// in Athena++.

		// (1.) Estimate the interface a_{i - 1/2}.
		//      Equivalent to step 1 in Athena++ [ppm_simple.cpp].

		// C&W Eq. (1.9) [parabola midpoint for the case of
		// equally-spaced zones]: a_{j+1/2} = (7/12)(a_j + a_{j+1}) -
		// (1/12)(a_{j+2} + a_{j-1}). Terms are grouped to preserve
		// exact symmetry in floating-point arithmetic, following
		// Athena++.

		const double coef_1 = (7. / 12.);
		const double coef_2 = (-1. / 12.);
		const double a_jhalf = (coef_1 * q(i) + coef_2 * q(i + 1)) +
				       (coef_1 * q(i - 1) + coef_2 * q(i - 2));

		// (2.) Constrain interface value to lie between adjacent
		//      cell-averaged values (equivalent to step 2b in
		//      Athena++ [ppm_simple.cpp]).

		std::pair<double, double> bounds = std::minmax(q(i), q(i - 1));
		const double interface =
		    std::clamp(a_jhalf, bounds.first, bounds.second);

		// a_R,(i-1) in C&W
		densityXLeft_(i) = interface;

		// a_L,i in C&W
		densityXRight_(i) = interface;
	}

	for (int i = lo; i < hi; ++i) {

		const double a_minus = densityXRight_(i);   // a_L,i in C&W
		const double a_plus = densityXLeft_(i + 1); // a_R,i in C&W
		const double a = q(i);			    // a_i in C&W

		const double dq_minus = (a - a_minus);
		const double dq_plus = (a_plus - a);

		double new_a_minus = a_minus;
		double new_a_plus = a_plus;

		// (3.) Monotonicity correction, using Eq. (1.10) in PPM paper.
		//      Equivalent to step 4b in Athena++ [ppm_simple.cpp].

		const double qa = dq_plus * dq_minus; // interface extrema

		if ((qa <= 0.0)) { // local extremum
			new_a_minus = a;
			new_a_plus = a;

		} else { // no local extrema

			// parabola overshoots near a_plus -> reset a_minus
			if (std::abs(dq_minus) >= 2.0 * std::abs(dq_plus)) {
				new_a_minus = a - 2.0 * dq_plus;
			}

			// parabola overshoots near a_minus -> reset a_plus
			if (std::abs(dq_plus) >= 2.0 * std::abs(dq_minus)) {
				new_a_plus = a + 2.0 * dq_minus;
			}
		}

		densityXRight_(i) = new_a_minus;
		densityXLeft_(i + 1) = new_a_plus;
	}
}

// TODO(ben): add flux limiter for positivity preservation.
void LinearAdvectionSystem::ComputeFluxes(const int lo, const int hi)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = lo; i < (hi + 1); ++i) {

		// For advection, simply choose upwind side of the interface.

		if (advectionVx_ < 0.0) { // upwind switch
			// upwind direction is the right-side of the interface
			densityXFlux_(i) = advectionVx_ * densityXRight_(i);

		} else {
			// upwind direction is the left-side of the interface
			densityXFlux_(i) = advectionVx_ * densityXLeft_(i);
		}
	}
}

void LinearAdvectionSystem::PredictHalfStep(const int lo, const int hi)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	for (int i = lo; i < hi; ++i) {
		densityPrediction_(i) =
		    density_(i) - (0.5 * dt_ / dx_) *
				      (densityXFlux_(i + 1) - densityXFlux_(i));
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
