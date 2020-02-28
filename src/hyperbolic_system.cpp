//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.cpp
/// \brief Implements classes and functions for use with hyperbolic systems of
/// conservation laws.

#include "hyperbolic_system.hpp"

auto HyperbolicSystem::time() -> double { return time_; }

auto HyperbolicSystem::nx() -> int { return nx_; }

auto HyperbolicSystem::nghost() -> int { return nghost_; }

auto HyperbolicSystem::nvars() -> int { return nvars_; }

void HyperbolicSystem::set_cflNumber(double cflNumber)
{
	assert((cflNumber > 0.0) && (cflNumber <= 1.0)); // NOLINT
	cflNumber_ = cflNumber;
}

void HyperbolicSystem::FillGhostZones(AthenaArray<double> &cons)
{
	// In general, this step will require MPI communication, and interaction
	// with the main AMR code.

#if 0
	// x1 right side boundary
	for (int n = 0; n < nvars_; ++n) {
		for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
			cons(n, i) = cons(n, i - nx_);
		}
	}

	// x1 left side boundary
	for (int n = 0; n < nvars_; ++n) {
		for (int i = 0; i < nghost_; ++i) {
			cons(n, i) = cons(n, i + nx_);
		}
	}
#endif

	// x1 right side boundary
	for (int n = 0; n < nvars_; ++n) {
		for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
			cons(n, i) = cons(n, nghost_ + nx_ - 1);
		}
	}

	// x1 left side boundary
	for (int n = 0; n < nvars_; ++n) {
		for (int i = 0; i < nghost_; ++i) {
			cons(n, i) = cons(n, nghost_ + 0);
		}
	}
}

void HyperbolicSystem::ReconstructStatesConstant(
    const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.
	for (int n = 0; n < nvars_; ++n) {
		for (int i = range.first; i < (range.second + 1); ++i) {

			// Use piecewise-constant reconstruction
			// (This converges at first order in spatial
			// resolution.)

			x1LeftState_(n, i) = primVar_(n, i - 1);
			x1RightState_(n, i) = primVar_(n, i);
		}
	}
}

template <typename F>
void HyperbolicSystem::ReconstructStatesPLM(F &&limiter,
					    const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int n = 0; n < nvars_; ++n) {
		for (int i = range.first; i < (range.second + 1); ++i) {

			// Use piecewise-linear reconstruction
			// (This converges at second order in spatial
			// resolution.)

			const auto lslope =
			    limiter(primVar_(n, i) - primVar_(n, i - 1),
				    primVar_(n, i - 1) - primVar_(n, i - 2));

			const auto rslope =
			    limiter(primVar_(n, i + 1) - primVar_(n, i),
				    primVar_(n, i) - primVar_(n, i - 1));

			x1LeftState_(n, i) =
			    primVar_(n, i - 1) + 0.25 * lslope; // NOLINT
			x1RightState_(n, i) =
			    primVar_(n, i) - 0.25 * rslope; // NOLINT
		}
	}
}

void HyperbolicSystem::ReconstructStatesPPM(AthenaArray<double> &q,
					    const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at the left
	// edge of zone i, and xright_(i) is the "right"-side of the interface
	// at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int n = 0; n < nvars_; ++n) {
		for (int i = range.first; i < (range.second + 1); ++i) {
			// PPM reconstruction following Colella & Woodward
			// (1984), with some modifications following Mignone
			// (2014), as implemented in Athena++.

			// (1.) Estimate the interface a_{i - 1/2}.
			//      Equivalent to step 1 in Athena++
			//      [ppm_simple.cpp].

			// C&W Eq. (1.9) [parabola midpoint for the case of
			// equally-spaced zones]: a_{j+1/2} = (7/12)(a_j +
			// a_{j+1}) - (1/12)(a_{j+2} + a_{j-1}). Terms are
			// grouped to preserve exact symmetry in floating-point
			// arithmetic, following Athena++.

			// const double coef_1 = (7. / 12.);
			// const double coef_2 = (-1. / 12.);
			// const double a_jhalf =
			//    (coef_1 * q(n, i) + coef_2 * q(n, i + 1)) +
			//    (coef_1 * q(n, i - 1) + coef_2 * q(n, i - 2));

			// Compute limited slopes
			const double dq0 =
			    MC(q(n, i + 1) - q(n, i), q(n, i) - q(n, i - 1));

			const double dq1 = MC(q(n, i) - q(n, i - 1),
					      q(n, i - 1) - q(n, i - 2));

			// Compute interface (i - 1/2)
			const double interface = q(n, i - 1) +
						 0.5 * (q(n, i) - q(n, i - 1)) -
						 (1. / 6.) * (dq0 - dq1);

			// (2.) Constrain interface value to lie between
			// adjacent cell-averaged values (equivalent to
			// step 2b in Athena++ [ppm_simple.cpp]).

			// std::pair<double, double> bounds =
			//    std::minmax(q(n, i), q(n, i - 1));
			// const double interface =
			//    std::clamp(a_jhalf, bounds.first,
			//    bounds.second);

			// a_R,(i-1) in C&W
			x1LeftState_(n, i) = interface;

			// a_L,i in C&W
			x1RightState_(n, i) = interface;
		}
	}

	for (int n = 0; n < nvars_; ++n) {
		for (int i = range.first; i < range.second; ++i) {

			const double a_minus =
			    x1RightState_(n, i); // a_L,i in C&W
			const double a_plus =
			    x1LeftState_(n, i + 1); // a_R,i in C&W
			const double a = q(n, i);   // a_i in C&W

			const double dq_minus = (a - a_minus);
			const double dq_plus = (a_plus - a);

			double new_a_minus = a_minus;
			double new_a_plus = a_plus;

			// (3.) Monotonicity correction, using Eq. (1.10) in PPM
			// paper. Equivalent to step 4b in Athena++
			// [ppm_simple.cpp].

			const double qa =
			    dq_plus * dq_minus; // interface extrema

			if ((qa <= 0.0)) { // local extremum

				const double dq0 = MC(q(n, i + 1) - q(n, i),
						      q(n, i) - q(n, i - 1));

				new_a_minus = a - 0.5 * dq0;
				new_a_plus = a + 0.5 * dq0;

				// new_a_minus = a;
				// new_a_plus = a;

			} else { // no local extrema

				// parabola overshoots near
				// a_plus -> reset a_minus
				if (std::abs(dq_minus) >=
				    2.0 * std::abs(dq_plus)) {
					new_a_minus = a - 2.0 * dq_plus;
				}

				// parabola overshoots near
				// a_minus -> reset a_plus
				if (std::abs(dq_plus) >=
				    2.0 * std::abs(dq_minus)) {
					new_a_plus = a + 2.0 * dq_minus;
				}
			}

			x1RightState_(n, i) = new_a_minus;
			x1LeftState_(n, i + 1) = new_a_plus;
		}
	}
}

void HyperbolicSystem::PredictStep(const std::pair<int, int> range)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	for (int n = 0; n < nvars_; ++n) {
		for (int i = range.first; i < range.second; ++i) {
			consVarPredictStep_(n, i) =
			    consVar_(n, i) -
			    (dt_ / dx_) * (x1Flux_(n, i + 1) - x1Flux_(n, i));
		}
	}
}

void HyperbolicSystem::AddFluxes()
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	for (int n = 0; n < nvars_; ++n) {
		for (int i = nghost_; i < nx_ + nghost_; ++i) {
			const double U_0 = consVar_(n, i);
			const double U_1 = consVarPredictStep_(n, i);
			const double FU_1 = -1.0 * (dt_ / dx_) *
					    (x1Flux_(n, i + 1) - x1Flux_(n, i));

			// RK-SSP2 integrator
			consVar_(n, i) = 0.5 * U_0 + 0.5 * U_1 + 0.5 * FU_1;
		}
	}
}

void HyperbolicSystem::ComputeTimestep()
{
	ComputeTimestep(std::numeric_limits<double>::max());
}

void HyperbolicSystem::AdvanceTimestep()
{
	AdvanceTimestep(std::numeric_limits<double>::max());
}

void HyperbolicSystem::AdvanceTimestep(const double dt_max)
{
	const auto ppm_range = std::make_pair(-1 + nghost_, nx_ + 1 + nghost_);
	const auto cell_range = std::make_pair(nghost_, nx_ + nghost_);

	// Initialize data
	FillGhostZones(consVar_);
	ConservedToPrimitive(consVar_, std::make_pair(0, dim1_));
	ComputeTimestep(dt_max);

	// Predictor step
	ReconstructStatesPPM(primVar_, ppm_range);
	ComputeFluxes(cell_range);
	PredictStep(cell_range);

	// Clear temporary arrays
	x1LeftState_.ZeroClear();
	x1RightState_.ZeroClear();
	x1Flux_.ZeroClear();

	// Corrector step
	FillGhostZones(consVarPredictStep_);
	ConservedToPrimitive(consVarPredictStep_, std::make_pair(0, dim1_));
	ReconstructStatesPPM(primVar_, ppm_range);
	ComputeFluxes(cell_range);
	AddFluxes();

	// Adjust our clock
	time_ += dt_;
}