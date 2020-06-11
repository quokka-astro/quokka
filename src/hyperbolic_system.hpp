#ifndef HYPERBOLIC_SYSTEM_HPP_ // NOLINT
#define HYPERBOLIC_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.hpp
/// \brief Defines classes and functions for use with hyperbolic systems of
/// conservation laws.
///
/// This file provides classes, data structures and functions for hyperbolic
/// systems of conservation laws.
///

// c++ headers
#include <cassert>
#include <cmath>

// library headers

// internal headers
#include "athena_arrays.hpp"

/// Provide type-safe global sign ('sgn') function.
template <typename T> auto sgn(T val) -> int
{
	return (T(0) < val) - (val < T(0));
}

typedef AthenaArray<double> array_t;

/// Class for a hyperbolic system of conservation laws (Cannot be instantiated,
/// must be subclassed.)
template <typename problem_t> class HyperbolicSystem
{
      public:
	array_t consVar_;

	/// Computes timestep and advances system
	void AdvanceTimestep();
	void AdvanceTimestep(double dt_max);
	void AdvanceTimestepRK2(double dt_max);
	void AdvanceTimestepSDC(double dt_max);

	// setter functions:

	void set_cflNumber(double cflNumber);

	// accessor functions:

	[[nodiscard]] auto nvars() const -> int;
	[[nodiscard]] auto nghost() const -> int;
	[[nodiscard]] auto nx() const -> int;
	[[nodiscard]] auto time() const -> double;
	[[nodiscard]] auto dt() const -> double;

	// inline functions:

	__attribute__((always_inline)) inline static auto minmod(double a,
								 double b)
	    -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) *
		       std::min(std::abs(a), std::abs(b));
	}

	__attribute__((always_inline)) inline static auto MC(double a, double b)
	    -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) *
		       std::min(0.5 * std::abs(a + b),
				std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	virtual void FillGhostZones(array_t &cons);
	virtual void ConservedToPrimitive(array_t &cons,
					  std::pair<int, int> range) = 0;
	virtual void AddSourceTerms(array_t &U, std::pair<int, int> range) = 0;
	virtual auto CheckStatesValid(array_t &cons,
				      std::pair<int, int> range) const -> bool;

      protected:
	array_t primVar_;
	array_t consVarPredictStep_;
	array_t consVarPredictStepPrev_;
	array_t x1LeftState_;
	array_t x1RightState_;
	array_t x1Flux_;
	array_t x1FluxDiffusive_;

	double cflNumber_ = 1.0;
	double dt_ = 0;
	const double dtExpandFactor_ = 1.2;
	double dtPrev_ = std::numeric_limits<double>::max();
	double time_ = 0.;
	double lx_;
	double dx_;
	int nx_;
	int dim1_;
	int nvars_;
	const int nghost_ = 4; // 4 ghost cells required for PPM

	HyperbolicSystem(int nx, double lx, double cflNumber, int nvars)
	    : nx_(nx), lx_(lx), dx_(lx / static_cast<double>(nx)),
	      cflNumber_(cflNumber), nvars_(nvars)
	{
		assert(lx_ > 0.0);				   // NOLINT
		assert(nx_ > 2);				   // NOLINT
		assert(nghost_ > 1);				   // NOLINT
		assert((cflNumber_ > 0.0) && (cflNumber_ <= 1.0)); // NOLINT

		dim1_ = nx_ + 2 * nghost_;

		consVar_.NewAthenaArray(nvars_, dim1_);
		primVar_.NewAthenaArray(nvars_, dim1_);
		consVarPredictStep_.NewAthenaArray(nvars_, dim1_);
		consVarPredictStepPrev_.NewAthenaArray(nvars_, dim1_);

		x1LeftState_.NewAthenaArray(nvars_, dim1_);
		x1RightState_.NewAthenaArray(nvars_, dim1_);
		x1Flux_.NewAthenaArray(nvars_, dim1_);
		x1FluxDiffusive_.NewAthenaArray(nvars_, dim1_);
	}

	virtual void AddFluxesRK2(array_t &U0, array_t &U1);
	virtual void AddFluxesSDC(array_t &U_new, array_t &U_0) = 0;

	void ReconstructStatesConstant(array_t &q, std::pair<int, int> range);
	void ReconstructStatesPPM(array_t &q, std::pair<int, int> range);

	template <typename F>
	void ReconstructStatesPLM(array_t &q, std::pair<int, int> range,
				  F &&limiter);

	virtual void PredictStep(std::pair<int, int> range);
	void ComputeTimestep();

	virtual void ComputeTimestep(double dt_max) = 0;
	virtual void ComputeFluxes(std::pair<int, int> range) = 0;
};

template <typename problem_t>
auto HyperbolicSystem<problem_t>::time() const -> double
{
	return time_;
}

template <typename problem_t>
auto HyperbolicSystem<problem_t>::dt() const -> double
{
	return dt_;
}

template <typename problem_t>
auto HyperbolicSystem<problem_t>::nx() const -> int
{
	return nx_;
}

template <typename problem_t>
auto HyperbolicSystem<problem_t>::nghost() const -> int
{
	return nghost_;
}

template <typename problem_t>
auto HyperbolicSystem<problem_t>::nvars() const -> int
{
	return nvars_;
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::set_cflNumber(double cflNumber)
{
	assert((cflNumber > 0.0) && (cflNumber <= 1.0)); // NOLINT
	cflNumber_ = cflNumber;
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::AddSourceTerms(array_t &U,
						 std::pair<int, int> range)
{
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::FillGhostZones(array_t &cons)
{
	// In general, this step will require MPI communication, and interaction
	// with the main AMR code.

	// periodic boundary conditions
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

	// extrapolate boundary conditions
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

template <typename problem_t>
void HyperbolicSystem<problem_t>::ReconstructStatesConstant(
    array_t &q, const std::pair<int, int> range)
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

			x1LeftState_(n, i) = q(n, i - 1);
			x1RightState_(n, i) = q(n, i);
		}
	}
}

template <typename problem_t>
template <typename F>
void HyperbolicSystem<problem_t>::ReconstructStatesPLM(
    array_t &q, const std::pair<int, int> range, F &&limiter)
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

			const auto lslope = limiter(q(n, i) - q(n, i - 1),
						    q(n, i - 1) - q(n, i - 2));

			const auto rslope = limiter(q(n, i + 1) - q(n, i),
						    q(n, i) - q(n, i - 1));

			x1LeftState_(n, i) =
			    q(n, i - 1) + 0.25 * lslope;	       // NOLINT
			x1RightState_(n, i) = q(n, i) - 0.25 * rslope; // NOLINT
		}
	}
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::ReconstructStatesPPM(
    array_t &q, const std::pair<int, int> range)
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

template <typename problem_t>
void HyperbolicSystem<problem_t>::PredictStep(const std::pair<int, int> range)
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

template <typename problem_t>
void HyperbolicSystem<problem_t>::AddFluxesRK2(array_t &U0, array_t &U1)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	for (int n = 0; n < nvars_; ++n) {
		for (int i = nghost_; i < nx_ + nghost_; ++i) {
			// RK-SSP2 integrator
			const double U_0 = U0(n, i);
			const double U_1 = U1(n, i);
			const double FU_1 = -1.0 * (dt_ / dx_) *
					    (x1Flux_(n, i + 1) - x1Flux_(n, i));

			// save results in U0
			U0(n, i) = 0.5 * U_0 + 0.5 * U_1 + 0.5 * FU_1;
		}
	}
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::ComputeTimestep()
{
	ComputeTimestep(std::numeric_limits<double>::max());
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::AdvanceTimestep()
{
	AdvanceTimestep(std::numeric_limits<double>::max());
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::AdvanceTimestep(const double dt_max)
{
	// use RK2 by default
	AdvanceTimestepRK2(dt_max);
}

template <typename problem_t>
auto HyperbolicSystem<problem_t>::CheckStatesValid(
    array_t &cons, const std::pair<int, int> range) const -> bool
{
	return true;
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::AdvanceTimestepRK2(const double dt_max)
{
	const auto ppm_range = std::make_pair(-1 + nghost_, nx_ + 1 + nghost_);
	const auto cell_range = std::make_pair(nghost_, nx_ + nghost_);

	// Initialize data
	FillGhostZones(consVar_);
	ConservedToPrimitive(consVar_, std::make_pair(0, dim1_));
	ComputeTimestep(std::min(dt_max, dtExpandFactor_ * dtPrev_));

	// Predictor step t_{n+1}
	FillGhostZones(consVar_);
	ConservedToPrimitive(consVar_, std::make_pair(0, dim1_));
	ReconstructStatesPPM(primVar_, ppm_range);
	ComputeFluxes(cell_range);
	PredictStep(cell_range);

	if (!CheckStatesValid(consVarPredictStep_, cell_range)) {
		std::cout << "[stage 1] This should not happen!\n";
		assert(false); // NOLINT
	}

	// Corrector step
	FillGhostZones(consVarPredictStep_);
	ConservedToPrimitive(consVarPredictStep_, std::make_pair(0, dim1_));
	ReconstructStatesPPM(primVar_, ppm_range);
	ComputeFluxes(cell_range);
	AddFluxesRK2(consVar_, consVarPredictStep_);

	if (!CheckStatesValid(consVarPredictStep_, cell_range)) {
		std::cout << "[stage 2] This should not happen!\n";
		assert(false); // NOLINT
	}

	// Add source terms via operator splitting
	AddSourceTerms(consVar_, cell_range);

#if 0
	if (CheckStatesValid(consVar_, cell_range)) {
		std::cout << "[source terms] This should not happen!\n";
		assert(false);
	}
#endif

	// Adjust our clock
	time_ += dt_;
	dtPrev_ = dt_;
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::AdvanceTimestepSDC(const double dt_max)
{
	const auto ppm_range = std::make_pair(-1 + nghost_, nx_ + 1 + nghost_);
	const auto cell_range = std::make_pair(nghost_, nx_ + nghost_);

	// 0. Initialize state U_{n}, fill ghost zones, compute timestep
	FillGhostZones(consVar_);
	ConservedToPrimitive(consVar_, std::make_pair(0, dim1_));
	ComputeTimestep(dt_max);

	/// Timestep reduction loop
	double L2norm_resid = NAN;
	double L2norm_resid_prev = NAN;
	double L2norm_initial = NAN;
	const double resid_rtol = 1e-5;
	const double resid_hang_atol = 1e-5;
	const int reduceDtMaxStages = 6;
	const double dt0 = dt_;
	int s = 0;
	int m = 0;
	for (s = 0; s <= reduceDtMaxStages; ++s) {

		// reduce initial timestep dt0 by a factor of 2^{s}
		dt_ = (s > 0) ? (dt0 * std::pow(s, -2.0)) : dt0;
		assert(dt_ > 0.0);     // NOLINT
		assert(dt_ <= dt_max); // NOLINT

		/// 1. Compute initial guess for state at time t_{n+1}, save in
		/// consVarPredictStep_

		// 1aa. Clear temporary arrays
		x1LeftState_.ZeroClear();
		x1RightState_.ZeroClear();
		x1Flux_.ZeroClear();

		// 1ab. Copy consVar_ -> consVarPredictStep_
		for (int n = 0; n < nvars_; ++n) {
			for (int i = 0; i < dim1_; ++i) {
				consVarPredictStep_(n, i) = consVar_(n, i);
			}
		}

		// 1a. Compute state \tilde{U}(t_{n+1})
		FillGhostZones(consVarPredictStep_);
		ConservedToPrimitive(consVarPredictStep_,
				     std::make_pair(0, dim1_));
		ReconstructStatesConstant(primVar_, ppm_range);
		ComputeFluxes(cell_range);
		AddFluxesSDC(consVarPredictStep_, consVar_);
		// PredictStep(cell_range); // saved to consVarPredictStep_

		// 1ab. Copy consVarPredictStep_ -> consVarPredictStepPrev_
		for (int n = 0; n < nvars_; ++n) {
			for (int i = 0; i < dim1_; ++i) {
				consVarPredictStepPrev_(n, i) =
				    consVarPredictStep_(n, i);
			}
		}

		// 1b. Compute state \hat{U}(t_{n+1})
		AddSourceTerms(consVarPredictStep_, cell_range);

		/// 2. Do SDC iteration

		// Compute size of initial 'correction'
		L2norm_initial = 0.0;
		for (int n = 0; n < nvars_; ++n) {
			for (int i = cell_range.first; i < cell_range.second;
			     ++i) {
				L2norm_initial +=
				    dx_ *
				    std::pow(consVarPredictStep_(n, i) -
						 consVarPredictStepPrev_(n, i),
					     2);
			}
		}

		const int maxIter = 200;
		for (m = 1; m < maxIter; ++m) {

			// 2b. Clear temporary arrays
			x1LeftState_.ZeroClear();
			x1RightState_.ZeroClear();
			x1Flux_.ZeroClear();

			// 2c. Copy consVarPredictStep_ ->
			// consVarPredictStepPrev_
			for (int n = 0; n < nvars_; ++n) {
				for (int i = 0; i < dim1_; ++i) {
					consVarPredictStepPrev_(n, i) =
					    consVarPredictStep_(n, i);
				}
			}

			// 2d. Compute F(\hat{U}_{t+1}), update U_{n} with
			// F(\hat{U}_{t+1}), save in consVarPredictStep_

			FillGhostZones(consVarPredictStep_);
			ConservedToPrimitive(consVarPredictStep_,
					     std::make_pair(0, dim1_));
			ReconstructStatesConstant(primVar_, ppm_range);
			ComputeFluxes(cell_range);
			AddFluxesSDC(consVarPredictStep_, consVar_);

			AddSourceTerms(consVarPredictStep_, cell_range);

			// 2e. Compute L2-norm difference from previous solution
			L2norm_resid_prev = L2norm_resid;
			L2norm_resid = 0.0;
			for (int n = 0; n < nvars_; ++n) {
				for (int i = cell_range.first;
				     i < cell_range.second; ++i) {
					L2norm_resid +=
					    dx_ *
					    std::pow(
						consVarPredictStep_(n, i) -
						    consVarPredictStepPrev_(n,
									    i),
						2);
				}
			}

			// 2a. convergence check
			if ((L2norm_resid / L2norm_initial) < resid_rtol) {
				break;
			}
			if (std::abs(L2norm_resid_prev - L2norm_resid) <
			    resid_hang_atol) {
				std::cout << "Solver hung after " << m
					  << " SDC iterations.\n";
				break;
			}
		}

		// 3. convergence check
		if ((L2norm_resid / L2norm_initial) < resid_rtol) {
			break;
		}

		std::cout << "Very stiff behavior, reducing timestep by 2x.\n";
	}

	if (std::abs(L2norm_resid_prev - L2norm_resid) < resid_hang_atol) {
		std::cout << "Solver *still* hung after reducing timestep by "
			     "factor of "
			  << std::pow(2, s) << "x.\n";
		// assert(std::abs(L2norm_resid_prev - L2norm_resid) >=
		// resid_hang_atol);
	} else {
		assert((L2norm_resid / L2norm_initial) < resid_rtol); // NOLINT
		std::cout << "Converged in " << m << " SDC iterations."
			  << std::endl;
	}

	/// 4. End SDC iteration, save result in consVar_

	for (int n = 0; n < nvars_; ++n) {
		for (int i = 0; i < dim1_; ++i) {
			consVar_(n, i) = consVarPredictStep_(n, i);
		}
	}

	// 4a. Adjust our clock
	time_ += dt_;
}

#endif // HYPERBOLIC_SYSTEM_HPP_
