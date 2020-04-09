#ifndef LINEAR_ADVECTION_HPP_ // NOLINT
#define LINEAR_ADVECTION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file linear_advection.hpp
/// \brief Defines a class for solving a scalar linear advection equation.
///

// c++ headers
#include <cassert>
#include <cmath>

// library headers

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
template <typename array_t>
class LinearAdvectionSystem : public HyperbolicSystem<array_t>
{
	using HyperbolicSystem<array_t>::lx_;
	using HyperbolicSystem<array_t>::nx_;
	using HyperbolicSystem<array_t>::dx_;
	using HyperbolicSystem<array_t>::dt_;
	using HyperbolicSystem<array_t>::cflNumber_;
	using HyperbolicSystem<array_t>::dim1_;
	using HyperbolicSystem<array_t>::nghost_;
	using HyperbolicSystem<array_t>::nvars_;

	using HyperbolicSystem<array_t>::x1LeftState_;
	using HyperbolicSystem<array_t>::x1RightState_;
	using HyperbolicSystem<array_t>::x1Flux_;
	using HyperbolicSystem<array_t>::x1FluxDiffusive_;
	using HyperbolicSystem<array_t>::primVar_;
	using HyperbolicSystem<array_t>::consVar_;
	using HyperbolicSystem<array_t>::consVarPredictStep_;

      public:
	struct LinearAdvectionArgs {
		int nx;
		double lx;
		double vx;
		double cflNumber;
		int nvars;
	};

	explicit LinearAdvectionSystem(LinearAdvectionArgs args);

	void AddSourceTerms(array_t &U, std::pair<int, int> range) override;
	auto ComputeMass() -> double;

	// accessor functions

	auto density(int i) -> double;	     // returns rvalue
	auto set_density(int i) -> double &; // returns lvalue

      protected:
	array_t density_; // shallow copy of consVars_(i,:)
	double advectionVx_;

	void ConservedToPrimitive(array_t &cons,
				  std::pair<int, int> range) override;
	void ComputeTimestep(double dt_max) override;
	void ComputeFluxes(std::pair<int, int> range) override;
	void AddFluxesSDC(array_t &U_new, array_t &U0) override;
};

template <typename array_t>
auto LinearAdvectionSystem<array_t>::density(const int i) -> double
{
	return density_(i);
}

template <typename array_t>
auto LinearAdvectionSystem<array_t>::set_density(const int i) -> double &
{
	return density_(i);
}

template <typename array_t>
auto LinearAdvectionSystem<array_t>::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		mass += density_(i) * dx_;
	}

	return mass;
}

template <typename array_t>
void LinearAdvectionSystem<array_t>::ComputeTimestep(const double dt_max)
{
	dt_ = std::min(cflNumber_ * (dx_ / advectionVx_), dt_max);
}

template <typename array_t>
void LinearAdvectionSystem<array_t>::ConservedToPrimitive(
    array_t &cons, const std::pair<int, int> range)
{
	for (int n = range.first; n < range.second; ++n) {
		for (int i = 0; i < dim1_; ++i) {
			primVar_(n, i) = cons(n, i);
		}
	}
}

// TODO(ben): add flux limiter for positivity preservation.
template <typename array_t>
void LinearAdvectionSystem<array_t>::ComputeFluxes(
    const std::pair<int, int> range)
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

template <typename array_t>
void LinearAdvectionSystem<array_t>::AddSourceTerms(array_t &U,
						    std::pair<int, int> range)
{
	// TODO(ben): to be implemented
}

template <typename array_t>
void LinearAdvectionSystem<array_t>::AddFluxesSDC(array_t &U_new, array_t &U0)
{
	// TODO(ben): to be implemented
}

#endif // LINEAR_ADVECTION_HPP_
