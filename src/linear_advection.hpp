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
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
template <typename problem_t> class LinearAdvectionSystem : public HyperbolicSystem<problem_t>
{
	using HyperbolicSystem<problem_t>::lx_;
	using HyperbolicSystem<problem_t>::nx_;
	using HyperbolicSystem<problem_t>::dx_;
	using HyperbolicSystem<problem_t>::dt_;
	using HyperbolicSystem<problem_t>::cflNumber_;
	using HyperbolicSystem<problem_t>::dim1_;
	using HyperbolicSystem<problem_t>::nghost_;
	using HyperbolicSystem<problem_t>::nvars_;

	using HyperbolicSystem<problem_t>::x1LeftState_;
	using HyperbolicSystem<problem_t>::x1RightState_;
	using HyperbolicSystem<problem_t>::x1Flux_;
	using HyperbolicSystem<problem_t>::x1FluxDiffusive_;
	using HyperbolicSystem<problem_t>::primVar_;
	using HyperbolicSystem<problem_t>::consVar_;
	using HyperbolicSystem<problem_t>::consVarPredictStep_;

      public:
	enum varIndex { density_index = 0 };

	struct LinearAdvectionArgs {
		int nx;
		double lx;
		double vx;
		double cflNumber;
		int nvars;
	};

	explicit LinearAdvectionSystem(LinearAdvectionArgs args);

	void AddSourceTerms(array_t &U_prev, array_t &U_new, std::pair<int, int> range) override;
	auto ComputeMass() -> double;
	void FillGhostZones(array_t &cons) override;

	// static member functions

	static void ConservedToPrimitive(array_t &cons, array_t &primVar, std::pair<int, int> range,
					 int nvars);
	static auto ComputeTimestep(double dt_max, double cflNumber, double dx, double advectionVx)
	    -> double;
	static void ComputeFluxes(array_t &x1Flux, array_t &x1LeftState, array_t &x1RightState,
				  double advectionVx, std::pair<int, int> range, int nvars);

	// accessor functions

	auto density(int i) -> double;	     // returns rvalue
	auto set_density(int i) -> double &; // returns lvalue

      protected:
	array_t density_; // shallow copy of consVars_(i,:)
	double advectionVx_;

	void ComputeFirstOrderFluxes(std::pair<int, int> range) override;
};

template <typename problem_t>
LinearAdvectionSystem<problem_t>::LinearAdvectionSystem(const LinearAdvectionArgs args)
    : advectionVx_(args.vx), HyperbolicSystem<problem_t>{args.nx, args.lx, args.cflNumber,
							 args.nvars}
{
	assert(advectionVx_ != 0.0); // NOLINT

	density_ = consVar_.SliceArray(density_index);
}

template <typename problem_t> auto LinearAdvectionSystem<problem_t>::density(const int i) -> double
{
	return density_(i);
}

template <typename problem_t>
auto LinearAdvectionSystem<problem_t>::set_density(const int i) -> double &
{
	return density_(i);
}

template <typename problem_t> auto LinearAdvectionSystem<problem_t>::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		mass += density_(i) * dx_;
	}

	return mass;
}

template <typename problem_t>
auto LinearAdvectionSystem<problem_t>::ComputeTimestep(const double dt_max,
							      const double cflNumber,
							      const double dx,
							      const double advectionVx) -> double
{
	auto dt = std::min(cflNumber * (dx / advectionVx), dt_max);
	return dt;
}

template <typename problem_t> void LinearAdvectionSystem<problem_t>::FillGhostZones(array_t &cons)
{
	// periodic boundary conditions

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
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ConservedToPrimitive(array_t &cons,
								array_t &primVar,
							    const std::pair<int, int> range,
								const int nvars)
{
	for (int n = 0; n < nvars; ++n) {
		for (int i = range.first; i < range.second; ++i) {
			primVar(n, i) = cons(n, i);
		}
	}
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ComputeFirstOrderFluxes(std::pair<int, int> range)
{
	// TODO(ben): implement
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ComputeFluxes(array_t &x1Flux, 
							 array_t &x1LeftState, array_t &x1RightState,
							 const double advectionVx,
						     const std::pair<int, int> range,
						     const int nvars)
{
	// By convention, the interfaces are defined on the left edge of each zone, i.e.
	// xinterface_(i) is the solution to the Riemann problem at the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int n = 0; n < nvars; ++n) {
		for (int i = range.first; i < (range.second + 1); ++i) {

			// For advection, simply choose upwind side of the interface.

			if (advectionVx < 0.0) { // upwind switch
				// upwind direction is the right-side of the interface
				x1Flux(n, i) = advectionVx * x1RightState(n, i);

			} else {
				// upwind direction is the left-side of the interface
				x1Flux(n, i) = advectionVx * x1LeftState(n, i);
			}
		}
	}
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::AddSourceTerms(array_t &U_prev, array_t &U_new,
						      std::pair<int, int> range)
{
	// TODO(ben): to be implemented
}

#endif // LINEAR_ADVECTION_HPP_
