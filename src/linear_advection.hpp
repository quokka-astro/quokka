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
#include <type_traits>

// library headers

// internal headers
#include "AMReX_BLassert.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArrayUtility.H"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
template <typename problem_t> class LinearAdvectionSystem : public HyperbolicSystem<problem_t>
{
      public:
	enum varIndex { density_index = 0 };

	// static member functions

	static void ConservedToPrimitive(arrayconst_t &cons, array_t &primVar,
					 amrex::Box const &indexRange, int nvars);
	static void ComputeMaxSignalSpeed(arrayconst_t &cons, array_t &maxSignal,
					  double advectionVx, amrex::Box const &indexRange);
	static void ComputeFluxes(array_t &x1Flux, arrayconst_t &x1LeftState,
				  arrayconst_t &x1RightState, double advectionVx,
				  amrex::Box const &indexRange, int nvars);
};

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(
    amrex::Array4<amrex::Real const> const & /*cons*/, amrex::Array4<amrex::Real> const &maxSignal,
    const double advectionVx, amrex::Box const &indexRange)
{
	const auto vx = advectionVx;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = std::abs(vx);
		// const double signal_max = advectionVx; // fails with 'invalid device function'
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ConservedToPrimitive(arrayconst_t &cons, array_t &primVar,
							    amrex::Box const &indexRange,
							    const int nvars)
{
	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		primVar(i, j, k, n) = cons(i, j, k, n);
	});
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ComputeFluxes(array_t &x1Flux, arrayconst_t &x1LeftState,
						     arrayconst_t &x1RightState,
						     const double advectionVx,
						     amrex::Box const &indexRange, const int nvars)
{
	// By convention, the interfaces are defined on the left edge of each zone, i.e.
	// xinterface_(i) is the solution to the Riemann problem at the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.
	const auto vx = advectionVx;

	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		// For advection, simply choose upwind side of the interface.
		if (vx < 0.0) { // upwind switch
			// upwind direction is the right-side of the interface
			x1Flux(i, j, k, n) = vx * x1RightState(i, j, k, n);

		} else {
			// upwind direction is the left-side of the interface
			x1Flux(i, j, k, n) = vx * x1LeftState(i, j, k, n);
		}
	});
}

#endif // LINEAR_ADVECTION_HPP_
