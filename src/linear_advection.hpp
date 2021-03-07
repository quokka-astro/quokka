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

	static void ConservedToPrimitive(arrayconst_t &cons, array_t &primVar, amrex::Box const &indexRange,
					 int nvars);
	static auto ComputeTimestep(double dt_max, double cflNumber, double dx, double advectionVx)
	    -> double;
	static void ComputeFluxes(array_t &x1Flux, arrayconst_t &x1LeftState, arrayconst_t &x1RightState,
				  double advectionVx, amrex::Box const &indexRange, int nvars);
	static auto ComputeMass(amrex::FArrayBox const &density, double dx,
						   amrex::Box const &indexRange) -> double;
	static void FillGhostZones(array_t &cons, int nx, int nghost,
						      int nvars);
};

template <typename problem_t>
auto LinearAdvectionSystem<problem_t>::ComputeMass(amrex::FArrayBox const &density, const double dx,
						   amrex::Box const &indexRange) -> double
{
	amrex::FArrayBox mass(indexRange, 1);
    mass.saxpy(dx, density);
	const amrex::Real mass_sum = mass.sum(0); // sum component 0
	return mass_sum;
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

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::FillGhostZones(array_t &cons, const int nx, const int nghost,
						      const int nvars)
{
	const int j = 1;
	const int k = 1;
	// periodic boundary conditions

	// x1 right side boundary
	for (int n = 0; n < nvars; ++n) {
		for (int i = nghost + nx; i < nghost + nx + nghost; ++i) {
			cons(i, j, k, n) = cons(i - nx, j, k, n);
		}
	}

	// x1 left side boundary
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < nghost; ++i) {
			cons(i, j, k, n) = cons(i + nx, j, k, n);
		}
	}
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ConservedToPrimitive(arrayconst_t &cons,
								array_t &primVar,
							    amrex::Box const &indexRange,
								const int nvars)
{
    amrex::ParallelFor(indexRange,
    [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
		for (int n = 0; n < nvars; ++n) {
			primVar(i, j, k, n) = cons(i, j, k, n);
		}
	});
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ComputeFluxes(array_t &x1Flux, 
							 arrayconst_t &x1LeftState, arrayconst_t &x1RightState,
							 const double advectionVx,
						     amrex::Box const &indexRange,
						     const int nvars)
{
	// By convention, the interfaces are defined on the left edge of each zone, i.e.
	// xinterface_(i) is the solution to the Riemann problem at the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

    amrex::ParallelFor(indexRange,
    [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
		for (int n = 0; n < nvars; ++n) {
			// For advection, simply choose upwind side of the interface.
			if (advectionVx < 0.0) { // upwind switch
				// upwind direction is the right-side of the interface
				x1Flux(i, j, k, n) = advectionVx * x1RightState(i, j, k, n);

			} else {
				// upwind direction is the left-side of the interface
				x1Flux(i, j, k, n) = advectionVx * x1LeftState(i, j, k, n);
			}
		}
	});
}

#endif // LINEAR_ADVECTION_HPP_
