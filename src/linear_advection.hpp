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

	static void ComputeMaxSignalSpeed(amrex::Array4<amrex::Real const> const & /*cons*/,
					  amrex::Array4<amrex::Real> const &maxSignal,
					  double advectionVx, double advectionVy,
					  double advectionVz, amrex::Box const &indexRange);

	AMREX_GPU_DEVICE
	static auto isStateValid(amrex::Array4<const amrex::Real> const &cons,
					  int i, int j, int k) -> bool;
	
	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew,
					  std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, double dt_in,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
					  int nvars, amrex::Array4<int> const &redoFlag);

	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1,
    				  std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, double dt_in,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
					  int nvars, amrex::Array4<int> const &redoFlag);

	template <FluxDir DIR>
	static void ComputeFluxes(array_t &x1Flux, arrayconst_t &x1LeftState,
				  arrayconst_t &x1RightState, double advectionVx,
				  amrex::Box const &indexRange, int nvars);
};

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(
    amrex::Array4<amrex::Real const> const & /*cons*/, amrex::Array4<amrex::Real> const &maxSignal,
    const double advectionVx, const double advectionVy, const double advectionVz,
    amrex::Box const &indexRange)
{
	const auto vx = advectionVx;
	const auto vy = advectionVy;
	const auto vz = advectionVz;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = std::sqrt(vx * vx + vy * vy + vz * vz);
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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto LinearAdvectionSystem<problem_t>::isStateValid(
		amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool {
	// check if cons(i, j, k) is a valid state
	const auto rho = cons(i, j, k, density_index);
	bool isDensityPositive = (rho > 0.);
	return isDensityPositive;
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::PredictStep(
    arrayconst_t &consVarOld, array_t &consVarNew,
    std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
    const int nvars, amrex::Array4<int> const &redoFlag)
{
	BL_PROFILE("LinearAdvectionSystem::PredictStep()");

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	auto const dx = dx_in[0];
	auto const x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	auto const dy = dx_in[1];
	auto const x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	auto const dz = dx_in[2];
	auto const x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(
	    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < nvars; ++n) {
				consVarNew(i, j, k, n) =
				consVarOld(i, j, k, n) +
				(AMREX_D_TERM( (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
							+ (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
							+ (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))
							));
			}

			// check if state is valid -- flag for re-do if not
			if (!isStateValid(consVarNew, i, j, k)) {
				redoFlag(i, j, k) = quokka::redoFlag::redo;
			} else {
				redoFlag(i, j, k) = quokka::redoFlag::none;
			}
	    });
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::AddFluxesRK2(
    array_t &U_new, arrayconst_t &U0, arrayconst_t &U1,
    std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
    const int nvars, amrex::Array4<int> const &redoFlag)
{
	BL_PROFILE("LinearAdvectionSystem::AddFluxesRK2()");

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	auto const dx = dx_in[0];
	auto const x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	auto const dy = dx_in[1];
	auto const x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	auto const dz = dx_in[2];
	auto const x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(
	    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < nvars; ++n) {
				// RK-SSP2 integrator
				const double U_0 = U0(i, j, k, n);
				const double U_1 = U1(i, j, k, n);

				const double FxU_1 = (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n));
	#if (AMREX_SPACEDIM >= 2)
				const double FyU_1 = (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n));
	#endif
	#if (AMREX_SPACEDIM == 3)
				const double FzU_1 = (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n));
	#endif

				// save results in U_new
				U_new(i, j, k, n) = (0.5 * U_0 + 0.5 * U_1) + (
					AMREX_D_TERM( 0.5 * FxU_1 ,
								+ 0.5 * FyU_1 ,
								+ 0.5 * FzU_1 )
								);
			}

			// check if state is valid -- flag for re-do if not
			if (!isStateValid(U_new, i, j, k)) {
				redoFlag(i, j, k) = quokka::redoFlag::redo;
			} else {
				redoFlag(i, j, k) = quokka::redoFlag::none;
			}
	    });
}

template <typename problem_t>
template <FluxDir DIR>
void LinearAdvectionSystem<problem_t>::ComputeFluxes(array_t &x1Flux_in,
						     arrayconst_t &x1LeftState_in,
						     arrayconst_t &x1RightState_in,
						     const double advectionVx,
						     amrex::Box const &indexRange, const int nvars)
{
	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<amrex::Real const, DIR> x1RightState(x1RightState_in);
	quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);

	const auto vx = advectionVx; // avoid CUDA invalid device function error (tracked as NVIDIA
				     // bug #3318015)
	// By convention, the interfaces are defined on the left edge of each zone, i.e.
	// xinterface_(i) is the solution to the Riemann problem at the left edge of zone i.
	// [Indexing note: There are (nx + 1) interfaces for nx zones.]

	amrex::ParallelFor(
	    indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		    // permute array indices according to dir
		    auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

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
