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
#include "AMReX_Array4.H"
#include "AMReX_Dim3.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_IntVect.H"
#include "AMReX_Math.H"
#include "AMReX_MultiFab.H"
#include "AMReX_SPACE.H"

// internal headers
#include "ArrayView.hpp"
#include "simulation.hpp"

/// Provide type-safe global sign ('sgn') function.
template <typename T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sgn(T val) -> int
{
	return (T(0) < val) - (val < T(0));
}

using array_t = amrex::Array4<amrex::Real> const;
using arrayconst_t = amrex::Array4<const amrex::Real> const;

/// Class for a hyperbolic system of conservation laws
template <typename problem_t> class HyperbolicSystem
{
      public:
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b)
	    -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) *
		       std::min(0.5 * std::abs(a + b),
				std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	template <FluxDir DIR>
	static void ReconstructStatesConstant(arrayconst_t &q, array_t &leftState,
					      array_t &rightState, amrex::Box const &indexRange,
					      int nvars);

	template <FluxDir DIR>
	static void ReconstructStatesPLM(arrayconst_t &q, array_t &leftState, array_t &rightState,
					 amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR>
	static void ReconstructStatesPPM(arrayconst_t &q, array_t &leftState, array_t &rightState,
					 amrex::Box const &cellRange,
					 amrex::Box const &interfaceRange, int nvars);

	__attribute__ ((__target__ ("no-fma")))
	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1,
				 std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				 double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
				 amrex::Box const &indexRange, int nvars);

	__attribute__ ((__target__ ("no-fma")))
	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew,
				std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
				amrex::Box const &indexRange, int nvars);

#if 0
	static void SaveFluxes(array_t &advectionFluxes, arrayconst_t &x1Flux, double dx,
			       amrex::Box const &indexRange, int nvars);
#endif
};

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesConstant(arrayconst_t &q_in,
							    array_t &leftState_in,
							    array_t &rightState_in,
							    amrex::Box const &indexRange,
							    const int nvars)
{
	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	// By convention, the interfaces are defined on the left edge of each zone, i.e. xleft_(i)
	// is the "left"-side of the interface at the left edge of zone i, and xright_(i) is the
	// "right"-side of the interface at the *left* edge of zone i. [Indexing note: There are (nx
	// + 1) interfaces for nx zones.]

	amrex::ParallelFor(
	    indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		    // permute array indices according to dir
		    auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		    // Use piecewise-constant reconstruction (This converges at first
		    // order in spatial resolution.)
		    leftState(i, j, k, n) = q(i - 1, j, k, n);
		    rightState(i, j, k, n) = q(i, j, k, n);
	    });
}

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesPLM(arrayconst_t &q_in, array_t &leftState_in,
						       array_t &rightState_in,
						       amrex::Box const &indexRange,
						       const int nvars)
{
	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	// Unlike PPM, PLM with the MC limiter is TVD.
	// (There are no spurious oscillations, *except* in the slow-moving shock problem,
	// which can produce unphysical oscillations even when using upwind Godunov fluxes.)
	// However, most tests fail when using PLM reconstruction because
	// the accuracy tolerances are very strict, and the L1 error is significantly
	// worse compared to PPM for a fixed number of mesh elements.

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	amrex::ParallelFor(
	    indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		    // permute array indices according to dir
		    auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		    // Use piecewise-linear reconstruction
		    // (This converges at second order in spatial resolution.)
		    const auto lslope = MC(q(i, j, k, n) - q(i - 1, j, k, n),
					   q(i - 1, j, k, n) - q(i - 2, j, k, n));
		    const auto rslope =
			MC(q(i + 1, j, k, n) - q(i, j, k, n), q(i, j, k, n) - q(i - 1, j, k, n));

		    leftState(i, j, k, n) = q(i - 1, j, k, n) + 0.25 * lslope; // NOLINT
		    rightState(i, j, k, n) = q(i, j, k, n) - 0.25 * rslope;    // NOLINT
	    });
}

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in,
						       array_t &rightState_in,
						       amrex::Box const &cellRange,
						       amrex::Box const &interfaceRange,
						       const int nvars)
{
	BL_PROFILE("HyperbolicSystem::ReconstructStatesPPM()");

	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at the left
	// edge of zone i, and xright_(i) is the "right"-side of the interface
	// at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// fuse loops into a single GPU kernel to avoid kernel launch overhead
	amrex::ParallelFor(
		interfaceRange, nvars, // interface-centered kernel
				[=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
				   // permute array indices according to dir
				   auto [i, j, k] =
				       quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

				   // PPM reconstruction following Colella & Woodward (1984), with
				   // some modifications following Mignone (2014), as implemented in
				   // Athena++.

				   // (1.) Estimate the interface a_{i - 1/2}. Equivalent to step 1
				   // in Athena++ [ppm_simple.cpp].

				   // C&W Eq. (1.9) [parabola midpoint for the case of
				   // equally-spaced zones]: a_{j+1/2} = (7/12)(a_j + a_{j+1}) -
				   // (1/12)(a_{j+2} + a_{j-1}). Terms are grouped to preserve exact
				   // symmetry in floating-point arithmetic, following Athena++.

				   const double coef_1 = (7. / 12.);
				   const double coef_2 = (-1. / 12.);
				   const double interface =
				       (coef_1 * q(i, j, k, n) + coef_2 * q(i + 1, j, k, n)) +
				       (coef_1 * q(i - 1, j, k, n) + coef_2 * q(i - 2, j, k, n));

				   // a_R,(i-1) in C&W
				   leftState(i, j, k, n) = interface;

				   // a_L,i in C&W
				   rightState(i, j, k, n) = interface;
			   },
		cellRange, nvars, // cell-centered kernel
		[=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		    // permute array indices according to dir
		    auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		    // (2.) Constrain interfaces to lie between surrounding cell-averaged
		    // values (equivalent to step 2b in Athena++ [ppm_simple.cpp]).
			// [See Eq. B8 of Mignone+ 2005.]

#ifdef MULTIDIM_EXTREMA_CHECK
			// N.B.: Checking all 27 nearest neighbors is *very* expensive on GPU
			// (presumably due to lots of cache misses), so it is hard-coded disabled.
			// Fortunately, almost all problems run stably without enabling this.
#if (AMREX_SPACEDIM == 1)
			// 1D: compute bounds from self + all 2 surrounding cells
			const std::pair<double, double> bounds =
				std::minmax({q(i, j, k, n), q(i - 1, j, k, n), q(i + 1, j, k, n)});
#elif (AMREX_SPACEDIM == 2)
			// 2D: compute bounds from self + all 8 surrounding cells
			const std::pair<double, double> bounds = std::minmax({q(i, j, k, n),
				q(i - 1, j, k, n), q(i + 1, j, k, n),
				q(i, j - 1, k, n), q(i, j + 1, k, n),
				q(i - 1, j - 1, k, n), q(i + 1, j - 1, k, n),
				q(i - 1, j + 1, k, n), q(i + 1, j + 1, k, n)});
#else // AMREX_SPACEDIM == 3
			// 3D: compute bounds from self + all 26 surrounding cells
			const std::pair<double, double> bounds = std::minmax({q(i, j, k, n),
				q(i - 1, j, k, n), q(i + 1, j, k, n),
				q(i, j - 1, k, n), q(i, j + 1, k, n),
				q(i, j, k - 1, n), q(i, j, k + 1, n),
				q(i - 1, j - 1, k, n), q(i + 1, j - 1, k, n),
				q(i - 1, j + 1, k, n), q(i + 1, j + 1, k, n),
				q(i, j - 1, k - 1, n), q(i, j + 1, k - 1, n),
				q(i, j - 1, k + 1, n), q(i, j + 1, k + 1, n),
				q(i - 1, j, k - 1, n), q(i + 1, j, k - 1, n),
				q(i - 1, j, k + 1, n), q(i + 1, j, k + 1, n),
				q(i - 1, j - 1, k - 1, n), q(i + 1, j - 1, k - 1, n),
				q(i - 1, j - 1, k + 1, n), q(i + 1, j - 1, k + 1, n),
				q(i - 1, j + 1, k - 1, n), q(i + 1, j + 1, k - 1, n),
				q(i - 1, j + 1, k + 1, n), q(i + 1, j + 1, k + 1, n)});
#endif // AMREX_SPACEDIM
#else // MULTIDIM_EXTREMA_CHECK
			// compute bounds from neighboring cell-averaged values along axis
			const std::pair<double, double> bounds =
				std::minmax({q(i, j, k, n), q(i - 1, j, k, n), q(i + 1, j, k, n)});
#endif // MULTIDIM_EXTREMA_CHECK

		    // get interfaces
		    const double a_minus = rightState(i, j, k, n);
		    const double a_plus = leftState(i + 1, j, k, n);

		    // left side of zone i
		    double new_a_minus = clamp(a_minus, bounds.first, bounds.second);

		    // right side of zone i
		    double new_a_plus = clamp(a_plus, bounds.first, bounds.second);

		    // (3.) Monotonicity correction, using Eq. (1.10) in PPM paper. Equivalent
		    // to step 4b in Athena++ [ppm_simple.cpp].
			
		    const double a = q(i, j, k, n);	// a_i in C&W
		    const double dq_minus = (a - new_a_minus);
		    const double dq_plus = (new_a_plus - a);

		    const double qa = dq_plus * dq_minus; // interface extrema

		    if (qa <= 0.0) { // local extremum

			    // Causes subtle, but very weird, oscillations in the Shu-Osher test
			    // problem. However, it is necessary to get a reasonable solution
			    // for the sawtooth advection problem.
			    const double dq0 = MC(q(i + 1, j, k, n) - q(i, j, k, n),
						  q(i, j, k, n) - q(i - 1, j, k, n));

			    // use linear reconstruction, following Balsara (2017) [Living Rev
			    // Comput Astrophys (2017) 3:2]
			    new_a_minus = a - 0.5 * dq0;
			    new_a_plus = a + 0.5 * dq0;

			    // original C&W method for this case
			    // new_a_minus = a;
			    // new_a_plus = a;

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

		    rightState(i, j, k, n) = new_a_minus;
		    leftState(i + 1, j, k, n) = new_a_plus;
	});
}

#if 0
template <typename problem_t>
void HyperbolicSystem<problem_t>::SaveFluxes(array_t &advectionFluxes, arrayconst_t &x1Flux,
					     const double dx, amrex::Box const &indexRange,
					     const int nvars)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
		advectionFluxes(i, j, k, n) =
		    (-1.0 / dx) * (x1Flux(i + 1, j, k, n) - x1Flux(i, j, k, n));
	});
}
#endif

template <typename problem_t>
void HyperbolicSystem<problem_t>::PredictStep(
    arrayconst_t &consVarOld, array_t &consVarNew,
    std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
    const int nvars)
{
	BL_PROFILE("HyperbolicSystem::PredictStep()");

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
	    indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
		    consVarNew(i, j, k, n) =
			consVarOld(i, j, k, n) +
			(AMREX_D_TERM( (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
				      	 + (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
				      	 + (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))
						   ));
	    });
}

template <typename problem_t>
void HyperbolicSystem<problem_t>::AddFluxesRK2(
    array_t &U_new, arrayconst_t &U0, arrayconst_t &U1,
    std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
    const int nvars)
{
	BL_PROFILE("HyperbolicSystem::AddFluxesRK2()");

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
	    indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
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
	    });
}

#endif // HYPERBOLIC_SYSTEM_HPP_
