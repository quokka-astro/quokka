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
#include <algorithm>
#include <cmath>

// library headers
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_SPACE.H"

// internal headers
#include "math/math_impl.hpp"
#include "util/ArrayView.hpp"

// #define MULTIDIM_EXTREMA_CHECK

namespace quokka
{
enum redoFlag { none = 0, redo = 1 };
} // namespace quokka

// Define enum for slope limiter type
enum class SlopeLimiter { minmod = 0, sweby, MC };

using array_t = amrex::Array4<amrex::Real> const;
using arrayconst_t = amrex::Array4<const amrex::Real> const;

/// Class for a hyperbolic system of conservation laws
template <typename problem_t> class HyperbolicSystem
{
      public:
	template <SlopeLimiter limiter> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SlopeFunc(amrex::Real x, amrex::Real y) -> amrex::Real
	{
		static_assert(limiter == SlopeLimiter::minmod || limiter == SlopeLimiter::sweby || limiter == SlopeLimiter::MC,
			      "Invalid slope limiter specified.");
		if constexpr (limiter == SlopeLimiter::minmod) {
			return Sweby(x, y, 1.0);
		}
		if constexpr (limiter == SlopeLimiter::sweby) {
			return Sweby(x, y, 1.5);
		}
		if constexpr (limiter == SlopeLimiter::MC) {
			return Sweby(x, y, 2.0);
		}
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double { return Sweby(a, b, 2.0); }

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(std::abs(a), std::abs(b));
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod3(double a, double b, double c) -> double
	{
		if ((sgn(a) == sgn(b)) && (sgn(a) == sgn(c))) {
			return sgn(a) * std::min({std::abs(a), std::abs(b), std::abs(c)});
		}
		return 0.0;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto Sweby(double a, double b, double sigma) -> double
	{
		const double sigma_a = sigma * a;
		const double sigma_b = sigma * b;
		const double centered = 0.5 * (a + b);
		return minmod3(sigma_a, centered, sigma_b);
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto median(double a, double b, double c) -> double
	{
		return std::max(std::min(a, b), std::min(std::max(a, b), c));
	}

	template <FluxDir DIR>
	static void ReconstructStatesConstant(amrex::MultiFab const &q, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int nghost, int nvars);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesConstant(arrayconst_t &q, array_t &leftState, array_t &rightState,
										       amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
	ReconstructStatesConstant(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState,
				  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in);

	template <FluxDir DIR, SlopeLimiter limiter>
	static void ReconstructStatesPLM(amrex::MultiFab const &q, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int nghost, int nvars);

	template <FluxDir DIR, SlopeLimiter limiter>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPLM(arrayconst_t &q, array_t &leftState, array_t &rightState,
										  amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR, SlopeLimiter limiter>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
	ReconstructStatesPLM(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState,
			     quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in);

	template <FluxDir DIR>
	static void ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, int nghost, int nvars,
					 int iReadFrom = 0, int iWriteFrom = 0);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
										  amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars,
										  int iReadFrom = 0, int iWriteFrom = 0);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q,
										  quokka::Array4View<amrex::Real, DIR> const &leftState,
										  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in,
										  int j_in, int k_in, int iReadFrom = 0, int iWriteFrom = 0);

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto MonotonizeEdges(double qL_in, double qR_in, double q, double qminus, double qplus)
	    -> std::pair<double, double>;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto ComputeSteepPPM(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n)
	    -> amrex::Real;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto ComputeWENOMoments(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n)
	    -> std::pair<amrex::Real, amrex::Real>;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto ComputeWENO(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n)
	    -> std::pair<amrex::Real, amrex::Real>;

	template <FluxDir DIR>
	static void ReconstructStatesPPM_EP(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, int nghost, int nvars,
					    int iReadFrom = 0, int iWriteFrom = 0);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM_EP(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
										     amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars,
										     int iReadFrom = 0, int iWriteFrom = 0);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM_EP(quokka::Array4View<amrex::Real const, DIR> const &q,
										     quokka::Array4View<amrex::Real, DIR> const &leftState,
										     quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in,
										     int j_in, int k_in, int iReadFrom = 0, int iWriteFrom = 0);

	template <typename F>
#if defined(__x86_64__)
	__attribute__((__target__("no-fma")))
#endif
	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, double dt_in,
				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, F &&isStateValid,
				 amrex::Array4<int> const &redoFlag);

	template <typename F>
#if defined(__x86_64__)
	__attribute__((__target__("no-fma")))
#endif
	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, double dt_in,
				amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, F &&isStateValid,
				amrex::Array4<int> const &redoFlag);
};

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesConstant(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf,
							    const int nghost, const int nvars)
{
	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
										  amrex::Box const &indexRange, const int nvars)
{
	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(quokka::Array4View<amrex::Real const, DIR> const &q,
										  quokka::Array4View<amrex::Real, DIR> const &leftState,
										  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in,
										  int j_in, int k_in)
{
	// permute array indices according to dir
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	// Use piecewise-constant reconstruction (This converges at first order in spatial resolution.)
	// By convention, the interfaces are defined on the left edge of each zone, i.e. xleft_(i)
	// is the "left"-side of the interface at the left edge of zone i, and xright_(i) is the
	// "right"-side of the interface at the *left* edge of zone i. [Indexing note: There are (nx
	// + 1) interfaces for nx zones.]
	leftState(i, j, k, n) = q(i - 1, j, k, n);
	rightState(i, j, k, n) = q(i, j, k, n);
}

template <typename problem_t>
template <FluxDir DIR, SlopeLimiter limiter>
void HyperbolicSystem<problem_t>::ReconstructStatesPLM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf,
						       const int nghost, const int nvars)
{
	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, limiter>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR, SlopeLimiter limiter>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPLM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
									     amrex::Box const &indexRange, const int nvars)
{
	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, limiter>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR, SlopeLimiter limiter>
AMREX_GPU_HOST_DEVICE void
HyperbolicSystem<problem_t>::ReconstructStatesPLM(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState,
						  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in)
{
	// permute array indices according to dir
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	// Unlike PPM, PLM with Sweby limiters (sigma=1 minmod, sigma=1.5 Sweby, sigma=2 MC) is TVD.
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

	// Use piecewise-linear reconstruction
	// (This converges at second order in spatial resolution.)
	const auto lslope = HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(q(i, j, k, n) - q(i - 1, j, k, n), q(i - 1, j, k, n) - q(i - 2, j, k, n));
	const auto rslope = HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(q(i + 1, j, k, n) - q(i, j, k, n), q(i, j, k, n) - q(i - 1, j, k, n));
	leftState(i, j, k, n) = q(i - 1, j, k, n) + 0.5 * lslope; // NOLINT
	rightState(i, j, k, n) = q(i, j, k, n) - 0.5 * rslope;	  // NOLINT
}

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf,
						       const int nghost, const int nvars, const int iReadFrom, const int iWriteFrom)
{
	const BL_PROFILE("HyperbolicSystem::ReconstructStatesPPM(MultiFabs)");

	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(q, leftState, rightState, n, i_in, j_in, k_in, iReadFrom, iWriteFrom);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
									     amrex::Box const &cellRange, amrex::Box const & /*interfaceRange*/,
									     const int nvars, const int iReadFrom, const int iWriteFrom)
{
	const BL_PROFILE("HyperbolicSystem::ReconstructStatesPPM(Arrays)");

	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	// cell-centered kernel
	amrex::ParallelFor(cellRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(q, leftState, rightState, n, i_in, j_in, k_in, iReadFrom, iWriteFrom);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q,
									     quokka::Array4View<amrex::Real, DIR> const &leftState,
									     quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in,
									     int k_in, int iReadFrom, int iWriteFrom)
{
	// permute array indices according to dir
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at the left
	// edge of zone i, and xright_(i) is the "right"-side of the interface
	// at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// (2.) Constrain interfaces to lie between surrounding cell-averaged
	// values (equivalent to step 2b in Athena++ [ppm_simple.cpp]).
	// [See Eq. B8 of Mignone+ 2005.]

	// compute bounds from neighboring cell-averaged values along axis
	const std::pair<double, double> bounds = std::minmax({q(i, j, k, iReadFrom + n), q(i - 1, j, k, iReadFrom + n), q(i + 1, j, k, iReadFrom + n)});

	// get interfaces

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
	const double a_minus = (coef_1 * q(i, j, k, iReadFrom + n) + coef_2 * q(i + 1, j, k, iReadFrom + n)) +
			       (coef_1 * q(i - 1, j, k, iReadFrom + n) + coef_2 * q(i - 2, j, k, iReadFrom + n));
	const double a_plus = (coef_1 * q(i + 1, j, k, iReadFrom + n) + coef_2 * q(i + 2, j, k, iReadFrom + n)) +
			      (coef_1 * q(i, j, k, iReadFrom + n) + coef_2 * q(i - 1, j, k, iReadFrom + n));

	// left side of zone i
	double new_a_minus = clamp(a_minus, bounds.first, bounds.second);

	// right side of zone i
	double new_a_plus = clamp(a_plus, bounds.first, bounds.second);

	// (3.) Monotonicity correction, using Eq. (1.10) in PPM paper. Equivalent
	// to step 4b in Athena++ [ppm_simple.cpp].

	const double a = q(i, j, k, iReadFrom + n); // a_i in C&W
	const double dq_minus = (a - new_a_minus);
	const double dq_plus = (new_a_plus - a);

	const double qa = dq_plus * dq_minus; // interface extrema

	if (qa <= 0.0) { // local extremum

		// Causes subtle, but very weird, oscillations in the Shu-Osher test
		// problem. However, it is necessary to get a reasonable solution
		// for the sawtooth advection problem.
		const double dq0 = MC(q(i + 1, j, k, iReadFrom + n) - q(i, j, k, iReadFrom + n), q(i, j, k, iReadFrom + n) - q(i - 1, j, k, iReadFrom + n));

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
	rightState(i, j, k, iWriteFrom + n) = new_a_minus;
	leftState(i + 1, j, k, iWriteFrom + n) = new_a_plus;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::MonotonizeEdges(double qL_in, double qR_in, double q, double qminus, double qplus)
    -> std::pair<double, double>
{
	// compute monotone edge values
	const double qL_star = median(q, qL_in, qminus);
	const double qR_star = median(q, qR_in, qplus);

	// this does something weird to the left side of the sawtooth advection
	// problem, but is absolutely essential for stability in other problems
	const double qL = median(q, qL_star, 3. * q - 2. * qR_star);
	const double qR = median(q, qR_star, 3. * q - 2. * qL_star);

	return std::make_pair(qL, qR);
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::ComputeSteepPPM(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k,
										      int n) -> amrex::Real
{
	// compute steepened PPM stencil value
	double S = 0.5 * (q(i + 1, j, k, n) - q(i - 1, j, k, n));
	double Sp = 0.5 * (q(i + 2, j, k, n) - q(i, j, k, n));
	double S_M = 2. * MC(q(i + 1, j, k, n) - q(i, j, k, n), q(i, j, k, n) - q(i - 1, j, k, n));
	double Sp_M = 2. * MC(q(i + 2, j, k, n) - q(i + 1, j, k, n), q(i + 1, j, k, n) - q(i, j, k, n));
	S = median(0., S, S_M);
	Sp = median(0., Sp, Sp_M);

	return 0.5 * (q(i, j, k, n) + q(i + 1, j, k, n)) - (1. / 6.) * (Sp - S);
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::ComputeWENOMoments(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j,
											 int k, int n) -> std::pair<amrex::Real, amrex::Real>
{
	/// compute WENO-Z reconstruction following Balsara (2017).

	/// compute moments for each stencil
	// left-biased stencil
	const double sL_x = -2.0 * q(i - 1, j, k, n) + 0.5 * q(i - 2, j, k, n) + 1.5 * q(i, j, k, n);
	const double sL_xx = 0.5 * q(i - 2, j, k, n) - q(i - 1, j, k, n) + 0.5 * q(i, j, k, n);

	// centered stencil
	const double sC_x = 0.5 * (q(i + 1, j, k, n) - q(i - 1, j, k, n));
	const double sC_xx = 0.5 * q(i - 1, j, k, n) - q(i, j, k, n) + 0.5 * q(i + 1, j, k, n);

	// right-biased stencil
	const double sR_x = -1.5 * q(i, j, k, n) + 2.0 * q(i + 1, j, k, n) - 0.5 * q(i + 2, j, k, n);
	const double sR_xx = 0.5 * q(i, j, k, n) - q(i + 1, j, k, n) + 0.5 * q(i + 2, j, k, n);

	// compute smoothness indicators
	const double IS_L = sL_x * sL_x + (13. / 3.) * (sL_xx * sL_xx);
	const double IS_C = sC_x * sC_x + (13. / 3.) * (sC_xx * sC_xx);
	const double IS_R = sR_x * sR_x + (13. / 3.) * (sR_xx * sR_xx);

	// use WENO-Z smoothness indicators with *symmetric* linear weights
	// (1-2-3 problem fails with the [asymmetric] 'optimal' weights)
	const double q_mean = (std::abs(q(i - 1, j, k, n)) + std::abs(q(i, j, k, n)) + std::abs(q(i + 1, j, k, n))) / 3.0;
	const double eps = std::max(1.0e-40 * q_mean, 1.0e-40); // prevent underflow
	const double tau = std::abs(IS_L - IS_R);
	double wL = 0.2 * (1. + tau / (IS_L + eps));
	double wC = 0.6 * (1. + tau / (IS_C + eps));
	double wR = 0.2 * (1. + tau / (IS_R + eps));

	// normalise weights
	const double norm = wL + wC + wR;
	wL /= norm;
	wC /= norm;
	wR /= norm;

	// compute weighted moments
	const double q_x = wL * sL_x + wC * sC_x + wR * sR_x;
	const double q_xx = wL * sL_xx + wC * sC_xx + wR * sR_xx;

	return std::make_pair(q_x, q_xx);
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::ComputeWENO(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k,
										  int n) -> std::pair<amrex::Real, amrex::Real>
{
	/// compute WENO-Z reconstruction following Balsara (2017).
	auto [q_x, q_xx] = ComputeWENOMoments(q, i, j, k, n);

	// evaluate i-(1/2) and i+(1/2) values
	const double qL = q(i, j, k, n) - 0.5 * q_x + (0.25 - 1. / 12.) * q_xx;
	const double qR = q(i, j, k, n) + 0.5 * q_x + (0.25 - 1. / 12.) * q_xx;

	return std::make_pair(qL, qR);
}

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesPPM_EP(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf,
							  const int nghost, const int nvars, const int iReadFrom, const int iWriteFrom)
{
	const BL_PROFILE("HyperbolicSystem::ReconstructStatesPPM(MultiFabs)");

	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(q, leftState, rightState, n, i_in, j_in, k_in, iReadFrom, iWriteFrom);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM_EP(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
										amrex::Box const &cellRange, amrex::Box const & /*interfaceRange*/,
										const int nvars, const int iReadFrom, const int iWriteFrom)
{
	const BL_PROFILE("HyperbolicSystem::ReconstructStatesPPM(Arrays)");

	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	// cell-centered kernel
	amrex::ParallelFor(cellRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(q, leftState, rightState, n, i_in, j_in, k_in, iReadFrom, iWriteFrom);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void
HyperbolicSystem<problem_t>::ReconstructStatesPPM_EP(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState,
						     quokka::Array4View<amrex::Real, DIR> const &rightState, const int n, const int i_in, const int j_in,
						     const int k_in, const int iReadFrom, const int iWriteFrom)
{
	/// Extrema-preserving hybrid PPM-WENO from Rider, Greenough & Kamm (2007).

	// permute array indices according to dir
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	// 0. 5-point interface-centered stencil (Suresh & Huynh, JCP 136, 83-99, 1997)
	const double c1 = 2. / 60.;
	const double c2 = -13. / 60.;
	const double c3 = 47. / 60.;
	const double c4 = 27. / 60.;
	const double c5 = -3. / 60.;

	const double a_minus = c1 * q(i + 2, j, k, iReadFrom + n) + c2 * q(i + 1, j, k, iReadFrom + n) + c3 * q(i, j, k, iReadFrom + n) +
			       c4 * q(i - 1, j, k, iReadFrom + n) + c5 * q(i - 2, j, k, iReadFrom + n);
	const double a_plus = c1 * q(i - 2, j, k, iReadFrom + n) + c2 * q(i - 1, j, k, iReadFrom + n) + c3 * q(i, j, k, iReadFrom + n) +
			      c4 * q(i + 1, j, k, iReadFrom + n) + c5 * q(i + 2, j, k, iReadFrom + n);

	// save neighboring values
	const double a = q(i, j, k, iReadFrom + n);
	const double am = q(i - 1, j, k, iReadFrom + n);
	const double ap = q(i + 1, j, k, iReadFrom + n);

	// 1. monotonize
	auto [new_a_minus, new_a_plus] = MonotonizeEdges(a_minus, a_plus, a, am, ap);

	// 2. check whether limiter was triggered on either side
	const double q_mean = (std::abs(q(i - 1, j, k, iReadFrom + n)) + std::abs(q(i, j, k, iReadFrom + n)) + std::abs(q(i + 1, j, k, iReadFrom + n))) / 3.0;
	const double eps = 1.0e-14 * q_mean;

	if (std::abs(new_a_minus - a_minus) > eps || std::abs(new_a_plus - a_plus) > eps) {

		// compute symmetric WENO-Z reconstruction
		auto [a_minus_weno, a_plus_weno] = ComputeWENO(q, i, j, k, iReadFrom + n);

		if (new_a_minus == a || new_a_plus == a) {
			// 3. to avoid clipping at extrema, use WENO value
			a_minus_weno = median(a, a_minus_weno, a_minus);
			a_plus_weno = median(a, a_plus_weno, a_plus);

			auto [a_minus_mweno, a_plus_mweno] = MonotonizeEdges(a_minus_weno, a_plus_weno, a, am, ap);

			new_a_minus = median(a_minus_weno, a_minus_mweno, a_minus);
			new_a_plus = median(a_plus_weno, a_plus_mweno, a_plus);
		} else {
			// 4. gradient is too steep, use one-sided 4th-order PPM stencil
			double a_minus_ppm = ComputeSteepPPM(q, i - 1, j, k, iReadFrom + n);
			double a_plus_ppm = ComputeSteepPPM(q, i, j, k, iReadFrom + n);

			a_minus_ppm = median(a_minus_weno, a_minus_ppm, a_minus);
			a_plus_ppm = median(a_plus_weno, a_plus_ppm, a_plus);

			auto [a_minus_mppm, a_plus_mppm] = MonotonizeEdges(a_minus_ppm, a_plus_ppm, a, am, ap);

			new_a_minus = median(a_minus_mppm, a_minus_weno, a_minus);
			new_a_plus = median(a_plus_mppm, a_plus_weno, a_plus);
		}
	}

	rightState(i, j, k, iWriteFrom + n) = new_a_minus;
	leftState(i + 1, j, k, iWriteFrom + n) = new_a_plus;
}

template <typename problem_t>
template <typename F>
void HyperbolicSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					      const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
					      const int nvars, F &&isStateValid, amrex::Array4<int> const &redoFlag)
{
	const BL_PROFILE("HyperbolicSystem::PredictStep()");

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

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			consVarNew(i, j, k, n) = consVarOld(i, j, k, n) + (AMREX_D_TERM((dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
											+(dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
											+(dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))));
		}

		// check if state is valid -- flag for re-do if not
		if (!std::forward<F>(isStateValid)(consVarNew, i, j, k)) {
			redoFlag(i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag(i, j, k) = quokka::redoFlag::none;
		}
	});
}

template <typename problem_t>
template <typename F>
void HyperbolicSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					       const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
					       const int nvars, F &&isStateValid, amrex::Array4<int> const &redoFlag)
{
	const BL_PROFILE("HyperbolicSystem::AddFluxesRK2()");

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

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
			U_new(i, j, k, n) = (0.5 * U_0 + 0.5 * U_1) + (AMREX_D_TERM(0.5 * FxU_1, +0.5 * FyU_1, +0.5 * FzU_1));
		}

		// check if state is valid -- flag for re-do if not
		if (!std::forward<F>(isStateValid)(U_new, i, j, k)) {
			redoFlag(i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag(i, j, k) = quokka::redoFlag::none;
		}
	});
}

#endif // HYPERBOLIC_SYSTEM_HPP_
