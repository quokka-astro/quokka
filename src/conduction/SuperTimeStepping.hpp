#ifndef SUPER_TIME_STEPPING_HPP_ // NOLINT
#define SUPER_TIME_STEPPING_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file SuperTimeStepping.hpp
/// \brief RKL2 super time stepping for thermal conduction (Meyer, Balsara & Aslam 2012, MNRAS 422, 2102).
///
/// An s-stage RKL2 step advances the internal energy over dt with
///   Y_1 = Y_0 + mu~_1 dt L(Y_0),
///   Y_j = mu_j Y_{j-1} + nu_j Y_{j-2} + (1 - mu_j - nu_j) Y_0 + mu~_j dt L(Y_{j-1}) + gamma~_j dt L(Y_0),  j = 2..s,
/// where L(Y) = -div q(Y). It is stable for dt <= dt_expl (s^2 + s - 2) / 4, with dt_expl the forward-Euler limit.
/// Since div is linear and the recursion coefficients sum to one, Y_j = Y_0 - dt div F_j exactly, where
///   F_1 = mu~_1 q(Y_0),
///   F_j = mu_j F_{j-1} + nu_j F_{j-2} + mu~_j q(Y_{j-1}) + gamma~_j q(Y_0),
/// so AdvanceRKL2 evolves the face fluxes F_j instead of the stage states. Each stage is then applied with
/// ApplyConductionUpdate, and F_s is the time-averaged flux to hand to the flux registers for refluxing.

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <utility>

#include "AMReX.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "conduction/ConductionHelpers.hpp"

namespace quokka::conduction
{

/// \brief Runtime parameters for RKL2 super time stepping.
struct STSParams {
	int max_stages = 51;	  // the timestep is capped so that a conduction step needs at most this many RKL2 stages
	amrex::Real safety = 0.9; // dt_expl = safety * dx_min^2 / (2 * AMREX_SPACEDIM * D_max)
	// fraction of the max_stages limit used when capping the timestep, leaving room for dt_expl to drop
	// between the timestep estimate and the second Strang half-step (after the hydro update)
	amrex::Real dt_cap_fraction = 0.8;
};

/// \brief RKL2 recursion coefficients for stages j = 1..s (index 0 is unused). Only mu_tilde[1] is used for j = 1.
struct RKL2Coefficients {
	amrex::Vector<amrex::Real> mu;
	amrex::Vector<amrex::Real> nu;
	amrex::Vector<amrex::Real> mu_tilde;
	amrex::Vector<amrex::Real> gamma_tilde;
};

/// \brief Returns the smallest odd number of RKL2 stages s >= 3 with dt <= dt_expl (s^2 + s - 2) / 4
///        (Meyer et al. 2012, Eq. 26).
inline auto ComputeRKL2Stages(amrex::Real dt, amrex::Real dt_expl) -> int
{
	const amrex::Real ratio = dt / dt_expl;
	int s = static_cast<int>(std::ceil(0.5 * (-1.0 + std::sqrt(9.0 + 16.0 * ratio))));
	s = std::max(s, 3);
	if (s % 2 == 0) {
		++s;
	}
	return s;
}

/// \brief Returns the largest dt / dt_expl for which ComputeRKL2Stages returns at most \p max_stages stages,
///        i.e. (s^2 + s - 2) / 4 for the largest odd s <= max_stages.
inline auto RKL2MaxTimestepRatio(int max_stages) -> amrex::Real
{
	const int s = (max_stages % 2 == 0) ? (max_stages - 1) : max_stages;
	const auto sr = static_cast<amrex::Real>(s);
	return 0.25 * (sr * sr + sr - 2.0);
}

/// \brief Returns the RKL2 coefficients for an \p s-stage step (Meyer et al. 2012, Sec. 3). With
///        b_0 = b_1 = b_2 = 1/3, b_j = (j^2 + j - 2) / (2 j (j + 1)), a_j = 1 - b_j and w_1 = 4 / (s^2 + s - 2):
///        mu~_1 = b_1 w_1, and for j >= 2, mu_j = (2j - 1)/j b_j/b_{j-1}, nu_j = -(j - 1)/j b_j/b_{j-2},
///        mu~_j = mu_j w_1, gamma~_j = -a_{j-1} mu~_j.
inline auto ComputeRKL2Coefficients(int s) -> RKL2Coefficients
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s >= 2, "RKL2 requires at least 2 stages.");

	auto const b = [](int j) -> amrex::Real {
		if (j <= 2) {
			return 1.0 / 3.0;
		}
		const auto jr = static_cast<amrex::Real>(j);
		return (jr * jr + jr - 2.0) / (2.0 * jr * (jr + 1.0));
	};

	const auto sr = static_cast<amrex::Real>(s);
	const amrex::Real w1 = 4.0 / (sr * sr + sr - 2.0);

	RKL2Coefficients coeffs;
	coeffs.mu.resize(s + 1, 0.0);
	coeffs.nu.resize(s + 1, 0.0);
	coeffs.mu_tilde.resize(s + 1, 0.0);
	coeffs.gamma_tilde.resize(s + 1, 0.0);

	coeffs.mu_tilde[1] = b(1) * w1;
	for (int j = 2; j <= s; ++j) {
		const auto jr = static_cast<amrex::Real>(j);
		coeffs.mu[j] = (2.0 * jr - 1.0) / jr * b(j) / b(j - 1);
		coeffs.nu[j] = -(jr - 1.0) / jr * b(j) / b(j - 2);
		coeffs.mu_tilde[j] = coeffs.mu[j] * w1;
		coeffs.gamma_tilde[j] = -(1.0 - b(j - 1)) * coeffs.mu_tilde[j];
	}
	return coeffs;
}

/// \brief Advances the internal energy of \p state over \p dt with RKL2 super time stepping, or with a single
///        forward-Euler step when \p dt <= \p dt_expl. Only the internal (and hence total) energy changes.
///
/// \param state       conserved state; its ghost cells must be filled on entry. On return, only valid cells are updated.
/// \param state_fc    face-centered state (MHD); not modified, so its ghost cells need only be filled once by the caller.
/// \param dt          time step to advance over (> 0).
/// \param dt_expl     forward-Euler stability limit for \p state (see ComputeConductionTimestep).
/// \param max_stages  aborts if the step would need more stages than this.
/// \param computeFlux callable `computeFlux(amrex::MultiFab const &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> &q)` that
///                    (re)defines q and fills it with the conductive heat flux of \p state on every valid face.
/// \param fillGhost   callable `fillGhost(amrex::MultiFab &state)` that fills the ghost cells of \p state.
/// \param heat_flux   on return, the flux F_s such that the update was Eint -= dt * div(F_s), for refluxing.
/// \return the number of stages taken (1 for the forward-Euler fallback).
template <typename problem_t, typename ComputeFluxFn, typename FillGhostFn>
auto AdvanceRKL2(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom, amrex::Real dt,
		 amrex::Real dt_expl, int max_stages, ComputeFluxFn const &computeFlux, FillGhostFn const &fillGhost,
		 std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux) -> int
{
	const auto dx = geom.CellSizeArray();

	if (dt <= dt_expl) {
		computeFlux(state, heat_flux);
		ApplyConductionUpdate<problem_t>(state, state_fc, heat_flux, dx, dt);
		return 1;
	}

	const int s = ComputeRKL2Stages(dt, dt_expl);
	if (s > max_stages) {
		amrex::Abort(std::format("RKL2 conduction step needs {} stages (dt / dt_expl = {:e}), more than conduction.sts_max_stages = {}.", s,
					 dt / dt_expl, max_stages));
	}
	const RKL2Coefficients coeffs = ComputeRKL2Coefficients(s);

	// Y_0, restored before applying each stage's flux
	amrex::MultiFab Eint0(state.boxArray(), state.DistributionMap(), 1, 0);
	ExtractInternalEnergy<problem_t>(state, state_fc, Eint0);

	std::array<amrex::MultiFab, AMREX_SPACEDIM> q0;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> q;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> F_jm2;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> F_jm1;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> F_j;

	computeFlux(state, q0);
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		F_jm2[idim].define(q0[idim].boxArray(), q0[idim].DistributionMap(), 1, 0);
		F_jm1[idim].define(q0[idim].boxArray(), q0[idim].DistributionMap(), 1, 0);
		F_j[idim].define(q0[idim].boxArray(), q0[idim].DistributionMap(), 1, 0);
		F_jm2[idim].setVal(0.0); // F_0 = 0
		amrex::MultiFab::Copy(F_jm1[idim], q0[idim], 0, 0, 1, 0);
		F_jm1[idim].mult(coeffs.mu_tilde[1]); // F_1 = mu~_1 q(Y_0)
	}

	// Y_j = Y_0 - dt div F_j
	auto const applyStage = [&](std::array<amrex::MultiFab, AMREX_SPACEDIM> const &F) {
		SetInternalEnergy<problem_t>(state, state_fc, Eint0);
		ApplyConductionUpdate<problem_t>(state, state_fc, F, dx, dt);
	};
	applyStage(F_jm1);

	for (int j = 2; j <= s; ++j) {
		fillGhost(state);
		computeFlux(state, q); // q(Y_{j-1})
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::MultiFab::LinComb(F_j[idim], coeffs.mu[j], F_jm1[idim], 0, coeffs.nu[j], F_jm2[idim], 0, 0, 1, 0);
			amrex::MultiFab::Saxpy(F_j[idim], coeffs.mu_tilde[j], q[idim], 0, 0, 1, 0);
			amrex::MultiFab::Saxpy(F_j[idim], coeffs.gamma_tilde[j], q0[idim], 0, 0, 1, 0);
		}
		applyStage(F_j);
		// rotate so that F_jm1 = F_j and F_jm2 = old F_jm1; F_j becomes scratch
		std::swap(F_jm2, F_jm1);
		std::swap(F_jm1, F_j);
	}

	std::swap(heat_flux, F_jm1); // F_s
	return s;
}

} // namespace quokka::conduction

#endif // SUPER_TIME_STEPPING_HPP_
