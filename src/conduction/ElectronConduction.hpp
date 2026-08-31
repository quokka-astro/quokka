#ifndef ELECTRON_CONDUCTION_HPP_ // NOLINT
#define ELECTRON_CONDUCTION_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ElectronConduction.hpp
/// \brief Explicit flux-limited electron thermal conduction update.

#include <array>
#include <cmath>
#include <limits>

#include "AMReX_Array4.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "hydro/hydro_system.hpp"
#include "hyperbolic_system.hpp"

namespace quokka::conduction
{

struct ElectronConductionParams {
	amrex::Real conductivity_prefactor = 3.e34; // units of erg cm^-1 s^-1 K^-1
	amrex::Real flux_limiter_phi = 0.1;
	amrex::Real saturation_factor = 5.0; // refer to equation 8 of Cowie & McKee 1977
	amrex::Real min_temperature = 0.0;   // default value will be overwritten by tempFloor_ during initialization
	bool spitzer_scaling = true;	      // if true, kappa(T) = conductivity_prefactor * T^2.5 (Spitzer);
					      // if false, kappa(T) = conductivity_prefactor (constant, isotropic)
	int reconstruction_order = 3;	      // 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM;
					      // mirrors the hydro solver's reconstruction_order/plm_limiter so that the
					      // (rho, T) states used to evaluate the face conductivity come from the
					      // same reconstruction as the rest of the hydro update.
	SlopeLimiter plm_limiter = SlopeLimiter::sweby;
	int ng_reconstruct = 2;	      // number of ghost faces to reconstruct beyond the valid box; must match
					      // the hydro solver's own reconstructGhost (nghost_Riemann + 1, see
					      // QuokkaSimulation::computeHydroFluxes) so the (rho, T) reconstruction gets
					      // the same stencil robustness as the hydro reconstruction it mirrors.
};

template <typename problem_t> class ElectronConduction
{
      public:
	// Dispatches to the reconstruction scheme selected by params.reconstruction_order/plm_limiter
	// (mirroring HyperbolicSystem's dispatch used for the hydro fluxes), reconstructing the 2-component
	// (rho, T) MultiFab primVar to left/right interface states in the DIR direction.
	template <FluxDir DIR>
	static void ReconstructPrimVar(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int ng_reconstruct,
					ElectronConductionParams const &params)
	{
		constexpr int nvars = 2;
		if (params.reconstruction_order == 5) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
		} else if (params.reconstruction_order == 3) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
		} else if (params.reconstruction_order == 2) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars, params.plm_limiter);
		} else if (params.reconstruction_order == 1) {
			HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
		} else {
			amrex::Abort("Invalid reconstruction order specified for electron conduction!");
		}
	}

	// Sound speed always comes from quokka::EOS (the fixed-mu ideal-gas formula, even for the
	// EOSTabulated backend), matching how hydro itself computes pressure/sound speed for every
	// problem — only temperature is actually table-driven. See EOSTabulated in hydro/EOS.hpp.
	static void ComputeExplicit(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    amrex::Real dt, ElectronConductionParams const &params, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		static_assert(Physics_Traits<problem_t>::is_hydro_enabled, "Electron conduction requires hydro to be enabled.");

		if ((dt <= 0.0) || (params.conductivity_prefactor <= 0.0)) {
			return;
		}

		if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
			amrex::ignore_unused(state_fc, geom, params);
			return;
		}

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.nGrow() >= 1, "Electron conduction requires at least 1 ghost cell.");

		const auto dx = geom.CellSizeArray();
		const amrex::Real flux_limiter_phi = params.flux_limiter_phi;
		const amrex::Real saturation_factor = params.saturation_factor;
		const amrex::Real t_min = params.min_temperature;
		const bool spitzer_scaling = params.spitzer_scaling;
		const amrex::Real kappa0 = params.conductivity_prefactor;
		const amrex::Real small = std::numeric_limits<amrex::Real>::min();
		constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;

		// Cell-centered (density, temperature); component 0 = rho, component 1 = T.
		// This is reconstructed to interfaces below (using the same reconstruction order/limiter
		// as the hydro solver) so that the face conductivity is evaluated from interface states
		// rather than from an average of the two neighboring cell-centered values.
		amrex::MultiFab primVar(state.boxArray(), state.DistributionMap(), 2, state.nGrow());
		primVar.setVal(0.0);

		auto const &state_x0 = state.const_arrays();
		auto const &state_fc_x0 = state_fc[0].const_arrays();
#if AMREX_SPACEDIM >= 2
		auto const &state_fc_x1 = state_fc[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
		auto const &state_fc_x2 = state_fc[2].const_arrays();
#endif
		auto primVar_arr = primVar.arrays();
		amrex::IntVect ng = amrex::IntVect(AMREX_D_DECL(state.nGrow(), state.nGrow(), state.nGrow()));

		amrex::ParallelFor(state, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_state_fc{};
			if (Physics_Traits<problem_t>::is_mhd_enabled) {
				local_state_fc[0] = state_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
				local_state_fc[1] = state_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
				local_state_fc[2] = state_fc_x2[bx];
#endif
			}

			auto const &cons = state_x0[bx];
			const amrex::Real rho = cons(i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real Eint = HydroSystem<problem_t>::ComputeInternalEnergy(cons, i, j, k, &local_state_fc);
			// Temperature always from EOS
			quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = {};
			const amrex::Real Tgas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);

			primVar_arr[bx](i, j, k, 0) = rho;
			primVar_arr[bx](i, j, k, 1) = amrex::max(Tgas, t_min);
		});

		// Reconstruct (rho, T) to the interfaces
		std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;
		const int ng_reconstruct = params.ng_reconstruct;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			leftState[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 2, ng_reconstruct);
			rightState[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 2, ng_reconstruct);
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}

		AMREX_D_TERM(ReconstructPrimVar<FluxDir::X1>(primVar, leftState[0], rightState[0], ng_reconstruct, params);
			     , ReconstructPrimVar<FluxDir::X2>(primVar, leftState[1], rightState[1], ng_reconstruct, params);
			     , ReconstructPrimVar<FluxDir::X3>(primVar, leftState[2], rightState[2], ng_reconstruct, params);)

		// Given left/right interface (rho, T) states, evaluate a single consistent face conductivity
		// and saturated flux: average rho and T across the interface first, then evaluate kappa(T),
		// P(rho, T), c_s(rho, P) at that single face state.
		auto const evaluateFace = [=] AMREX_GPU_DEVICE(amrex::Real rho_L, amrex::Real T_L, amrex::Real rho_R, amrex::Real T_R,
								amrex::Real &kappa_face, amrex::Real &qsat_face) noexcept {
			const amrex::Real rho_face = 0.5 * (rho_L + rho_R);
			const amrex::Real T_face = amrex::max(0.5 * (T_L + T_R), t_min);
			quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = {};
			const amrex::Real Eint_face = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho_face, T_face, massScalars);
			// Sound speed always from EOS (see comment on ComputeExplicit above)
			const amrex::Real Pgas_face = ::quokka::EOS<problem_t>::ComputePressure(rho_face, Eint_face, massScalars);
			const amrex::Real cs_face = ::quokka::EOS<problem_t>::ComputeSoundSpeed(rho_face, Pgas_face, massScalars);

			kappa_face = spitzer_scaling ? (kappa0 * std::pow(T_face, 2.5)) : kappa0;
			qsat_face = amrex::max(saturation_factor * flux_limiter_phi * rho_face * cs_face * cs_face * cs_face, small);
		};

		auto const &temp = primVar.const_arrays();
		auto const &left_x = leftState[0].const_arrays();
		auto const &right_x = rightState[0].const_arrays();
		auto flux_x = heat_flux[0].arrays();
		amrex::ParallelFor(heat_flux[0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real gradT = (temp[bx](i, j, k, 1) - temp[bx](i - 1, j, k, 1)) / dx[0];
			amrex::Real kappa_face = 0.0;
			amrex::Real q_sat_face = 0.0;
			evaluateFace(left_x[bx](i, j, k, 0), left_x[bx](i, j, k, 1), right_x[bx](i, j, k, 0), right_x[bx](i, j, k, 1), kappa_face, q_sat_face);
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_x[bx](i, j, k) = q_classical / limiter;
		});

#if AMREX_SPACEDIM >= 2
		auto const &left_y = leftState[1].const_arrays();
		auto const &right_y = rightState[1].const_arrays();
		auto flux_y = heat_flux[1].arrays();
		amrex::ParallelFor(heat_flux[1], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real gradT = (temp[bx](i, j, k, 1) - temp[bx](i, j - 1, k, 1)) / dx[1];
			amrex::Real kappa_face = 0.0;
			amrex::Real q_sat_face = 0.0;
			evaluateFace(left_y[bx](i, j, k, 0), left_y[bx](i, j, k, 1), right_y[bx](i, j, k, 0), right_y[bx](i, j, k, 1), kappa_face, q_sat_face);
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_y[bx](i, j, k) = q_classical / limiter;
		});
#endif

#if AMREX_SPACEDIM == 3
		auto const &left_z = leftState[2].const_arrays();
		auto const &right_z = rightState[2].const_arrays();
		auto flux_z = heat_flux[2].arrays();
		amrex::ParallelFor(heat_flux[2], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real gradT = (temp[bx](i, j, k, 1) - temp[bx](i, j, k - 1, 1)) / dx[2];
			amrex::Real kappa_face = 0.0;
			amrex::Real q_sat_face = 0.0;
			evaluateFace(left_z[bx](i, j, k, 0), left_z[bx](i, j, k, 1), right_z[bx](i, j, k, 0), right_z[bx](i, j, k, 1), kappa_face, q_sat_face);
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_z[bx](i, j, k) = q_classical / limiter;
		});
#endif

		auto state_out = state.arrays();
		auto const &flux_x_const = heat_flux[0].const_arrays();
#if AMREX_SPACEDIM >= 2
		auto const &flux_y_const = heat_flux[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
		auto const &flux_z_const = heat_flux[2].const_arrays();
#endif

		amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_state_fc{};
			if (Physics_Traits<problem_t>::is_mhd_enabled) {
				local_state_fc[0] = state_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
				local_state_fc[1] = state_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
				local_state_fc[2] = state_fc_x2[bx];
#endif
			}

			const amrex::Real rho = state_out[bx](i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real px = state_out[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const amrex::Real py = state_out[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const amrex::Real pz = state_out[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);

			const amrex::Real Ekin = 0.5 * (px * px + py * py + pz * pz) / rho;
			const amrex::Real Eint_old = HydroSystem<problem_t>::ComputeInternalEnergy(state_out[bx], i, j, k, &local_state_fc);
			const amrex::Real Emag = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &local_state_fc);
			amrex::Real div_flux = (flux_x_const[bx](i + 1, j, k) - flux_x_const[bx](i, j, k)) / dx[0];
#if AMREX_SPACEDIM >= 2
			div_flux += (flux_y_const[bx](i, j + 1, k) - flux_y_const[bx](i, j, k)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
			div_flux += (flux_z_const[bx](i, j, k + 1) - flux_z_const[bx](i, j, k)) / dx[2];
#endif

			amrex::Real Eint_new = Eint_old - dt * div_flux;

			state_out[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Eint_new + Ekin + Emag;
			state_out[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index) = Eint_new;
		});
	}
};

} // namespace quokka::conduction

#endif // ELECTRON_CONDUCTION_HPP_
