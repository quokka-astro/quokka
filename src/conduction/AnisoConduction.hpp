#ifndef ANISO_CONDUCTION_HPP_ // NOLINT
#define ANISO_CONDUCTION_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file AnisoConduction.hpp
/// \brief Explicit thermal conduction update using kappa_parallel (no magnetic field), computed
///        identically to ElectronConduction::ComputeExplicit. kappa_perp is retained in
///        AnisoConductionParams but not yet used by the flux calculation.

#include <array>
#include <cmath>
#include <limits>

#include "AMReX_Array4.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include "hyperbolic_system.hpp"

namespace quokka::conduction
{

struct AnisoConductionParams {
	amrex::Real kappa_parallel = 0.0;   // conductivity applied to the face-tangential (in-plane) gradient, units erg cm^-1 s^-1 K^-1
	amrex::Real kappa_perp = 0.0;	     // conductivity applied to the face-normal gradient, units erg cm^-1 s^-1 K^-1
	amrex::Real flux_limiter_phi = 0.1;
	amrex::Real saturation_factor = 5.0; // refer to equation 8 of Cowie & McKee 1977
	amrex::Real min_temperature = 0.0;   // default value will be overwritten by tempFloor_ during initialization
	int reconstruction_order = 3;	     // 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM;
	SlopeLimiter plm_limiter = SlopeLimiter::sweby;
	int ng_reconstruct = 2; // number of ghost faces to reconstruct beyond the valid box
};

template <typename problem_t> class AnisoConduction
{
      public:
	// Reconstruct rho and T at the interfaces (identical to ElectronConduction::ReconstructPrimVar)
	template <FluxDir DIR>
	static void ReconstructPrimVar(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int ng_reconstruct,
				       AnisoConductionParams const &params)
	{
		constexpr int nvars = 2;
		if (params.reconstruction_order == 5) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
		} else if (params.reconstruction_order == 3) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
		} else if (params.reconstruction_order == 2) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars,
											params.plm_limiter);
		} else if (params.reconstruction_order == 1) {
			HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
		} else {
			amrex::Abort("Invalid reconstruction order specified for anisotropic conduction!");
		}
	}

	static void ComputeExplicit(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    amrex::Real dt, AnisoConductionParams const &params, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		static_assert(Physics_Traits<problem_t>::is_hydro_enabled, "Anisotropic conduction requires hydro to be enabled.");

		if ((dt <= 0.0) || (params.kappa_parallel <= 0.0)) {
			return;
		}

		if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
			amrex::ignore_unused(geom, params);
			return;
		}

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.nGrow() >= 1, "Anisotropic conduction requires at least 1 ghost cell.");

		const auto dx = geom.CellSizeArray();
		const amrex::Real flux_limiter_phi = params.flux_limiter_phi;
		const amrex::Real saturation_factor = params.saturation_factor;
		const amrex::Real t_min = params.min_temperature;
		const amrex::Real kappa_par = params.kappa_parallel;
		const amrex::Real small = std::numeric_limits<amrex::Real>::min();
		constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;

		amrex::MultiFab primVar(state.boxArray(), state.DistributionMap(), 2, state.nGrow());
		primVar.setVal(0.0);

		// Cell-centered B, averaged from the two bounding faces per component (same convention as
		// ComputeCellCenteredMagneticEnergy in physics_info.hpp).
		amrex::MultiFab Bcc(state.boxArray(), state.DistributionMap(), AMREX_SPACEDIM, state.nGrow());
		Bcc.setVal(0.0);

		auto const &state_x0 = state.const_arrays();
		auto primVar_arr = primVar.arrays();
		auto Bcc_arr = Bcc.arrays();
		amrex::IntVect const ng = amrex::IntVect(AMREX_D_DECL(state.nGrow(), state.nGrow(), state.nGrow()));

		// Per-box face-centered B, gathered into an array so it can be handed to
		// HydroSystem<problem_t>::ComputeInternalEnergy/ComputeMagneticEnergy below.
		auto const &state_fc_x0 = state_fc[0].const_arrays();
#if AMREX_SPACEDIM >= 2
		auto const &state_fc_x1 = state_fc[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
		auto const &state_fc_x2 = state_fc[2].const_arrays();
#endif

		amrex::ParallelFor(state, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_state_fc{};
			amrex::ignore_unused(state_fc_x0
#if AMREX_SPACEDIM >= 2
					     ,
					     state_fc_x1
#endif
#if AMREX_SPACEDIM == 3
					     ,
					     state_fc_x2
#endif
			);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				local_state_fc[0] = state_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
				local_state_fc[1] = state_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
				local_state_fc[2] = state_fc_x2[bx];
#endif

				constexpr int mhdIdx = Physics_Indices<problem_t>::mhdFirstIndex;
				Bcc_arr[bx](i, j, k, 0) = 0.5 * (local_state_fc[0](i, j, k, mhdIdx) + local_state_fc[0](i + 1, j, k, mhdIdx));
#if AMREX_SPACEDIM >= 2
				Bcc_arr[bx](i, j, k, 1) = 0.5 * (local_state_fc[1](i, j, k, mhdIdx) + local_state_fc[1](i, j + 1, k, mhdIdx));
#endif
#if AMREX_SPACEDIM == 3
				Bcc_arr[bx](i, j, k, 2) = 0.5 * (local_state_fc[2](i, j, k, mhdIdx) + local_state_fc[2](i, j, k + 1, mhdIdx));
#endif
			} else {
				for (int n = 0; n < AMREX_SPACEDIM; ++n) {
					Bcc_arr[bx](i, j, k, n) = 0.0;
				}
			}
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

		// Same evaluateFace structure as ElectronConduction::ComputeExplicit, but kappa_face is
		// simply kappa_parallel (a fixed constant here, rather than a Spitzer-scaled function of T_face).
		auto const evaluateFace = [=] AMREX_GPU_DEVICE(amrex::Real rho_L, amrex::Real T_L, amrex::Real rho_R, amrex::Real T_R,
							       amrex::GpuArray<amrex::Real, nmscalars_> const &massScalars_L,
							       amrex::GpuArray<amrex::Real, nmscalars_> const &massScalars_R, amrex::Real &kappa_face,
							       amrex::Real &qsat_face) noexcept {
			const amrex::Real rho_face = 0.5 * (rho_L + rho_R);
			const amrex::Real T_face = amrex::max(0.5 * (T_L + T_R), t_min);
			amrex::GpuArray<amrex::Real, nmscalars_> massArray_face{};
			for (int n = 0; n < nmscalars_; ++n) {
				massArray_face[n] = 0.5 * (massScalars_L[n] + massScalars_R[n]);
			}
			quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = massArray_face;
			const amrex::Real Eint_face = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho_face, T_face, massScalars);
			const amrex::Real Pgas_face = ::quokka::EOS<problem_t>::ComputePressure(rho_face, Eint_face, massScalars);
			const amrex::Real cs_face = ::quokka::EOS<problem_t>::ComputeSoundSpeed(rho_face, Pgas_face, massScalars);

			kappa_face = kappa_par;
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
			auto const massScalars_L = RadSystem<problem_t>::ComputeMassScalars(state_x0[bx], i - 1, j, k);
			auto const massScalars_R = RadSystem<problem_t>::ComputeMassScalars(state_x0[bx], i, j, k);
			evaluateFace(left_x[bx](i, j, k, 0), left_x[bx](i, j, k, 1), right_x[bx](i, j, k, 0), right_x[bx](i, j, k, 1), massScalars_L,
				     massScalars_R, kappa_face, q_sat_face);
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
			auto const massScalars_L = RadSystem<problem_t>::ComputeMassScalars(state_x0[bx], i, j - 1, k);
			auto const massScalars_R = RadSystem<problem_t>::ComputeMassScalars(state_x0[bx], i, j, k);
			evaluateFace(left_y[bx](i, j, k, 0), left_y[bx](i, j, k, 1), right_y[bx](i, j, k, 0), right_y[bx](i, j, k, 1), massScalars_L,
				     massScalars_R, kappa_face, q_sat_face);
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
			auto const massScalars_L = RadSystem<problem_t>::ComputeMassScalars(state_x0[bx], i, j, k - 1);
			auto const massScalars_R = RadSystem<problem_t>::ComputeMassScalars(state_x0[bx], i, j, k);
			evaluateFace(left_z[bx](i, j, k, 0), left_z[bx](i, j, k, 1), right_z[bx](i, j, k, 0), right_z[bx](i, j, k, 1), massScalars_L,
				     massScalars_R, kappa_face, q_sat_face);
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
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
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
			const amrex::Real Eint_old = state_out[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index);
			const amrex::Real Emag = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &local_state_fc);
			amrex::Real div_flux = (flux_x_const[bx](i + 1, j, k) - flux_x_const[bx](i, j, k)) / dx[0];
#if AMREX_SPACEDIM >= 2
			div_flux += (flux_y_const[bx](i, j + 1, k) - flux_y_const[bx](i, j, k)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
			div_flux += (flux_z_const[bx](i, j, k + 1) - flux_z_const[bx](i, j, k)) / dx[2];
#endif

			amrex::Real const Eint_new = Eint_old - dt * div_flux;

			state_out[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Eint_new + Ekin + Emag;
			state_out[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index) = Eint_new;
		});
	}
};

} // namespace quokka::conduction

#endif // ANISO_CONDUCTION_HPP_
