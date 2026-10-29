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
	bool spitzer_scaling = true;	     // if true, kappa(T) = conductivity_prefactor * T^2.5 (Spitzer);
					     // if false, kappa(T) = conductivity_prefactor (constant, isotropic)
	int reconstruction_order = 3;	     // 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM;
	SlopeLimiter plm_limiter = SlopeLimiter::sweby;
	int ng_reconstruct = 2; // number of ghost faces to reconstruct beyond the valid box
};

template <typename problem_t> class ElectronConduction
{
      public:
	// Reconstruct rho, T, and mass fractions at the interfaces
	template <FluxDir DIR>
	static void ReconstructPrimVar(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int ng_reconstruct,
				       ElectronConductionParams const &params)
	{
		constexpr int nvars = 2 + Physics_Traits<problem_t>::numMassScalars;
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
			amrex::Abort("Invalid reconstruction order specified for electron conduction!");
		}
	}

	/// \brief Face value at i-1/2 from the four cells i-2, i-1, i, i+1: the centred 4th-order interpolant
	/// (7(q_{i-1} + q_i) - (q_{i-2} + q_{i+1}))/12 (the unlimited PPM interface value), clamped to the range of the two
	/// adjacent cells so it never overshoots at discontinuities.
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto FaceValue4(amrex::Real q_mm, amrex::Real q_m, amrex::Real q_p, amrex::Real q_pp) -> amrex::Real
	{
		const amrex::Real q_face = (7.0 * (q_m + q_p) - (q_mm + q_pp)) / 12.0;
		return amrex::Clamp(q_face, amrex::min(q_m, q_p), amrex::max(q_m, q_p));
	}

	static void ComputeExplicit(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    amrex::Real dt, ElectronConductionParams const &params, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		if ((dt <= 0.0) || (params.conductivity_prefactor <= 0.0)) {
			return;
		}

		if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
			amrex::ignore_unused(state_fc, geom, params);
			return;
		}

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.nGrow() >= 2, "Electron conduction requires at least 2 ghost cells.");

		const auto dx = geom.CellSizeArray();
		const amrex::Real flux_limiter_phi = params.flux_limiter_phi;
		const amrex::Real saturation_factor = params.saturation_factor;
		const amrex::Real t_min = params.min_temperature;
		const bool spitzer_scaling = params.spitzer_scaling;
		const amrex::Real kappa0 = params.conductivity_prefactor;
		const amrex::Real small = std::numeric_limits<amrex::Real>::min();
		constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;

		amrex::MultiFab primVar(state.boxArray(), state.DistributionMap(), 2 + nmscalars_, state.nGrow());
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
			auto const &cons = state_x0[bx];
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
			}

			const amrex::Real rho = cons(i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real Eint = HydroSystem<problem_t>::ComputeInternalEnergy(cons, i, j, k, &local_state_fc);
			// Temperature always from EOS
			auto const massScalarsRaw = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
			quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = massScalarsRaw;
			const amrex::Real Tgas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);

			primVar_arr[bx](i, j, k, 0) = rho;
			primVar_arr[bx](i, j, k, 1) = amrex::max(Tgas, t_min);
			for (int n = 0; n < nmscalars_; ++n) {
				// store mass fraction (not partial density) so it reconstructs consistently with rho and T
				primVar_arr[bx](i, j, k, 2 + n) = massScalarsRaw[n] / rho;
			}
		});

		// src-02b(-fix): no reconstruction pass and no face MultiFabs; rho, T and mass fractions are brought to faces inside the
		// flux kernels by the clamped 4th-order centred interpolant FaceValue4 (needs 2 ghost cells). A plain arithmetic mean of
		// the two adjacent cells was tried first but is ~3x less accurate for kappa ~ T^2.5 on steep profiles.
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}

		auto const evaluateFace = [=] AMREX_GPU_DEVICE(amrex::Real rho_f, amrex::Real T_f, amrex::GpuArray<amrex::Real, nmscalars_> const &massFrac_f,
							       amrex::Real &kappa_face, amrex::Real &qsat_face) noexcept {
			const amrex::Real rho_face = rho_f;
			const amrex::Real T_face = amrex::max(T_f, t_min);
			amrex::GpuArray<amrex::Real, nmscalars_> massArray_face{};
			amrex::Real sumFrac = 0.0;
			for (int n = 0; n < nmscalars_; ++n) {
				massArray_face[n] = massFrac_f[n];
				sumFrac += massArray_face[n];
			}
			const amrex::Real invSum = 1.0 / amrex::max(sumFrac, small);
			for (int n = 0; n < nmscalars_; ++n) {
				massArray_face[n] *= invSum * rho_face;
			}
			quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = massArray_face;
			const amrex::Real Eint_face = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho_face, T_face, massScalars);
			const amrex::Real Pgas_face = ::quokka::EOS<problem_t>::ComputePressure(rho_face, Eint_face, massScalars);
			const amrex::Real cs_face = ::quokka::EOS<problem_t>::ComputeSoundSpeed(rho_face, Pgas_face, massScalars);

			kappa_face = spitzer_scaling ? (kappa0 * std::pow(T_face, 2.5)) : kappa0;
			qsat_face = amrex::max(saturation_factor * flux_limiter_phi * rho_face * cs_face * cs_face * cs_face, small);
		};

		auto const &temp = primVar.const_arrays();
		auto flux_x = heat_flux[0].arrays();
		amrex::ParallelFor(heat_flux[0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			auto const &q = temp[bx];
			const amrex::Real gradT = (q(i, j, k, 1) - q(i - 1, j, k, 1)) / dx[0];
			amrex::Real kappa_face = 0.0;
			amrex::Real q_sat_face = 0.0;
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_f{};
			for (int n = 0; n < nmscalars_; ++n) {
				massFrac_f[n] = FaceValue4(q(i - 2, j, k, 2 + n), q(i - 1, j, k, 2 + n), q(i, j, k, 2 + n), q(i + 1, j, k, 2 + n));
			}
			evaluateFace(FaceValue4(q(i - 2, j, k, 0), q(i - 1, j, k, 0), q(i, j, k, 0), q(i + 1, j, k, 0)),
				     FaceValue4(q(i - 2, j, k, 1), q(i - 1, j, k, 1), q(i, j, k, 1), q(i + 1, j, k, 1)), massFrac_f, kappa_face, q_sat_face);
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_x[bx](i, j, k) = q_classical / limiter;
		});

#if AMREX_SPACEDIM >= 2
		auto flux_y = heat_flux[1].arrays();
		amrex::ParallelFor(heat_flux[1], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			auto const &q = temp[bx];
			const amrex::Real gradT = (q(i, j, k, 1) - q(i, j - 1, k, 1)) / dx[1];
			amrex::Real kappa_face = 0.0;
			amrex::Real q_sat_face = 0.0;
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_f{};
			for (int n = 0; n < nmscalars_; ++n) {
				massFrac_f[n] = FaceValue4(q(i, j - 2, k, 2 + n), q(i, j - 1, k, 2 + n), q(i, j, k, 2 + n), q(i, j + 1, k, 2 + n));
			}
			evaluateFace(FaceValue4(q(i, j - 2, k, 0), q(i, j - 1, k, 0), q(i, j, k, 0), q(i, j + 1, k, 0)),
				     FaceValue4(q(i, j - 2, k, 1), q(i, j - 1, k, 1), q(i, j, k, 1), q(i, j + 1, k, 1)), massFrac_f, kappa_face, q_sat_face);
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_y[bx](i, j, k) = q_classical / limiter;
		});
#endif

#if AMREX_SPACEDIM == 3
		auto flux_z = heat_flux[2].arrays();
		amrex::ParallelFor(heat_flux[2], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			auto const &q = temp[bx];
			const amrex::Real gradT = (q(i, j, k, 1) - q(i, j, k - 1, 1)) / dx[2];
			amrex::Real kappa_face = 0.0;
			amrex::Real q_sat_face = 0.0;
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_f{};
			for (int n = 0; n < nmscalars_; ++n) {
				massFrac_f[n] = FaceValue4(q(i, j, k - 2, 2 + n), q(i, j, k - 1, 2 + n), q(i, j, k, 2 + n), q(i, j, k + 1, 2 + n));
			}
			evaluateFace(FaceValue4(q(i, j, k - 2, 0), q(i, j, k - 1, 0), q(i, j, k, 0), q(i, j, k + 1, 0)),
				     FaceValue4(q(i, j, k - 2, 1), q(i, j, k - 1, 1), q(i, j, k, 1), q(i, j, k + 1, 1)), massFrac_f, kappa_face, q_sat_face);
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
			const amrex::Real Emag = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &local_state_fc);
			// derive Eint from total energy rather than the auxiliary internalEnergy_index field,
			// which is not kept in sync with total energy when hydro.use_dual_energy == 0
			const amrex::Real Eint_old = HydroSystem<problem_t>::ComputeInternalEnergy(state_out[bx], i, j, k, &local_state_fc);
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
