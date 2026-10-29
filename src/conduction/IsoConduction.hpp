#ifndef ISO_CONDUCTION_HPP_ // NOLINT
#define ISO_CONDUCTION_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file IsoConduction.hpp
/// \brief Explicit flux-limited isotropic thermal conduction update.

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
#include "conduction/ConductionHelpers.hpp"
#include "conduction/conductivity.hpp"
#include "hydro/hydro_system.hpp"
#include "hyperbolic_system.hpp"

namespace quokka::conduction
{

struct IsoConductionParams {
	ConductivityParams conductivity{}; // prefactors for ConductionModel::constant/spitzer (see conductivity.hpp)
	amrex::Real flux_limiter_phi = 0.1;
	amrex::Real saturation_factor = 5.0; // refer to equation 8 of Cowie & McKee 1977
	amrex::Real min_temperature = 0.0;   // default value will be overwritten by tempFloor_ during initialization
	int reconstruction_order = 3;	     // 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM;
	SlopeLimiter plm_limiter = SlopeLimiter::sweby;
	int ng_reconstruct = 2; // number of ghost faces to reconstruct beyond the valid box
};

template <typename problem_t> class IsoConduction
{
      public:
	// Reconstruct rho, T, and mass fractions at the interfaces
	template <FluxDir DIR>
	static void ReconstructPrimVar(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int ng_reconstruct,
				       IsoConductionParams const &params)
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
			amrex::Abort("Invalid reconstruction order specified for isotropic conduction!");
		}
	}

	/// \brief Returns false when conduction has no effect for this problem: ConductionModel::none, an isothermal EOS,
	///        or a built-in conductivity model with a non-positive prefactor.
	static auto IsActive(IsoConductionParams const &params) -> bool
	{
		constexpr ConductionModel model = Physics_Traits<problem_t>::conduction_model;
		if constexpr (model == ConductionModel::none || HydroSystem<problem_t>::is_eos_isothermal()) {
			amrex::ignore_unused(params);
			return false;
		} else if constexpr (model == ConductionModel::constant || model == ConductionModel::spitzer) {
			return params.conductivity.kappa0_par > 0.0;
		} else {
			amrex::ignore_unused(params);
			return true;
		}
	}

	/// \brief Defines \p heat_flux (1 component, no ghost faces) on the faces of \p state's BoxArray and zeroes it.
	static void DefineHeatFlux(amrex::MultiFab const &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}
	}

	/// \brief Forward-Euler conduction update of \p state over \p dt. On return, \p heat_flux holds the face fluxes
	///        used for the update (zero if conduction is inactive or \p dt <= 0).
	static void ComputeExplicit(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    amrex::Real dt, IsoConductionParams const &params, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		if ((dt <= 0.0) || !IsActive(params)) {
			DefineHeatFlux(state, heat_flux);
			return;
		}
		ComputeHeatFlux(state, state_fc, geom, params, heat_flux);
		ApplyConductionUpdate<problem_t>(state, state_fc, heat_flux, geom.CellSizeArray(), dt);
	}

	/// \brief Computes the flux-limited, saturated conductive heat flux q on every face of \p state's valid boxes.
	///        Does not modify \p state; its ghost cells must already be filled. \p heat_flux is (re)defined here.
	static void ComputeHeatFlux(amrex::MultiFab const &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    IsoConductionParams const &params, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		DefineHeatFlux(state, heat_flux);
		if (!IsActive(params)) {
			amrex::ignore_unused(state_fc, geom);
			return;
		}

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.nGrow() >= 1, "Isotropic conduction requires at least 1 ghost cell.");

		const auto dx = geom.CellSizeArray();
		const amrex::Real flux_limiter_phi = params.flux_limiter_phi;
		const amrex::Real saturation_factor = params.saturation_factor;
		const amrex::Real t_min = params.min_temperature;
		const ConductivityParams conductivity_params = params.conductivity;
		const amrex::Real small = std::numeric_limits<amrex::Real>::min();
		constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;

		// mass fractions (not partial densities), so they reconstruct consistently with rho and T
		amrex::MultiFab primVar(state.boxArray(), state.DistributionMap(), 2 + nmscalars_, state.nGrow());
		ComputeConductionPrimVar<problem_t, MassScalarStorage::mass_fraction>(state, state_fc, t_min, primVar);

		// Reconstruct (rho, T, mass fractions) to the interfaces
		std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;
		const int ng_reconstruct = params.ng_reconstruct;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			leftState[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 2 + nmscalars_, ng_reconstruct);
			rightState[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 2 + nmscalars_, ng_reconstruct);
		}

		AMREX_D_TERM(ReconstructPrimVar<FluxDir::X1>(primVar, leftState[0], rightState[0], ng_reconstruct, params);
			     , ReconstructPrimVar<FluxDir::X2>(primVar, leftState[1], rightState[1], ng_reconstruct, params);
			     , ReconstructPrimVar<FluxDir::X3>(primVar, leftState[2], rightState[2], ng_reconstruct, params);)

		auto const evaluateFace = [=] AMREX_GPU_DEVICE(amrex::Real rho_L, amrex::Real T_L, amrex::Real rho_R, amrex::Real T_R,
							       amrex::GpuArray<amrex::Real, nmscalars_> const &massFrac_L,
							       amrex::GpuArray<amrex::Real, nmscalars_> const &massFrac_R, amrex::Real &kappa_face,
							       amrex::Real &qsat_face) noexcept {
			const amrex::Real rho_face = 0.5 * (rho_L + rho_R);
			const amrex::Real T_face = amrex::max(0.5 * (T_L + T_R), t_min);
			amrex::GpuArray<amrex::Real, nmscalars_> massArray_face{};
			amrex::Real sumFrac = 0.0;
			for (int n = 0; n < nmscalars_; ++n) {
				massArray_face[n] = 0.5 * (massFrac_L[n] + massFrac_R[n]);
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

			// kappa from the conductivity model; see conductivity.hpp
			kappa_face = EvaluateConductivity<problem_t>(rho_face, T_face, conductivity_params)[0];
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
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_L{};
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_R{};
			for (int n = 0; n < nmscalars_; ++n) {
				massFrac_L[n] = left_x[bx](i, j, k, 2 + n);
				massFrac_R[n] = right_x[bx](i, j, k, 2 + n);
			}
			evaluateFace(left_x[bx](i, j, k, 0), left_x[bx](i, j, k, 1), right_x[bx](i, j, k, 0), right_x[bx](i, j, k, 1), massFrac_L, massFrac_R,
				     kappa_face, q_sat_face);
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
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_L{};
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_R{};
			for (int n = 0; n < nmscalars_; ++n) {
				massFrac_L[n] = left_y[bx](i, j, k, 2 + n);
				massFrac_R[n] = right_y[bx](i, j, k, 2 + n);
			}
			evaluateFace(left_y[bx](i, j, k, 0), left_y[bx](i, j, k, 1), right_y[bx](i, j, k, 0), right_y[bx](i, j, k, 1), massFrac_L, massFrac_R,
				     kappa_face, q_sat_face);
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
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_L{};
			amrex::GpuArray<amrex::Real, nmscalars_> massFrac_R{};
			for (int n = 0; n < nmscalars_; ++n) {
				massFrac_L[n] = left_z[bx](i, j, k, 2 + n);
				massFrac_R[n] = right_z[bx](i, j, k, 2 + n);
			}
			evaluateFace(left_z[bx](i, j, k, 0), left_z[bx](i, j, k, 1), right_z[bx](i, j, k, 0), right_z[bx](i, j, k, 1), massFrac_L, massFrac_R,
				     kappa_face, q_sat_face);
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_z[bx](i, j, k) = q_classical / limiter;
		});
#endif
	}
};

} // namespace quokka::conduction

#endif // ISO_CONDUCTION_HPP_
