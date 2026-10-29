#ifndef CONDUCTION_HELPERS_HPP_ // NOLINT
#define CONDUCTION_HELPERS_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ConductionHelpers.hpp
/// \brief Kernels shared by IsoConduction and AnisoConduction: gathering the face-centered state for
///        MHD, building the cell-centered (rho, T, mass scalars) primVar, and applying the heat flux
///        divergence to the internal and total energy.

#include <array>
#include <limits>

#include "AMReX_Array4.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParReduce.H"
#include "AMReX_ParallelReduce.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_ValLocPair.H"
#include "conduction/conductivity.hpp"
#include "hydro/hydro_system.hpp"

namespace quokka::conduction
{

/// \brief Per-direction MultiArray4s of the face-centered state, capturable by value in a GPU lambda.
using FaceStateArrays = amrex::GpuArray<amrex::MultiArray4<const amrex::Real>, AMREX_SPACEDIM>;

inline auto MakeFaceStateArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) -> FaceStateArrays
{
	FaceStateArrays arrs{};
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		arrs[idim] = state_fc[idim].const_arrays();
	}
	return arrs;
}

/// \brief Gathers box \p bx of the face-centered state so it can be handed to
///        HydroSystem<problem_t>::ComputeInternalEnergy/ComputeMagneticEnergy. Returns empty
///        Array4s when MHD is disabled.
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto GatherFaceFields(FaceStateArrays const &state_fc, int bx)
    -> std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>
{
	std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_state_fc{};
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			local_state_fc[idim] = state_fc[idim][bx];
		}
	} else {
		amrex::ignore_unused(state_fc, bx);
	}
	return local_state_fc;
}

/// \brief How ComputeConductionPrimVar stores the mass scalars in primVar. IsoConduction reconstructs
///        them to faces, so it needs mass fractions (which reconstruct consistently with rho and T);
///        AnisoConduction averages them to corners and passes them to the EOS as partial densities.
enum class MassScalarStorage { mass_fraction, partial_density };

/// \brief Fills \p primVar (2 + numMassScalars components, including ghost cells) with
///        (rho, max(T, t_min), mass scalars...) computed from the conserved \p state via the EOS.
template <typename problem_t, MassScalarStorage storage>
void ComputeConductionPrimVar(amrex::MultiFab const &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Real t_min,
			      amrex::MultiFab &primVar)
{
	constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	primVar.setVal(0.0);

	auto const &state_in = state.const_arrays();
	FaceStateArrays const state_fc_in = MakeFaceStateArrays(state_fc);
	auto primVar_arr = primVar.arrays();
	amrex::IntVect const ng = amrex::IntVect(AMREX_D_DECL(state.nGrow(), state.nGrow(), state.nGrow()));

	amrex::ParallelFor(state, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		auto const &cons = state_in[bx];
		auto local_state_fc = GatherFaceFields<problem_t>(state_fc_in, bx);

		const amrex::Real rho = cons(i, j, k, HydroSystem<problem_t>::density_index);
		const amrex::Real Eint = HydroSystem<problem_t>::ComputeInternalEnergy(cons, i, j, k, &local_state_fc);
		// Temperature always from EOS
		amrex::GpuArray<amrex::Real, nmscalars_> const massScalarsArr = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
		quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const massScalars = massScalarsArr;
		const amrex::Real Tgas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);

		primVar_arr[bx](i, j, k, 0) = rho;
		primVar_arr[bx](i, j, k, 1) = amrex::max(Tgas, t_min);
		for (int n = 0; n < nmscalars_; ++n) {
			if constexpr (storage == MassScalarStorage::mass_fraction) {
				primVar_arr[bx](i, j, k, 2 + n) = massScalarsArr[n] / rho;
			} else {
				primVar_arr[bx](i, j, k, 2 + n) = massScalarsArr[n];
			}
		}
	});
}

/// \brief Applies Eint_new = Eint_old - dt * div(heat_flux) to every valid cell of \p state, updating
///        both the total energy (Eint_new + Ekin + Emag) and the auxiliary internal energy.
template <typename problem_t>
void ApplyConductionUpdate(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc,
			   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &heat_flux, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real dt)
{
	auto state_out = state.arrays();
	FaceStateArrays const state_fc_in = MakeFaceStateArrays(state_fc);
	auto const &flux_x = heat_flux[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &flux_y = heat_flux[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &flux_z = heat_flux[2].const_arrays();
#endif

	amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		auto local_state_fc = GatherFaceFields<problem_t>(state_fc_in, bx);

		const amrex::Real rho = state_out[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const amrex::Real px = state_out[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const amrex::Real py = state_out[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const amrex::Real pz = state_out[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);

		const amrex::Real Ekin = 0.5 * (px * px + py * py + pz * pz) / rho;
		const amrex::Real Emag = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &local_state_fc);
		// derive Eint from total energy rather than the auxiliary internalEnergy_index field,
		// which is not kept in sync with total energy when hydro.use_dual_energy == 0
		const amrex::Real Eint_old = HydroSystem<problem_t>::ComputeInternalEnergy(state_out[bx], i, j, k, &local_state_fc);
		amrex::Real div_flux = (flux_x[bx](i + 1, j, k) - flux_x[bx](i, j, k)) / dx[0];
#if AMREX_SPACEDIM >= 2
		div_flux += (flux_y[bx](i, j + 1, k) - flux_y[bx](i, j, k)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
		div_flux += (flux_z[bx](i, j, k + 1) - flux_z[bx](i, j, k)) / dx[2];
#endif

		const amrex::Real Eint_new = Eint_old - dt * div_flux;

		state_out[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Eint_new + Ekin + Emag;
		state_out[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index) = Eint_new;
	});
}

/// \brief Copies the internal energy of every valid cell of \p state into the 1-component MultiFab \p Eint
///        (same BoxArray and DistributionMapping as \p state), derived from total energy as in ApplyConductionUpdate.
template <typename problem_t>
void ExtractInternalEnergy(amrex::MultiFab const &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::MultiFab &Eint)
{
	auto const &state_in = state.const_arrays();
	FaceStateArrays const state_fc_in = MakeFaceStateArrays(state_fc);
	auto Eint_out = Eint.arrays();

	amrex::ParallelFor(Eint, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		auto local_state_fc = GatherFaceFields<problem_t>(state_fc_in, bx);
		Eint_out[bx](i, j, k) = HydroSystem<problem_t>::ComputeInternalEnergy(state_in[bx], i, j, k, &local_state_fc);
	});
}

/// \brief Sets the internal energy of every valid cell of \p state to \p Eint, updating both the total energy
///        (Eint + Ekin + Emag) and the auxiliary internal energy, as in ApplyConductionUpdate.
template <typename problem_t>
void SetInternalEnergy(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::MultiFab const &Eint)
{
	auto state_out = state.arrays();
	FaceStateArrays const state_fc_in = MakeFaceStateArrays(state_fc);
	auto const &Eint_in = Eint.const_arrays();

	amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		auto local_state_fc = GatherFaceFields<problem_t>(state_fc_in, bx);

		const amrex::Real rho = state_out[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const amrex::Real px = state_out[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const amrex::Real py = state_out[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const amrex::Real pz = state_out[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);

		const amrex::Real Ekin = 0.5 * (px * px + py * py + pz * pz) / rho;
		const amrex::Real Emag = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &local_state_fc);
		const amrex::Real Eint_new = Eint_in[bx](i, j, k);

		state_out[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Eint_new + Ekin + Emag;
		state_out[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index) = Eint_new;
	});
}

/// \brief Returns the minimum over the valid cells of \p state (reduced across all MPI ranks) of the explicit
///        conduction timestep cfl * dx_min^2 / D, with D = kappa_eff / (dEint/dT), and the cell where it occurs.
///        With ConductionGeometry::anisotropic, kappa_eff = 2 (kappa_par + kappa_perp). Cells with D <= 0 do not
///        limit the timestep.
template <typename problem_t>
auto ComputeConductionTimestep(amrex::MultiFab const &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Real dx_min,
			       ConductivityParams const &conductivity_params, amrex::Real t_min, amrex::Real cfl)
    -> amrex::ValLocPair<amrex::Real, amrex::IntVect>
{
	auto const &state_mf = state.const_arrays();
	FaceStateArrays const state_fc_in = MakeFaceStateArrays(state_fc);

	// Use amrex::ParReduce to find the minimum dt and its location across all GPU threads
	amrex::ValLocPair<amrex::Real, amrex::IntVect> conduction_dt =
	    amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMin>{}, amrex::TypeList<amrex::ValLocPair<amrex::Real, amrex::IntVect>>{}, state, amrex::IntVect(0),
			     [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) -> amrex::ValLocPair<amrex::Real, amrex::IntVect> {
				     auto const &cons = state_mf[bx];
				     auto local_state_fc = GatherFaceFields<problem_t>(state_fc_in, bx);

				     const amrex::Real rho = cons(i, j, k, HydroSystem<problem_t>::density_index);
				     const amrex::Real Eint = HydroSystem<problem_t>::ComputeInternalEnergy(cons, i, j, k, &local_state_fc);
				     auto const massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
				     const amrex::Real T = amrex::max(t_min, quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars));

				     // kappa from the conductivity model; see conductivity.hpp
				     const auto kappa = EvaluateConductivity<problem_t>(rho, T, conductivity_params);
				     amrex::Real kappa_eff = kappa[0];
				     if constexpr (Physics_Traits<problem_t>::conduction_geometry == ConductionGeometry::anisotropic) {
					     // bound the largest eigenvalue of the conductivity tensor by kappa_par + kappa_perp, keeping the
					     // factor-of-2 safety margin previously used for the anisotropic scheme
					     kappa_eff = 2.0 * (kappa[0] + kappa[1]);
				     }
				     const amrex::Real heat_capacity = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T, massScalars);
				     const amrex::Real diffusion_coefficient = kappa_eff / heat_capacity;

				     // Avoid division by zero for unphysical states
				     amrex::Real cell_dt = std::numeric_limits<amrex::Real>::max();
				     if (diffusion_coefficient > 0.0) {
					     cell_dt = cfl * (dx_min * dx_min) / diffusion_coefficient;
				     }

				     return {.value = cell_dt, .index = amrex::IntVect{AMREX_D_DECL(i, j, k)}};
			     });

	amrex::ParallelAllReduce::Min(conduction_dt, amrex::ParallelContext::CommunicatorSub());
	return conduction_dt;
}

} // namespace quokka::conduction

#endif // CONDUCTION_HELPERS_HPP_
