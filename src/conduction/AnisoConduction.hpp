#ifndef ANISO_CONDUCTION_HPP_ // NOLINT
#define ANISO_CONDUCTION_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file AnisoConduction.hpp
/// \brief Explicit anisotropic thermal conduction update. At each face, gradT is decomposed into
///        components parallel/perpendicular to the local unit B-field (bhat) and scaled by
///        kappa_parallel/kappa_perp respectively, then flux-limited/saturated as in
///        ElectronConduction::ComputeExplicit. The saturation flux itself is not yet computed
///        (a large sentinel is used, making the limiter a no-op).

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

// Declarations only -- see the definitions below AnisoConduction for what each of these actually
// does. Declared here so that AnisoConduction::ComputeExplicit (which calls all four) can be read
// first; scroll down past the class for the bodies.
template <typename problem_t, FluxDir DIR>
void ComputeFaceUnitBField(amrex::MultiFab &bhat_fc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int reconstructionOrder,
			   SlopeLimiter plmLimiter, int nghost);

template <FluxDir DIR>
void ComputeFaceGradT(amrex::MultiFab &gradT_fc_mf, amrex::MultiFab const &temperature, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, int comp);

template <FluxDir DIR>
void ComputeFaceNumberDensity(amrex::MultiFab &n_fc_mf, amrex::MultiFab const &density, amrex::Real mean_molecular_weight, int comp);

template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &bhat_fc, amrex::MultiFab const &gradT_fc, amrex::MultiFab const &primVar,
			    amrex::Real mean_molecular_weight, amrex::Real kappa_parallel, amrex::Real kappa_perp, amrex::MultiFab const &q_sat_fc);

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

		if ((dt <= 0.0) || ((params.kappa_parallel <= 0.0) && (params.kappa_perp <= 0.0))) {
			return;
		}

		if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
			amrex::ignore_unused(geom, params);
			return;
		}

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.nGrow() >= 1, "Anisotropic conduction requires at least 1 ghost cell.");

		const auto dx = geom.CellSizeArray();

		amrex::MultiFab primVar(state.boxArray(), state.DistributionMap(), 2, state.nGrow());
		auto const &state_x0 = state.const_arrays();
		primVar.setVal(0.0);

		auto primVar_arr = primVar.arrays();
		constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
		const amrex::Real t_min = params.min_temperature;

		// Per-box face-centered B, gathered into an array so it can be handed to
		// HydroSystem<problem_t>::ComputeInternalEnergy/ComputeMagneticEnergy below.
		auto const &state_fc_x0 = state_fc[0].const_arrays();
#if AMREX_SPACEDIM >= 2
		auto const &state_fc_x1 = state_fc[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
		auto const &state_fc_x2 = state_fc[2].const_arrays();
#endif

		amrex::IntVect const ng = amrex::IntVect(AMREX_D_DECL(state.nGrow(), state.nGrow(), state.nGrow()));

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
			quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
			const amrex::Real Tgas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);

			primVar_arr[bx](i, j, k, 0) = rho;
			primVar_arr[bx](i, j, k, 1) = amrex::max(Tgas, t_min);
		});

		// Unit B-field at each face (bx, by, bz), populated only when MHD is enabled (zeroed
		// otherwise, so a non-MHD build gets zero flux rather than reading uninitialized data).
		std::array<amrex::MultiFab, AMREX_SPACEDIM> bhat_fc;
		// gradT at each face (dT/dx, dT/dy, dT/dz), fixed physical (x,y,z) order -- see ComputeFaceGradT.
		std::array<amrex::MultiFab, AMREX_SPACEDIM> gradT_fc;
		const int ng_reconstruct = params.ng_reconstruct;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			bhat_fc[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 3, 0);
			bhat_fc[idim].setVal(0.0);
			gradT_fc[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 3, 0);
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			AMREX_D_TERM(ComputeFaceUnitBField<problem_t, FluxDir::X1>(bhat_fc[0], state_fc, params.reconstruction_order, params.plm_limiter,
										    ng_reconstruct);
				     , ComputeFaceUnitBField<problem_t, FluxDir::X2>(bhat_fc[1], state_fc, params.reconstruction_order, params.plm_limiter,
										      ng_reconstruct);
				     , ComputeFaceUnitBField<problem_t, FluxDir::X3>(bhat_fc[2], state_fc, params.reconstruction_order, params.plm_limiter,
										      ng_reconstruct);)
		}

		// primVar component 1 holds T.
		AMREX_D_TERM(ComputeFaceGradT<FluxDir::X1>(gradT_fc[0], primVar, dx, 1);, ComputeFaceGradT<FluxDir::X2>(gradT_fc[1], primVar, dx, 1);
			     , ComputeFaceGradT<FluxDir::X3>(gradT_fc[2], primVar, dx, 1);)

		// Saturation flux at each face (Cowie & McKee 1977) -- NOT computed yet, since that needs
		// (rho, T) reconstructed to the interfaces via EOS calls, which isn't wired up in this file
		// right now. Filled with a large sentinel so the limiter below is a no-op (q_classical
		// unmodified) until that piece is added.
		std::array<amrex::MultiFab, AMREX_SPACEDIM> q_sat_fc;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			q_sat_fc[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 1, 0);
			q_sat_fc[idim].setVal(std::numeric_limits<amrex::Real>::max());
		}

		// Heat flux at each face: full parallel/perpendicular decomposition of gradT relative to
		// bhat, projected onto the face normal -- see ComputeAnisotropicFlux (which computes number
		// density internally via ComputeFaceNumberDensity).
		const amrex::Real mmw = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
		AMREX_D_TERM(ComputeAnisotropicFlux<FluxDir::X1>(heat_flux[0], bhat_fc[0], gradT_fc[0], primVar, mmw, params.kappa_parallel,
								  params.kappa_perp, q_sat_fc[0]);
			     , ComputeAnisotropicFlux<FluxDir::X2>(heat_flux[1], bhat_fc[1], gradT_fc[1], primVar, mmw, params.kappa_parallel,
								    params.kappa_perp, q_sat_fc[1]);
			     , ComputeAnisotropicFlux<FluxDir::X3>(heat_flux[2], bhat_fc[2], gradT_fc[2], primVar, mmw, params.kappa_parallel,
								    params.kappa_perp, q_sat_fc[2]);)

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

// Estimate the unit vector of B at the DIR-faces: bhat_fc_mf (output) is a 3-component (bx,by,bz)
// MultiFab on the DIR-face grid. The face-normal component comes directly from the staggered
// state_fc representation (exact); the two transverse components are averaged to cell centers from
// their own faces, reconstructed along DIR to the interface (same cyclic perp0/perp1 mapping as
// QuokkaSimulation::computeCCPerpBfieldComps: X1->(perp0=y,perp1=z), X2->(z,x), X3->(x,y)), then
// combined with the normal component and normalized.
template <typename problem_t, FluxDir DIR>
void ComputeFaceUnitBField(amrex::MultiFab &bhat_fc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int reconstructionOrder,
			   SlopeLimiter plmLimiter, int nghost)
{
	static_assert(Physics_Traits<problem_t>::is_mhd_enabled);
	constexpr int b_comp = Physics_Indices<problem_t>::mhdFirstIndex;

	// ------------------------------------------------------------------
	// Step (a): cell-centered transverse-B, same cyclic mapping as computeCCPerpBfieldComps.
	// ------------------------------------------------------------------
	amrex::BoxArray const ba_cc = amrex::convert(bhat_fc_mf.boxArray(), amrex::IntVect::TheZeroVector());
	amrex::MultiFab cc_bperp(ba_cc, bhat_fc_mf.DistributionMap(), 2, nghost);

	{
		std::array<int, 3> delta_perp0{0, 0, 0};
		std::array<int, 3> delta_perp1{0, 0, 0};
		amrex::MultiArray4<const amrex::Real> perp0_fc;
		amrex::MultiArray4<const amrex::Real> perp1_fc;
		if constexpr (DIR == FluxDir::X1) {
			perp0_fc = state_fc[1].const_arrays(); // y-faces
			perp1_fc = state_fc[2].const_arrays(); // z-faces
			delta_perp0[1] = 1;
			delta_perp1[2] = 1;
		} else if constexpr (DIR == FluxDir::X2) {
			perp0_fc = state_fc[2].const_arrays(); // z-faces
			perp1_fc = state_fc[0].const_arrays(); // x-faces
			delta_perp0[2] = 1;
			delta_perp1[0] = 1;
		} else { // FluxDir::X3
			perp0_fc = state_fc[0].const_arrays(); // x-faces
			perp1_fc = state_fc[1].const_arrays(); // y-faces
			delta_perp0[0] = 1;
			delta_perp1[1] = 1;
		}

		auto cc_out = cc_bperp.arrays();
		amrex::IntVect const ng_cc{AMREX_D_DECL(nghost, nghost, nghost)};
		amrex::ParallelFor(cc_bperp, ng_cc, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real b_perp0_m = perp0_fc[bx](i, j, k, b_comp);
			const amrex::Real b_perp0_p = perp0_fc[bx](i + delta_perp0[0], j + delta_perp0[1], k + delta_perp0[2], b_comp);
			cc_out[bx](i, j, k, 0) = 0.5 * (b_perp0_m + b_perp0_p);

			const amrex::Real b_perp1_m = perp1_fc[bx](i, j, k, b_comp);
			const amrex::Real b_perp1_p = perp1_fc[bx](i + delta_perp1[0], j + delta_perp1[1], k + delta_perp1[2], b_comp);
			cc_out[bx](i, j, k, 1) = 0.5 * (b_perp1_m + b_perp1_p);
		});
	}

	// ------------------------------------------------------------------
	// Step (b): reconstruct cc_bperp along DIR -> left/right at the face.
	// ------------------------------------------------------------------
	amrex::MultiFab leftState_b(bhat_fc_mf.boxArray(), bhat_fc_mf.DistributionMap(), 2, nghost);
	amrex::MultiFab rightState_b(bhat_fc_mf.boxArray(), bhat_fc_mf.DistributionMap(), 2, nghost);

	if (reconstructionOrder == 5) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(cc_bperp, leftState_b, rightState_b, nghost, 2);
	} else if (reconstructionOrder == 3) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(cc_bperp, leftState_b, rightState_b, nghost, 2);
	} else if (reconstructionOrder == 2) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR>(cc_bperp, leftState_b, rightState_b, nghost, 2, plmLimiter);
	} else {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(cc_bperp, leftState_b, rightState_b, nghost, 2);
	}

	// ------------------------------------------------------------------
	// Step (c): average left/right, pull in the exact normal component, un-permute back into
	// fixed physical (bx,by,bz) order, normalize.
	// ------------------------------------------------------------------
	auto const &left_in = leftState_b.const_arrays();
	auto const &right_in = rightState_b.const_arrays();
	auto const &bn_in = state_fc[static_cast<int>(DIR)].const_arrays();
	auto bhat_out = bhat_fc_mf.arrays();

	amrex::ParallelFor(bhat_fc_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real bn = bn_in[bx](i, j, k, b_comp);				       // exact, step (a)'s face data
		const amrex::Real bt1 = 0.5 * (left_in[bx](i, j, k, 0) + right_in[bx](i, j, k, 0)); // averaged, step (b) result
		const amrex::Real bt2 = 0.5 * (left_in[bx](i, j, k, 1) + right_in[bx](i, j, k, 1));

		const amrex::Real bmag = std::sqrt(bn * bn + bt1 * bt1 + bt2 * bt2);
		const amrex::Real inv_b = (bmag > 0.0) ? (1.0 / bmag) : 0.0;

		// un-permute: place (bn, bt1, bt2) back into fixed (x,y,z) slots, the exact inverse of
		// the table used in step (a).
		if constexpr (DIR == FluxDir::X1) {
			bhat_out[bx](i, j, k, 0) = bn * inv_b;	 // x
			bhat_out[bx](i, j, k, 1) = bt1 * inv_b; // y
			bhat_out[bx](i, j, k, 2) = bt2 * inv_b; // z
		} else if constexpr (DIR == FluxDir::X2) {
			bhat_out[bx](i, j, k, 0) = bt2 * inv_b; // x (perp1 was x for X2)
			bhat_out[bx](i, j, k, 1) = bn * inv_b;	 // y
			bhat_out[bx](i, j, k, 2) = bt1 * inv_b; // z (perp0 was z for X2)
		} else {					 // X3
			bhat_out[bx](i, j, k, 0) = bt1 * inv_b; // x (perp0 was x for X3)
			bhat_out[bx](i, j, k, 1) = bt2 * inv_b; // y (perp1 was y for X3)
			bhat_out[bx](i, j, k, 2) = bn * inv_b;	 // z
		}
	});
}

// Compute the temperature gradient vector (dT/dx, dT/dy, dT/dz) at the DIR-faces of a cell-centered
// temperature field: gradT_fc_mf (output) is a 3-component MultiFab on the DIR-face grid, in fixed
// physical (x,y,z) order. `temperature` is a (possibly multi-component) cell-centered MultiFab;
// `comp` selects which component holds T (e.g. component 1 of the (rho, T) primVar used elsewhere
// in this file). The face-normal component is the usual two-point difference across the face; each
// transverse component is the average of the centered difference computed in the two cells bounding
// the face -- the same diamond-averaging used for the transverse B components in
// ComputeFaceUnitBField.
// NOTE: this transverse estimate is the step flagged earlier as needing a Sharma & Hammett-style
// limiter, to avoid an unphysical flux direction near sharp field bends -- not yet applied here.
template <FluxDir DIR>
void ComputeFaceGradT(amrex::MultiFab &gradT_fc_mf, amrex::MultiFab const &temperature, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, int comp)
{
	auto const &temp = temperature.const_arrays();
	auto gradT_out = gradT_fc_mf.arrays();

	amrex::ParallelFor(gradT_fc_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real gradT_x = 0.0;
		amrex::Real gradT_y = 0.0;
		amrex::Real gradT_z = 0.0;

		if constexpr (DIR == FluxDir::X1) {
			gradT_x = (temp[bx](i, j, k, comp) - temp[bx](i - 1, j, k, comp)) / dx[0];
#if AMREX_SPACEDIM >= 2
			gradT_y = 0.5 *
				  ((temp[bx](i - 1, j + 1, k, comp) - temp[bx](i - 1, j - 1, k, comp)) +
				   (temp[bx](i, j + 1, k, comp) - temp[bx](i, j - 1, k, comp))) /
				  (2.0 * dx[1]);
#endif
#if AMREX_SPACEDIM == 3
			gradT_z = 0.5 *
				  ((temp[bx](i - 1, j, k + 1, comp) - temp[bx](i - 1, j, k - 1, comp)) +
				   (temp[bx](i, j, k + 1, comp) - temp[bx](i, j, k - 1, comp))) /
				  (2.0 * dx[2]);
#endif
		} else if constexpr (DIR == FluxDir::X2) {
			gradT_y = (temp[bx](i, j, k, comp) - temp[bx](i, j - 1, k, comp)) / dx[1];
			gradT_x = 0.5 *
				  ((temp[bx](i + 1, j - 1, k, comp) - temp[bx](i - 1, j - 1, k, comp)) +
				   (temp[bx](i + 1, j, k, comp) - temp[bx](i - 1, j, k, comp))) /
				  (2.0 * dx[0]);
#if AMREX_SPACEDIM == 3
			gradT_z = 0.5 *
				  ((temp[bx](i, j - 1, k + 1, comp) - temp[bx](i, j - 1, k - 1, comp)) +
				   (temp[bx](i, j, k + 1, comp) - temp[bx](i, j, k - 1, comp))) /
				  (2.0 * dx[2]);
#endif
		} else { // FluxDir::X3
			gradT_z = (temp[bx](i, j, k, comp) - temp[bx](i, j, k - 1, comp)) / dx[2];
			gradT_x = 0.5 *
				  ((temp[bx](i + 1, j, k - 1, comp) - temp[bx](i - 1, j, k - 1, comp)) +
				   (temp[bx](i + 1, j, k, comp) - temp[bx](i - 1, j, k, comp))) /
				  (2.0 * dx[0]);
			gradT_y = 0.5 *
				  ((temp[bx](i, j + 1, k - 1, comp) - temp[bx](i, j - 1, k - 1, comp)) +
				   (temp[bx](i, j + 1, k, comp) - temp[bx](i, j - 1, k, comp))) /
				  (2.0 * dx[1]);
		}

		gradT_out[bx](i, j, k, 0) = gradT_x;
		gradT_out[bx](i, j, k, 1) = gradT_y;
		gradT_out[bx](i, j, k, 2) = gradT_z;
	});
}

// Compute the number density at the DIR-faces from a cell-centered mass-density field (e.g.
// primVar component 0): the average of the two cells bounding each face, divided by the mean
// molecular weight (already in mass units, e.g. EOS_Traits<problem_t>::mean_molecular_weight).
template <FluxDir DIR>
void ComputeFaceNumberDensity(amrex::MultiFab &n_fc_mf, amrex::MultiFab const &density, amrex::Real mean_molecular_weight, int comp)
{
	auto const &rho_in = density.const_arrays();
	auto n_out = n_fc_mf.arrays();

	amrex::ParallelFor(n_fc_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real rho_face = 0.0;
		if constexpr (DIR == FluxDir::X1) {
			rho_face = 0.5 * (rho_in[bx](i, j, k, comp) + rho_in[bx](i - 1, j, k, comp));
		} else if constexpr (DIR == FluxDir::X2) {
			rho_face = 0.5 * (rho_in[bx](i, j, k, comp) + rho_in[bx](i, j - 1, k, comp));
		} else { // FluxDir::X3
			rho_face = 0.5 * (rho_in[bx](i, j, k, comp) + rho_in[bx](i, j, k - 1, comp));
		}
		n_out[bx](i, j, k) = rho_face / mean_molecular_weight;
	});
}

// Compute the anisotropic heat flux crossing the DIR-faces, given the unit B-field (bhat_fc, from
// ComputeFaceUnitBField) and the temperature gradient (gradT_fc, from ComputeFaceGradT) already
// estimated at those faces. Number density at the face is computed internally via
// ComputeFaceNumberDensity, from primVar (component 0 is rho) and mean_molecular_weight. For now
// this only applies the field-aligned term:
//   q . nhat = -kappa_parallel * (bhat . gradT) * (bhat . nhat) * n
// (kappa_perp is accepted but not yet used -- the perpendicular term and q_sat are still to be
// fixed), then flux-limited/saturated exactly as in ElectronConduction::ComputeExplicit:
//   flux = q_classical / (1 + |q_classical| / max(q_sat, small)).
// q_sat_fc is the (precomputed) saturation flux at each face -- this function does not compute it.
template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &bhat_fc, amrex::MultiFab const &gradT_fc, amrex::MultiFab const &primVar,
			    amrex::Real mean_molecular_weight, amrex::Real kappa_parallel, amrex::Real kappa_perp, amrex::MultiFab const &q_sat_fc)
{
	amrex::MultiFab n_fc(heat_flux_fc.boxArray(), heat_flux_fc.DistributionMap(), 1, 0);
	ComputeFaceNumberDensity<DIR>(n_fc, primVar, mean_molecular_weight, 0);

	constexpr int normal_comp = static_cast<int>(DIR);
	const amrex::Real small = std::numeric_limits<amrex::Real>::min();

	auto const &bhat_in = bhat_fc.const_arrays();
	auto const &gradT_in = gradT_fc.const_arrays();
	auto const &qsat_in = q_sat_fc.const_arrays();
	auto const &n_in = n_fc.const_arrays();
	auto flux_out = heat_flux_fc.arrays();

	amrex::ParallelFor(heat_flux_fc, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real bdotgradT = bhat_in[bx](i, j, k, 0) * gradT_in[bx](i, j, k, 0) + bhat_in[bx](i, j, k, 1) * gradT_in[bx](i, j, k, 1) +
					      bhat_in[bx](i, j, k, 2) * gradT_in[bx](i, j, k, 2);
		const amrex::Real bn = bhat_in[bx](i, j, k, normal_comp);
		const amrex::Real n = n_in[bx](i, j, k);

		const amrex::Real q_classical = -kappa_parallel * bdotgradT * bn * n;
		const amrex::Real q_sat = qsat_in[bx](i, j, k);
		const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat, small);
		flux_out[bx](i, j, k) = q_classical / limiter;
	});
}

} // namespace quokka::conduction

#endif // ANISO_CONDUCTION_HPP_
