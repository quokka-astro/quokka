#ifndef ANISO_CONDUCTION_HPP_ // NOLINT
#define ANISO_CONDUCTION_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file AnisoConduction.hpp
/// \brief Explicit anisotropic thermal conduction update. At each face, gradT is projected onto
///        the local unit B-field (bhat) and scaled by kappa_parallel (kappa_perp is not yet
///        folded in), then flux-limited/saturated exactly as in ElectronConduction::ComputeExplicit,
///        using a saturation flux computed from (rho, T) averaged from the two cells bounding the
///        face (see ComputeFaceNumberDensityAndSaturationFlux).

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

template <FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::IntVect FaceMinusOne(int i, int j, int k)
{
	if constexpr (DIR == FluxDir::X1) { return {i - 1, j, k}; }
	else if constexpr (DIR == FluxDir::X2) { return {i, j - 1, k}; }
	else { return {i, j, k - 1}; }
}

// Declarations only -- see the definitions below AnisoConduction for what each of these actually
// does. Declared here so that AnisoConduction::ComputeExplicit (which calls all four) can be read
// first; scroll down past the class for the bodies.
template <typename problem_t, FluxDir DIR>
void ComputeFaceUnitBField(amrex::MultiFab &bhat_fc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int reconstructionOrder,
			   SlopeLimiter plmLimiter, int nghost);

template <FluxDir DIR>
void ComputeFaceGradT(amrex::MultiFab &gradT_fc_mf, amrex::MultiFab const &temperature, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, int comp);

template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &bhat_fc, amrex::MultiFab const &gradT_fc, amrex::MultiFab const &n_fc,
			    amrex::Real kappa_parallel, amrex::Real kappa_perp, amrex::MultiFab const &q_sat_fc);

// Computes both the face number density and the saturation flux in a single pass, since both are
// derived from the same (rho, T) face average -- see the definition below for details.
template <typename problem_t, FluxDir DIR>
void ComputeFaceNumberDensityAndSaturationFlux(amrex::MultiFab &n_fc_mf, amrex::MultiFab &qsat_fc_mf, amrex::MultiFab const &primVar,
					       amrex::Real mean_molecular_weight, amrex::Real saturation_factor, amrex::Real flux_limiter_phi);

template <typename problem_t> class AnisoConduction
{
      public:

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

		constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
		// primVar holds (rho, T, massScalars...) so that mass scalars, like rho/T, are computed
		// once per cell here and reused (via face-averaging) instead of being re-derived from
		// `state` at every face below.
		amrex::MultiFab primVar(state.boxArray(), state.DistributionMap(), 2 + nmscalars_, state.nGrow());
		auto const &state_x0 = state.const_arrays();
		primVar.setVal(0.0);

		auto primVar_arr = primVar.arrays();
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
			amrex::GpuArray<amrex::Real, nmscalars_> const massScalarsArr = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
			quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const massScalars = massScalarsArr;
			const amrex::Real Tgas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);

			primVar_arr[bx](i, j, k, 0) = rho;
			primVar_arr[bx](i, j, k, 1) = amrex::max(Tgas, t_min);
			for (int n = 0; n < nmscalars_; ++n) {
				primVar_arr[bx](i, j, k, 2 + n) = massScalarsArr[n];
			}
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
			ComputeFaceUnitBField<problem_t, FluxDir::X1>(bhat_fc[0], state_fc, params.reconstruction_order, params.plm_limiter, ng_reconstruct);
#if AMREX_SPACEDIM >= 2
			ComputeFaceUnitBField<problem_t, FluxDir::X2>(bhat_fc[1], state_fc, params.reconstruction_order, params.plm_limiter, ng_reconstruct);
#endif
#if AMREX_SPACEDIM == 3
			ComputeFaceUnitBField<problem_t, FluxDir::X3>(bhat_fc[2], state_fc, params.reconstruction_order, params.plm_limiter, ng_reconstruct);
#endif
		}
		AMREX_D_TERM(ComputeFaceGradT<FluxDir::X1>(gradT_fc[0], primVar, dx, 1);, ComputeFaceGradT<FluxDir::X2>(gradT_fc[1], primVar, dx, 1);
			     , ComputeFaceGradT<FluxDir::X3>(gradT_fc[2], primVar, dx, 1);)

		// Number density and saturation flux at each face (Cowie & McKee 1977), both derived from
		// the same (rho, T, massScalars...) averaged from the two bounding cells -- see
		// ComputeFaceNumberDensityAndSaturationFlux.
		const amrex::Real mmw = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> n_fc;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> q_sat_fc;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			n_fc[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 1, 0);
			q_sat_fc[idim] = amrex::MultiFab(ba_face, state.DistributionMap(), 1, 0);
		}
		AMREX_D_TERM((ComputeFaceNumberDensityAndSaturationFlux<problem_t, FluxDir::X1>(n_fc[0], q_sat_fc[0], primVar, mmw, params.saturation_factor,
												params.flux_limiter_phi));
			     , (ComputeFaceNumberDensityAndSaturationFlux<problem_t, FluxDir::X2>(n_fc[1], q_sat_fc[1], primVar, mmw,
													    params.saturation_factor, params.flux_limiter_phi));
			     , (ComputeFaceNumberDensityAndSaturationFlux<problem_t, FluxDir::X3>(n_fc[2], q_sat_fc[2], primVar, mmw,
													    params.saturation_factor, params.flux_limiter_phi));)

		// Heat flux at each face: full parallel/perpendicular decomposition of gradT relative to bhat,
		// projected onto the face normal -- see ComputeAnisotropicFlux.
		AMREX_D_TERM(ComputeAnisotropicFlux<FluxDir::X1>(heat_flux[0], bhat_fc[0], gradT_fc[0], n_fc[0], params.kappa_parallel, params.kappa_perp,
								  q_sat_fc[0]);
			     , ComputeAnisotropicFlux<FluxDir::X2>(heat_flux[1], bhat_fc[1], gradT_fc[1], n_fc[1], params.kappa_parallel, params.kappa_perp,
								    q_sat_fc[1]);
			     , ComputeAnisotropicFlux<FluxDir::X3>(heat_flux[2], bhat_fc[2], gradT_fc[2], n_fc[2], params.kappa_parallel, params.kappa_perp,
								    q_sat_fc[2]);)

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
		amrex::IntVect const ivm = FaceMinusOne<DIR>(i, j, k);

		if constexpr (DIR == FluxDir::X1) {
			gradT_x = (temp[bx](i, j, k, comp) - temp[bx](ivm, comp)) / dx[0];
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
			gradT_y = (temp[bx](i, j, k, comp) - temp[bx](ivm, comp)) / dx[1];
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
			gradT_z = (temp[bx](i, j, k, comp) - temp[bx](ivm, comp)) / dx[2];
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

// Compute the anisotropic heat flux crossing the DIR-faces, given the unit B-field (bhat_fc, from
// ComputeFaceUnitBField), the temperature gradient (gradT_fc, from ComputeFaceGradT), and the
// number density (n_fc, from ComputeFaceNumberDensityAndSaturationFlux) already estimated at those
// faces. This applies the full parallel + perpendicular decomposition:
//   q_vec = -(kappa_parallel - kappa_perp) * (bhat . gradT) * n * bhat  -  n * kappa_perp * gradT
// and stores the face-normal component q . nhat, then flux-limited/saturated exactly as in
// ElectronConduction::ComputeExplicit:
//   flux = q_classical / (1 + |q_classical| / max(q_sat, small)).
// q_sat_fc is the (precomputed) saturation flux at each face -- this function does not compute it.
template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &bhat_fc, amrex::MultiFab const &gradT_fc, amrex::MultiFab const &n_fc,
			    amrex::Real kappa_parallel, amrex::Real kappa_perp, amrex::MultiFab const &q_sat_fc)
{
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

		const amrex::Real q_classical = -(kappa_parallel - kappa_perp) * bdotgradT * bn * n - n * kappa_perp * gradT_in[bx](i, j, k, normal_comp);
		const amrex::Real q_sat = qsat_in[bx](i, j, k);
		const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat, small);
		flux_out[bx](i, j, k) = q_classical / limiter;
	});
}

// Compute the face number density and the saturation flux (Cowie & McKee 1977) at the DIR-faces in
// a single pass, since both derive from the same (rho, T, massScalars...) average of primVar over
// the two cells bounding each face. n_fc is (rho_face / mean_molecular_weight); q_sat_fc is the
// saturation flux used to limit/saturate the classical flux in ComputeAnisotropicFlux.
template <typename problem_t, FluxDir DIR>
void ComputeFaceNumberDensityAndSaturationFlux(amrex::MultiFab &n_fc_mf, amrex::MultiFab &qsat_fc_mf, amrex::MultiFab const &primVar,
					       amrex::Real mean_molecular_weight, amrex::Real saturation_factor, amrex::Real flux_limiter_phi)
{
	constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	const amrex::Real small = std::numeric_limits<amrex::Real>::min();

	auto const &primVar_in = primVar.const_arrays();
	auto n_out = n_fc_mf.arrays();
	auto qsat_out = qsat_fc_mf.arrays();

	amrex::ParallelFor(qsat_fc_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::IntVect const ivm = FaceMinusOne<DIR>(i, j, k);
		auto const face_avg = [=](int comp) { return 0.5 * (primVar_in[bx](i, j, k, comp) + primVar_in[bx](ivm, comp)); };
		const amrex::Real rho_face = face_avg(0);
		const amrex::Real T_face = face_avg(1);

		n_out[bx](i, j, k) = rho_face / mean_molecular_weight;

		amrex::GpuArray<amrex::Real, nmscalars_> massArray_face{};
		for (int n = 0; n < nmscalars_; ++n) {
			massArray_face[n] = face_avg(2 + n);
		}
		quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = massArray_face;

		const amrex::Real Eint_face = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho_face, T_face, massScalars);
		const amrex::Real Pgas_face = ::quokka::EOS<problem_t>::ComputePressure(rho_face, Eint_face, massScalars);
		const amrex::Real cs_face = ::quokka::EOS<problem_t>::ComputeSoundSpeed(rho_face, Pgas_face, massScalars);

		qsat_out[bx](i, j, k) = amrex::max(saturation_factor * flux_limiter_phi * rho_face * cs_face * cs_face * cs_face, small);
	});
}

} // namespace quokka::conduction

#endif // ANISO_CONDUCTION_HPP_
