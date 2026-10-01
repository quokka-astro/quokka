#ifndef ANISO_CONDUCTION_HPP_ // NOLINT
#define ANISO_CONDUCTION_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file AnisoConduction.hpp
/// \brief Explicit anisotropic thermal conduction update, following the symmetric flux-limiting
///        scheme of Sharma & Hammett (2007). Each face's flux is split into a normal ("diagonal",
///        q_xx-type) term and one transverse ("cross", q_xy/q_xz-type) term per transverse
///        direction; the normal term is limited with the biased L2 limiter (their Eq. 21, sign-
///        definite so it tolerates a biased limiter), while each cross term is limited with a
///        nested symmetric limiter (L(L(a,b),L(c,d))), MC or minmod, since it is sign-indefinite
///        (selected by AnisoConductionParams::flux_limiter_type). Both terms read bhat/kappa at
///        mesh vertices ("corners") and gradT via direct two-point differences of cell-centered T,
///        never a precomputed face/corner gradT field. kappa_par and kappa_perp are stored at each
///        corner, kappa = n k_B chi (Sharma & Hammett's n chi), with chi from EvaluateDiffusivity
///        (conductivity.hpp) evaluated at the corner-averaged (rho, T). The flux is
///        q = -kappa_perp grad T - (kappa_par - kappa_perp) bhat (bhat . grad T): the diagonal and
///        cross terms above carry (kappa_par - kappa_perp), and the isotropic kappa_perp term is an
///        unlimited two-point difference across the face. The result is saturated exactly as in
///        IsoConduction::ComputeExplicit, using a saturation flux computed from (rho, T)
///        averaged from the corners bounding the face.

#include <array>
#include <cmath>
#include <limits>

#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_Enum.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "conduction/conductivity.hpp"
#include "hydro/hydro_system.hpp"
#include "hyperbolic_system.hpp"

namespace quokka::conduction
{

// Selects the limiter LimitUpperLowerFlux applies to the transverse temperature gradients of the cross
// term (Sharma & Hammett 2007), read from conduction.aniso_flux_limiter. Minmod and MC (monotonized
// central: default) are implemented.
AMREX_ENUM(AnisoFluxLimiterType, minmod, mc); // NOLINT

struct AnisoConductionParams {
	ConductivityParams conductivity{}; // prefactors for ConductionModel::constant/spitzer (see conductivity.hpp)
	amrex::Real l2_alpha = 2.0;	   // bias factor for the L2 limiter (Sharma & Hammett 2007 Eq. 21) applied to the normal (q_xx-type) term
	amrex::Real flux_limiter_phi = 0.1;
	amrex::Real saturation_factor = 5.0; // refer to equation 8 of Cowie & McKee 1977
	amrex::Real min_temperature = 0.0;   // default value will be overwritten by tempFloor_ during initialization
	int reconstruction_order = 3;	     // 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM;
	SlopeLimiter plm_limiter = SlopeLimiter::sweby;
	int ng_reconstruct = 2;						   // number of ghost faces to reconstruct beyond the valid box
	AnisoFluxLimiterType flux_limiter_type = AnisoFluxLimiterType::mc; // transverse-gradient limiter (conduction.aniso_flux_limiter)
};

template <FluxDir DIR> AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::IntVect FaceMinusOne(int i, int j, int k)
{
	amrex::ignore_unused(i, j, k);
	return amrex::IntVect(AMREX_D_DECL(i, j, k)) - amrex::IntVect::TheDimensionVector(static_cast<int>(DIR));
}

// The "upper" corner bounding a DIR-face, given the face's own (lower-corner) index (i,j,k) -- i.e.
// shifted by +1 in every TRANSVERSE direction (never in DIR's own direction, since the face is
// already exactly at the right position along DIR). In 2D there is only one transverse direction, so
// this is unambiguous; in 3D a face has 4 corners, and this picks the one diagonally opposite the
// lower corner (i,j,k), leaving the other two corners (offset in only one transverse direction) unused.
template <FluxDir DIR> AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::IntVect UpperCorner(int i, int j, int k)
{
	amrex::ignore_unused(i, j, k);
	return amrex::IntVect(AMREX_D_DECL(i + 1, j + 1, k + 1)) - amrex::IntVect::TheDimensionVector(static_cast<int>(DIR));
}

// Combines two per-corner or per-neighbor estimates (q_lower, q_upper) into a single limited
// value
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real LimitUpperLowerFlux(amrex::Real q_lower, amrex::Real q_upper, AnisoFluxLimiterType limiter_type)
{
	switch (limiter_type) {
		case AnisoFluxLimiterType::mc:
			if (q_lower * q_upper <= 0.0) {
				return 0.0;
			}
			return std::copysign(amrex::min(2.0 * std::abs(q_lower), 2.0 * std::abs(q_upper), 0.5 * std::abs(q_lower + q_upper)), q_lower);
		case AnisoFluxLimiterType::minmod:
		default:
			if (q_lower * q_upper <= 0.0) {
				return 0.0;
			}
			return (q_lower > 0.0) ? amrex::min(q_lower, q_upper) : amrex::max(q_lower, q_upper);
	}
}

// (kappa_par - kappa_perp) * bn^2 -- the field-aligned part of the diagonal tensor entry (q_xx-type term).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real DiagCoeff(amrex::Real bn, amrex::Real kappa_aniso) { return kappa_aniso * bn * bn; }

// (kappa_par - kappa_perp) * bi * bj -- the off-diagonal tensor entry (q_xy/q_xz-type term).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real CrossCoeff(amrex::Real bi, amrex::Real bj, amrex::Real kappa_aniso) { return kappa_aniso * bi * bj; }

// Sharma & Hammett (2007) Eq. (21): a biased limiter
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real L2(amrex::Real anchor, amrex::Real neighbor, amrex::Real alpha)
{
	const amrex::Real avg = 0.5 * (anchor + neighbor);
	const amrex::Real lo = amrex::min(alpha * anchor, anchor / alpha);
	const amrex::Real hi = amrex::max(alpha * anchor, anchor / alpha);
	if (avg > lo && avg < hi) {
		return avg;
	}
	return (avg <= lo) ? lo : hi;
}

// L(L(a,b), L(c,d)) -- Eq. (17) / Sec. 6.1's nested-limiter pattern for the (sign-indefinite)
// transverse term, with L = minmod or MC per `limiter_type`.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real NestedLimit(amrex::Real a, amrex::Real b, amrex::Real c, amrex::Real d, AnisoFluxLimiterType limiter_type)
{
	return LimitUpperLowerFlux(LimitUpperLowerFlux(a, b, limiter_type), LimitUpperLowerFlux(c, d, limiter_type), limiter_type);
}

// q/(1 + |q|/qsat) -- shared by ComputeAnisotropicFlux so the saturation formula lives in one place.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real SaturateFlux(amrex::Real q_classical, amrex::Real q_sat, amrex::Real small)
{
	return q_classical / (1.0 + std::abs(q_classical) / amrex::max(q_sat, small));
}

// Unit IntVect shift along a single axis (0=x, 1=y, 2=z). Used to step from a face's own "lower"
// bounding corner to its immediate neighbor along exactly one transverse axis -- unlike
// UpperCorner<DIR>, which steps along every transverse axis of the face simultaneously.
template <int Axis> AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::IntVect AxisUnit()
{
	static_assert(Axis >= 0 && Axis < AMREX_SPACEDIM, "AxisUnit: Axis must be a valid spatial direction");
	return amrex::IntVect::TheDimensionVector(Axis);
}

// q_xx-type normal term for a DIR-face at (i,j,k) (cells cm, cp), limited across the `Axis`-
// transverse direction via L2. `corner_lo` is the face's lower bounding corner along every axis
// OTHER than `Axis` (fixed by the caller, e.g. to average over a second transverse axis in 3D);
// `corner_lo` and `corner_lo + AxisUnit<Axis>()` are the two corners this term reads bhat/kappa from.
template <FluxDir DIR, int Axis>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
ComputeDiagTerm(amrex::Array4<const amrex::Real> const &T, amrex::Array4<const amrex::Real> const &bhat, amrex::Array4<const amrex::Real> const &kappa,
		amrex::IntVect const &cm, amrex::IntVect const &cp, amrex::IntVect const &corner_lo, amrex::Real dx_n, amrex::Real l2_alpha)
{
	constexpr int normal_comp = static_cast<int>(DIR);
	constexpr int T_comp = 1;
	amrex::IntVect const shift = AxisUnit<Axis>();

	const amrex::Real anchor = (T(cp, T_comp) - T(cm, T_comp)) / dx_n;
	const amrex::Real north = (T(cp + shift, T_comp) - T(cm + shift, T_comp)) / dx_n;
	const amrex::Real south = (T(cp - shift, T_comp) - T(cm - shift, T_comp)) / dx_n;
	const amrex::Real grad_N = L2(anchor, north, l2_alpha);
	const amrex::Real grad_S = L2(anchor, south, l2_alpha);

	amrex::IntVect const corner_N = corner_lo + shift;
	amrex::IntVect const corner_S = corner_lo;

	const amrex::Real q_N = -DiagCoeff(bhat(corner_N, normal_comp), kappa(corner_N, 0) - kappa(corner_N, 1)) * grad_N;
	const amrex::Real q_S = -DiagCoeff(bhat(corner_S, normal_comp), kappa(corner_S, 0) - kappa(corner_S, 1)) * grad_S;
	return 0.5 * (q_N + q_S);
}

// q_xy/q_xz-type cross term for a DIR-face at (i,j,k) (cells cm, cp), limited across the `Axis`-
// transverse direction via NestedLimit. `corner_lo`/`corner_lo + AxisUnit<Axis>()` are the two
// corners bhat/kappa are face-averaged from (see the same `corner_lo` convention as ComputeDiagTerm).
template <FluxDir DIR, int Axis>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
ComputeCrossTerm(amrex::Array4<const amrex::Real> const &T, amrex::Array4<const amrex::Real> const &bhat, amrex::Array4<const amrex::Real> const &kappa,
		 amrex::IntVect const &cm, amrex::IntVect const &cp, amrex::IntVect const &corner_lo, amrex::Real dx_t, AnisoFluxLimiterType limiter_type)
{
	constexpr int normal_comp = static_cast<int>(DIR);
	constexpr int T_comp = 1;
	amrex::IntVect const shift = AxisUnit<Axis>();

	const amrex::Real d_cm_lo = (T(cm, T_comp) - T(cm - shift, T_comp)) / dx_t;
	const amrex::Real d_cm_hi = (T(cm + shift, T_comp) - T(cm, T_comp)) / dx_t;
	const amrex::Real d_cp_lo = (T(cp, T_comp) - T(cp - shift, T_comp)) / dx_t;
	const amrex::Real d_cp_hi = (T(cp + shift, T_comp) - T(cp, T_comp)) / dx_t;
	const amrex::Real slope = NestedLimit(d_cm_lo, d_cm_hi, d_cp_lo, d_cp_hi, limiter_type);

	amrex::IntVect const corner_hi = corner_lo + shift;
	const amrex::Real bn_face = 0.5 * (bhat(corner_lo, normal_comp) + bhat(corner_hi, normal_comp));
	const amrex::Real bt_face = 0.5 * (bhat(corner_lo, Axis) + bhat(corner_hi, Axis));
	const amrex::Real kappa_face = 0.5 * ((kappa(corner_lo, 0) - kappa(corner_lo, 1)) + (kappa(corner_hi, 0) - kappa(corner_hi, 1)));

	return -CrossCoeff(bn_face, bt_face, kappa_face) * slope;
}

// Unit B-field at mesh vertices ("corners"), built from FACE-CENTERED input data (state_fc)
template <typename problem_t> void ComputeCornerFC(amrex::MultiFab &bhat_corner_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int nghost);

// kappa and qsat at mesh vertices ("corners"), built from CELL-CENTERED input data (primVar)
template <typename problem_t>
void ComputeCornerCC(amrex::MultiFab &kappa_corner_mf, amrex::MultiFab &qsat_corner_mf, amrex::MultiFab const &primVar, ConductivityParams const &conductivity,
		     amrex::Real saturation_factor, amrex::Real flux_limiter_phi, int nghost);

template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &primVar, amrex::MultiFab const &bhat_corner,
			    amrex::MultiFab const &kappa_corner, amrex::MultiFab const &qsat_corner, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			    amrex::Real l2_alpha, AnisoFluxLimiterType limiter_type);

template <typename problem_t> class AnisoConduction
{
      public:
	static void ComputeExplicit(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    amrex::Real dt, AnisoConductionParams const &params, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		constexpr ConductionModel model = Physics_Traits<problem_t>::conduction_model;
		if constexpr (model == ConductionModel::none) {
			amrex::ignore_unused(state, state_fc, geom, dt, params, heat_flux);
			return;
		}
		if (dt <= 0.0) {
			return;
		}
		if constexpr (model == ConductionModel::constant || model == ConductionModel::spitzer) {
			if ((params.conductivity.kappa0_par <= 0.0) && (params.conductivity.kappa0_perp <= 0.0)) {
				return;
			}
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

		// heat_flux at each face -- the actual per-direction output of this routine.
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}

		// Unit B-field, kappa, and qsat at mesh vertices ("corners")
		amrex::BoxArray const ba_corner = amrex::convert(state.boxArray(), amrex::IntVect::TheUnitVector());
		amrex::MultiFab bhat_corner(ba_corner, state.DistributionMap(), 3, 0);
		amrex::MultiFab kappa_corner(ba_corner, state.DistributionMap(), 2, 0); // (kappa_par, kappa_perp)
		amrex::MultiFab qsat_corner(ba_corner, state.DistributionMap(), 1, 0);
		bhat_corner.setVal(0.0); // zeroed when MHD is disabled, so a non-MHD build gets zero flux rather than reading uninitialized data.

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			ComputeCornerFC<problem_t>(bhat_corner, state_fc, 0);
		}

		ComputeCornerCC<problem_t>(kappa_corner, qsat_corner, primVar, params.conductivity, params.saturation_factor, params.flux_limiter_phi, 0);

		// Heat flux at each face: ComputeAnisotropicFlux splits the flux into a normal (q_xx-type)
		// term limited via L2 and one transverse (q_xy/q_xz-type) term per transverse direction
		// limited via NestedLimit, then saturates the sum (Sharma & Hammett 2007) -- see the file
		// doc-comment and ComputeDiagTerm/ComputeCrossTerm for details.
		AMREX_D_TERM(ComputeAnisotropicFlux<FluxDir::X1>(heat_flux[0], primVar, bhat_corner, kappa_corner, qsat_corner, dx, params.l2_alpha,
								 params.flux_limiter_type);
			     , ComputeAnisotropicFlux<FluxDir::X2>(heat_flux[1], primVar, bhat_corner, kappa_corner, qsat_corner, dx, params.l2_alpha,
								   params.flux_limiter_type);
			     , ComputeAnisotropicFlux<FluxDir::X3>(heat_flux[2], primVar, bhat_corner, kappa_corner, qsat_corner, dx, params.l2_alpha,
								   params.flux_limiter_type);)

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

template <typename problem_t> void ComputeCornerFC(amrex::MultiFab &bhat_corner_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int nghost)
{
	static_assert(Physics_Traits<problem_t>::is_mhd_enabled);
	constexpr int b_comp = Physics_Indices<problem_t>::mhdFirstIndex;

	auto const &bx_fc = state_fc[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &by_fc = state_fc[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &bz_fc = state_fc[2].const_arrays();
#endif
	auto bhat_out = bhat_corner_mf.arrays();

	amrex::IntVect const ng{AMREX_D_DECL(nghost, nghost, nghost)};
	amrex::ParallelFor(bhat_corner_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real Bx = 0.0;
		amrex::Real By = 0.0;
		amrex::Real Bz = 0.0;

#if AMREX_SPACEDIM == 3
		Bx = 0.25 * (bx_fc[bx](i, j - 1, k - 1, b_comp) + bx_fc[bx](i, j, k - 1, b_comp) + bx_fc[bx](i, j - 1, k, b_comp) + bx_fc[bx](i, j, k, b_comp));
		By = 0.25 * (by_fc[bx](i - 1, j, k - 1, b_comp) + by_fc[bx](i, j, k - 1, b_comp) + by_fc[bx](i - 1, j, k, b_comp) + by_fc[bx](i, j, k, b_comp));
		Bz = 0.25 * (bz_fc[bx](i - 1, j - 1, k, b_comp) + bz_fc[bx](i, j - 1, k, b_comp) + bz_fc[bx](i - 1, j, k, b_comp) + bz_fc[bx](i, j, k, b_comp));
#elif AMREX_SPACEDIM == 2
		Bx = 0.5 * (bx_fc[bx](i, j - 1, k, b_comp) + bx_fc[bx](i, j, k, b_comp));
		By = 0.5 * (by_fc[bx](i - 1, j, k, b_comp) + by_fc[bx](i, j, k, b_comp));
#else
		Bx = bx_fc[bx](i, j, k, b_comp);
#endif

		const amrex::Real bmag = std::sqrt(Bx * Bx + By * By + Bz * Bz);
		const amrex::Real inv_b = (bmag > 0.0) ? (1.0 / bmag) : 0.0;

		bhat_out[bx](i, j, k, 0) = Bx * inv_b;
		bhat_out[bx](i, j, k, 1) = By * inv_b;
		bhat_out[bx](i, j, k, 2) = Bz * inv_b;
	});
}

// Estimate kappa and qsat at mesh vertices ("corners") from CELL-CENTERED input data (primVar), on the
// same fully-nodal box array and vertex-indexing convention as ComputeCornerFC (vertex (i,j,k) draws
// from the 8 (3D) or 4 (2D) cells with indices in {i-1,i}x{j-1,j}x{k-1,k}). (rho,T,massScalars) are
// the plain average over those cells; kappa = n k_B (chi_parallel, chi_perp) with chi from EvaluateDiffusivity, and
// qsat follows the same EOS chain as ComputeFaceNumberDensityAndSaturationFlux (Cowie & McKee 1977). gradT is intentionally not computed
// here -- ComputeAnisotropicFlux differences primVar directly (see ComputeDiagTerm/ComputeCrossTerm).
template <typename problem_t>
void ComputeCornerCC(amrex::MultiFab &kappa_corner_mf, amrex::MultiFab &qsat_corner_mf, amrex::MultiFab const &primVar, ConductivityParams const &conductivity,
		     amrex::Real saturation_factor, amrex::Real flux_limiter_phi, int nghost)
{
	constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	constexpr int T_comp = 1;
	const amrex::Real small = std::numeric_limits<amrex::Real>::min();

	auto const &primVar_in = primVar.const_arrays();
	const ConductivityParams conductivity_params = conductivity;
	const amrex::Real mean_molecular_weight = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
	const amrex::Real k_B = quokka::EOS<problem_t>::boltzmann_constant_;
	auto kappa_out = kappa_corner_mf.arrays();
	auto qsat_out = qsat_corner_mf.arrays();

	amrex::IntVect const ng{AMREX_D_DECL(nghost, nghost, nghost)};
	amrex::ParallelFor(kappa_corner_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// kappa / qsat: plain average of (rho, T, massScalars) over the {i-1,i}x{j-1,j}x{k-1,k} cells
		// touching this vertex, then the same EOS chain ComputeFaceNumberDensityAndSaturationFlux uses.
		auto const corner_avg = [=](int comp) {
#if AMREX_SPACEDIM == 3
			return 0.125 * (primVar_in[bx](i - 1, j - 1, k - 1, comp) + primVar_in[bx](i, j - 1, k - 1, comp) +
					primVar_in[bx](i - 1, j, k - 1, comp) + primVar_in[bx](i, j, k - 1, comp) + primVar_in[bx](i - 1, j - 1, k, comp) +
					primVar_in[bx](i, j - 1, k, comp) + primVar_in[bx](i - 1, j, k, comp) + primVar_in[bx](i, j, k, comp));
#elif AMREX_SPACEDIM == 2
			return 0.25 * (primVar_in[bx](i - 1, j - 1, k, comp) + primVar_in[bx](i, j - 1, k, comp) + primVar_in[bx](i - 1, j, k, comp) +
				       primVar_in[bx](i, j, k, comp));
#else
			return 0.5 * (primVar_in[bx](i - 1, j, k, comp) + primVar_in[bx](i, j, k, comp));
#endif
		};
		const amrex::Real rho_corner = corner_avg(0);
		const amrex::Real T_corner = corner_avg(T_comp);

		// kappa = n k_B chi (Sharma & Hammett's n chi); see conductivity.hpp
		const auto chi_corner = EvaluateDiffusivity<problem_t>(rho_corner, T_corner, conductivity_params);
		const amrex::Real nkB_corner = (rho_corner / mean_molecular_weight) * k_B;
		kappa_out[bx](i, j, k, 0) = nkB_corner * chi_corner[0];
		// the L2-limited (kappa_par - kappa_perp) b_n^2 term must stay non-negative; the input-file prefactors are
		// asserted at startup, so this can only fire with ConductionModel::problem_defined (debug builds only)
		AMREX_ASSERT(chi_corner[1] <= chi_corner[0]);
		kappa_out[bx](i, j, k, 1) = nkB_corner * chi_corner[1];

		amrex::GpuArray<amrex::Real, nmscalars_> massArray_corner{};
		for (int n = 0; n < nmscalars_; ++n) {
			massArray_corner[n] = corner_avg(2 + n);
		}
		quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const massScalars = massArray_corner;

		const amrex::Real Eint_corner = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho_corner, T_corner, massScalars);
		const amrex::Real Pgas_corner = ::quokka::EOS<problem_t>::ComputePressure(rho_corner, Eint_corner, massScalars);
		const amrex::Real cs_corner = ::quokka::EOS<problem_t>::ComputeSoundSpeed(rho_corner, Pgas_corner, massScalars);

		qsat_out[bx](i, j, k) = amrex::max(saturation_factor * flux_limiter_phi * rho_corner * cs_corner * cs_corner * cs_corner, small);
	});
}

// Compute the anisotropic heat flux crossing the DIR-faces (Sharma & Hammett 2007's symmetric
// scheme), given bhat/kappa/qsat at mesh vertices (ComputeCornerFC/ComputeCornerCC) and T at cell centers
// (primVar). Splits q_classical into a normal (q_xx-type) term (ComputeDiagTerm, limited via L2), one
// transverse (q_xy/q_xz-type) term per transverse direction (ComputeCrossTerm, limited via
// NestedLimit), and an unlimited isotropic -kappa_perp dT/dn term (kappa_perp face-averaged from the
// corners, like qsat); in 3D each term is evaluated once per position along the *other* transverse axis and
// averaged, so every term ultimately draws from all 4 corners bounding the face. qsat is likewise the
// plain average of all corners bounding the face. See the file doc-comment for the overall scheme and
// ComputeDiagTerm/ComputeCrossTerm's doc-comments for the per-term derivations.
template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &primVar, amrex::MultiFab const &bhat_corner,
			    amrex::MultiFab const &kappa_corner, amrex::MultiFab const &qsat_corner, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			    amrex::Real l2_alpha, AnisoFluxLimiterType limiter_type)
{
	constexpr int normal_comp = static_cast<int>(DIR);
	constexpr int T_comp = 1;
	const amrex::Real small = std::numeric_limits<amrex::Real>::min();
	const amrex::Real dx_n = dx[normal_comp];

	auto const &T_in = primVar.const_arrays();
	auto const &bhat_in = bhat_corner.const_arrays();
	auto const &kappa_in = kappa_corner.const_arrays();
	auto const &qsat_in = qsat_corner.const_arrays();
	auto flux_out = heat_flux_fc.arrays();

	amrex::ParallelFor(heat_flux_fc, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::IntVect const cm = FaceMinusOne<DIR>(i, j, k);
		amrex::IntVect const cp(AMREX_D_DECL(i, j, k));
		amrex::IntVect const corner_lo(AMREX_D_DECL(i, j, k));

#if AMREX_SPACEDIM == 1
		amrex::ignore_unused(corner_lo);
		const amrex::Real kappa_perp_face = kappa_in[bx](i, j, k, 1);
		const amrex::Real gradT_n = (T_in[bx](cp, T_comp) - T_in[bx](cm, T_comp)) / dx_n;
		const amrex::Real q_classical =
		    -DiagCoeff(bhat_in[bx](i, j, k, normal_comp), kappa_in[bx](i, j, k, 0) - kappa_perp_face) * gradT_n - kappa_perp_face * gradT_n;
		const amrex::Real qsat_face = qsat_in[bx](i, j, k, 0);
#else
#if AMREX_SPACEDIM == 2
		// The only transverse axis of an x-face is y, and of a y-face is x.
		constexpr int Ax0 = (DIR == FluxDir::X1) ? 1 : 0;
		const amrex::Real dx0 = dx[Ax0];

		const amrex::Real q_xx = ComputeDiagTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_lo, dx_n, l2_alpha);
		const amrex::Real q_cross = ComputeCrossTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_lo, dx0, limiter_type);
		amrex::IntVect const corner_hi = UpperCorner<DIR>(i, j, k);
		const amrex::Real kappa_perp_face = 0.5 * (kappa_in[bx](corner_lo, 1) + kappa_in[bx](corner_hi, 1));
		const amrex::Real q_perp = -kappa_perp_face * ((T_in[bx](cp, T_comp) - T_in[bx](cm, T_comp)) / dx_n);
		const amrex::Real q_classical = q_xx + q_cross + q_perp;

		const amrex::Real qsat_face = 0.5 * (qsat_in[bx](corner_lo, 0) + qsat_in[bx](corner_hi, 0));
#else // AMREX_SPACEDIM == 3

		// Cyclic transverse-axis convention (matches ComputeFaceUnitBField elsewhere in this file):
		// X1 -> (y,z), X2 -> (z,x), X3 -> (x,y).
		constexpr int Ax0 = (DIR == FluxDir::X1) ? 1 : (DIR == FluxDir::X2) ? 2 : 0;
		constexpr int Ax1 = (DIR == FluxDir::X1) ? 2 : (DIR == FluxDir::X2) ? 0 : 1;
		const amrex::Real dx0 = dx[Ax0];
		const amrex::Real dx1 = dx[Ax1];
		amrex::IntVect const corner_0 = corner_lo + AxisUnit<Ax0>();
		amrex::IntVect const corner_1 = corner_lo + AxisUnit<Ax1>();
		amrex::IntVect const corner_01 = UpperCorner<DIR>(i, j, k); // == corner_0 + AxisUnit<Ax1>()

		const amrex::Real q_xx = 0.5 * (ComputeDiagTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_lo, dx_n, l2_alpha) +
						ComputeDiagTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_1, dx_n, l2_alpha));
		const amrex::Real q_cross0 =
		    0.5 * (ComputeCrossTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_lo, dx0, limiter_type) +
			   ComputeCrossTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_1, dx0, limiter_type));
		const amrex::Real q_cross1 =
		    0.5 * (ComputeCrossTerm<DIR, Ax1>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_lo, dx1, limiter_type) +
			   ComputeCrossTerm<DIR, Ax1>(T_in[bx], bhat_in[bx], kappa_in[bx], cm, cp, corner_0, dx1, limiter_type));
		const amrex::Real kappa_perp_face =
		    0.25 * (kappa_in[bx](corner_lo, 1) + kappa_in[bx](corner_0, 1) + kappa_in[bx](corner_1, 1) + kappa_in[bx](corner_01, 1));
		const amrex::Real q_perp = -kappa_perp_face * ((T_in[bx](cp, T_comp) - T_in[bx](cm, T_comp)) / dx_n);
		const amrex::Real q_classical = q_xx + q_cross0 + q_cross1 + q_perp;

		const amrex::Real qsat_face =
		    0.25 * (qsat_in[bx](corner_lo, 0) + qsat_in[bx](corner_0, 0) + qsat_in[bx](corner_1, 0) + qsat_in[bx](corner_01, 0));
#endif
#endif

		flux_out[bx](i, j, k) = SaturateFlux(q_classical, qsat_face, small);
	});
}

} // namespace quokka::conduction

#endif // ANISO_CONDUCTION_HPP_
