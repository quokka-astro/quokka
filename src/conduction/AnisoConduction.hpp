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
///        nested minmod (L(L(a,b),L(c,d))) since it is sign-indefinite. Both terms read bhat/n at
///        mesh vertices ("corners") and gradT via direct two-point differences of cell-centered T,
///        never a precomputed face/corner gradT field. Only kappa_parallel enters the flux
///        (kappa_perp is currently unused -- see AnisoConductionParams); the result is saturated
///        exactly as in ElectronConduction::ComputeExplicit, using a saturation flux computed from
///        (rho, T) averaged from the corners bounding the face.

#include <array>
#include <cmath>
#include <limits>
#include <string>

#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MFIter.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include "hyperbolic_system.hpp"

namespace quokka::conduction
{

// Selects how LimitUpperLowerFlux combines a face's two per-corner flux estimates (Sharma & Hammett
// 2007). Only Minmod is implemented so far; add new cases here (and in LimitUpperLowerFlux's switch)
// to support more limiter types.
enum class AnisoFluxLimiterType { Minmod };

struct AnisoConductionParams {
	amrex::Real kappa_parallel = 0.0; // field-aligned conductivity, units erg cm^-1 s^-1 K^-1
	amrex::Real kappa_perp = 0.0;	  // currently unused by ComputeAnisotropicFlux -- see file doc-comment
	amrex::Real l2_alpha = 2.0;	  // bias factor for the L2 limiter (Sharma & Hammett 2007 Eq. 21) applied to the normal (q_xx-type) term
	amrex::Real flux_limiter_phi = 0.1;
	amrex::Real saturation_factor = 5.0; // refer to equation 8 of Cowie & McKee 1977
	amrex::Real min_temperature = 0.0;   // default value will be overwritten by tempFloor_ during initialization
	int reconstruction_order = 3;	     // 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM;
	SlopeLimiter plm_limiter = SlopeLimiter::sweby;
	int ng_reconstruct = 2;						       // number of ghost faces to reconstruct beyond the valid box
	AnisoFluxLimiterType flux_limiter_type = AnisoFluxLimiterType::Minmod; // combines a face's lower/upper corner flux estimates; see LimitUpperLowerFlux
};

template <FluxDir DIR> AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::IntVect FaceMinusOne(int i, int j, int k)
{
	if constexpr (DIR == FluxDir::X1) {
		return {i - 1, j, k};
	} else if constexpr (DIR == FluxDir::X2) {
		return {i, j - 1, k};
	} else {
		return {i, j, k - 1};
	}
}

// The "upper" corner bounding a DIR-face, given the face's own (lower-corner) index (i,j,k) -- i.e.
// shifted by +1 in every TRANSVERSE direction (never in DIR's own direction, since the face is
// already exactly at the right position along DIR). In 2D there is only one transverse direction, so
// this is unambiguous; in 3D a face has 4 corners, and this picks the one diagonally opposite the
// lower corner (i,j,k), leaving the other two corners (offset in only one transverse direction) unused.
template <FluxDir DIR> AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::IntVect UpperCorner(int i, int j, int k)
{
	if constexpr (DIR == FluxDir::X1) {
		return {i, j + (AMREX_SPACEDIM >= 2 ? 1 : 0), k + (AMREX_SPACEDIM == 3 ? 1 : 0)};
	} else if constexpr (DIR == FluxDir::X2) {
		return {i + 1, j, k + (AMREX_SPACEDIM == 3 ? 1 : 0)};
	} else {
		return {i + 1, j + 1, k};
	}
}

// Host-only: parses a limiter name (e.g. from an input parameter) into AnisoFluxLimiterType. Not
// GPU-safe -- call this once on the host (e.g. while populating AnisoConductionParams), never from
// inside a GPU kernel; std::string comparisons are not usable in device code, which is why
// LimitUpperLowerFlux below takes the already-parsed enum instead of a string.
inline auto ParseAnisoFluxLimiterType(std::string const &name) -> AnisoFluxLimiterType
{
	if (name == "minmod") {
		return AnisoFluxLimiterType::Minmod;
	}
	amrex::Abort("Unknown anisotropic-conduction flux limiter \"" + name + "\" (valid: \"minmod\")");
	return AnisoFluxLimiterType::Minmod; // unreachable; amrex::Abort does not return
}

// Combines two per-corner or per-neighbor estimates (q_lower, q_upper) into a single minmod-limited
// value, per the limiter selected by `limiter_type` -- used both directly (the transverse-term
// NestedLimit) and, in earlier revisions of this scheme, to combine per-corner flux estimates. Takes
// the pre-parsed enum (see ParseAnisoFluxLimiterType) rather than a string, since this runs per-face
// inside a device kernel and std::string comparisons are not GPU-safe. Marked host+device (unlike
// UpperCorner/FaceMinusOne) since it is pure arithmetic with no device-only memory access.
//
// Minmod: if q_lower and q_upper disagree in sign (or either is exactly zero), returns 0 -- this is
// the signature of a spurious, non-physical contribution (e.g. near a field bend or a field-direction
// null), so it is suppressed entirely rather than averaged into a nonzero residual. If they agree in
// sign, returns whichever has the smaller magnitude (with that shared sign), to avoid overshoot.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real LimitUpperLowerFlux(amrex::Real q_lower, amrex::Real q_upper, AnisoFluxLimiterType limiter_type)
{
	switch (limiter_type) {
		case AnisoFluxLimiterType::Minmod:
		default:
			if (q_lower * q_upper <= 0.0) {
				return 0.0;
			}
			return (q_lower > 0.0) ? amrex::min(q_lower, q_upper) : amrex::max(q_lower, q_upper);
	}
}

// kappa_par * bn^2 -- the diagonal tensor entry (q_xx-type term). Always non-negative, since it's
// kappa_par times a square -- this is why the normal term is safe to limit with the biased L2 rather
// than minmod (see L2 below).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real DiagCoeff(amrex::Real bn, amrex::Real kappa_par) { return kappa_par * bn * bn; }

// kappa_par * bi * bj -- the off-diagonal tensor entry (q_xy/q_xz-type term). Sign-indefinite (flips
// with bi*bj), so it needs minmod-style limiting (see NestedLimit below), not L2.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real CrossCoeff(amrex::Real bi, amrex::Real bj, amrex::Real kappa_par) { return kappa_par * bi * bj; }

// Sharma & Hammett (2007) Eq. (21): a biased limiter for the (sign-definite) normal term. Not
// symmetric in its arguments -- `anchor` must be the face's own gradient estimate, `neighbor` the
// transverse-neighbor face's. Returns the neighbor-inclusive average when it falls within
// [alpha*anchor, anchor/alpha] (in either order), otherwise clamps to whichever bound was crossed.
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

// L(L(a,b), L(c,d)) -- Eq. (17) / Sec. 6.1's nested-minmod pattern for the (sign-indefinite)
// transverse term. Reuses LimitUpperLowerFlux rather than re-deriving minmod a second time.
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
	if constexpr (Axis == 0) {
		return {1, 0, 0};
	} else if constexpr (Axis == 1) {
		return {0, 1, 0};
	} else {
		return {0, 0, 1};
	}
}

// q_xx-type normal term for a DIR-face at (i,j,k) (cells cm, cp), limited across the `Axis`-
// transverse direction via L2. `corner_lo` is the face's lower bounding corner along every axis
// OTHER than `Axis` (fixed by the caller, e.g. to average over a second transverse axis in 3D);
// `corner_lo` and `corner_lo + AxisUnit<Axis>()` are the two corners this term reads bhat/n from.
template <FluxDir DIR, int Axis>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real ComputeDiagTerm(amrex::Array4<const amrex::Real> const &T, amrex::Array4<const amrex::Real> const &bhat,
								amrex::Array4<const amrex::Real> const &n, amrex::IntVect const &cm, amrex::IntVect const &cp,
								amrex::IntVect const &corner_lo, amrex::Real dx_n, amrex::Real kappa_parallel,
								amrex::Real l2_alpha)
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

	const amrex::Real q_N = -n(corner_N, 0) * DiagCoeff(bhat(corner_N, normal_comp), kappa_parallel) * grad_N;
	const amrex::Real q_S = -n(corner_S, 0) * DiagCoeff(bhat(corner_S, normal_comp), kappa_parallel) * grad_S;
	return 0.5 * (q_N + q_S);
}

// q_xy/q_xz-type cross term for a DIR-face at (i,j,k) (cells cm, cp), limited across the `Axis`-
// transverse direction via NestedLimit. `corner_lo`/`corner_lo + AxisUnit<Axis>()` are the two
// corners bhat/n are face-averaged from (see the same `corner_lo` convention as ComputeDiagTerm).
template <FluxDir DIR, int Axis>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real ComputeCrossTerm(amrex::Array4<const amrex::Real> const &T, amrex::Array4<const amrex::Real> const &bhat,
								 amrex::Array4<const amrex::Real> const &n, amrex::IntVect const &cm, amrex::IntVect const &cp,
								 amrex::IntVect const &corner_lo, amrex::Real dx_t, amrex::Real kappa_parallel,
								 AnisoFluxLimiterType limiter_type)
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
	const amrex::Real n_face = 0.5 * (n(corner_lo, 0) + n(corner_hi, 0));

	return -n_face * CrossCoeff(bn_face, bt_face, kappa_parallel) * slope;
}

// Declarations only -- see the definitions below AnisoConduction for what each of these actually
// does. Declared here so that AnisoConduction::ComputeExplicit (which calls all four) can be read
// first; scroll down past the class for the bodies.

// Unit B-field at mesh vertices ("corners"), built from FACE-CENTERED input data (state_fc), shared
// by all three flux directions -- see the definition below for details. bhat_corner_mf must already
// be defined by the caller on a fully-nodal box array, e.g.
// amrex::convert(state.boxArray(), amrex::IntVect::TheUnitVector()).
template <typename problem_t> void ComputeCornerFC(amrex::MultiFab &bhat_corner_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int nghost);

// n and qsat at mesh vertices ("corners"), built from CELL-CENTERED input data (primVar), on the
// same fully-nodal box array as ComputeCornerFC's bhat_corner_mf -- see the definition below for
// details. gradT is no longer computed here: ComputeAnisotropicFlux differences primVar directly.
template <typename problem_t>
void ComputeCornerCC(amrex::MultiFab &n_corner_mf, amrex::MultiFab &qsat_corner_mf, amrex::MultiFab const &primVar, amrex::Real mean_molecular_weight,
		     amrex::Real saturation_factor, amrex::Real flux_limiter_phi, int nghost);

template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &primVar, amrex::MultiFab const &bhat_corner, amrex::MultiFab const &n_corner,
			    amrex::MultiFab const &qsat_corner, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real kappa_parallel,
			    amrex::Real l2_alpha, AnisoFluxLimiterType limiter_type);

// DEBUG: prints the face-centered heat flux components (Fx, Fy, Fz) at the fixed grid index
// (i=192, j=128, k=0), to check that flux is non-zero only along the field direction.
void PrintHeatFluxAtPoint(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &heat_flux, amrex::Geometry const &geom);

// DEBUG: prints the 3 components of a fixed (x,y,z)-ordered vector MultiFab (e.g. bhat_corner) at
// the fixed grid index (i=192, j=128, k=0), prefixed by `label`.
void PrintVectorAtPoint(amrex::MultiFab const &vec_fc, amrex::Geometry const &geom, char const *label);

// DEBUG: prints T (from the cell touching the domain-center vertex) and bhat AT the domain-center
// vertex (i=128, j=128, k=0) -- the exact grid point where this problem's B-field direction is
// mathematically singular (Bx=-y/r, By=x/r is 0/0 at r=0) -- to check whether an erratic/arbitrary
// discrete bhat there is acting as a leak path for heat to reach the center.
void PrintCenterDiagnostics(amrex::MultiFab const &primVar, amrex::MultiFab const &bhat_corner);

template <typename problem_t> class AnisoConduction
{
      public:
	static void ComputeExplicit(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    amrex::Real dt, AnisoConductionParams const &params, std::array<amrex::MultiFab, AMREX_SPACEDIM> &heat_flux)
	{
		static_assert(Physics_Traits<problem_t>::is_hydro_enabled, "Anisotropic conduction requires hydro to be enabled.");

		if ((dt <= 0.0) || (params.kappa_parallel <= 0.0)) { // kappa_perp currently unused -- see file doc-comment
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

		// heat_flux at each face -- the actual per-direction output of this routine.
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}

		// Unit B-field, n, and qsat at mesh vertices ("corners") -- direction-independent, so computed
		// once (unlike the old per-face bhat_fc/n_fc/q_sat_fc) and shared by all three DIR-face flux
		// calculations below. Each box's own fully-nodal valid region already extends one node beyond
		// its cell range in every direction, covering both the "lower" and "upper" transverse corners
		// of every face in that box, so no ghost cells are needed on these corner MultiFabs themselves
		// -- only the *inputs* (state_fc, primVar) need their own pre-existing ghost cells to fill a
		// box's boundary corners. ComputeAnisotropicFlux additionally differences primVar directly at
		// the DIR-neighbor faces one cell to either transverse side of each face it computes (see
		// ComputeDiagTerm/ComputeCrossTerm); since heat_flux itself is only ever evaluated on the
		// valid (non-ghost) nodal region, that neighbor is always within 1 ghost ring of primVar, so
		// state.nGrow() >= 1 (asserted below) remains sufficient.
		amrex::BoxArray const ba_corner = amrex::convert(state.boxArray(), amrex::IntVect::TheUnitVector());
		amrex::MultiFab bhat_corner(ba_corner, state.DistributionMap(), 3, 0);
		amrex::MultiFab n_corner(ba_corner, state.DistributionMap(), 1, 0);
		amrex::MultiFab qsat_corner(ba_corner, state.DistributionMap(), 1, 0);
		bhat_corner.setVal(0.0); // zeroed when MHD is disabled, so a non-MHD build gets zero flux rather than reading uninitialized data.

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			ComputeCornerFC<problem_t>(bhat_corner, state_fc, 0);
		}

		const amrex::Real mmw = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
		ComputeCornerCC<problem_t>(n_corner, qsat_corner, primVar, mmw, params.saturation_factor, params.flux_limiter_phi, 0);

		// Heat flux at each face: ComputeAnisotropicFlux splits the flux into a normal (q_xx-type)
		// term limited via L2 and one transverse (q_xy/q_xz-type) term per transverse direction
		// limited via NestedLimit, then saturates the sum (Sharma & Hammett 2007) -- see the file
		// doc-comment and ComputeDiagTerm/ComputeCrossTerm for details.
		AMREX_D_TERM(ComputeAnisotropicFlux<FluxDir::X1>(heat_flux[0], primVar, bhat_corner, n_corner, qsat_corner, dx, params.kappa_parallel,
								 params.l2_alpha, params.flux_limiter_type);
			     , ComputeAnisotropicFlux<FluxDir::X2>(heat_flux[1], primVar, bhat_corner, n_corner, qsat_corner, dx, params.kappa_parallel,
								   params.l2_alpha, params.flux_limiter_type);
			     , ComputeAnisotropicFlux<FluxDir::X3>(heat_flux[2], primVar, bhat_corner, n_corner, qsat_corner, dx, params.kappa_parallel,
								   params.l2_alpha, params.flux_limiter_type);)

		PrintHeatFluxAtPoint(heat_flux, geom);
		PrintVectorAtPoint(bhat_corner, geom, "[AnisoConduction] bhat (corner)");
		PrintCenterDiagnostics(primVar, bhat_corner);

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

// Estimate n and qsat at mesh vertices ("corners") from CELL-CENTERED input data (primVar), on the
// same fully-nodal box array and vertex-indexing convention as ComputeCornerFC (vertex (i,j,k) draws
// from the 8 (3D) or 4 (2D) cells with indices in {i-1,i}x{j-1,j}x{k-1,k}). n and qsat are the plain
// average of (rho,T,massScalars) over those cells, then the same EOS chain as
// ComputeFaceNumberDensityAndSaturationFlux (Cowie & McKee 1977). gradT is intentionally not computed
// here -- ComputeAnisotropicFlux differences primVar directly (see ComputeDiagTerm/ComputeCrossTerm).
template <typename problem_t>
void ComputeCornerCC(amrex::MultiFab &n_corner_mf, amrex::MultiFab &qsat_corner_mf, amrex::MultiFab const &primVar, amrex::Real mean_molecular_weight,
		     amrex::Real saturation_factor, amrex::Real flux_limiter_phi, int nghost)
{
	constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	constexpr int T_comp = 1;
	const amrex::Real small = std::numeric_limits<amrex::Real>::min();

	auto const &primVar_in = primVar.const_arrays();
	auto n_out = n_corner_mf.arrays();
	auto qsat_out = qsat_corner_mf.arrays();

	amrex::IntVect const ng{AMREX_D_DECL(nghost, nghost, nghost)};
	amrex::ParallelFor(n_corner_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// n / qsat: plain average of (rho, T, massScalars) over the {i-1,i}x{j-1,j}x{k-1,k} cells
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

		n_out[bx](i, j, k) = rho_corner / mean_molecular_weight;

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
// scheme), given bhat/n/qsat at mesh vertices (ComputeCornerFC/ComputeCornerCC) and T at cell centers
// (primVar). Splits q_classical into a normal (q_xx-type) term (ComputeDiagTerm, limited via L2) plus
// one transverse (q_xy/q_xz-type) term per transverse direction (ComputeCrossTerm, limited via
// NestedLimit); in 3D each term is evaluated once per position along the *other* transverse axis and
// averaged, so every term ultimately draws from all 4 corners bounding the face. qsat is likewise the
// plain average of all corners bounding the face. See the file doc-comment for the overall scheme and
// ComputeDiagTerm/ComputeCrossTerm's doc-comments for the per-term derivations.
template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &primVar, amrex::MultiFab const &bhat_corner, amrex::MultiFab const &n_corner,
			    amrex::MultiFab const &qsat_corner, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real kappa_parallel,
			    amrex::Real l2_alpha, AnisoFluxLimiterType limiter_type)
{
	constexpr int normal_comp = static_cast<int>(DIR);
	constexpr int T_comp = 1;
	const amrex::Real small = std::numeric_limits<amrex::Real>::min();
	const amrex::Real dx_n = dx[normal_comp];

	auto const &T_in = primVar.const_arrays();
	auto const &bhat_in = bhat_corner.const_arrays();
	auto const &n_in = n_corner.const_arrays();
	auto const &qsat_in = qsat_corner.const_arrays();
	auto flux_out = heat_flux_fc.arrays();

	amrex::ParallelFor(heat_flux_fc, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::IntVect const cm = FaceMinusOne<DIR>(i, j, k);
		amrex::IntVect const cp{i, j, k};
		amrex::IntVect const corner_lo{i, j, k};

#if AMREX_SPACEDIM == 1
		amrex::ignore_unused(corner_lo);
		const amrex::Real q_classical = -n_in[bx](i, j, k, 0) * DiagCoeff(bhat_in[bx](i, j, k, normal_comp), kappa_parallel) *
						((T_in[bx](cp, T_comp) - T_in[bx](cm, T_comp)) / dx_n);
		const amrex::Real qsat_face = qsat_in[bx](i, j, k, 0);
#else
		// Cyclic transverse-axis convention (matches ComputeFaceUnitBField elsewhere in this file):
		// X1 -> (y,z), X2 -> (z,x), X3 -> (x,y).
		constexpr int Ax0 = (DIR == FluxDir::X1) ? 1 : (DIR == FluxDir::X2) ? 2 : 0;
		constexpr int Ax1 = (DIR == FluxDir::X1) ? 2 : (DIR == FluxDir::X2) ? 0 : 1;
		const amrex::Real dx0 = dx[Ax0];

#if AMREX_SPACEDIM == 2
		const amrex::Real q_xx = ComputeDiagTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_lo, dx_n, kappa_parallel, l2_alpha);
		const amrex::Real q_cross = ComputeCrossTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_lo, dx0, kappa_parallel, limiter_type);
		const amrex::Real q_classical = q_xx + q_cross;

		amrex::IntVect const corner_hi = UpperCorner<DIR>(i, j, k);
		const amrex::Real qsat_face = 0.5 * (qsat_in[bx](corner_lo, 0) + qsat_in[bx](corner_hi, 0));
#else // AMREX_SPACEDIM == 3
		const amrex::Real dx1 = dx[Ax1];
		amrex::IntVect const corner_0 = corner_lo + AxisUnit<Ax0>();
		amrex::IntVect const corner_1 = corner_lo + AxisUnit<Ax1>();
		amrex::IntVect const corner_01 = UpperCorner<DIR>(i, j, k); // == corner_0 + AxisUnit<Ax1>()

		const amrex::Real q_xx = 0.5 * (ComputeDiagTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_lo, dx_n, kappa_parallel, l2_alpha) +
						ComputeDiagTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_1, dx_n, kappa_parallel, l2_alpha));
		const amrex::Real q_cross0 =
		    0.5 * (ComputeCrossTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_lo, dx0, kappa_parallel, limiter_type) +
			   ComputeCrossTerm<DIR, Ax0>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_1, dx0, kappa_parallel, limiter_type));
		const amrex::Real q_cross1 =
		    0.5 * (ComputeCrossTerm<DIR, Ax1>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_lo, dx1, kappa_parallel, limiter_type) +
			   ComputeCrossTerm<DIR, Ax1>(T_in[bx], bhat_in[bx], n_in[bx], cm, cp, corner_0, dx1, kappa_parallel, limiter_type));
		const amrex::Real q_classical = q_xx + q_cross0 + q_cross1;

		const amrex::Real qsat_face =
		    0.25 * (qsat_in[bx](corner_lo, 0) + qsat_in[bx](corner_0, 0) + qsat_in[bx](corner_1, 0) + qsat_in[bx](corner_01, 0));
#endif
#endif

		flux_out[bx](i, j, k) = SaturateFlux(q_classical, qsat_face, small);
	});
}

// DEBUG: the fixed grid index (i=192, j=128, k=0) probed by PrintHeatFluxAtPoint and
// PrintVectorAtPoint -- shared so both sample the exact same face/vertex.
amrex::IntVect DebugProbeIntVect(amrex::Geometry const &geom)
{
	amrex::ignore_unused(geom);
	return amrex::IntVect{AMREX_D_DECL(192, 128, 0)};
}

// DEBUG: MPI-safe extraction of a single component at a single cell/face/corner of a MultiFab. Uses
// Max (not Sum) because `mf` may be nodal in some direction (heat_flux/bhat_corner both are): at a
// box-decomposition boundary, the shared nodal point is duplicated verbatim in both
// neighboring boxes' valid regions, so summing would double-count it. All duplicate copies hold the
// identical value, so Max recovers it exactly regardless of how many boxes/ranks contain a copy.
amrex::Real DebugReduceComponentAt(amrex::MultiFab const &mf, amrex::IntVect const &iv, int comp)
{
	amrex::Box const target_box(iv, iv);

	amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	using ReduceTuple = typename decltype(reduce_data)::Type;

	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		amrex::Box const bx = mfi.validbox() & target_box;
		if (bx.ok()) {
			auto const &arr = mf.const_array(mfi);
			reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple { return {arr(i, j, k, comp)}; });
		}
	}

	amrex::Real local_val = amrex::get<0>(reduce_data.value());
	amrex::ParallelDescriptor::ReduceRealMax(local_val);
	return local_val;
}

// DEBUG: extracts and prints the (Fx, Fy, Fz) face-centered heat flux at the fixed grid index
// (i=192, j=128, k=0). For a pure B-aligned conduction test with B along y, only Fy should be nonzero;
// nonzero Fx/Fz indicates leakage of flux perpendicular to the field. Remove once verified.
void PrintHeatFluxAtPoint(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &heat_flux, amrex::Geometry const &geom)
{
	amrex::IntVect const iv = DebugProbeIntVect(geom);

	amrex::Real flux_vals[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		flux_vals[idim] = DebugReduceComponentAt(heat_flux[idim], iv, 0);
	}

	amrex::Print() << "[AnisoConduction] heat flux at (i=192,j=128,k=0): Fx=" << flux_vals[0]
#if AMREX_SPACEDIM >= 2
		       << " Fy=" << flux_vals[1]
#endif
#if AMREX_SPACEDIM == 3
		       << " Fz=" << flux_vals[2]
#endif
		       << std::endl;
}

// DEBUG: prints the 3 components of a fixed (x,y,z)-ordered vector MultiFab (e.g. bhat_corner) at
// the fixed grid index (i=192, j=128, k=0), prefixed by `label`.
void PrintVectorAtPoint(amrex::MultiFab const &vec_fc, amrex::Geometry const &geom, char const *label)
{
	amrex::IntVect const iv = DebugProbeIntVect(geom);

	const amrex::Real vx = DebugReduceComponentAt(vec_fc, iv, 0);
	const amrex::Real vy = DebugReduceComponentAt(vec_fc, iv, 1);
	const amrex::Real vz = DebugReduceComponentAt(vec_fc, iv, 2);

	amrex::Print() << label << " at (i=192,j=128,k=0): (" << vx << ", " << vy << ", " << vz << ")" << std::endl;
}

// DEBUG: prints T (from the cell touching the domain-center vertex) and bhat AT the domain-center
// vertex (i=128, j=128, k=0) -- the exact grid point where this problem's B-field direction is
// mathematically singular (Bx=-y/r, By=x/r is 0/0 at r=0) -- to check whether an erratic/arbitrary
// discrete bhat there is acting as a leak path for heat to reach the center.
void PrintCenterDiagnostics(amrex::MultiFab const &primVar, amrex::MultiFab const &bhat_corner)
{
	constexpr int T_comp = 1;
	amrex::IntVect const iv_vertex{AMREX_D_DECL(128, 128, 0)};
	amrex::IntVect const iv_cell = iv_vertex; // one of the 4 (2D) / 8 (3D) cells touching that vertex

	const amrex::Real T = DebugReduceComponentAt(primVar, iv_cell, T_comp);

	const amrex::Real bx = DebugReduceComponentAt(bhat_corner, iv_vertex, 0);
	const amrex::Real by = DebugReduceComponentAt(bhat_corner, iv_vertex, 1);
	const amrex::Real bz = DebugReduceComponentAt(bhat_corner, iv_vertex, 2);

	amrex::Print() << "[AnisoConduction] domain center (i=128,j=128,k=0): T=" << T << " bhat=(" << bx << ", " << by << ", " << bz << ")" << std::endl;
}

} // namespace quokka::conduction

#endif // ANISO_CONDUCTION_HPP_
