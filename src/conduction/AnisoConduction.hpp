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

// Unit B-field at mesh vertices ("corners"), built from FACE-CENTERED input data (state_fc), shared
// by all three flux directions -- see the definition below for details. bhat_corner_mf must already
// be defined by the caller on a fully-nodal box array, e.g.
// amrex::convert(state.boxArray(), amrex::IntVect::TheUnitVector()).
template <typename problem_t>
void ComputeCornerFC(amrex::MultiFab &bhat_corner_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int nghost);

// (dT/dx, dT/dy, dT/dz), n, and qsat at mesh vertices ("corners"), built from CELL-CENTERED input
// data (primVar), on the same fully-nodal box array as ComputeCornerFC's bhat_corner_mf -- see the
// definition below for details.
template <typename problem_t>
void ComputeCornerCC(amrex::MultiFab &gradT_corner_mf, amrex::MultiFab &n_corner_mf, amrex::MultiFab &qsat_corner_mf, amrex::MultiFab const &primVar,
		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real mean_molecular_weight, amrex::Real saturation_factor,
		     amrex::Real flux_limiter_phi, int nghost);

template <FluxDir DIR>
void ComputeAnisotropicFlux(amrex::MultiFab &heat_flux_fc, amrex::MultiFab const &bhat_fc, amrex::MultiFab const &gradT_fc, amrex::MultiFab const &n_fc,
			    amrex::Real kappa_parallel, amrex::Real kappa_perp, amrex::MultiFab const &q_sat_fc);

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

		// heat_flux at each face -- the actual per-direction output of this routine.
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray const ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}

		// Unit B-field, gradT, n, and qsat at mesh vertices ("corners") -- direction-independent, so
		// computed once (unlike the old per-face bhat_fc/gradT_fc/n_fc/q_sat_fc) and shared by all
		// three DIR-face flux calculations below. Each box's own fully-nodal valid region already
		// extends one node beyond its cell range in every direction, covering both the "lower" and
		// "upper" transverse corners of every face in that box, so no ghost cells are needed on these
		// corner MultiFabs themselves -- only the *inputs* (state_fc, primVar) need their own
		// pre-existing ghost cells to fill a box's boundary corners.
		amrex::BoxArray const ba_corner = amrex::convert(state.boxArray(), amrex::IntVect::TheUnitVector());
		amrex::MultiFab bhat_corner(ba_corner, state.DistributionMap(), 3, 0);
		amrex::MultiFab gradT_corner(ba_corner, state.DistributionMap(), 3, 0);
		amrex::MultiFab n_corner(ba_corner, state.DistributionMap(), 1, 0);
		amrex::MultiFab qsat_corner(ba_corner, state.DistributionMap(), 1, 0);
		bhat_corner.setVal(0.0); // zeroed when MHD is disabled, so a non-MHD build gets zero flux rather than reading uninitialized data.

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			ComputeCornerFC<problem_t>(bhat_corner, state_fc, 0);
		}

		const amrex::Real mmw = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
		ComputeCornerCC<problem_t>(gradT_corner, n_corner, qsat_corner, primVar, dx, mmw, params.saturation_factor, params.flux_limiter_phi, 0);

		// Heat flux at each face, from the LOWER bounding corner only: bhat_corner/gradT_corner/
		// n_corner/qsat_corner are read at the face's own (i,j,k) index, which is that face's lower
		// corner (see ComputeCornerCC's doc-comment).
		// TODO: average with the UPPER corner (index+1 in the transverse direction) to complete the
		// Sharma & Hammett symmetric scheme.
		AMREX_D_TERM(ComputeAnisotropicFlux<FluxDir::X1>(heat_flux[0], bhat_corner, gradT_corner, n_corner, params.kappa_parallel, params.kappa_perp,
								  qsat_corner);
			     , ComputeAnisotropicFlux<FluxDir::X2>(heat_flux[1], bhat_corner, gradT_corner, n_corner, params.kappa_parallel, params.kappa_perp,
								    qsat_corner);
			     , ComputeAnisotropicFlux<FluxDir::X3>(heat_flux[2], bhat_corner, gradT_corner, n_corner, params.kappa_parallel, params.kappa_perp,
								    qsat_corner);)

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

template <typename problem_t>
void ComputeCornerFC(amrex::MultiFab &bhat_corner_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int nghost)
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

// Estimate (dT/dx, dT/dy, dT/dz), n, and qsat at mesh vertices ("corners") from CELL-CENTERED input
// data (primVar), on the same fully-nodal box array and vertex-indexing convention as ComputeCornerFC
// (vertex (i,j,k) draws from the 8 (3D) or 4 (2D) cells with indices in {i-1,i}x{j-1,j}x{k-1,k}).
// gradT and (n, qsat) are computed together because they are derived from the very same surrounding
// cells, just combined differently: gradT needs each cell's own (rho,T,massScalars) via the two-point
// centered difference along its own axis, averaged over the neighboring-cell-pair combinations of the
// OTHER axes -- e.g. in 3D, dT/dx at vertex (i,j,k) averages the x-difference
// (T(i,*,*)-T(i-1,*,*))/dx over (j-1,j) x (k-1,k), the vertex generalization of the diamond-averaged
// transverse gradient already used in ComputeFaceGradT -- while n and qsat need the plain average of
// (rho,T,massScalars) over those same cells, then the same EOS chain as
// ComputeFaceNumberDensityAndSaturationFlux (Cowie & McKee 1977).
template <typename problem_t>
void ComputeCornerCC(amrex::MultiFab &gradT_corner_mf, amrex::MultiFab &n_corner_mf, amrex::MultiFab &qsat_corner_mf, amrex::MultiFab const &primVar,
		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real mean_molecular_weight, amrex::Real saturation_factor,
		     amrex::Real flux_limiter_phi, int nghost)
{
	constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	constexpr int T_comp = 1;
	const amrex::Real small = std::numeric_limits<amrex::Real>::min();

	auto const &primVar_in = primVar.const_arrays();
	auto gradT_out = gradT_corner_mf.arrays();
	auto n_out = n_corner_mf.arrays();
	auto qsat_out = qsat_corner_mf.arrays();

	amrex::IntVect const ng{AMREX_D_DECL(nghost, nghost, nghost)};
	amrex::ParallelFor(gradT_corner_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real gradT_x = 0.0;
		amrex::Real gradT_y = 0.0;
		amrex::Real gradT_z = 0.0;

#if AMREX_SPACEDIM == 3
		gradT_x = 0.25 *
			  ((primVar_in[bx](i, j - 1, k - 1, T_comp) - primVar_in[bx](i - 1, j - 1, k - 1, T_comp)) +
			   (primVar_in[bx](i, j, k - 1, T_comp) - primVar_in[bx](i - 1, j, k - 1, T_comp)) +
			   (primVar_in[bx](i, j - 1, k, T_comp) - primVar_in[bx](i - 1, j - 1, k, T_comp)) +
			   (primVar_in[bx](i, j, k, T_comp) - primVar_in[bx](i - 1, j, k, T_comp))) /
			  dx[0];
		gradT_y = 0.25 *
			  ((primVar_in[bx](i - 1, j, k - 1, T_comp) - primVar_in[bx](i - 1, j - 1, k - 1, T_comp)) +
			   (primVar_in[bx](i, j, k - 1, T_comp) - primVar_in[bx](i, j - 1, k - 1, T_comp)) +
			   (primVar_in[bx](i - 1, j, k, T_comp) - primVar_in[bx](i - 1, j - 1, k, T_comp)) +
			   (primVar_in[bx](i, j, k, T_comp) - primVar_in[bx](i, j - 1, k, T_comp))) /
			  dx[1];
		gradT_z = 0.25 *
			  ((primVar_in[bx](i - 1, j - 1, k, T_comp) - primVar_in[bx](i - 1, j - 1, k - 1, T_comp)) +
			   (primVar_in[bx](i, j - 1, k, T_comp) - primVar_in[bx](i, j - 1, k - 1, T_comp)) +
			   (primVar_in[bx](i - 1, j, k, T_comp) - primVar_in[bx](i - 1, j, k - 1, T_comp)) +
			   (primVar_in[bx](i, j, k, T_comp) - primVar_in[bx](i, j, k - 1, T_comp))) /
			  dx[2];
#elif AMREX_SPACEDIM == 2
		gradT_x = 0.5 *
			  ((primVar_in[bx](i, j - 1, k, T_comp) - primVar_in[bx](i - 1, j - 1, k, T_comp)) +
			   (primVar_in[bx](i, j, k, T_comp) - primVar_in[bx](i - 1, j, k, T_comp))) /
			  dx[0];
		gradT_y = 0.5 *
			  ((primVar_in[bx](i - 1, j, k, T_comp) - primVar_in[bx](i - 1, j - 1, k, T_comp)) +
			   (primVar_in[bx](i, j, k, T_comp) - primVar_in[bx](i, j - 1, k, T_comp))) /
			  dx[1];
#else
		gradT_x = (primVar_in[bx](i, j, k, T_comp) - primVar_in[bx](i - 1, j, k, T_comp)) / dx[0];
#endif

		gradT_out[bx](i, j, k, 0) = gradT_x;
		gradT_out[bx](i, j, k, 1) = gradT_y;
		gradT_out[bx](i, j, k, 2) = gradT_z;

		// n / qsat: plain average of (rho, T, massScalars) over the same {i-1,i}x{j-1,j}x{k-1,k}
		// cells, then the same EOS chain ComputeFaceNumberDensityAndSaturationFlux uses.
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

// Compute the anisotropic heat flux crossing the DIR-faces, given the unit B-field (bhat_fc, from
// This algorithm implements equivalent of asymmetric scheme in Sharma & Hammett 2007 without any limiting.
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

} // namespace quokka::conduction

#endif // ANISO_CONDUCTION_HPP_
