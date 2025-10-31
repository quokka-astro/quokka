#ifndef HYDRO_SYSTEM_HPP_ // NOLINT
#define HYDRO_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hydro_system.hpp
/// \brief Defines a class for solving the Euler equations.
///

// c++ headers
#include <algorithm>
#include <cmath>

// library headers
#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_iMultiFab.H"

// internal headers
#include "EOS.hpp"
#include "HLLC.hpp"
#include "HLLD.hpp"
#include "LLF.hpp"
#include "LLF_mhd.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayView.hpp"
#include "util/valarray.hpp"

// Microphysics headers
#include "extern_parameters.H"

// this struct is specialized by the user application code
//
template <typename problem_t> struct HydroSystem_Traits {
	// if true, reconstruct e_int instead of pressure
	static constexpr bool reconstruct_eint = true;
};

enum class RiemannSolver { HLLC, LLF, LLF_MHD, HLLD };

/// Class for the Euler equations of inviscid hydrodynamics
///
template <typename problem_t> class HydroSystem : public HyperbolicSystem<problem_t>
{
      public:
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr int nscalars_ = Physics_Traits<problem_t>::numPassiveScalars;
	static constexpr int nvar_ = Physics_NumVars::numHydroVars + nscalars_;

	enum consVarIndex {
		density_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1Momentum_index,
		x2Momentum_index,
		x3Momentum_index,
		energy_index,
		internalEnergy_index, // auxiliary internal energy (rho * e)
		scalar0_index	      // first passive scalar (only present if nscalars > 0!)
	};

	enum primVarIndex {
		primDensity_index = 0,
		x1Velocity_index,
		x2Velocity_index,
		x3Velocity_index,
		pressure_index,
		primEint_index,	   // auxiliary internal energy (rho * e)
		primScalar0_index, // first passive scalar (only present if nscalars > 0!)
	};

	static void ConservedToPrimitive(amrex::MultiFab const &cons_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf,
					 amrex::MultiFab &primVar_mf, int nghost);

	static auto maxSignalSpeedLocal(amrex::MultiFab const &cons, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> amrex::Real;

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc, std::array<amrex::Array4<const amrex::Real>, 3> const &cons_fc,
					  array_t &maxSignal, amrex::Box const &indexRange);

	static auto CheckStatesValid(amrex::MultiFab const &cons_mf,  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> bool;

	AMREX_GPU_DEVICE static auto ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, std::array<amrex::Array4<const amrex::Real>, 3> const &cons_fc, int i, int j, int k) -> quokka::valarray<amrex::Real, nvar_>;

	AMREX_GPU_DEVICE static auto ComputeConsVars(quokka::valarray<amrex::Real, nvar_> const &prim) -> quokka::valarray<amrex::Real, nvar_>;

	AMREX_GPU_DEVICE static auto ComputePressure(amrex::Array4<const amrex::Real> const &cons, std::array<amrex::Array4<const amrex::Real>, 3> const &cons_fc, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, std::array<amrex::Array4<const amrex::Real>, 3> const &cons_fc, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool;

	static void ComputeRhsFromFluxes(amrex::MultiFab &rhs_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int nvars);

	static void PredictStep(amrex::MultiFab const &consVarOld, amrex::MultiFab &consVarNew, amrex::MultiFab const &rhs, double dt, int nvars,
				amrex::iMultiFab &redoFlag_mf);

	static void AddFluxesRK2(amrex::MultiFab &Unew_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, amrex::MultiFab const &rhs_mf, double dt,
				 int nvars, amrex::iMultiFab &redoFlag_mf);

	AMREX_GPU_DEVICE static auto GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;

	static void EnforceLimits(amrex::Real densityFloor, amrex::Real tempFloor, amrex::MultiFab &state_mf);

	static void AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf);

	static void SyncDualEnergy(amrex::MultiFab &consVar_mf, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &faceVar_mf);

	template <RiemannSolver RIEMANN, FluxDir DIR>
	static void ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf,
				  amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &leftState_bfield_mf,
				  amrex::MultiFab const &rightState_bfield_mf, amrex::MultiFab const &primVar_mf, amrex::Real K_visc,
				  amrex::MultiFab *x1FSpds_mf = nullptr, amrex::MultiFab const *x1ConsVar_fc_mf = nullptr, int nghost_vel = 2);

	template <FluxDir DIR>
	static void ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar, array_t &x1FluxDiffusive, amrex::Box const &indexRange);

	template <FluxDir DIR> static void ComputeFlatteningCoefficients(amrex::MultiFab const &primVar_mf, amrex::MultiFab &x1Chi_mf, int nghost);

	template <FluxDir DIR>
	static void FlattenShocks(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf,
				  amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf, int nghost, int nvars);

	// C++ does not allow constexpr to be uninitialized, even in a templated
	// class!
	static constexpr double gamma_ = quokka::EOS_Traits<problem_t>::gamma;
	static constexpr double cs_iso_ = quokka::EOS_Traits<problem_t>::cs_isothermal;
	static constexpr auto is_eos_isothermal() -> bool { return (gamma_ == 1.0); }

	static constexpr bool reconstruct_eint = HydroSystem_Traits<problem_t>::reconstruct_eint;
};

template <typename problem_t>
void HydroSystem<problem_t>::ConservedToPrimitive(amrex::MultiFab const &cons_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf,
						  amrex::MultiFab &primVar_mf, const int nghost)
{
	// convert conserved to primitive variables
	auto const &cons_fc_x0 = cons_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = cons_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = cons_fc_mf[2].const_arrays();
#endif
	auto const &cons_cc = cons_cc_mf.const_arrays();
	auto const &primVar = primVar_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(cons_cc_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		// First-capture the magnetic field arrays by accessing them early
		// This forces NVCC to capture them before the constexpr if block
		std::remove_cv_t<std::remove_reference_t<decltype(cons_fc_x0[bx])>> fc_x0_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			// If not wrapped in this if clause, this will fail in Debug mode due to nan values
			fc_x0_ref = cons_fc_x0[bx];
		}
#if AMREX_SPACEDIM >= 2
		std::remove_cv_t<std::remove_reference_t<decltype(cons_fc_x1[bx])>> fc_x1_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			fc_x1_ref = cons_fc_x1[bx];
		}
#endif
#if AMREX_SPACEDIM == 3
		std::remove_cv_t<std::remove_reference_t<decltype(cons_fc_x2[bx])>> fc_x2_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			fc_x2_ref = cons_fc_x2[bx];
		}
#endif

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		cons_fc[0] = fc_x0_ref;
#if AMREX_SPACEDIM >= 2
		cons_fc[1] = fc_x1_ref;
#else
		cons_fc[1] = amrex::Array4<const amrex::Real>{};
#endif
#if AMREX_SPACEDIM == 3
		cons_fc[2] = fc_x2_ref;
#else
		cons_fc[2] = amrex::Array4<const amrex::Real>{};
#endif
		const amrex::Real rho = cons_cc[bx](i, j, k, density_index);
		const amrex::Real px = cons_cc[bx](i, j, k, x1Momentum_index);
		const amrex::Real py = cons_cc[bx](i, j, k, x2Momentum_index);
		const amrex::Real pz = cons_cc[bx](i, j, k, x3Momentum_index);
		const amrex::Real E = cons_cc[bx](i, j, k, energy_index); // *total* gas energy per unit volume
		const amrex::Real Eint_aux = cons_cc[bx](i, j, k, internalEnergy_index);

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(px));
		AMREX_ASSERT(!std::isnan(py));
		AMREX_ASSERT(!std::isnan(pz));
		AMREX_ASSERT(!std::isnan(E));

		const amrex::Real vx = px / rho;
		const amrex::Real vy = py / rho;
		const amrex::Real vz = pz / rho;
		const amrex::Real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		amrex::Real magnetic_energy = 0.;

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			// note, bx is the box, and bxi is the magnetic-field component
			const amrex::Real b_x0_m = fc_x0_ref(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x0_p = fc_x0_ref(i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x0 = 0.5 * (b_x0_m + b_x0_p);
#if AMREX_SPACEDIM >= 2
			const amrex::Real b_x1_m = fc_x1_ref(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x1_p = fc_x1_ref(i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x1 = 0.5 * (b_x1_m + b_x1_p);
#endif
#if AMREX_SPACEDIM == 3
			const amrex::Real b_x2_m = fc_x2_ref(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x2_p = fc_x2_ref(i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
			const amrex::Real b_x2 = 0.5 * (b_x2_m + b_x2_p);
#endif
			magnetic_energy = 0.5 * (AMREX_D_TERM(b_x0 * b_x0, +b_x1 * b_x1, +b_x2 * b_x2));
		}
		const amrex::Real Eint_cons = E - kinetic_energy - magnetic_energy;

		const amrex::Real Pgas = ComputePressure(cons_cc[bx], cons_fc, i, j, k);
		const amrex::Real eint_cons = Eint_cons / rho;
		const amrex::Real eint_aux = Eint_aux / rho;

		AMREX_ASSERT(rho > 0.);
		if constexpr (!is_eos_isothermal()) {
			AMREX_ASSERT(Pgas > 0.);
		}

		primVar[bx](i, j, k, primDensity_index) = rho;
		primVar[bx](i, j, k, x1Velocity_index) = vx;
		primVar[bx](i, j, k, x2Velocity_index) = vy;
		primVar[bx](i, j, k, x3Velocity_index) = vz;

		if constexpr (reconstruct_eint) {
			// save specific internal energy (SIE) == (Etot - KE) / rho
			primVar[bx](i, j, k, pressure_index) = eint_cons;
			// save auxiliary specific internal energy (SIE) == Eint_aux / rho
			primVar[bx](i, j, k, primEint_index) = eint_aux;
		} else {
			// save pressure
			primVar[bx](i, j, k, pressure_index) = Pgas;
			// save auxiliary internal energy (rho * e)
			primVar[bx](i, j, k, primEint_index) = Eint_aux;
		}

		// copy any passive scalars
		for (int nc = 0; nc < nscalars_; ++nc) {
			primVar[bx](i, j, k, primScalar0_index + nc) = cons_cc[bx](i, j, k, scalar0_index + nc);
		}
	});
}

template <typename problem_t> auto HydroSystem<problem_t>::maxSignalSpeedLocal(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, 3> const &cons_fc_mf) -> amrex::Real
{
	// return maximum signal speed on local grids
	auto const &cons_fc_x0 = cons_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = cons_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = cons_fc_mf[2].const_arrays();
#endif
	
auto const &cons = cons_mf.const_arrays();
	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{}, cons_mf,
				amrex::IntVect(0), // no ghost cells
				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
					std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
					std::remove_cv_t<std::remove_reference_t<decltype(cons_fc_x0[bx])>> fc_x0_ref{};
					if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
						cons_fc[0] = cons_fc_x0[bx];
					#if AMREX_SPACEDIM >= 2
						cons_fc[1] = cons_fc_x1[bx];
					#endif
					#if AMREX_SPACEDIM == 3
						cons_fc[2] = cons_fc_x2[bx];
					#endif
					}
					const auto rho = cons[bx](i, j, k, HydroSystem<problem_t>::density_index);
					const auto px = cons[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
					const auto py = cons[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
					const auto pz = cons[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
					const auto kinetic_energy = (px * px + py * py + pz * pz) / (2.0 * rho);
					const double abs_vel = std::sqrt(2.0 * kinetic_energy / rho);
					double cs = NAN;

					if constexpr (is_eos_isothermal()) {
						cs = cs_iso_;
					} else {
						cs = ComputeSoundSpeed(cons[bx], cons_fc, i, j, k);
					}
					return {cs + abs_vel};
				});
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc,
						   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &cons_fc, array_t &maxSignal,
						   amrex::Box const &indexRange)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// First-capture by early access
		const auto rho = cons_cc(i, j, k, density_index);
		const auto px = cons_cc(i, j, k, x1Momentum_index);
		const auto py = cons_cc(i, j, k, x2Momentum_index);
		const auto pz = cons_cc(i, j, k, x3Momentum_index);
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(px));
		AMREX_ASSERT(!std::isnan(py));
		AMREX_ASSERT(!std::isnan(pz));

		const auto vx = px / rho;
		const auto vy = py / rho;
		const auto vz = pz / rho;
		const double vel_magnitude = std::sqrt(vx * vx + vy * vy + vz * vz);
		double fastest_wavespeed = NAN;

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons_cc, i, j, k);
			const auto total_energy = cons_cc(i, j, k, energy_index); // *total* gas energy per unit volume
			const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
			const auto bx1_m = cons_fc[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const auto bx1_p = cons_fc[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const auto bx2_m = cons_fc[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const auto bx2_p = cons_fc[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const auto bx3_m = cons_fc[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const auto bx3_p = cons_fc[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
			const auto bx1 = 0.5 * (bx1_m + bx1_p);
			const auto bx2 = 0.5 * (bx2_m + bx2_p);
			const auto bx3 = 0.5 * (bx3_m + bx3_p);
			double b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
			const auto magnetic_energy = 0.5 * b_sq;
			const auto thermal_energy = total_energy - kinetic_energy - magnetic_energy;
			const auto pressure = quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
			double gp = quokka::EOS<problem_t>::gamma_ * pressure;


			double bgp_p = b_sq + gp;
			double const bgp_m = b_sq - gp;
			fastest_wavespeed = std::max({std::sqrt(0.5 * (bgp_p + std::sqrt(bgp_m * bgp_m + 4.0 * gp * (bx2 * bx2 + bx3 * bx3))) / rho),
						      std::sqrt(0.5 * (bgp_p + std::sqrt(bgp_m * bgp_m + 4.0 * gp * (bx1 * bx1 + bx3 * bx3))) / rho),
						      std::sqrt(0.5 * (bgp_p + std::sqrt(bgp_m * bgp_m + 4.0 * gp * (bx1 * bx1 + bx2 * bx2))) / rho)});
		} else {
			if constexpr (is_eos_isothermal()) {
				fastest_wavespeed = cs_iso_;
			} else {
				fastest_wavespeed = ComputeSoundSpeed(cons_cc, cons_fc, i, j, k);
			}
			AMREX_ASSERT(fastest_wavespeed > 0.);
		}

		const double signal_max = fastest_wavespeed + vel_magnitude;
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t> auto HydroSystem<problem_t>::CheckStatesValid(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> bool
{
	// check whether density or pressure are negative
	auto const &cons = cons_mf.const_arrays();
		auto const &cons_fc_x0 = cons_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = cons_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = cons_fc_mf[2].const_arrays();
#endif


	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalAnd>{}, amrex::TypeList<bool>{}, cons_mf,
				amrex::IntVect(0), // no ghost cells
				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<bool> {
					
					std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
					amrex::Real magnetic_energy = 0.0;
					if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
						cons_fc[0] = cons_fc_x0[bx];
					#if AMREX_SPACEDIM >= 2
						cons_fc[1] = cons_fc_x1[bx];
					#endif
					#if AMREX_SPACEDIM == 3
						cons_fc[2] = cons_fc_x2[bx];
					#endif
						const auto bx1_m = cons_fc[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
						const auto bx1_p = cons_fc[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
						const auto bx2_m = cons_fc[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
						const auto bx2_p = cons_fc[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
						const auto bx3_m = cons_fc[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
						const auto bx3_p = cons_fc[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
						const auto bx1 = 0.5 * (bx1_m + bx1_p);
						const auto bx2 = 0.5 * (bx2_m + bx2_p);
						const auto bx3 = 0.5 * (bx3_m + bx3_p);
						double b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
						magnetic_energy = 0.5 * b_sq;
					}
					const auto rho = cons[bx](i, j, k, density_index);
					const auto px = cons[bx](i, j, k, x1Momentum_index);
					const auto py = cons[bx](i, j, k, x2Momentum_index);
					const auto pz = cons[bx](i, j, k, x3Momentum_index);
					const auto E = cons[bx](i, j, k, energy_index);
					const auto vx = px / rho;
					const auto vy = py / rho;
					const auto vz = pz / rho;
					const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
					const auto thermal_energy = E - kinetic_energy - magnetic_energy;
					const auto P = ComputePressure(cons[bx], cons_fc, i, j, k);

					bool negativeDensity = (rho <= 0.);
					bool const negativePressure = (P <= 0.);

					if constexpr (is_eos_isothermal()) {
						if (negativeDensity) {
							printf("invalid state at (%d, %d, %d): rho %g\n", i, j, k, rho);
							return {false};
						}
					} else {
						if (negativeDensity || negativePressure) {
							printf("invalid state at (%d, %d, %d): rho %g, Etot %g, Eint %g, P %g\n", i, j, k, rho, E,
							       thermal_energy, P);
							return {false};
						}
					}
					return {true};
				});
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, std::array<amrex::Array4<const amrex::Real>, 3> const &cons_fc, int i, int j, int k)
    -> quokka::valarray<amrex::Real, nvar_>
{
	// convert to primitive vars
	amrex::Real magnetic_energy = 0.0;
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto Eint_aux = cons(i, j, k, internalEnergy_index);
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		const auto bx1_m = cons_fc[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1_p = cons_fc[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_m = cons_fc[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_p = cons_fc[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_m = cons_fc[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_p = cons_fc[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1 = 0.5 * (bx1_m + bx1_p);
		const auto bx2 = 0.5 * (bx2_m + bx2_p);
		const auto bx3 = 0.5 * (bx3_m + bx3_p);
		double b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
		magnetic_energy = 0.5 * b_sq;
	} 
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy - magnetic_energy;

	amrex::Real P = NAN;
	if constexpr (is_eos_isothermal()) {
		P = rho * cs_iso_ * cs_iso_;
	} else {
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
		P = quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
	}

	quokka::valarray<amrex::Real, nvar_> primVars{rho, vx, vy, vz, P, Eint_aux};

	for (int n = 0; n < nscalars_; ++n) {
		primVars[primScalar0_index + n] = cons(i, j, k, scalar0_index + n);
	}
	return primVars;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeConsVars(quokka::valarray<amrex::Real, nvar_> const &prim)
    -> quokka::valarray<amrex::Real, nvar_>
{
	// convert to conserved vars
	Real const rho = prim[0];
	Real const v1 = prim[1];
	Real const v2 = prim[2];
	Real const v3 = prim[3];
	Real const P = prim[4];
	Real const Eint_aux = prim[5];

	Real const Eint = quokka::EOS<problem_t>::ComputeEintFromPres(rho, P);
	Real const Egas = Eint + 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3);

	quokka::valarray<amrex::Real, nvar_> consVars{rho, rho * v1, rho * v2, rho * v3, Egas, Eint_aux};

	for (int i = 0; i < nscalars_; ++i) {
		consVars[scalar0_index + i] = prim[primScalar0_index + i];
	}
	return consVars;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePressure(amrex::Array4<const amrex::Real> const &cons, std::array<amrex::Array4<const amrex::Real>, 3> const &cons_fc, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real magnetic_energy = 0.0;
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		const auto bx1_m = cons_fc[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1_p = cons_fc[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_m = cons_fc[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_p = cons_fc[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_m = cons_fc[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_p = cons_fc[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1 = 0.5 * (bx1_m + bx1_p);
		const auto bx2 = 0.5 * (bx2_m + bx2_p);
		const auto bx3 = 0.5 * (bx3_m + bx3_p);
		double b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
		magnetic_energy = 0.5 * b_sq;
	}
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy - magnetic_energy;
	amrex::Real P = NAN;

	if constexpr (is_eos_isothermal()) {
		P = rho * cs_iso_ * cs_iso_;
	} else {
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
		P = quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
	}
	return P;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, std::array<amrex::Array4<const amrex::Real>, 3> const &cons_fc, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real magnetic_energy = 0.0;
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		const auto bx1_m = cons_fc[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1_p = cons_fc[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_m = cons_fc[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx2_p = cons_fc[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_m = cons_fc[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx3_p = cons_fc[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
		const auto bx1 = 0.5 * (bx1_m + bx1_p);
		const auto bx2 = 0.5 * (bx2_m + bx2_p);
		const auto bx3 = 0.5 * (bx3_m + bx3_p);
		double b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
		magnetic_energy = 0.5 * b_sq;
	}
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy - magnetic_energy;

	amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
	amrex::Real P = quokka::EOS<problem_t>::ComputePressure(rho, thermal_energy, massScalars);
	amrex::Real cs = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P, massScalars);

	return cs;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real const rho = cons(i, j, k, density_index);
	amrex::Real const vel_x = cons(i, j, k, x1Momentum_index) / rho;
	return vel_x;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real const rho = cons(i, j, k, density_index);
	amrex::Real const vel_y = cons(i, j, k, x2Momentum_index) / rho;
	return vel_y;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real
{
	amrex::Real const rho = cons(i, j, k, density_index);
	amrex::Real const vel_z = cons(i, j, k, x3Momentum_index) / rho;
	return vel_z;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool
{
	// check if cons(i, j, k) is a valid state
	const amrex::Real rho = cons(i, j, k, density_index);
	bool const isDensityPositive = (rho > 0.);

	bool isMassScalarPositive = true;
	if constexpr (nmscalars_ > 0) {
		amrex::GpuArray<Real, nmscalars_> massScalars_ = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);

		for (int idx = 0; idx < nmscalars_; ++idx) {
			if (massScalars_[idx] < 0.0) {
				isMassScalarPositive = false;
				break; // Exit the loop early if any element is not positive
			}
		}
	}
	// when the dual energy method is used, we *cannot* reset on pressure
	// failures. on the other hand, we don't need to -- the auxiliary internal
	// energy is used instead!

	return isDensityPositive && isMassScalarPositive;
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeRhsFromFluxes(amrex::MultiFab &rhs_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const int nvars)
{
	// compute the total right-hand-side for the MOL integration

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const x1Flux = fluxArray[0].const_arrays();
#if (AMREX_SPACEDIM >= 2)
	auto const x2Flux = fluxArray[1].const_arrays();
#endif
#if (AMREX_SPACEDIM == 3)
	auto const x3Flux = fluxArray[2].const_arrays();
#endif
	auto rhs = rhs_mf.arrays();

	amrex::ParallelFor(rhs_mf, amrex::IntVect{0}, nvars, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
		rhs[bx](i, j, k, n) = AMREX_D_TERM((1.0 / dx[0]) * (x1Flux[bx](i, j, k, n) - x1Flux[bx](i + 1, j, k, n)),
						   +(1.0 / dx[1]) * (x2Flux[bx](i, j, k, n) - x2Flux[bx](i, j + 1, k, n)),
						   +(1.0 / dx[2]) * (x3Flux[bx](i, j, k, n) - x3Flux[bx](i, j, k + 1, n)));
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::PredictStep(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab const &rhs_mf, const double dt,
					 const int nvars, amrex::iMultiFab &redoFlag_mf)
{
	const BL_PROFILE("HydroSystem::PredictStep()");

	auto const &consVarOld = consVarOld_mf.const_arrays();
	auto const &rhs = rhs_mf.const_arrays();
	auto consVarNew = consVarNew_mf.arrays();
	auto redoFlag = redoFlag_mf.arrays();

	amrex::ParallelFor(consVarNew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			consVarNew[bx](i, j, k, n) = consVarOld[bx](i, j, k, n) + dt * rhs[bx](i, j, k, n);
		}
		// check if state is valid -- flag for re-do if not
		if (!isStateValid(consVarNew[bx], i, j, k)) {
			redoFlag[bx](i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag[bx](i, j, k) = quokka::redoFlag::none;
		}
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::AddFluxesRK2(amrex::MultiFab &Unew_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, amrex::MultiFab const &rhs_mf,
					  const double dt, const int nvars, amrex::iMultiFab &redoFlag_mf)
{
	const BL_PROFILE("HydroSystem::AddFluxesRK2()");

	auto const &U0 = U0_mf.const_arrays();
	auto const &U1 = U1_mf.const_arrays();
	auto const &rhs = rhs_mf.const_arrays();
	auto U_new = Unew_mf.arrays();
	auto redoFlag = redoFlag_mf.arrays();

	amrex::ParallelFor(Unew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			// RK-SSP2 integrator
			const double U_0 = U0[bx](i, j, k, n);
			const double U_1 = U1[bx](i, j, k, n);
			const double FU = dt * rhs[bx](i, j, k, n);

			// save results in U_new
			U_new[bx](i, j, k, n) = (0.5 * U_0 + 0.5 * U_1) + 0.5 * FU;
		}

		// check if state is valid -- flag for re-do if not
		if (!isStateValid(U_new[bx], i, j, k)) {
			redoFlag[bx](i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag[bx](i, j, k) = quokka::redoFlag::none;
		}
	});
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::ComputeFlatteningCoefficients(amrex::MultiFab const &primVar_mf, amrex::MultiFab &x1Chi_mf, const int nghost)
{
	// compute the PPM shock flattening coefficient following
	//   Appendix B1 of Mignone+ 2005 [this description has typos].
	// Method originally from Miller & Colella,
	//   Journal of Computational Physics 183, 26–82 (2002) [no typos].

	constexpr double beta_max = 0.85;
	constexpr double beta_min = 0.75;
	constexpr double Zmax = 0.75;
	constexpr double Zmin = 0.25;

	auto const &primVar_in = primVar_mf.const_arrays();
	auto x1Chi_in = x1Chi_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(primVar_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
		quokka::Array4View<const amrex::Real, DIR> primVar(primVar_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1Chi(x1Chi_in[bx]);
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		amrex::Real Pplus2 = primVar(i + 2, j, k, pressure_index);
		amrex::Real Pplus1 = primVar(i + 1, j, k, pressure_index);
		amrex::Real P = primVar(i, j, k, pressure_index);
		amrex::Real Pminus1 = primVar(i - 1, j, k, pressure_index);
		amrex::Real Pminus2 = primVar(i - 2, j, k, pressure_index);

		if constexpr (reconstruct_eint) {
			// compute (rho e) (gamma - 1)
			amrex::GpuArray<Real, nmscalars_> massScalars_plus2 = RadSystem<problem_t>::ComputeMassScalars(primVar, i + 2, j, k);
			Pplus2 = quokka::EOS<problem_t>::ComputePressure(primVar(i + 2, j, k, primDensity_index),
									 primVar(i + 2, j, k, primDensity_index) * Pplus2, massScalars_plus2);
			amrex::GpuArray<Real, nmscalars_> massScalars_plus1 = RadSystem<problem_t>::ComputeMassScalars(primVar, i + 1, j, k);
			Pplus1 = quokka::EOS<problem_t>::ComputePressure(primVar(i + 1, j, k, primDensity_index),
									 primVar(i + 1, j, k, primDensity_index) * Pplus1, massScalars_plus1);
			amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(primVar, i, j, k);
			P = quokka::EOS<problem_t>::ComputePressure(primVar(i, j, k, primDensity_index), primVar(i, j, k, primDensity_index) * P, massScalars);
			amrex::GpuArray<Real, nmscalars_> massScalars_minus1 = RadSystem<problem_t>::ComputeMassScalars(primVar, i - 1, j, k);
			Pminus1 = quokka::EOS<problem_t>::ComputePressure(primVar(i - 1, j, k, primDensity_index),
									  primVar(i - 1, j, k, primDensity_index) * Pminus1, massScalars_minus1);
			amrex::GpuArray<Real, nmscalars_> massScalars_minus2 = RadSystem<problem_t>::ComputeMassScalars(primVar, i - 2, j, k);
			Pminus2 = quokka::EOS<problem_t>::ComputePressure(primVar(i - 2, j, k, primDensity_index),
									  primVar(i - 2, j, k, primDensity_index) * Pminus2, massScalars_minus2);
		}

		if constexpr (is_eos_isothermal()) {
			const amrex::Real cs_sq = cs_iso_ * cs_iso_;
			Pplus2 = primVar(i + 2, j, k, primDensity_index) * cs_sq;
			Pplus1 = primVar(i + 1, j, k, primDensity_index) * cs_sq;
			P = primVar(i, j, k, primDensity_index) * cs_sq;
			Pminus1 = primVar(i - 1, j, k, primDensity_index) * cs_sq;
			Pminus2 = primVar(i - 2, j, k, primDensity_index) * cs_sq;
		}

		// beta is a measure of shock resolution (Eq. 74 of Miller & Colella 2002)
		// Miller & Collela note: "If beta is 1/2, then pressure is linear across
		//   four computational cells. If beta is small enough, then we assume that
		//   any discontinuity is already sufficiently well resolved that additional
		//   dissipation (flattening) is not required."
		const double beta_denom = std::abs(Pplus2 - Pminus2);
		// avoid division by zero (in this case, chi = 1 anyway)
		const double beta = (beta_denom != 0) ? (std::abs(Pplus1 - Pminus1) / beta_denom) : 0;

		// Eq. 75 of Miller & Colella 2002
		const double chi_min = std::max(0., std::min(1., (beta_max - beta) / (beta_max - beta_min)));

		// Z is a measure of shock strength (Eq. 76 of Miller & Colella 2002)
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(primVar, i, j, k);
		double K_S = std::pow(quokka::EOS<problem_t>::ComputeSoundSpeed(primVar(i, j, k, primDensity_index), P, massScalars), 2) *
			     primVar(i, j, k, primDensity_index);
		if constexpr (is_eos_isothermal()) {
			K_S = primVar(i, j, k, primDensity_index) * cs_iso_ * cs_iso_;
		}

		const double Z = std::abs(Pplus1 - Pminus1) / K_S;

		// check for converging flow along the normal direction DIR (Eq. 77)
		int velocity_index = 0;
		if constexpr (DIR == FluxDir::X1) {
			velocity_index = x1Velocity_index;
		} else if constexpr (DIR == FluxDir::X2) {
			velocity_index = x2Velocity_index;
		} else if constexpr (DIR == FluxDir::X3) {
			velocity_index = x3Velocity_index;
		}
		double chi = 1.0;
		if (primVar(i + 1, j, k, velocity_index) < primVar(i - 1, j, k, velocity_index)) {
			chi = std::max(chi_min, std::min(1., (Zmax - Z) / (Zmax - Zmin)));
		}

		x1Chi(i, j, k) = chi;
	});
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::FlattenShocks(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf,
					   amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf, const int nghost,
					   const int nvars)
{
	// Apply shock flattening based on Miller & Colella (2002)
	// [This is necessary to get a reasonable solution to the slow-moving
	// shock problem, and reduces post-shock oscillations in other cases.]

	auto const &q_in = q_mf.const_arrays();
	auto const &x1Chi_in = x1Chi_mf.const_arrays();
	auto const &x2Chi_in = x2Chi_mf.const_arrays();
	auto const &x3Chi_in = x3Chi_mf.const_arrays();
	auto x1LeftState_in = x1LeftState_mf.arrays();
	auto x1RightState_in = x1RightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) {
		quokka::Array4View<const amrex::Real, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1LeftState(x1LeftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1RightState(x1RightState_in[bx]);

		// compute coefficient as the minimum from adjacent cells along *each
		// axis*
		//  (Eq. 86 of Miller & Colella 2001; Eq. 78 of Miller & Colella 2002)
		double const chi_ijk = std::min({
		    x1Chi_in[bx](i_in - 1, j_in, k_in),
		    x1Chi_in[bx](i_in, j_in, k_in),
		    x1Chi_in[bx](i_in + 1, j_in, k_in),
#if (AMREX_SPACEDIM >= 2)
		    x2Chi_in[bx](i_in, j_in - 1, k_in),
		    x2Chi_in[bx](i_in, j_in, k_in),
		    x2Chi_in[bx](i_in, j_in + 1, k_in),
#endif
#if (AMREX_SPACEDIM == 3)
		    x3Chi_in[bx](i_in, j_in, k_in - 1),
		    x3Chi_in[bx](i_in, j_in, k_in),
		    x3Chi_in[bx](i_in, j_in, k_in + 1),
#endif
		});

		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// get interfaces
		const double a_minus = x1RightState(i, j, k, n);
		const double a_plus = x1LeftState(i + 1, j, k, n);
		const double a_mean = q(i, j, k, n);

		// left side of zone i (Eq. 70a)
		const double new_a_minus = chi_ijk * a_minus + (1. - chi_ijk) * a_mean;

		// right side of zone i (Eq. 70b)
		const double new_a_plus = chi_ijk * a_plus + (1. - chi_ijk) * a_mean;

		x1RightState(i, j, k, n) = new_a_minus;
		x1LeftState(i + 1, j, k, n) = new_a_plus;
	});

#if (AMREX_SPACEDIM < 2)
	amrex::ignore_unused(x2Chi_in);
#endif
#if (AMREX_SPACEDIM < 3)
	amrex::ignore_unused(x3Chi_in);
#endif
}

// to ensure that physical quantities are within reasonable
// floors and ceilings which can be set in the param file
template <typename problem_t> void HydroSystem<problem_t>::EnforceLimits(amrex::Real const densityFloor, amrex::Real const tempFloor, amrex::MultiFab &state_mf)
{
	auto state = state_mf.arrays();

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// Enforce density floor (do not adjust energies here!!)
		amrex::Real rho_new = NAN;
		{
			amrex::Real const rho = state[bx](i, j, k, density_index);
			rho_new = rho;

			if (rho < densityFloor) {
				rho_new = densityFloor;
				state[bx](i, j, k, density_index) = rho_new;

				if (nscalars_ > 0) {
					for (int n = 0; n < nscalars_; ++n) {
						if (rho_new == 0.0) {
							state[bx](i, j, k, scalar0_index + n) = 0.0;
						} else {
							state[bx](i, j, k, scalar0_index + n) *= rho / rho_new;
						}
					}
				}
			}
		}

		// Enforce mass scalars floor
		if (nmscalars_ > 0) {
			amrex::Real sp_sum = 0.0;
			for (int idx = 0; idx < nmscalars_; ++idx) {
				if (state[bx](i, j, k, scalar0_index + idx) < 0.0) {
					state[bx](i, j, k, scalar0_index + idx) = network_rp::small_x * rho_new;
				}
				// get sum to renormalize
				sp_sum += state[bx](i, j, k, scalar0_index + idx);
			}

			if ((sp_sum > std::numeric_limits<amrex::Real>::min()) && (rho_new > std::numeric_limits<amrex::Real>::min())) {
				sp_sum /= rho_new; // get mass fractions
				for (int idx = 0; idx < nmscalars_; ++idx) {
					// renormalize
					state[bx](i, j, k, scalar0_index + idx) /= sp_sum;
				}
			}
		}

		// Enforce temperature floor
		if ((rho_new > std::numeric_limits<amrex::Real>::min()) && !HydroSystem<problem_t>::is_eos_isothermal()) {
			amrex::Real const vx1 = state[bx](i, j, k, x1Momentum_index) / rho_new;
			amrex::Real const vx2 = state[bx](i, j, k, x2Momentum_index) / rho_new;
			amrex::Real const vx3 = state[bx](i, j, k, x3Momentum_index) / rho_new;
			amrex::Real const Ekin = 0.5 * rho_new * (vx1 * vx1 + vx2 * vx2 + vx3 * vx3);

			// Enforce temperature floor (for total energy)
			amrex::GpuArray<Real, nmscalars_> const massScalars = RadSystem<problem_t>::ComputeMassScalars(state[bx], i, j, k);
			amrex::Real const Etot = state[bx](i, j, k, energy_index);
			amrex::Real const primTemp = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_new, (Etot - Ekin), massScalars);

			if (primTemp < tempFloor) {
				amrex::Real const prim_eint = quokka::EOS<problem_t>::ComputeEintFromTgas(rho_new, tempFloor, massScalars);
				state[bx](i, j, k, energy_index) = Ekin + prim_eint;
			}

			// Enforce temperature floor (for auxiliary internal energy)
			amrex::Real const auxEint = state[bx](i, j, k, internalEnergy_index);
			amrex::Real const auxTemp = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_new, auxEint, massScalars);

			if (auxTemp < tempFloor) {
				amrex::Real const new_Eint = quokka::EOS<problem_t>::ComputeEintFromTgas(rho_new, tempFloor, massScalars);
				state[bx](i, j, k, internalEnergy_index) = new_Eint;
				// total energy should NOT be updated here
			}
		}
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf)
{
	// compute P dV source term for the internal energy equation,
	// using the face-centered velocities in faceVelArray and the pressure
	auto const &cons_fc_x0 = cons_fc_mf[0].const_arrays();
	auto vel_x = faceVelArray[0].const_arrays();

#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = cons_fc_mf[1].const_arrays();
	auto vel_y = faceVelArray[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = cons_fc_mf[2].const_arrays();
	auto vel_z = faceVelArray[2].const_arrays();
#endif


	auto const &consVar = consVar_mf.const_arrays();
	auto const &redoFlag = redoFlag_mf.const_arrays();
	auto rhs = rhs_mf.arrays();

amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
    // get cell-centered pressure
    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
    
    if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
        cons_fc[0] = cons_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
        cons_fc[1] = cons_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
        cons_fc[2] = cons_fc_x2[bx];
#endif
    }
		const amrex::Real Pgas = ComputePressure(consVar[bx], cons_fc, i, j, k);

		// compute div v from face-centered velocities
		amrex::Real div_v = NAN;

		if (redoFlag[bx](i, j, k) == quokka::redoFlag::none) {
			div_v = AMREX_D_TERM((vel_x[bx](i + 1, j, k) - vel_x[bx](i, j, k)) / dx[0], +(vel_y[bx](i, j + 1, k) - vel_y[bx](i, j, k)) / dx[1],
					     +(vel_z[bx](i, j, k + 1) - vel_z[bx](i, j, k)) / dx[2]);
		} else {
			div_v = 0.5 * (AMREX_D_TERM((ComputeVelocityX1(consVar[bx], i + 1, j, k) - ComputeVelocityX1(consVar[bx], i - 1, j, k)) / dx[0],
						    +(ComputeVelocityX2(consVar[bx], i, j + 1, k) - ComputeVelocityX2(consVar[bx], i, j - 1, k)) / dx[1],
						    +(ComputeVelocityX3(consVar[bx], i, j, k + 1) - ComputeVelocityX3(consVar[bx], i, j, k - 1)) / dx[2]));
		}

		// add P dV term to rhs array
		rhs[bx](i, j, k, internalEnergy_index) += -Pgas * div_v;
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::SyncDualEnergy(amrex::MultiFab &consVar_mf, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &faceVar_mf)
{
	// sync internal energy and total energy
	// this step must be done as an operator-split step after *each* RK stage

	const amrex::Real eta = 1.0e-3; // dual energy parameter 'eta'

	auto consVar = consVar_mf.arrays();
	auto faceVar_x = faceVar_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto faceVar_y = faceVar_mf[1].const_arrays();
#if AMREX_SPACEDIM == 3
	auto faceVar_z = faceVar_mf[2].const_arrays();
#endif
#endif

	amrex::ParallelFor(consVar_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		amrex::Real const rho = consVar[bx](i, j, k, density_index);
		amrex::Real const px = consVar[bx](i, j, k, x1Momentum_index);
		amrex::Real const py = consVar[bx](i, j, k, x2Momentum_index);
		amrex::Real const pz = consVar[bx](i, j, k, x3Momentum_index);
		amrex::Real const Etot = consVar[bx](i, j, k, energy_index);
		amrex::Real const Eint_aux = consVar[bx](i, j, k, internalEnergy_index);

		// abort if density is negative (can't compute kinetic energy)
		if (rho <= 0.) {
			amrex::Abort("density is negative in SyncDualEnergy! abort!!");
		}

		// compute kinetic energy
		amrex::Real const Ekin = (px * px + py * py + pz * pz) / (2.0 * rho);

		// compute magnetic energy
		amrex::Real Emag = 0;
		if (Physics_Traits<problem_t>::is_mhd_enabled) { // cannot be 'if constexpr' due to CUDA limitation
			constexpr int mhd_idx = Physics_Indices<problem_t>::mhdFirstIndex;
			amrex::Real const Bx = 0.5 * (faceVar_x[bx](i, j, k, mhd_idx) + faceVar_x[bx](i + 1, j, k, mhd_idx));
#if AMREX_SPACEDIM >= 2
			amrex::Real const By = 0.5 * (faceVar_y[bx](i, j, k, mhd_idx) + faceVar_y[bx](i, j + 1, k, mhd_idx));
#endif
#if AMREX_SPACEDIM == 3
			amrex::Real const Bz = 0.5 * (faceVar_z[bx](i, j, k, mhd_idx) + faceVar_z[bx](i, j, k + 1, mhd_idx));
#endif
			Emag = 0.5 * (AMREX_D_TERM(Bx * Bx, +By * By, +Bz * Bz));
		}

		// compute internal energy from conserved vars
		amrex::Real const Eint_cons = Etot - Ekin - Emag;

		// Li et al. sync method
		// replace Eint with Eint_cons == (Etot - Ekin) if (Eint_cons / E) > eta
		if (Eint_cons > eta * Etot) {
			consVar[bx](i, j, k, internalEnergy_index) = Eint_cons;
		} else { // non-conservative sync
			consVar[bx](i, j, k, internalEnergy_index) = Eint_aux;
			consVar[bx](i, j, k, energy_index) = Eint_aux + Ekin + Emag;
		}
	});
}

template <typename problem_t>
template <RiemannSolver RIEMANN, FluxDir DIR>
void HydroSystem<problem_t>::ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf,
					   amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &x1LeftState_bfield_mf,
					   amrex::MultiFab const &x1RightState_bfield_mf, amrex::MultiFab const &primVar_mf, const amrex::Real K_visc,
					   amrex::MultiFab *x1FSpds_mf, amrex::MultiFab const *x1ConsVar_fc_mf, const int nghost_vel)
{

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	auto const &x1LeftState_in = x1LeftState_mf.const_arrays();
	auto const &x1RightState_in = x1RightState_mf.const_arrays();
	auto const &x1LeftState_bfield_in = x1LeftState_bfield_mf.const_arrays();
	auto const &x1RightState_bfield_in = x1RightState_bfield_mf.const_arrays();
	auto const &primVar_in = primVar_mf.const_arrays();
	auto x1Flux_in = x1Flux_mf.arrays();
	auto x1FaceVel_in = x1FaceVel_mf.arrays();

	amrex::MultiArray4<double> x1FSpds_in;
	amrex::MultiArray4<const double> x1ConsVar_fc_in;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		if (RIEMANN == RiemannSolver::HLLD || RIEMANN == RiemannSolver::LLF_MHD) {
			x1FSpds_in = (*x1FSpds_mf).arrays();
		}
		x1ConsVar_fc_in = (*x1ConsVar_fc_mf).const_arrays();
	}

	// Include ghost cells when computing face velocities
	amrex::IntVect ng{AMREX_D_DECL(nghost_vel, nghost_vel, nghost_vel)};
	amrex::ParallelFor(x1Flux_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
		// first capture
		[[maybe_unused]] std::remove_cv_t<std::remove_reference_t<decltype(x1ConsVar_fc_in[bx])>> x1ConsVar_fc_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			x1ConsVar_fc_ref = x1ConsVar_fc_in[bx];
		}
		[[maybe_unused]] std::remove_cv_t<std::remove_reference_t<decltype(x1FSpds_in[bx])>> x1FSpds_ref{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			x1FSpds_ref = x1FSpds_in[bx];
		}

		quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in[bx]);
		quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in[bx]);
		quokka::Array4View<const amrex::Real, DIR> x1LeftState_bfield(x1LeftState_bfield_in[bx]);
		quokka::Array4View<const amrex::Real, DIR> x1RightState_bfield(x1RightState_bfield_in[bx]);
		quokka::Array4View<const amrex::Real, DIR> q(primVar_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1FaceVel(x1FaceVel_in[bx]);

		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// gather left- and right- state variables

		const double rho_L = x1LeftState(i, j, k, primDensity_index);
		const double rho_R = x1RightState(i, j, k, primDensity_index);

		const double vx_L = x1LeftState(i, j, k, x1Velocity_index);
		const double vx_R = x1RightState(i, j, k, x1Velocity_index);

		const double vy_L = x1LeftState(i, j, k, x2Velocity_index);
		const double vy_R = x1RightState(i, j, k, x2Velocity_index);

		const double vz_L = x1LeftState(i, j, k, x3Velocity_index);
		const double vz_R = x1RightState(i, j, k, x3Velocity_index);

		const double ke_L = 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L);
		const double ke_R = 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R);

		// auxiliary Eint (rho * e)
		// this is evolved as a passive scalar by the Riemann solver
		double Eint_L = NAN;
		double Eint_R = NAN;

		double P_L = NAN;
		double P_R = NAN;

		double E_L = NAN;
		double E_R = NAN;

		double cs_L = NAN;
		double cs_R = NAN;

		double bx1 = 0.0;
		double by_L = 0.0;
		double bz_L = 0.0;
		double by_R = 0.0;
		double bz_R = 0.0;
		double magnetic_energy_L = 0.0;
		double magnetic_energy_R = 0.0;
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			quokka::Array4View<const amrex::Real, DIR> x1ConsVar_fc(x1ConsVar_fc_ref);
			bx1 = x1ConsVar_fc(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			by_L = x1LeftState_bfield(i, j, k, 0);
			bz_L = x1LeftState_bfield(i, j, k, 1);
			by_R = x1RightState_bfield(i, j, k, 0);
			bz_R = x1RightState_bfield(i, j, k, 1);
			magnetic_energy_L = 0.5 * (bx1 * bx1 + by_L * by_L + bz_L * bz_L);
			magnetic_energy_R = 0.5 * (bx1 * bx1 + by_R * by_R + bz_R * bz_R);
		}

		if constexpr (is_eos_isothermal()) {
			P_L = rho_L * (cs_iso_ * cs_iso_);
			P_R = rho_R * (cs_iso_ * cs_iso_);

			cs_L = cs_iso_;
			cs_R = cs_iso_;
		} else {
			if constexpr (reconstruct_eint) {
				// compute pressure from specific internal energy
				// (pressure_index is actually eint)
				const double eint_L = x1LeftState(i, j, k, pressure_index);
				const double eint_R = x1RightState(i, j, k, pressure_index);
				amrex::GpuArray<Real, nmscalars_> massScalars_L = RadSystem<problem_t>::ComputeMassScalars(x1LeftState, i, j, k);
				P_L = quokka::EOS<problem_t>::ComputePressure(rho_L, eint_L * rho_L, massScalars_L);
				amrex::GpuArray<Real, nmscalars_> massScalars_R = RadSystem<problem_t>::ComputeMassScalars(x1RightState, i, j, k);
				P_R = quokka::EOS<problem_t>::ComputePressure(rho_R, eint_R * rho_R, massScalars_R);

				// auxiliary Eint is actually (auxiliary) specific internal energy
				Eint_L = rho_L * x1LeftState(i, j, k, primEint_index);
				Eint_R = rho_R * x1RightState(i, j, k, primEint_index);
			} else {
				// pressure_index is actually pressure
				P_L = x1LeftState(i, j, k, pressure_index);
				P_R = x1RightState(i, j, k, pressure_index);

				// primEint_index is actually (rho * e)
				Eint_L = x1LeftState(i, j, k, primEint_index);
				Eint_R = x1RightState(i, j, k, primEint_index);
			}

			amrex::GpuArray<Real, nmscalars_> massScalars_L = RadSystem<problem_t>::ComputeMassScalars(x1LeftState, i, j, k);
			cs_L = quokka::EOS<problem_t>::ComputeSoundSpeed(rho_L, P_L, massScalars_L);
			E_L = quokka::EOS<problem_t>::ComputeEintFromPres(rho_L, P_L, massScalars_L) + ke_L + magnetic_energy_L;

			amrex::GpuArray<Real, nmscalars_> massScalars_R = RadSystem<problem_t>::ComputeMassScalars(x1RightState, i, j, k);
			cs_R = quokka::EOS<problem_t>::ComputeSoundSpeed(rho_R, P_R, massScalars_R);
			E_R = quokka::EOS<problem_t>::ComputeEintFromPres(rho_R, P_R, massScalars_R) + ke_R + magnetic_energy_R;
		}

		AMREX_ASSERT(cs_L > 0.0);
		AMREX_ASSERT(cs_R > 0.0);

		// assign normal component of velocity according to DIR

		int velN_index = x1Velocity_index;
		int velV_index = x2Velocity_index;
		int velW_index = x3Velocity_index;

		if constexpr (DIR == FluxDir::X1) {
			velN_index = x1Velocity_index;
			velV_index = x2Velocity_index;
			velW_index = x3Velocity_index;
		} else if constexpr (DIR == FluxDir::X2) {
#if (AMREX_SPACEDIM == 2)
			velN_index = x2Velocity_index;
			velV_index = x1Velocity_index;
			velW_index = x3Velocity_index; // unchanged in 2D
#endif
#if (AMREX_SPACEDIM == 3)
			velN_index = x2Velocity_index;
			velV_index = x3Velocity_index;
			velW_index = x1Velocity_index;
#endif
		} else if constexpr (DIR == FluxDir::X3) {
			velN_index = x3Velocity_index;
			velV_index = x1Velocity_index;
			velW_index = x2Velocity_index;
		}

		quokka::HydroState<nscalars_, nmscalars_> sL{};
		sL.rho = rho_L;
		sL.u = x1LeftState(i, j, k, velN_index);
		sL.v = x1LeftState(i, j, k, velV_index);
		sL.w = x1LeftState(i, j, k, velW_index);
		sL.P = P_L;
		sL.cs = cs_L;
		sL.E = E_L;
		sL.Eint = Eint_L;
		sL.by = by_L;
		sL.bz = bz_L;

		quokka::HydroState<nscalars_, nmscalars_> sR{};
		sR.rho = rho_R;
		sR.u = x1RightState(i, j, k, velN_index);
		sR.v = x1RightState(i, j, k, velV_index);
		sR.w = x1RightState(i, j, k, velW_index);
		sR.P = P_R;
		sR.cs = cs_R;
		sR.E = E_R;
		sR.Eint = Eint_R;
		sR.by = by_R;
		sR.bz = bz_R;

		// The remaining components are mass scalars and passive scalars, so just copy them from
		// x1LeftState and x1RightState into the (left, right) state vectors U_L and
		// U_R
		for (int n = 0; n < nscalars_; ++n) {
			sL.scalar[n] = x1LeftState(i, j, k, scalar0_index + n);
			sR.scalar[n] = x1RightState(i, j, k, scalar0_index + n);
			// also store mass scalars separately
			if (n < nmscalars_) {
				sL.massScalar[n] = x1LeftState(i, j, k, scalar0_index + n);
				sR.massScalar[n] = x1RightState(i, j, k, scalar0_index + n);
			}
		}

		// difference in normal velocity along normal axis
		const double du = q(i, j, k, velN_index) - q(i - 1, j, k, velN_index);

		// difference in transverse velocity
#if AMREX_SPACEDIM == 1
		const double dw = 0.;
#else
	  	amrex::Real dvl = std::min(q(i - 1, j + 1, k, velV_index) - q(i - 1, j, k, velV_index), q(i - 1, j, k, velV_index) - q(i - 1, j - 1, k, velV_index));
	  	amrex::Real dvr = std::min(q(i, j + 1, k, velV_index) - q(i, j, k, velV_index), q(i, j, k, velV_index) - q(i, j - 1, k, velV_index));
	  	double dw = std::min(dvl, dvr);
#endif
#if AMREX_SPACEDIM == 3
		amrex::Real dwl =
		    std::min(q(i - 1, j, k + 1, velW_index) - q(i - 1, j, k, velW_index), q(i - 1, j, k, velW_index) - q(i - 1, j, k - 1, velW_index));
		amrex::Real dwr = std::min(q(i, j, k + 1, velW_index) - q(i, j, k, velW_index), q(i, j, k, velW_index) - q(i, j, k - 1, velW_index));
		dw = std::min({dwl, dwr, dw});
#endif

		// solve the Riemann problem in canonical form (i.e., where the x-dir is the normal direction)
		quokka::valarray<double, nvar_> F_canonical{};

		if constexpr (RIEMANN == RiemannSolver::HLLC) {
			static_assert(!Physics_Traits<problem_t>::is_mhd_enabled, "Cannot use HLLC solver for MHD problems!");
			F_canonical = quokka::Riemann::HLLC<problem_t, nscalars_, nmscalars_, nvar_>(sL, sR, gamma_, du, dw);
		} else if constexpr (RIEMANN == RiemannSolver::LLF) {
			F_canonical = quokka::Riemann::LLF<problem_t, nscalars_, nmscalars_, nvar_>(sL, sR);
		} else if constexpr (RIEMANN == RiemannSolver::LLF_MHD) {
			quokka::Array4View<amrex::Real, DIR> x1FSpds(x1FSpds_ref);
			auto [F_canonical_tmp, fspd_m, fspd_p] = quokka::Riemann::LLF_MHD<problem_t, nscalars_, nmscalars_, nvar_>(sL, sR, gamma_, bx1);
			F_canonical = F_canonical_tmp;
			x1FSpds(i, j, k, 0) = fspd_m;
			x1FSpds(i, j, k, 1) = fspd_p;
		} else if constexpr (RIEMANN == RiemannSolver::HLLD) {
			quokka::Array4View<amrex::Real, DIR> x1FSpds(x1FSpds_ref);
			auto [F_canonical_tmp, fspd_m, fspd_p] = quokka::Riemann::HLLD<problem_t, nscalars_, nmscalars_, nvar_>(sL, sR, gamma_, bx1, dw);
			F_canonical = F_canonical_tmp;
			x1FSpds(i, j, k, 0) = fspd_m;
			x1FSpds(i, j, k, 1) = fspd_p;
		}

		quokka::valarray<double, nvar_> F = F_canonical;

		// add artificial viscosity
		// following Colella & Woodward (1984), eq. (4.2)
		const double div_v = AMREX_D_TERM(du, +0.5 * (dvl + dvr), +0.5 * (dwl + dwr));
		const double viscosity = K_visc * std::max(-div_v, 0.);

		quokka::valarray<double, nvar_> U_L = {sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, sL.E, sL.Eint};
		quokka::valarray<double, nvar_> U_R = {sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, sR.E, sR.Eint};

		// conserve flux of mass scalars
		// based on Plewa and Muller 1999, A&A, 342, 179 (equations 8 and 12)
		amrex::Real fluxSum_U_L = 0;
		amrex::Real fluxSum_U_R = 0;

		for (int n = 0; n < nscalars_; ++n) {
			const int nstart = nvar_ - nscalars_;
			U_L[nstart + n] = sL.scalar[n];
			U_R[nstart + n] = sR.scalar[n];

			if (n < nmscalars_) {
				fluxSum_U_L += U_L[nstart + n];
				fluxSum_U_R += U_R[nstart + n];
			}
		}

		F = F + viscosity * (U_L - U_R);

		// permute momentum components according to flux direction DIR
		F[velN_index] = F_canonical[x1Momentum_index];
		F[velV_index] = F_canonical[x2Momentum_index];
		F[velW_index] = F_canonical[x3Momentum_index];

		// set energy fluxes to zero if EOS is isothermal
		if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
			F[energy_index] = 0;
			F[internalEnergy_index] = 0;
		}

		// compute face-centered normal velocity
		double v_norm = 0.0;
		if (F[density_index] >= 0.) {
			if (rho_R > 0.) {
				v_norm = F[density_index] / rho_R;
			}
		} else {
			if (rho_L > 0.) {
				v_norm = F[density_index] / rho_L;
			}
		}
		x1FaceVel(i, j, k) = v_norm;

		// use the same logic as above to scale and conserve specie fluxes
		if (F[density_index] >= 0.) {
			for (int n = 0; n < nmscalars_; ++n) {
				const int nstart = nvar_ - nscalars_;
				if (fluxSum_U_L > 0.) {
					F[nstart + n] = F[density_index] * U_L[nstart + n] / fluxSum_U_L;
				} else {
					F[nstart + n] = 0.;
				}
			}
		} else {
			for (int n = 0; n < nmscalars_; ++n) {
				const int nstart = nvar_ - nscalars_;
				if (fluxSum_U_R > 0.) {
					F[nstart + n] = F[density_index] * U_R[nstart + n] / fluxSum_U_R;
				} else {
					F[nstart + n] = 0.;
				}
			}
		}

		// copy all flux components to the flux array
		for (int nc = 0; nc < nvar_; ++nc) {
			AMREX_ASSERT(!std::isnan(F[nc])); // check flux is valid
			x1Flux(i, j, k, nc) = F[nc];
		}
	});
}

#endif // HYDRO_SYSTEM_HPP_
