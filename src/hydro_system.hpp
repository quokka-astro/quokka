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
#include <cmath>

// library headers
#include "AMReX_Arena.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Loop.H"
#include "AMReX_REAL.H"

// internal headers
#include "ArrayView.hpp"
#include "HLLC.hpp"
#include "hyperbolic_system.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"

// this struct is specialized by the user application code
//
template <typename problem_t> struct HydroSystem_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	// if true, reconstruct e_int instead of pressure
	static constexpr bool reconstruct_eint = true;
};

/// Class for the Euler equations of inviscid hydrodynamics
///
template <typename problem_t> class HydroSystem : public HyperbolicSystem<problem_t>
{
      public:
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
		primEint_index,	  // auxiliary internal energy (rho * e)
		primScalar0_index // first passive scalar (only present if nscalars > 0!)
	};

	static void ConservedToPrimitive(amrex::MultiFab const &cons_mf, amrex::MultiFab &primVar_mf, int nghost);

	static auto maxSignalSpeedLocal(amrex::MultiFab const &cons) -> amrex::Real;

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange);

	static auto CheckStatesValid(amrex::MultiFab const &cons_mf) -> bool;

	AMREX_GPU_DEVICE static auto ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

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

	static void AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray);

	static void SyncDualEnergy(amrex::MultiFab &consVar_mf);

	template <FluxDir DIR>
	static void ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf,
				  amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &primVar_mf);

	template <FluxDir DIR>
	static void ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar, array_t &x1FluxDiffusive, amrex::Box const &indexRange);

	template <FluxDir DIR> static void ComputeFlatteningCoefficients(amrex::MultiFab const &primVar_mf, amrex::MultiFab &x1Chi_mf, int nghost);

	template <FluxDir DIR>
	static void FlattenShocks(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf,
				  amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf, int nghost, int nvars);

	// C++ does not allow constexpr to be uninitialized, even in a templated
	// class!
	static constexpr double gamma_ = HydroSystem_Traits<problem_t>::gamma;
	static constexpr double cs_iso_ = HydroSystem_Traits<problem_t>::cs_isothermal;
	static constexpr bool reconstruct_eint = HydroSystem_Traits<problem_t>::reconstruct_eint;

	static constexpr auto is_eos_isothermal() -> bool { return (gamma_ == 1.0); }
};

template <typename problem_t> void HydroSystem<problem_t>::ConservedToPrimitive(amrex::MultiFab const &cons_mf, amrex::MultiFab &primVar_mf, const int nghost)
{
	// convert conserved to primitive variables
	auto const &cons = cons_mf.const_arrays();
	auto const &primVar = primVar_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(cons_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const auto rho = cons[bx](i, j, k, density_index);
		const auto px = cons[bx](i, j, k, x1Momentum_index);
		const auto py = cons[bx](i, j, k, x2Momentum_index);
		const auto pz = cons[bx](i, j, k, x3Momentum_index);
		const auto E = cons[bx](i, j, k, energy_index); // *total* gas energy per unit volume
		const auto Eint_aux = cons[bx](i, j, k, internalEnergy_index);

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(px));
		AMREX_ASSERT(!std::isnan(py));
		AMREX_ASSERT(!std::isnan(pz));
		AMREX_ASSERT(!std::isnan(E));

		const auto vx = px / rho;
		const auto vy = py / rho;
		const auto vz = pz / rho;
		const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		const auto Eint_cons = E - kinetic_energy;

		const amrex::Real Pgas = Eint_cons * (HydroSystem<problem_t>::gamma_ - 1.0);
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
			primVar[bx](i, j, k, primScalar0_index + nc) = cons[bx](i, j, k, scalar0_index + nc);
		}
	});
}

template <typename problem_t> auto HydroSystem<problem_t>::maxSignalSpeedLocal(amrex::MultiFab const &cons_mf) -> amrex::Real
{
	// return maximum signal speed on local grids

	auto const &cons = cons_mf.const_arrays();
	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{}, cons_mf, amrex::IntVect(0), // no ghost cells
				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
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
						const auto Etot = cons[bx](i, j, k, HydroSystem<problem_t>::energy_index);
						const auto Eint = Etot - kinetic_energy;
						const auto P = Eint * (HydroSystem<problem_t>::gamma_ - 1.0);
						cs = std::sqrt(HydroSystem<problem_t>::gamma_ * P / rho);
					}
					return {cs + abs_vel};
				});
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto rho = cons(i, j, k, density_index);
		const auto px = cons(i, j, k, x1Momentum_index);
		const auto py = cons(i, j, k, x2Momentum_index);
		const auto pz = cons(i, j, k, x3Momentum_index);
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(px));
		AMREX_ASSERT(!std::isnan(py));
		AMREX_ASSERT(!std::isnan(pz));

		const auto vx = px / rho;
		const auto vy = py / rho;
		const auto vz = pz / rho;
		const double vel_mag = std::sqrt(vx * vx + vy * vy + vz * vz);
		double cs = NAN;

		if constexpr (is_eos_isothermal()) {
			cs = cs_iso_;
		} else {
			const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
			AMREX_ASSERT(!std::isnan(E));
			const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
			const auto thermal_energy = E - kinetic_energy;
			const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);
			cs = std::sqrt(HydroSystem<problem_t>::gamma_ * P / rho);
		}
		AMREX_ASSERT(cs > 0.);

		const double signal_max = cs + vel_mag;
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t> auto HydroSystem<problem_t>::CheckStatesValid(amrex::MultiFab const &cons_mf) -> bool
{
	// check whether density or pressure are negative
	auto const &cons = cons_mf.const_arrays();

	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalAnd>{}, amrex::TypeList<bool>{}, cons_mf, amrex::IntVect(0), // no ghost cells
				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<bool> {
					const auto rho = cons[bx](i, j, k, density_index);
					const auto px = cons[bx](i, j, k, x1Momentum_index);
					const auto py = cons[bx](i, j, k, x2Momentum_index);
					const auto pz = cons[bx](i, j, k, x3Momentum_index);
					const auto E = cons[bx](i, j, k, energy_index);
					const auto vx = px / rho;
					const auto vy = py / rho;
					const auto vz = pz / rho;
					const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
					const auto thermal_energy = E - kinetic_energy;
					const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);

					bool negativeDensity = (rho <= 0.);
					bool negativePressure = (P <= 0.);

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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real
{
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy;
	const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);
	return P;
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
	bool isDensityPositive = (rho > 0.);

	// when the dual energy method is used, we *cannot* reset on pressure
	// failures. on the other hand, we don't need to -- the auxiliary internal
	// energy is used instead!
#if 0
  bool isPressurePositive = false;
  if constexpr (!is_eos_isothermal()) {
    const amrex::Real P = ComputePressure(cons, i, j, k);
    isPressurePositive = (P > 0.);
  } else {
    isPressurePositive = true;
  }
#endif
	// return (isDensityPositive && isPressurePositive);

	return isDensityPositive;
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
#if AMREX_SPACEDIM >= 2
	auto const x2Flux = fluxArray[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
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
	BL_PROFILE("HydroSystem::PredictStep()");

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
	BL_PROFILE("HyperbolicSystem::AddFluxesRK2()");

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
			Pplus2 *= primVar(i + 2, j, k, primDensity_index) * (gamma_ - 1.0);
			Pplus1 *= primVar(i + 1, j, k, primDensity_index) * (gamma_ - 1.0);
			P *= primVar(i, j, k, primDensity_index) * (gamma_ - 1.0);
			Pminus1 *= primVar(i - 1, j, k, primDensity_index) * (gamma_ - 1.0);
			Pminus2 *= primVar(i - 2, j, k, primDensity_index) * (gamma_ - 1.0);
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
		const double K_S = gamma_ * P; // equal to \rho c_s^2
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
		double chi_ijk = std::min({
			x1Chi_in[bx](i_in - 1, j_in, k_in), x1Chi_in[bx](i_in, j_in, k_in), x1Chi_in[bx](i_in + 1, j_in, k_in),
#if (AMREX_SPACEDIM >= 2)
			    x2Chi_in[bx](i_in, j_in - 1, k_in), x2Chi_in[bx](i_in, j_in, k_in), x2Chi_in[bx](i_in, j_in + 1, k_in),
#endif
#if (AMREX_SPACEDIM == 3)
			    x3Chi_in[bx](i_in, j_in, k_in - 1), x3Chi_in[bx](i_in, j_in, k_in), x3Chi_in[bx](i_in, j_in, k_in + 1),
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
}

template <typename problem_t>
void HydroSystem<problem_t>::AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray)
{
	// compute P dV source term for the internal energy equation,
	// using the face-centered velocities in faceVelArray and the pressure

	auto vel_x = faceVelArray[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto vel_y = faceVelArray[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto vel_z = faceVelArray[2].const_arrays();
#endif

	auto const &consVar = consVar_mf.const_arrays();
	auto rhs = rhs_mf.arrays();

	amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		// get cell-centered pressure
		const amrex::Real Pgas = ComputePressure(consVar[bx], i, j, k);

		// compute div v from face-centered velocities
		amrex::Real div_v = AMREX_D_TERM((vel_x[bx](i + 1, j, k) - vel_x[bx](i, j, k)) / dx[0], +(vel_y[bx](i, j + 1, k) - vel_y[bx](i, j, k)) / dx[1],
						 +(vel_z[bx](i, j, k + 1) - vel_z[bx](i, j, k)) / dx[2]);

#if 0                
    if (redoFlag(i,j,k) == quokka::redoFlag::none) {
      div_v = AMREX_D_TERM(  ( vel_x(i+1, j  , k  ) - vel_x(i, j, k) ) / dx[0],
                           + ( vel_y(i  , j+1, k  ) - vel_y(i, j, k) ) / dx[1],
                           + ( vel_z(i  , j  , k+1) - vel_z(i, j, k) ) / dx[2]  );
    } else {
      div_v = 0.5 * ( AMREX_D_TERM(
                ( ComputeVelocityX1(consVar, i+1, j, k) - ComputeVelocityX1(consVar, i-1, j, k) ) / dx[0],
              + ( ComputeVelocityX2(consVar, i, j+1, k) - ComputeVelocityX2(consVar, i, j-1, k) ) / dx[1],
              + ( ComputeVelocityX3(consVar, i, j, k+1) - ComputeVelocityX3(consVar, i, j, k-1) ) / dx[2]
              ) );
    }
#endif

		// add P dV term to rhs array
		rhs[bx](i, j, k, internalEnergy_index) += -Pgas * div_v;
	});
}

template <typename problem_t> void HydroSystem<problem_t>::SyncDualEnergy(amrex::MultiFab &consVar_mf)
{
	// sync internal energy and total energy
	// this step must be done as an operator-split step after *each* RK stage

	const amrex::Real eta = 1.0e-3; // dual energy parameter 'eta'

	auto consVar = consVar_mf.arrays();

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

		amrex::Real const Ekin = (px * px + py * py + pz * pz) / (2.0 * rho);
		amrex::Real const Eint_cons = Etot - Ekin;

		// Li et al. sync method
		// replace Eint with Eint_cons == (Etot - Ekin) if (Eint_cons / E) > eta
		if (Eint_cons > eta * Etot) {
			consVar[bx](i, j, k, internalEnergy_index) = Eint_cons;
		} else { // non-conservative sync
			consVar[bx](i, j, k, internalEnergy_index) = Eint_aux;
			consVar[bx](i, j, k, energy_index) = Eint_aux + Ekin;
		}
	});
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf,
					   amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &primVar_mf)
{

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	auto const &x1LeftState_in = x1LeftState_mf.const_arrays();
	auto const &x1RightState_in = x1RightState_mf.const_arrays();
	auto const &primVar_in = primVar_mf.const_arrays();
	auto x1Flux_in = x1Flux_mf.arrays();
	auto x1FaceVel_in = x1FaceVel_mf.arrays();

	amrex::ParallelFor(x1Flux_mf, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
		quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in[bx]);
		quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1FaceVel(x1FaceVel_in[bx]);
		quokka::Array4View<const amrex::Real, DIR> q(primVar_in[bx]);

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
				P_L = rho_L * eint_L * (gamma_ - 1.0);
				P_R = rho_R * eint_R * (gamma_ - 1.0);

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

			cs_L = std::sqrt(gamma_ * P_L / rho_L);
			cs_R = std::sqrt(gamma_ * P_R / rho_R);

			E_L = P_L / (gamma_ - 1.0) + ke_L;
			E_R = P_R / (gamma_ - 1.0) + ke_R;
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
			if constexpr (AMREX_SPACEDIM == 2) {
				velN_index = x2Velocity_index;
				velV_index = x1Velocity_index;
				velW_index = x3Velocity_index; // unchanged in 2D
			} else if constexpr (AMREX_SPACEDIM == 3) {
				velN_index = x2Velocity_index;
				velV_index = x3Velocity_index;
				velW_index = x1Velocity_index;
			}
		} else if constexpr (DIR == FluxDir::X3) {
			velN_index = x3Velocity_index;
			velV_index = x1Velocity_index;
			velW_index = x2Velocity_index;
		}

		quokka::HydroState<nscalars_> sL{};
		sL.rho = rho_L;
		sL.u = x1LeftState(i, j, k, velN_index);
		sL.v = x1LeftState(i, j, k, velV_index);
		sL.w = x1LeftState(i, j, k, velW_index);
		sL.P = P_L;
		sL.cs = cs_L;
		sL.E = E_L;
		sL.Eint = Eint_L;

		quokka::HydroState<nscalars_> sR{};
		sR.rho = rho_R;
		sR.u = x1RightState(i, j, k, velN_index);
		sR.v = x1RightState(i, j, k, velV_index);
		sR.w = x1RightState(i, j, k, velW_index);
		sR.P = P_R;
		sR.cs = cs_R;
		sR.E = E_R;
		sR.Eint = Eint_R;

		// The remaining components are passive scalars, so just copy them from
		// x1LeftState and x1RightState into the (left, right) state vectors U_L and
		// U_R
		for (int n = 0; n < nscalars_; ++n) {
			sL.scalar[n] = x1LeftState(i, j, k, scalar0_index + n);
			sR.scalar[n] = x1RightState(i, j, k, scalar0_index + n);
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
		dw = std::min(std::min(dwl, dwr), dw);
#endif

		// solve the Riemann problem in canonical form
		quokka::valarray<double, nvar_> F_canonical = quokka::Riemann::HLLC<nscalars_, nvar_>(sL, sR, gamma_, du, dw);
		quokka::valarray<double, nvar_> F = F_canonical;

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
		const double v_norm = (F[density_index] >= 0.) ? (F[density_index] / rho_R) : (F[density_index] / rho_L);
		x1FaceVel(i, j, k) = v_norm;

		// copy all flux components to the flux array
		for (int nc = 0; nc < nvar_; ++nc) {
			AMREX_ASSERT(!std::isnan(F[nc])); // check flux is valid
			x1Flux(i, j, k, nc) = F[nc];
		}
	});
}

#endif // HYDRO_SYSTEM_HPP_
