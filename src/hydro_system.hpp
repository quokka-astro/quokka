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
#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_REAL.H"
#include "AMReX_iMultiFab.H"

// internal headers
#include "ArrayView.hpp" // NOLINT(unused-includes)
#include "EOS.hpp"
#include "HLLC.hpp"
#include "HLLD.hpp"
#include "LLF.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"

// Microphysics headers
#include "extern_parameters.H"

enum class RiemannSolver { HLLC, LLF, HLLD };

/// Class for the Euler equations of inviscid hydrodynamics
///
template <typename problem_t> class HydroSystem : public HyperbolicSystem<problem_t>
{
      public:
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(0.5 * std::abs(a + b), std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

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
		primEint_index,	  // auxiliary internal energy (rho * e)
		primScalar0_index // first passive scalar (only present if nscalars > 0!)
	};

	static void ConservedToPrimitive(amrex::MultiFab const &cons_mf, amrex::MultiFab &primVar_mf, int nghost);

	static auto maxSignalSpeedLocal(amrex::MultiFab const &cons) -> amrex::Real;

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange);

	static auto CheckStatesValid(amrex::MultiFab const &cons_mf) -> bool;

	AMREX_GPU_DEVICE static auto ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> quokka::valarray<amrex::Real, nvar_>;

	AMREX_GPU_DEVICE static auto ComputeConsVars(quokka::valarray<amrex::Real, nvar_> const &prim) -> quokka::valarray<amrex::Real, nvar_>;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto PrimToReconstructVars(quokka::valarray<amrex::Real, nvar_> const &q, quokka::valarray<amrex::Real, nvar_> const &q_i)
	    -> quokka::valarray<amrex::Real, nvar_>;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ReconstructToPrimVars(quokka::valarray<amrex::Real, nvar_> &beta, quokka::valarray<amrex::Real, nvar_> const &q_i)
	    -> quokka::valarray<amrex::Real, nvar_>;

	AMREX_FORCE_INLINE AMREX_GPU_DEVICE static auto
	ReconstructCellPPM(quokka::valarray<Real, nvar_> const &r_im2, quokka::valarray<Real, nvar_> const &r_im1, quokka::valarray<Real, nvar_> const &r_i,
			   quokka::valarray<Real, nvar_> const &r_ip1, quokka::valarray<Real, nvar_> const &r_ip2)
	    -> std::pair<quokka::valarray<Real, nvar_>, quokka::valarray<Real, nvar_>>;

	AMREX_GPU_DEVICE static auto ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;

	AMREX_GPU_DEVICE static auto isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool;

	template <FluxDir DIR>
	static void ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, int nghost, int nvars,
					 double dx, double dt);

	static void ComputeRhsFromFluxes(amrex::MultiFab &rhs_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int nvars);

	static void PredictStep(amrex::MultiFab const &consVarOld, amrex::MultiFab &consVarNew, amrex::MultiFab const &rhs, double dt, int nvars,
				amrex::iMultiFab &redoFlag_mf);

	static void AddFluxesRK2(amrex::MultiFab &Unew_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, amrex::MultiFab const &rhs_mf, double dt,
				 int nvars, amrex::iMultiFab &redoFlag_mf);

	AMREX_GPU_DEVICE static auto GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;

	static void EnforceLimits(amrex::Real densityFloor, amrex::Real pressureFloor, amrex::Real speedCeiling, amrex::Real tempCeiling, amrex::Real tempFloor,
				  amrex::MultiFab &state_mf);

	static void AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf);

	static void SyncDualEnergy(amrex::MultiFab &consVar_mf);

	template <RiemannSolver RIEMANN, FluxDir DIR>
	static void ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf,
				  amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &primVar_mf, amrex::Real K_visc);

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

		const amrex::Real Pgas = ComputePressure(cons[bx], i, j, k);
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

		// save pressure
		primVar[bx](i, j, k, pressure_index) = Pgas;
		// save auxiliary internal energy (rho * e)
		primVar[bx](i, j, k, primEint_index) = Eint_aux;

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
	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{}, cons_mf,
				amrex::IntVect(0), // no ghost cells
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
						cs = ComputeSoundSpeed(cons[bx], i, j, k);
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
			cs = ComputeSoundSpeed(cons, i, j, k);
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

	return amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalAnd>{}, amrex::TypeList<bool>{}, cons_mf,
				amrex::IntVect(0), // no ghost cells
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
					const auto P = ComputePressure(cons[bx], i, j, k);

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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> quokka::valarray<amrex::Real, nvar_>
{
	// convert to primitive vars
	const auto rho = cons(i, j, k, density_index);
	const auto px = cons(i, j, k, x1Momentum_index);
	const auto py = cons(i, j, k, x2Momentum_index);
	const auto pz = cons(i, j, k, x3Momentum_index);
	const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume
	const auto Eint_aux = cons(i, j, k, internalEnergy_index);
	const auto vx = px / rho;
	const auto vy = py / rho;
	const auto vz = pz / rho;
	const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
	const auto thermal_energy = E - kinetic_energy;

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
template <FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::PrimToReconstructVars(quokka::valarray<amrex::Real, nvar_> const &q,
										       quokka::valarray<amrex::Real, nvar_> const &q_i)
    -> quokka::valarray<amrex::Real, nvar_>
{
	// Get rho, cs from q_i
	const amrex::Real rho = q_i[primDensity_index];
	const amrex::Real P = q_i[pressure_index];
	const amrex::Real cs = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P);

	const amrex::Real rho_hat = q[primDensity_index];
	const amrex::Real P_hat = q[pressure_index];
	const amrex::Real e_hat = q[primEint_index] / rho_hat;

	// Pick normal velocity component
	amrex::Real u_hat{};
	amrex::Real v_hat{};
	amrex::Real w_hat{};
	if constexpr (DIR == FluxDir::X1) {
		u_hat = q[x1Velocity_index];
		v_hat = q[x2Velocity_index];
		w_hat = q[x3Velocity_index];
	} else if constexpr (DIR == FluxDir::X2) {
		u_hat = q[x2Velocity_index];
		v_hat = q[x3Velocity_index];
		w_hat = q[x1Velocity_index];
	} else if constexpr (DIR == FluxDir::X3) {
		u_hat = q[x3Velocity_index];
		v_hat = q[x1Velocity_index];
		w_hat = q[x2Velocity_index];
	}

	// Project primitive vars onto the right eigenvectors
	// (tau, u, p, e) eigensystem
	const amrex::Real tau = 1. / rho;
	const amrex::Real tau_hat = 1. / rho_hat;
	const amrex::Real x0 = P_hat * tau;
	const amrex::Real x1 = cs * u_hat;
	const amrex::Real x2 = 1. / (cs * cs);
	const amrex::Real x3 = 0.5 * tau * x2;
	const amrex::Real x4 = 1. / P;
	const amrex::Real x5 = P_hat * (tau * tau) * x2;
	quokka::valarray<amrex::Real, nvar_> charVars{};
	charVars[0] = x3 * (-x0 + x1);
	charVars[1] = -tau_hat * x4 - x4 * x5;
	charVars[2] = v_hat;
	charVars[3] = w_hat;
	charVars[4] = -e_hat * x4 + x5;
	charVars[5] = x3 * (-x0 - x1);

	for (int i = 0; i < nscalars_; ++i) {
		charVars[scalar0_index + i] = q[primScalar0_index + i];
	}
	return charVars;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ReconstructToPrimVars(quokka::valarray<amrex::Real, nvar_> &beta,
										       quokka::valarray<amrex::Real, nvar_> const &q_i)
    -> quokka::valarray<amrex::Real, nvar_>
{
	// Get rho, cs from q_i
	const amrex::Real rho = q_i[primDensity_index];
	const amrex::Real P = q_i[pressure_index];
	const amrex::Real cs = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P);

	// Project characteristic vars onto left eigenvectors
	// (tau, u, p, e) eigensystem
	const amrex::Real tau = 1. / rho;
	const amrex::Real x0 = beta[0] + beta[5];
	quokka::valarray<amrex::Real, nvar_> primVars{};
	primVars[0] = -P * beta[1] + x0;
	primVars[1] = cs * (beta[0] - beta[5]) / tau;
	primVars[2] = beta[2];
	primVars[3] = beta[3];
	primVars[4] = -(cs * cs) * x0 / (tau * tau);
	primVars[5] = -P * (beta[4] + x0);

	// convert tau -> rho
	primVars[0] = 1.0 / primVars[0];

	for (int i = 0; i < nscalars_; ++i) {
		primVars[scalar0_index + i] = beta[primScalar0_index + i];
	}

	amrex::Real u{};
	amrex::Real v{};
	amrex::Real w{};
	if constexpr (DIR == FluxDir::X1) {
		u = primVars[x1Velocity_index];
		v = primVars[x2Velocity_index];
		w = primVars[x3Velocity_index];
	} else if constexpr (DIR == FluxDir::X2) {
		u = primVars[x3Velocity_index];
		v = primVars[x1Velocity_index];
		w = primVars[x2Velocity_index];
	} else if constexpr (DIR == FluxDir::X3) {
		u = primVars[x2Velocity_index];
		v = primVars[x3Velocity_index];
		w = primVars[x1Velocity_index];
	}
	primVars[x1Velocity_index] = u;
	primVars[x2Velocity_index] = v;
	primVars[x3Velocity_index] = w;

	// convert e_int -> rho*e_int
	primVars[primEint_index] *= primVars[primDensity_index];

	return primVars;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto
HydroSystem<problem_t>::ReconstructCellPPM(quokka::valarray<Real, nvar_> const &r_im2, quokka::valarray<Real, nvar_> const &r_im1,
					   quokka::valarray<Real, nvar_> const &r_i, quokka::valarray<Real, nvar_> const &r_ip1,
					   quokka::valarray<Real, nvar_> const &r_ip2)
    -> std::pair<quokka::valarray<Real, nvar_>, quokka::valarray<Real, nvar_>>
{
	// reconstructed edge states (in characteristic variables)
	quokka::valarray<Real, nvar_> r_minus{};
	quokka::valarray<Real, nvar_> r_plus{};

	for (int n = 0; n < nvar_; ++n) {
		const double q_im2 = r_im2[n];
		const double q_im1 = r_im1[n];
		const double q_i = r_i[n];
		const double q_ip1 = r_ip1[n];
		const double q_ip2 = r_ip2[n];

		// (1.) Estimate the interface a_{i - 1/2}. Equivalent to step 1
		// in Athena++ [ppm_simple.cpp].

		// C&W Eq. (1.9) [parabola midpoint for the case of
		// equally-spaced zones]: a_{j+1/2} = (7/12)(a_j + a_{j+1}) -
		// (1/12)(a_{j+2} + a_{j-1}). Terms are grouped to preserve exact
		// symmetry in floating-point arithmetic, following Athena++.
		const double coef_1 = (7. / 12.);
		const double coef_2 = (-1. / 12.);
		const double a_minus = (coef_1 * q_i + coef_2 * q_ip1) + (coef_1 * q_im1 + coef_2 * q_im2);
		const double a_plus = (coef_1 * q_ip1 + coef_2 * q_ip2) + (coef_1 * q_i + coef_2 * q_im1);

		// (2.) Constrain interfaces to lie between surrounding cell-averaged
		// values (equivalent to step 2b in Athena++ [ppm_simple.cpp]).
		// [See Eq. B8 of Mignone+ 2005.]

		// compute bounds from neighboring cell-averaged values along axis
		const std::pair<double, double> bounds = std::minmax({q_im1, q_i, q_ip1});

		// left side of zone i
		double new_a_minus = clamp(a_minus, bounds.first, bounds.second);
		// right side of zone i
		double new_a_plus = clamp(a_plus, bounds.first, bounds.second);

		// (3.) Monotonicity correction, using Eq. (1.10) in PPM paper. Equivalent
		// to step 4b in Athena++ [ppm_simple.cpp].

		const double a = q_i; // a_i in C&W
		const double dq_minus = (a - new_a_minus);
		const double dq_plus = (new_a_plus - a);
		const double qa = dq_plus * dq_minus; // interface extrema

		if (qa <= 0.0) { // local extremum
			const double dq0 = MC(q_ip1 - q_i, q_i - q_im1);
			// use linear reconstruction, following Balsara (2017) [Living Rev Comput Astrophys (2017) 3:2]
			new_a_minus = a - 0.5 * dq0;
			new_a_plus = a + 0.5 * dq0;

		} else { // no local extrema
			// parabola overshoots near a_plus -> reset a_minus
			if (std::abs(dq_minus) >= 2.0 * std::abs(dq_plus)) {
				new_a_minus = a - 2.0 * dq_plus;
			}

			// parabola overshoots near a_minus -> reset a_plus
			if (std::abs(dq_plus) >= 2.0 * std::abs(dq_minus)) {
				new_a_plus = a + 2.0 * dq_minus;
			}
		}

		r_minus[n] = new_a_minus;
		r_plus[n] = new_a_plus;
	}

	return std::make_pair(r_minus, r_plus);
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, const int nghost,
						  const int nvars, const double dx, const double dt)
{
	BL_PROFILE("HydroSystem::ReconstructStatesPPM()");

	// PPM reconstruction following Colella & Woodward (1984), with
	// some modifications following Mignone (2014), as implemented in
	// Athena++.

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at the left
	// edge of zone i, and xright_(i) is the "right"-side of the interface
	// at the *left* edge of zone i.

	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(q_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		// permute array indices according to dir
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// primitive variable states
		quokka::valarray<Real, nvar_> q_im2{};
		quokka::valarray<Real, nvar_> q_im1{};
		quokka::valarray<Real, nvar_> q_i{};
		quokka::valarray<Real, nvar_> q_ip1{};
		quokka::valarray<Real, nvar_> q_ip2{};

		for (int n = 0; n < nvar_; ++n) {
			q_im2[n] = q(i - 2, j, k, n);
			q_im1[n] = q(i - 1, j, k, n);
			q_i[n] = q(i, j, k, n);
			q_ip1[n] = q(i + 1, j, k, n);
			q_ip2[n] = q(i + 2, j, k, n);
		}

		auto [q_minus, q_plus] = ReconstructCellPPM(q_im2, q_im1, q_i, q_ip1, q_ip2);

		// Get rho, cs from q_i
		const amrex::Real rho = q_i[primDensity_index];
		const amrex::Real P = q_i[pressure_index];
		const amrex::Real cs = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P);

		// Pick normal velocity component
		amrex::Real u{};
		if constexpr (DIR == FluxDir::X1) {
			u = q_i[x1Velocity_index];
		} else if constexpr (DIR == FluxDir::X2) {
			u = q_i[x2Velocity_index];
		} else if constexpr (DIR == FluxDir::X3) {
			u = q_i[x3Velocity_index];
		}

		// compute CFL numbers for characteristics
		double sigma_plus = std::abs(u + cs) * dt / dx;
		double sigma_minus = std::abs(u - cs) * dt / dx;

		// extrapolate characteristics (these are the reference states from CW84)
		quokka::valarray<Real, nvar_> q_6 = 6.0 * (q_i - 0.5 * (q_minus + q_plus));
		quokka::valarray<Real, nvar_> dq_i = q_plus - q_minus;

		quokka::valarray<Real, nvar_> I_plus = q_plus - 0.5 * sigma_plus * (dq_i - q_6 * (1. - (2. / 3.) * sigma_plus));
		quokka::valarray<Real, nvar_> I_minus = q_minus + 0.5 * sigma_minus * (dq_i + q_6 * (1. - (2. / 3.) * sigma_minus));

		for (int n = 0; n < nvars; ++n) {
			rightState(i, j, k, n) = ((u - cs) < 0.) ? I_minus[n] : q_minus[n];
			leftState(i + 1, j, k, n) = ((u + cs) > 0.) ? I_plus[n] : q_plus[n];
		}
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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
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
	bool isDensityPositive = (rho > 0.);

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
		const double K_S = std::pow(quokka::EOS<problem_t>::ComputeSoundSpeed(primVar(i, j, k, primDensity_index), P, massScalars), 2) *
				   primVar(i, j, k, primDensity_index);
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

	if constexpr (AMREX_SPACEDIM < 2) {
		amrex::ignore_unused(x2Chi_in);
	}
	if constexpr (AMREX_SPACEDIM < 3) {
		amrex::ignore_unused(x3Chi_in);
	}
}

// to ensure that physical quantities are within reasonable
// floors and ceilings which can be set in the param file
template <typename problem_t>
void HydroSystem<problem_t>::EnforceLimits(amrex::Real const densityFloor, amrex::Real const pressureFloor, amrex::Real const speedCeiling,
					   amrex::Real const tempCeiling, amrex::Real const tempFloor, amrex::MultiFab &state_mf)
{

	auto state = state_mf.arrays();

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real const rho = state[bx](i, j, k, density_index);
		amrex::Real const vx1 = state[bx](i, j, k, x1Momentum_index) / rho;
		amrex::Real const vx2 = state[bx](i, j, k, x2Momentum_index) / rho;
		amrex::Real const vx3 = state[bx](i, j, k, x3Momentum_index) / rho;
		amrex::Real const vsq = (vx1 * vx1 + vx2 * vx2 + vx3 * vx3);
		amrex::Real const v_abs = std::sqrt(vx1 * vx1 + vx2 * vx2 + vx3 * vx3);
		amrex::Real Etot = state[bx](i, j, k, energy_index);
		amrex::Real Eint = state[bx](i, j, k, internalEnergy_index);
		amrex::Real Ekin = rho * vsq / 2.;
		amrex::Real rho_new = rho;

		if (rho < densityFloor) {
			rho_new = densityFloor;
			state[bx](i, j, k, density_index) = rho_new;
			state[bx](i, j, k, internalEnergy_index) = Eint * rho_new / rho;
			state[bx](i, j, k, energy_index) = rho_new * vsq / 2. + (Etot - Ekin);
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

		if (v_abs > speedCeiling) {
			amrex::Real rescale_factor = speedCeiling / v_abs;
			state[bx](i, j, k, x1Momentum_index) *= rescale_factor;
			state[bx](i, j, k, x2Momentum_index) *= rescale_factor;
			state[bx](i, j, k, x3Momentum_index) *= rescale_factor;
		}

		// Enforcing Limits on mass scalars
		if (nmscalars_ > 0) {

			amrex::Real sp_sum = 0.0;
			for (int idx = 0; idx < nmscalars_; ++idx) {
				if (state[bx](i, j, k, scalar0_index + idx) < 0.0) {
					state[bx](i, j, k, scalar0_index + idx) = network_rp::small_x * rho;
				}

				// get sum to renormalize
				sp_sum += state[bx](i, j, k, scalar0_index + idx);
			}

			if (sp_sum > 0) {
				sp_sum /= rho; // get mass fractions
				for (int idx = 0; idx < nmscalars_; ++idx) {
					// renormalize
					state[bx](i, j, k, scalar0_index + idx) /= sp_sum;
				}
			}
		}

		// Enforcing Limits on temperature estimated from Etot and Ekin
		if (!HydroSystem<problem_t>::is_eos_isothermal()) {
			// recompute gas energy (to prevent P < 0)
			amrex::Real const Eint_star = Etot - 0.5 * rho_new * vsq;
			amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(state[bx], i, j, k);
			amrex::Real const P_star = quokka::EOS<problem_t>::ComputePressure(rho_new, Eint_star, massScalars);
			amrex::Real P_new = P_star;
			if (P_star < pressureFloor) {
				P_new = pressureFloor;
#pragma nv_diag_suppress divide_by_zero
				amrex::Real const Etot_new = quokka::EOS<problem_t>::ComputeEintFromPres(rho_new, P_new, massScalars) + 0.5 * rho_new * vsq;
				state[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Etot_new;
			}
		}
		// re-obtain Ekin and Etot for putting limits on Temperature
		if (rho_new == 0.0) {
			Ekin = Etot = 0.0;
		} else {
			Ekin = std::pow(state[bx](i, j, k, x1Momentum_index), 2.) / state[bx](i, j, k, density_index) / 2.;
			Ekin += std::pow(state[bx](i, j, k, x2Momentum_index), 2.) / state[bx](i, j, k, density_index) / 2.;
			Ekin += std::pow(state[bx](i, j, k, x3Momentum_index), 2.) / state[bx](i, j, k, density_index) / 2.;
			Etot = state[bx](i, j, k, energy_index);
		}
		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(state[bx], i, j, k);
		amrex::Real primTemp = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, (Etot - Ekin), massScalars);

		if (primTemp > tempCeiling) {
			amrex::Real prim_eint = quokka::EOS<problem_t>::ComputeEintFromTgas(state[bx](i, j, k, density_index), tempCeiling, massScalars);
			state[bx](i, j, k, energy_index) = Ekin + prim_eint;
		}

		if (primTemp < tempFloor) {
			amrex::Real prim_eint = quokka::EOS<problem_t>::ComputeEintFromTgas(state[bx](i, j, k, density_index), tempFloor, massScalars);
			state[bx](i, j, k, energy_index) = Ekin + prim_eint;
		}

		// Enforcing Limits on Auxiliary temperature estimated from Eint
		Eint = state[bx](i, j, k, internalEnergy_index);
		amrex::Real auxTemp = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);

		if (auxTemp > tempCeiling) {
			state[bx](i, j, k, internalEnergy_index) =
			    quokka::EOS<problem_t>::ComputeEintFromTgas(state[bx](i, j, k, density_index), tempCeiling, massScalars);
			state[bx](i, j, k, energy_index) = Ekin + state[bx](i, j, k, internalEnergy_index);
		}

		if (auxTemp < tempFloor) {
			state[bx](i, j, k, internalEnergy_index) =
			    quokka::EOS<problem_t>::ComputeEintFromTgas(state[bx](i, j, k, density_index), tempFloor, massScalars);
			state[bx](i, j, k, energy_index) = Ekin + state[bx](i, j, k, internalEnergy_index);
		}
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf)
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
	auto const &redoFlag = redoFlag_mf.const_arrays();
	auto rhs = rhs_mf.arrays();

	amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		// get cell-centered pressure
		const amrex::Real Pgas = ComputePressure(consVar[bx], i, j, k);

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

template <typename problem_t> void HydroSystem<problem_t>::SyncDualEnergy(amrex::MultiFab &consVar_mf)
{
	// sync internal energy and total energy
	// this step must be done as an operator-split step after *each* RK stage

	const amrex::Real eta = 1.0e-2; // dual energy parameter 'eta'

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
template <RiemannSolver RIEMANN, FluxDir DIR>
void HydroSystem<problem_t>::ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf,
					   amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &primVar_mf, const amrex::Real K_visc)
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
			P_L = x1LeftState(i, j, k, pressure_index);
			P_R = x1RightState(i, j, k, pressure_index);

			Eint_L = x1LeftState(i, j, k, primEint_index);
			Eint_R = x1RightState(i, j, k, primEint_index);

			amrex::GpuArray<Real, nmscalars_> massScalars_L = RadSystem<problem_t>::ComputeMassScalars(x1LeftState, i, j, k);
			cs_L = quokka::EOS<problem_t>::ComputeSoundSpeed(rho_L, P_L, massScalars_L);
			E_L = quokka::EOS<problem_t>::ComputeEintFromPres(rho_L, P_L, massScalars_L) + ke_L;

			amrex::GpuArray<Real, nmscalars_> massScalars_R = RadSystem<problem_t>::ComputeMassScalars(x1RightState, i, j, k);
			cs_R = quokka::EOS<problem_t>::ComputeSoundSpeed(rho_R, P_R, massScalars_R);
			E_R = quokka::EOS<problem_t>::ComputeEintFromPres(rho_R, P_R, massScalars_R) + ke_R;
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

		quokka::HydroState<nscalars_, nmscalars_> sL{};
		sL.rho = rho_L;
		sL.u = x1LeftState(i, j, k, velN_index);
		sL.v = x1LeftState(i, j, k, velV_index);
		sL.w = x1LeftState(i, j, k, velW_index);
		sL.P = P_L;
		sL.cs = cs_L;
		sL.E = E_L;
		sL.Eint = Eint_L;
		// the following has been set to zero to test that the HLLD solver works with hydro only
		// TODO(Neco): set correct magnetic field values once magnetic fields are enabled
		sL.by = 0.0;
		sL.bz = 0.0;

		quokka::HydroState<nscalars_, nmscalars_> sR{};
		sR.rho = rho_R;
		sR.u = x1RightState(i, j, k, velN_index);
		sR.v = x1RightState(i, j, k, velV_index);
		sR.w = x1RightState(i, j, k, velW_index);
		sR.P = P_R;
		sR.cs = cs_R;
		sR.E = E_R;
		sR.Eint = Eint_R;
		// as above, set to zero for testing purposes
		sR.by = 0.0;
		sR.bz = 0.0;

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
		dw = std::min(std::min(dwl, dwr), dw);
#endif

		// solve the Riemann problem in canonical form (i.e., where the x-dir is the normal direction)
		quokka::valarray<double, nvar_> F_canonical{};

		if constexpr (RIEMANN == RiemannSolver::HLLC) {
			static_assert(!Physics_Traits<problem_t>::is_mhd_enabled, "Cannot use HLLC solver for MHD problems!");
			F_canonical = quokka::Riemann::HLLC<problem_t, nscalars_, nmscalars_, nvar_>(sL, sR, gamma_, du, dw);
		} else if constexpr (RIEMANN == RiemannSolver::LLF) {
			F_canonical = quokka::Riemann::LLF<problem_t, nscalars_, nmscalars_, nvar_>(sL, sR);
		} else if constexpr (RIEMANN == RiemannSolver::HLLD) {
			// bx = 0 for testing purposes
			// TODO(Neco): pass correct bx value once magnetic fields are enabled
			F_canonical = quokka::Riemann::HLLD<problem_t, nscalars_, nmscalars_, nvar_>(sL, sR, gamma_, 0.0);
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
		const double v_norm = (F[density_index] >= 0.) ? (F[density_index] / rho_R) : (F[density_index] / rho_L);
		x1FaceVel(i, j, k) = v_norm;

		// use the same logic as above to scale and conserve specie fluxes
		if (F[density_index] >= 0.) {
			for (int n = 0; n < nmscalars_; ++n) {
				const int nstart = nvar_ - nscalars_;
				F[nstart + n] = F[density_index] * U_L[nstart + n] / fluxSum_U_L;
			}
		} else {
			for (int n = 0; n < nmscalars_; ++n) {
				const int nstart = nvar_ - nscalars_;
				F[nstart + n] = F[density_index] * U_R[nstart + n] / fluxSum_U_R;
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
