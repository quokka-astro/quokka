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

// library headers
#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Loop.H"
#include "AMReX_REAL.H"

// internal headers
#include "ArrayView.hpp"
#include "hyperbolic_system.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"

// this struct is specialized by the user application code
//
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.; // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
};

/// Class for the Euler equations of inviscid hydrodynamics
///
template <typename problem_t> class HydroSystem : public HyperbolicSystem<problem_t>
{
      public:
	enum consVarIndex {
		density_index = 0,
		x1Momentum_index = 1,
		x2Momentum_index = 2,
		x3Momentum_index = 3,
		energy_index = 4
	};
	enum primVarIndex {
		primDensity_index = 0,
		x1Velocity_index = 1,
		x2Velocity_index = 2,
		x3Velocity_index = 3,
		pressure_index = 4
	};

	static constexpr int nvar_ = 5;

	static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons,
					 array_t &primVar, amrex::Box const &indexRange);

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons,
					  array_t &maxSignal, amrex::Box const &indexRange);
	// requires GPU reductions
	static auto CheckStatesValid(amrex::Box const &indexRange, amrex::Array4<const amrex::Real> const &cons)
    				  -> bool;
	static void	EnforcePressureFloor(amrex::Real densityFloor, amrex::Real pressureFloor, 
												  amrex::Box const &indexRange,
												  amrex::Array4<amrex::Real> const &state);

	AMREX_GPU_DEVICE static auto ComputePressure(amrex::Array4<const amrex::Real> const &cons,
						     int i, int j, int k) -> amrex::Real;

	template <FluxDir DIR>
	static void ComputeFluxes(array_t &x1Flux,
				  amrex::Array4<const amrex::Real> const &x1LeftState,
				  amrex::Array4<const amrex::Real> const &x1RightState,
				  amrex::Box const &indexRange);

	template <FluxDir DIR>
	static void ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar,
					    array_t &x1FluxDiffusive, amrex::Box const &indexRange);

	template <FluxDir DIR>
	static void ComputeFlatteningCoefficients(amrex::Array4<const amrex::Real> const &primVar,
						  array_t &x1Chi, amrex::Box const &indexRange);

	template <FluxDir DIR>
	static void FlattenShocks(amrex::Array4<const amrex::Real> const &q_in,
				   amrex::Array4<const amrex::Real> const &x1Chi_in,
   				   amrex::Array4<const amrex::Real> const &x2Chi_in,
				   amrex::Array4<const amrex::Real> const &x3Chi_in,
				   array_t &x1LeftState_in, array_t &x1RightState_in,
				   amrex::Box const &indexRange, int nvars);

	// C++ does not allow constexpr to be uninitialized, even in a templated class!
	static constexpr double gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr double cs_iso_ = EOS_Traits<problem_t>::cs_isothermal;

	static constexpr auto is_eos_isothermal() -> bool {
		return (gamma_ == 1.0);
	}
};

template <typename problem_t>
void HydroSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons,
						  array_t &primVar, amrex::Box const &indexRange)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto rho = cons(i, j, k, density_index);
		const auto px = cons(i, j, k, x1Momentum_index);
		const auto py = cons(i, j, k, x2Momentum_index);
		const auto pz = cons(i, j, k, x3Momentum_index);
		const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(px));
		AMREX_ASSERT(!std::isnan(py));
		AMREX_ASSERT(!std::isnan(pz));
		AMREX_ASSERT(!std::isnan(E));

		const auto vx = px / rho;
		const auto vy = py / rho;
		const auto vz = pz / rho;
		const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		const auto thermal_energy = E - kinetic_energy;

		const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);

		AMREX_ASSERT(rho > 0.);
		AMREX_ASSERT(P > 0.);

		primVar(i, j, k, primDensity_index) = rho;
		primVar(i, j, k, x1Velocity_index) = vx;
		primVar(i, j, k, x2Velocity_index) = vy;
		primVar(i, j, k, x3Velocity_index) = vz;
		primVar(i, j, k, pressure_index) = P;
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons,
						   array_t &maxSignal, amrex::Box const &indexRange)
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


template <typename problem_t>
auto HydroSystem<problem_t>::CheckStatesValid(amrex::Box const &indexRange, amrex::Array4<const amrex::Real> const &cons)
    -> bool
{
	bool areValid = true;
	AMREX_LOOP_3D(indexRange, i, j, k, {
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

		bool negativeDensity = (rho <= 0.);
		bool negativePressure = (P <= 0.);

		if (negativeDensity || negativePressure) {
			areValid = false;
		}
	})

	return areValid;
}

template <typename problem_t>
void HydroSystem<problem_t>::EnforcePressureFloor(amrex::Real const densityFloor, amrex::Real const pressureFloor, 
												  amrex::Box const &indexRange,
												  amrex::Array4<amrex::Real> const &state)
{
	// prevent vacuum creation
	amrex::Real const rho_floor = densityFloor; // workaround nvcc bug
	amrex::Real const P_floor = pressureFloor;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const rho = state(i, j, k, density_index);
		amrex::Real const vx1 = state(i, j, k, x1Momentum_index) / rho;
		amrex::Real const vx2 = state(i, j, k, x2Momentum_index) / rho;
		amrex::Real const vx3 = state(i, j, k, x3Momentum_index) / rho;
		amrex::Real const vsq = (vx1*vx1 + vx2*vx2 + vx3*vx3);
		amrex::Real const Etot = state(i, j, k, energy_index);

		amrex::Real rho_new = rho;
		if (rho < rho_floor) {
			rho_new = rho_floor;
			state(i, j, k, density_index) = rho_new;
		}

		if (!is_eos_isothermal()) {
			// recompute gas energy (to prevent P < 0)
			amrex::Real const Eint_star = Etot - 0.5 * rho_new * vsq;
			amrex::Real const P_star = Eint_star * (gamma_ - 1.);
			amrex::Real P_new = P_star;
			if (P_star < P_floor) {
				P_new = P_floor;
				amrex::Real const Etot_new = P_new / (gamma_ - 1.) + 0.5 * rho_new * vsq;
				state(i, j, k, energy_index) = Etot_new;
			}
		}
	});
}

template <typename problem_t>
AMREX_GPU_DEVICE auto
HydroSystem<problem_t>::ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j,
					int k) -> amrex::Real
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
template <FluxDir DIR>
void HydroSystem<problem_t>::ComputeFlatteningCoefficients(
    amrex::Array4<const amrex::Real> const &primVar_in, array_t &x1Chi_in,
    amrex::Box const &indexRange)
{
	quokka::Array4View<const amrex::Real, DIR> primVar(primVar_in);
	quokka::Array4View<amrex::Real, DIR> x1Chi(x1Chi_in);

	// compute the PPM shock flattening coefficient following
	//   Appendix B1 of Mignone+ 2005 [this description has typos].
	// Method originally from Miller & Colella,
	//   Journal of Computational Physics 183, 26–82 (2002) [no typos].

	constexpr double beta_max = 0.85;
	constexpr double beta_min = 0.75;
	constexpr double Zmax = 0.75;
	constexpr double Zmin = 0.25;

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// beta is a measure of shock resolution (Eq. 74 of Miller & Colella 2002)
		const double beta = std::abs(primVar(i + 1, j, k, pressure_index) -
					     primVar(i - 1, j, k, pressure_index)) /
				    std::abs(primVar(i + 2, j, k, pressure_index) -
					     primVar(i - 2, j, k, pressure_index));

		// Eq. 75 of Miller & Colella 2002
		const double chi_min =
		    std::max(0., std::min(1., (beta_max - beta) / (beta_max - beta_min)));

		// Z is a measure of shock strength (Eq. 76 of Miller & Colella 2002)
		const double K_S = gamma_ * primVar(i, j, k, pressure_index); // equal to \rho c_s^2
		const double Z = std::abs(primVar(i + 1, j, k, pressure_index) -
					  primVar(i - 1, j, k, pressure_index)) /
				 K_S;

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
void HydroSystem<problem_t>::FlattenShocks(amrex::Array4<const amrex::Real> const &q_in,
					   amrex::Array4<const amrex::Real> const &x1Chi_in,
   					   amrex::Array4<const amrex::Real> const &x2Chi_in,
					   amrex::Array4<const amrex::Real> const &x3Chi_in,
					   array_t &x1LeftState_in, array_t &x1RightState_in,
					   amrex::Box const &indexRange, const int nvars)
{
	quokka::Array4View<const amrex::Real, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<amrex::Real, DIR> x1RightState(x1RightState_in);

	// Apply shock flattening based on Miller & Colella (2002)
	// [This is necessary to get a reasonable solution to the slow-moving
	// shock problem, and reduces post-shock oscillations in other cases.]

	// cell-centered kernel
	amrex::ParallelFor(
	    indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) {
		    // compute coefficient as the minimum from adjacent cells along *each axis*
		    //  (Eq. 86 of Miller & Colella 2001; Eq. 78 of Miller & Colella 2002)
		    double chi_ijk = std::min({
			    x1Chi_in(i_in - 1, j_in, k_in), x1Chi_in(i_in, j_in, k_in),
				x1Chi_in(i_in + 1, j_in, k_in),
#if (AMREX_SPACEDIM >= 2)
				x2Chi_in(i_in, j_in - 1, k_in), x2Chi_in(i_in, j_in, k_in),
				x2Chi_in(i_in, j_in + 1, k_in),
#endif
#if (AMREX_SPACEDIM == 3)
				x3Chi_in(i_in, j_in, k_in - 1), x3Chi_in(i_in, j_in, k_in),
				x3Chi_in(i_in, j_in, k_in + 1),
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
template <FluxDir DIR>
void HydroSystem<problem_t>::ComputeFluxes(array_t &x1Flux_in,
					   amrex::Array4<const amrex::Real> const &x1LeftState_in,
					   amrex::Array4<const amrex::Real> const &x1RightState_in,
					   amrex::Box const &indexRange)
{
	quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in);
	quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// HLLC solver following Toro (1998) and Balsara (2017).

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

			cs_L = std::sqrt(gamma_ * P_L / rho_L);
			cs_R = std::sqrt(gamma_ * P_R / rho_R);

			E_L = P_L / (gamma_ - 1.0) + ke_L;
			E_R = P_R / (gamma_ - 1.0) + ke_R;
		}

		AMREX_ASSERT(cs_L > 0.0);
		AMREX_ASSERT(cs_R > 0.0);

		// assign normal component of velocity according to DIR

		double u_L = NAN;
		double u_R = NAN;

		if constexpr (DIR == FluxDir::X1) {
			u_L = vx_L;
			u_R = vx_R;
		} else if constexpr (DIR == FluxDir::X2) {
			u_L = vy_L;
			u_R = vy_R;
		} else if constexpr (DIR == FluxDir::X3) {
			u_L = vz_L;
			u_R = vz_R;
		}

		// compute PVRS states (Toro 10.5.2)

		const double rho_bar = 0.5 * (rho_L + rho_R);
		const double cs_bar = 0.5 * (cs_L + cs_R);
		const double P_PVRS = 0.5 * (P_L + P_R) - 0.5 * (u_R - u_L) * (rho_bar * cs_bar);
		const double P_star = std::max(P_PVRS, 0.0);

		const double q_L = (P_star <= P_L)
				       ? 1.0
				       : std::sqrt(1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
							     ((P_star / P_L) - 1.0));

		const double q_R = (P_star <= P_R)
				       ? 1.0
				       : std::sqrt(1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
							     ((P_star / P_R) - 1.0));

		// compute wave speeds

		double S_L = u_L - q_L * cs_L;
		double S_R = u_R + q_R * cs_R;
		const double S_star =
		    ((P_R - P_L) + (rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))) /
		    (rho_L * (S_L - u_L) - rho_R * (S_R - u_R));

		const double P_LR = 0.5 * (P_L + P_R + rho_L * (S_L - u_L) * (S_star - u_L) +
					   rho_R * (S_R - u_R) * (S_star - u_R));

		// compute fluxes
		constexpr int fluxdim = nvar_;

		quokka::valarray<double, fluxdim> D_L{};
		quokka::valarray<double, fluxdim> D_R{};
		quokka::valarray<double, fluxdim> D_star{};

		if constexpr (DIR == FluxDir::X1) {
			D_L = {0., 1., 0., 0., u_L};
			D_R = {0., 1., 0., 0., u_R};
			D_star = {0., 1., 0., 0., S_star};
		} else if constexpr (DIR == FluxDir::X2) {
			D_L = {0., 0., 1., 0., u_L};
			D_R = {0., 0., 1., 0., u_R};
			D_star = {0., 0., 1., 0., S_star};
		} else if constexpr (DIR == FluxDir::X3) {
			D_L = {0., 0., 0., 1., u_L};
			D_R = {0., 0., 0., 1., u_R};
			D_star = {0., 0., 0., 1., S_star};
		}

		const quokka::valarray<double, fluxdim> U_L = {rho_L, rho_L * vx_L, rho_L * vy_L,
							       rho_L * vz_L, E_L};

		const quokka::valarray<double, fluxdim> U_R = {rho_R, rho_R * vx_R, rho_R * vy_R,
							       rho_R * vz_R, E_R};

		quokka::valarray<double, fluxdim> F_L = u_L * U_L + P_L * D_L;
		quokka::valarray<double, fluxdim> F_R = u_R * U_R + P_R * D_R;

		const quokka::valarray<double, fluxdim> F_starL =
		    (S_star * (S_L * U_L - F_L) + S_L * P_LR * D_star) / (S_L - S_star);

		const quokka::valarray<double, fluxdim> F_starR =
		    (S_star * (S_R * U_R - F_R) + S_R * P_LR * D_star) / (S_R - S_star);

		// open the Riemann fan
		quokka::valarray<double, fluxdim> F{};

		// HLLC flux
		if (S_L > 0.0) {
			F = F_L;
		} else if ((S_star > 0.0) && (S_L <= 0.0)) {
			F = F_starL;
		} else if ((S_star <= 0.0) && (S_R >= 0.0)) {
			F = F_starR;
		} else { // S_R < 0.0
			F = F_R;
		}

		// check states are valid
		AMREX_ASSERT(!std::isnan(F[0]));
		AMREX_ASSERT(!std::isnan(F[1]));
		AMREX_ASSERT(!std::isnan(F[2]));
		AMREX_ASSERT(!std::isnan(F[3]));

		x1Flux(i, j, k, density_index) = F[0];
		x1Flux(i, j, k, x1Momentum_index) = F[1];
		x1Flux(i, j, k, x2Momentum_index) = F[2];
		x1Flux(i, j, k, x3Momentum_index) = F[3];
		if constexpr (!is_eos_isothermal()) {
			AMREX_ASSERT(!std::isnan(F[4]));
			x1Flux(i, j, k, energy_index) = F[4];
		} else {
			x1Flux(i, j, k, energy_index) = 0;
		}
	});
}

#endif // HYDRO_SYSTEM_HPP_
