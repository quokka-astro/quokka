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
#include <cassert>
#include <cmath>
#include <limits>
#include <valarray>

// library headers
#include <fmt/format.h>

// internal headers
#include "AMReX_BLassert.H"
#include "hyperbolic_system.hpp"
#include "valarray.hpp"

// this struct is specialized by the user application code
//
template <typename problem_t> struct EOS_Traits
{
	static constexpr double gamma = 5. / 3.; // default value
};

/// Class for the Euler equations of inviscid hydrodynamics
///
template <typename problem_t>
class HydroSystem : public HyperbolicSystem<problem_t>
{
      public:
	enum consVarIndex { density_index = 0, x1Momentum_index = 1, energy_index = 2 };
	enum primVarIndex { primDensity_index = 0, x1Velocity_index = 1, pressure_index = 2 };

	static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons,
					 array_t &primVar, amrex::Box const &indexRange);

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons,
					  array_t &maxSignal, amrex::Box const &indexRange);
	// requires GPU reductions
	// static auto CheckStatesValid(array_t &cons, const std::pair<int, int> range) -> bool;

	static void ComputeFluxes(array_t &x1Flux,
				  amrex::Array4<const amrex::Real> const &x1LeftState,
				  amrex::Array4<const amrex::Real> const &x1RightState,
				  amrex::Box const &indexRange);
	static void ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar,
					    array_t &x1FluxDiffusive, amrex::Box const &indexRange);

	static void ComputeFlatteningCoefficients(amrex::Array4<const amrex::Real> const &primVar,
						  array_t &x1Chi, amrex::Box const &indexRange);
	static void FlattenShocks(amrex::Array4<const amrex::Real> const &q,
				  amrex::Array4<const amrex::Real> const &x1Chi,
				  array_t &x1LeftState, array_t &x1RightState,
				  amrex::Box const &indexRange, int nvars);

	static constexpr double gamma_ = EOS_Traits<problem_t>::gamma;
	// static constexpr double gamma_; // C++ standard does not allow constexpr to be
	// uninitialized, even in a templated class!
};

template <typename problem_t>
void HydroSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons,
						  array_t &primVar, amrex::Box const &indexRange)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto rho = cons(i, j, k, density_index);
		const auto px = cons(i, j, k, x1Momentum_index);
		const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(px));
		AMREX_ASSERT(!std::isnan(E));

		const auto vx = px / rho;
		const auto kinetic_energy = 0.5 * rho * std::pow(vx, 2);
		const auto thermal_energy = E - kinetic_energy;

		const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);

		assert(rho > 0.); // NOLINT
		assert(P > 0.);	  // NOLINT

		primVar(i, j, k, primDensity_index) = rho;
		primVar(i, j, k, x1Velocity_index) = vx;
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
		const auto E = cons(i, j, k, energy_index); // *total* gas energy per unit volume

		const auto vx = px / rho;
		const auto kinetic_energy = 0.5 * rho * std::pow(vx, 2);
		const auto thermal_energy = E - kinetic_energy;

		const auto P = thermal_energy * (gamma_ - 1.0);
		const double cs = std::sqrt(HydroSystem<problem_t>::gamma_ * P / rho);
		assert(cs > 0.); // NOLINT

		const double signal_max = std::max(std::abs(vx - cs), std::abs(vx + cs));
		maxSignal(i, j, k) = signal_max;
	});
}

#if 0
template <typename problem_t>
auto HydroSystem<problem_t>::CheckStatesValid(amrex::Array4<const amrex::Real> const &cons, amrex::Box const &indexRange)
    -> bool
{
	bool areValid = true;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto rho = cons(density_index, i);
		const auto px = cons(x1Momentum_index, i);
		const auto E = cons(energy_index, i); // *total* gas energy per unit volume

		const auto vx = px / rho;
		const auto kinetic_energy = 0.5 * rho * std::pow(vx, 2);
		const auto thermal_energy = E - kinetic_energy;

		const auto P = thermal_energy * (gamma_ - 1.0);

		if (rho <= 0.) {
			amrex::Print() << "Bad cell i = " << i << " (negative density = " << rho
				       << ")." << std::endl;
			areValid = false;
		}
		if (P <= 0.) {
			amrex::Print() << "Bad cell i = " << i << " (negative pressure = " << P
				       << ")." << std::endl;
			areValid = false;
		}
	});
	return areValid;
}
#endif

template <typename problem_t>
void HydroSystem<problem_t>::ComputeFlatteningCoefficients(
    amrex::Array4<const amrex::Real> const &primVar, array_t &x1Chi, amrex::Box const &indexRange)
{
	// compute the PPM shock flattening coefficient following
	//   Appendix B1 of Mignone+ 2005 [this description has typos].
	// Method originally from Miller & Colella,
	//   Journal of Computational Physics 183, 26–82 (2002) [no typos].

	constexpr double beta_max = 0.85;
	constexpr double beta_min = 0.75;
	constexpr double Zmax = 0.75;
	constexpr double Zmin = 0.25;

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
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

		// check for converging flow (Eq. 77)
		double chi = 1.0;
		if (primVar(i + 1, j, k, x1Velocity_index) <
		    primVar(i - 1, j, k, x1Velocity_index)) {
			chi = std::max(chi_min, std::min(1., (Zmax - Z) / (Zmax - Zmin)));
		}

		x1Chi(i, j, k) = chi;
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::FlattenShocks(amrex::Array4<const amrex::Real> const &q,
					   amrex::Array4<const amrex::Real> const &x1Chi,
					   array_t &x1LeftState, array_t &x1RightState,
					   amrex::Box const &indexRange, const int nvars)
{
	// Apply shock flattening based on Miller & Colella (2002)
	// [This is necessary to get a reasonable solution to the slow-moving
	// shock problem, and reduces post-shock oscillations in other cases.]

	// cell-centered kernel
	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		// compute coefficient as the minimum from all surrounding cells
		//  (Eq. 78 of Miller & Colella 2002)
		double chi = std::min(
		    {x1Chi(i - 1, j, k), x1Chi(i, j, k), x1Chi(i + 1, j, k)}); // modify in 3d !!

		// get interfaces
		const double a_minus = x1RightState(i, j, k, n);
		const double a_plus = x1LeftState(i + 1, j, k, n);
		const double a_mean = q(i, j, k, n);

		// left side of zone i (Eq. 70a)
		const double new_a_minus = chi * a_minus + (1. - chi) * a_mean;

		// right side of zone i (Eq. 70b)
		const double new_a_plus = chi * a_plus + (1. - chi) * a_mean;

		x1RightState(i, j, k, n) = new_a_minus;
		x1LeftState(i + 1, j, k, n) = new_a_plus;
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeFirstOrderFluxes(
    amrex::Array4<const amrex::Real> const &consVar, array_t &x1FluxDiffusive,
    amrex::Box const &indexRange)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. x1Flux_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// compute Lax-Friedrichs fluxes for use in flux-limiting to ensure realizable states

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// gather L/R states
		const double rho_L = consVar(i - 1, j, k, density_index);
		const double rho_R = consVar(i, j, k, density_index);

		const double mom_L = consVar(i - 1, j, k, x1Momentum_index);
		const double mom_R = consVar(i, j, k, x1Momentum_index);

		const double E_L = consVar(i - 1, j, k, energy_index);
		const double E_R = consVar(i, j, k, energy_index);

		// compute primitive variables
		const double vx_L = mom_L / rho_L;
		const double vx_R = mom_R / rho_R;

		const double ke_L = 0.5 * rho_L * (vx_L * vx_L);
		const double ke_R = 0.5 * rho_R * (vx_R * vx_R);

		const double P_L = (E_L - ke_L) * (gamma_ - 1.0);
		const double P_R = (E_R - ke_R) * (gamma_ - 1.0);

		const double cs_L = std::sqrt(gamma_ * P_L / rho_L);
		const double cs_R = std::sqrt(gamma_ * P_R / rho_R);

		const double s_L = std::abs(vx_L) + cs_L;
		const double s_R = std::abs(vx_R) + cs_R;
		const double sstar = std::max(s_L, s_R);

		// compute (using local signal speed) Lax-Friedrichs flux
		constexpr int dim = 3;
		
		const quokka::valarray<double, dim> F_L = {rho_L * vx_L, rho_L * (vx_L * vx_L) + P_L,
						   (E_L + P_L) * vx_L};

		const quokka::valarray<double, dim> F_R = {rho_R * vx_R, rho_R * (vx_R * vx_R) + P_R,
						   (E_R + P_R) * vx_R};

		const quokka::valarray<double, dim> U_L = {rho_L, mom_L, E_L};
		const quokka::valarray<double, dim> U_R = {rho_R, mom_R, E_R};

		const quokka::valarray<double, dim> LLF = 0.5 * (F_L + F_R - sstar * (U_R - U_L));

		x1FluxDiffusive(i, j, k, density_index) = LLF[0];
		x1FluxDiffusive(i, j, k, x1Momentum_index) = LLF[1];
		x1FluxDiffusive(i, j, k, energy_index) = LLF[2];
	});
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeFluxes(array_t &x1Flux,
					   amrex::Array4<const amrex::Real> const &x1LeftState,
					   amrex::Array4<const amrex::Real> const &x1RightState,
					   amrex::Box const &indexRange)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// HLLC solver following Toro (1998) and Balsara (2017).

		// gather left- and right- state variables

		const double rho_L = x1LeftState(i, j, k, primDensity_index);
		const double rho_R = x1RightState(i, j, k, primDensity_index);

		const double vx_L = x1LeftState(i, j, k, x1Velocity_index);
		const double vx_R = x1RightState(i, j, k, x1Velocity_index);

		const double P_L = x1LeftState(i, j, k, pressure_index);
		const double P_R = x1RightState(i, j, k, pressure_index);

		const double ke_L = 0.5 * rho_L * (vx_L * vx_L);
		const double ke_R = 0.5 * rho_R * (vx_R * vx_R);

		const double E_L = P_L / (gamma_ - 1.0) + ke_L;
		const double E_R = P_R / (gamma_ - 1.0) + ke_R;

		// const double H_L = (E_L + P_L) / rho_L; // enthalpy
		// const double H_R = (E_R + P_R) / rho_R;

		const double cs_L = std::sqrt(gamma_ * P_L / rho_L);
		const double cs_R = std::sqrt(gamma_ * P_R / rho_R);

		assert(cs_L > 0.0); // NOLINT
		assert(cs_R > 0.0); // NOLINT

		// compute PVRS states (Toro 10.5.2)

		const double rho_bar = 0.5 * (rho_L + rho_R);
		const double cs_bar = 0.5 * (cs_L + cs_R);
		const double P_PVRS = 0.5 * (P_L + P_R) - 0.5 * (vx_R - vx_L) * (rho_bar * cs_bar);
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

		const double S_L = vx_L - q_L * cs_L;
		const double S_R = vx_R + q_R * cs_R;
		const double S_star =
		    ((P_R - P_L) + (rho_L * vx_L * (S_L - vx_L) - rho_R * vx_R * (S_R - vx_R))) /
		    (rho_L * (S_L - vx_L) - rho_R * (S_R - vx_R));

		const double P_LR = 0.5 * (P_L + P_R + rho_L * (S_L - vx_L) * (S_star - vx_L) +
					   rho_R * (S_R - vx_R) * (S_star - vx_R));

		assert(S_L <= S_R);

		// compute fluxes
		constexpr int fluxdim = 3;
		const quokka::valarray<double, fluxdim> F_L = {rho_L * vx_L, rho_L * (vx_L * vx_L) + P_L,
						   (E_L + P_L) * vx_L};

		const quokka::valarray<double, fluxdim> F_R = {rho_R * vx_R, rho_R * (vx_R * vx_R) + P_R,
						   (E_R + P_R) * vx_R};

		const quokka::valarray<double, fluxdim> U_L = {
		    rho_L, rho_L * vx_L, P_L / (gamma_ - 1.0) + 0.5 * rho_L * (vx_L * vx_L)};

		const quokka::valarray<double, fluxdim> U_R = {
		    rho_R, rho_R * vx_R, P_R / (gamma_ - 1.0) + 0.5 * rho_R * (vx_R * vx_R)};

		const quokka::valarray<double, fluxdim> D_star = {0., 1., S_star};

		const quokka::valarray<double, fluxdim> F_starL =
		    (S_star * (S_L * U_L - F_L) + S_L * P_LR * D_star) / (S_L - S_star);

		const quokka::valarray<double, fluxdim> F_starR =
		    (S_star * (S_R * U_R - F_R) + S_R * P_LR * D_star) / (S_R - S_star);

		// open the Riemann fan
		quokka::valarray<double, fluxdim> F;

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
		assert(!std::isnan(F[0])); // NOLINT
		assert(!std::isnan(F[1])); // NOLINT
		assert(!std::isnan(F[2])); // NOLINT

		x1Flux(i, j, k, density_index) = F[0];
		x1Flux(i, j, k, x1Momentum_index) = F[1];
		x1Flux(i, j, k, energy_index) = F[2];
	});
}

#endif // HYDRO_SYSTEM_HPP_
