#ifndef HYDRO_SYSTEM_HPP_ // NOLINT
#define HYDRO_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hydro_system.hpp
/// \brief Defines a class for solving the (1d) Euler equations.
///

// c++ headers
#include <cassert>
#include <cmath>
#include <valarray>

// library headers
#include <fmt/format.h>

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
template <typename problem_t>
class HydroSystem : public HyperbolicSystem<problem_t>
{
	// See
	// [https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members]
	using HyperbolicSystem<problem_t>::lx_;
	using HyperbolicSystem<problem_t>::nx_;
	using HyperbolicSystem<problem_t>::dx_;
	using HyperbolicSystem<problem_t>::dt_;
	using HyperbolicSystem<problem_t>::cflNumber_;
	using HyperbolicSystem<problem_t>::dim1_;
	using HyperbolicSystem<problem_t>::nghost_;
	using HyperbolicSystem<problem_t>::nvars_;

	using HyperbolicSystem<problem_t>::x1LeftState_;
	using HyperbolicSystem<problem_t>::x1RightState_;
	using HyperbolicSystem<problem_t>::x1Flux_;
	using HyperbolicSystem<problem_t>::x1FluxDiffusive_;
	using HyperbolicSystem<problem_t>::primVar_;
	using HyperbolicSystem<problem_t>::consVarPredictStep_;

      public:
	using HyperbolicSystem<problem_t>::consVar_;

	enum consVarIndex {
		density_index = 0,
		x1Momentum_index = 1,
		energy_index = 2
	};

	enum primVarIndex {
		primDensity_index = 0,
		x1Velocity_index = 1,
		pressure_index = 2
	};

	struct HydroSystemArgs {
		int nx;
		double lx;
		double cflNumber;
		double gamma;
	};

	explicit HydroSystem(HydroSystemArgs args);

	void AddSourceTerms(array_t &U, std::pair<int, int> range) override;
	void ConservedToPrimitive(array_t &cons,
				  std::pair<int, int> range) override;
	auto ComputeTimestep(double dt_max) -> double override;
	void AdvanceTimestep(double dt_max) override;
	void FillGhostZones(array_t &cons) override;


	// setter functions:

	void set_cflNumber(double cflNumber);
	auto set_density(int i) -> double &;
	auto set_x1Momentum(int i) -> double &;
	auto set_energy(int i) -> double &;

	// accessor functions:

	auto density(int i) -> double;
	auto x1Momentum(int i) -> double;
	auto energy(int i) -> double;

	auto primDensity(int i) -> double;
	auto x1Velocity(int i) -> double;
	auto pressure(int i) -> double;

	auto ComputeMass() -> double;
	auto ComputeEnergy() -> double;

      protected:
	array_t density_;
	array_t x1Momentum_;
	array_t energy_;

	array_t primDensity_;
	array_t x1Velocity_;
	array_t pressure_;

	double gamma_;

	void ComputeFluxes(std::pair<int, int> range) override;
};

template <typename problem_t>
HydroSystem<problem_t>::HydroSystem(HydroSystemArgs args)
    : gamma_(args.gamma), HyperbolicSystem<problem_t>{args.nx, args.lx,
						      args.cflNumber, 3}
{
	assert((gamma_ > 1.0)); // NOLINT

	density_.InitWithShallowSlice(consVar_, 2, density_index, 0);
	x1Momentum_.InitWithShallowSlice(consVar_, 2, x1Momentum_index, 0);
	energy_.InitWithShallowSlice(consVar_, 2, energy_index, 0);

	primDensity_.InitWithShallowSlice(primVar_, 2, primDensity_index, 0);
	x1Velocity_.InitWithShallowSlice(primVar_, 2, x1Velocity_index, 0);
	pressure_.InitWithShallowSlice(primVar_, 2, pressure_index, 0);
}

template <typename problem_t>
auto HydroSystem<problem_t>::density(const int i) -> double
{
	return density_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::set_density(const int i) -> double &
{
	return density_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::x1Momentum(const int i) -> double
{
	return x1Momentum_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::set_x1Momentum(const int i) -> double &
{
	return x1Momentum_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::energy(const int i) -> double
{
	return energy_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::set_energy(const int i) -> double &
{
	return energy_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::primDensity(const int i) -> double
{
	return primDensity_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::x1Velocity(const int i) -> double
{
	return x1Velocity_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::pressure(const int i) -> double
{
	return pressure_(i);
}

template <typename problem_t>
auto HydroSystem<problem_t>::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		mass += density_(i) * dx_;
	}

	return mass;
}

template <typename problem_t>
auto HydroSystem<problem_t>::ComputeEnergy() -> double
{
	double energy = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		energy += energy_(i) * dx_;
	}

	return energy;
}

template <typename problem_t>
void HydroSystem<problem_t>::ConservedToPrimitive(
    array_t &cons, const std::pair<int, int> range)
{
	for (int i = range.first; i < range.second; ++i) {
		const auto rho = cons(density_index, i);
		const auto px = cons(x1Momentum_index, i);
		const auto E =
		    cons(energy_index, i); // *total* gas energy per unit volume

		const auto vx = px / rho;
		const auto kinetic_energy = 0.5 * rho * std::pow(vx, 2);
		const auto thermal_energy = E - kinetic_energy;

		const auto P = thermal_energy * (gamma_ - 1.0);

		assert(rho > 0.); // NOLINT
		assert(P > 0.);	  // NOLINT

		primDensity_(i) = rho;
		x1Velocity_(i) = vx;
		pressure_(i) = P;
	}
}

template <typename problem_t>
auto HydroSystem<problem_t>::ComputeTimestep(const double dt_max) -> double
{
	double dt = dt_max;

	for (int i = 0; i < dim1_; ++i) {
		const double rho = primDensity_(i);
		const double vx = x1Velocity_(i);
		const double P = pressure_(i);
		const double cs = std::sqrt(gamma_ * P / rho);
		assert(cs > 0.); // NOLINT

		const double signal_max =
		    std::max(std::abs(vx - cs), std::abs(vx + cs));
		const double thisDt = cflNumber_ * (dx_ / signal_max);
		dt = std::min(dt, thisDt);
	}

	dt_ = dt;
	return dt;
}

template <typename problem_t>
void HydroSystem<problem_t>::FillGhostZones(array_t &cons)
{
	HyperbolicSystem<problem_t>::FillGhostZones(cons);
}

template <typename problem_t>
void HydroSystem<problem_t>::AdvanceTimestep(double dt_max)
{
	HyperbolicSystem<problem_t>::AdvanceTimestep(dt_max);
}

// TODO(ben): add flux limiter for positivity preservation.
template <typename problem_t>
void HydroSystem<problem_t>::ComputeFluxes(const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = range.first; i < (range.second + 1); ++i) {
		// HLL solver following Toro (1998) and Balsara (2017).

		// gather left- and right- state variables

		const double rho_L = x1LeftState_(primDensity_index, i);
		const double rho_R = x1RightState_(primDensity_index, i);

		const double vx_L = x1LeftState_(x1Velocity_index, i);
		const double vx_R = x1RightState_(x1Velocity_index, i);

		const double P_L = x1LeftState_(pressure_index, i);
		const double P_R = x1RightState_(pressure_index, i);

		const double ke_L = 0.5 * rho_L * (vx_L * vx_L);
		const double ke_R = 0.5 * rho_R * (vx_R * vx_R);

		const double E_L = P_L / (gamma_ - 1.0) + ke_L;
		const double E_R = P_R / (gamma_ - 1.0) + ke_R;

		const double H_L = (E_L + P_L) / rho_L; // enthalpy
		const double H_R = (E_R + P_R) / rho_R;

		const double cs_L = std::sqrt(gamma_ * P_L / rho_L);
		const double cs_R = std::sqrt(gamma_ * P_R / rho_R);

		assert(cs_L > 0.0); // NOLINT
		assert(cs_R > 0.0); // NOLINT

		// compute Roe averages
		const double roe_norm = (std::sqrt(rho_L) + std::sqrt(rho_R));

		const double vx_roe =
		    (std::sqrt(rho_L) * vx_L + std::sqrt(rho_R) * vx_R) / roe_norm; // Roe-average vx

		const double vroe_sq = vx_roe * vx_roe; // modify for 3d!!!

		const double H_roe =
		    (std::sqrt(rho_L) * H_L + std::sqrt(rho_R) * H_R) / roe_norm; // Roe-average H

		const double cs_roe = std::sqrt((gamma_ - 1.0)*(H_roe - 0.5*vroe_sq));

		// compute PVRS states (Toro 10.5.2)

		const double rho_bar = 0.5 * (rho_L + rho_R);
		const double cs_bar = 0.5 * (cs_L + cs_R);
		const double P_PVRS = 0.5 * (P_L + P_R) - 0.5 * (vx_R - vx_L) * (rho_bar * cs_bar);
		const double P_star = std::max(P_PVRS, 0.0);

		const double q_L =
		    (P_star <= P_L)
			? 1.0
			: std::sqrt( 1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
					      ((P_star / P_L) - 1.0) );

		const double q_R =
		    (P_star <= P_R)
			? 1.0
			: std::sqrt( 1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
					      ((P_star / P_R) - 1.0) );

		// compute wave speeds
		
		const double s_L = vx_L - q_L*cs_L;
		const double s_R = vx_R + q_R*cs_R;

		const double s_Lroe = (vx_roe - cs_roe); // Davis, Einfeldt wave speeds
		const double s_Rroe = (vx_roe + cs_roe);

		const double S_L = std::min(s_L, s_Lroe);
		const double S_R = std::max(s_R, s_Rroe);

		assert( S_L <= S_R );

		// compute fluxes

		const std::valarray<double> F_L = {rho_L * vx_L,
						   rho_L * (vx_L * vx_L) + P_L,
						   (E_L + P_L) * vx_L};

		const std::valarray<double> F_R = {rho_R * vx_R,
						   rho_R * (vx_R * vx_R) + P_R,
						   (E_R + P_R) * vx_R};

		const std::valarray<double> U_L = {
		    rho_L, rho_L * vx_L,
		    P_L / (gamma_ - 1.0) + 0.5 * rho_L * (vx_L * vx_L)};

		const std::valarray<double> U_R = {
		    rho_R, rho_R * vx_R,
		    P_R / (gamma_ - 1.0) + 0.5 * rho_R * (vx_R * vx_R)};

		const std::valarray<double> F_star =
		    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R +
		    (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

		std::valarray<double> F(3);

		// open the Riemann fan

		if (S_L > 0.0) {
			F = F_L;
		} else if (S_R < 0.0) {
			F = F_R;
		} else { // S_L <= 0.0 <= S_R
			F = F_star;
		}

		// add artificial viscosity following C&W Eq. (4.2), (4.5)
		const double avisc_coef = 1.0;
		const double div_v = (vx_R - vx_L); // modify for 3d!!!

		// activate artificial viscosity only in converging flows, e.g.
		// shocks
		const double avisc = avisc_coef * std::max(-div_v, 0.0);
		F = F + avisc * (U_L - U_R);

		// check states are valid
		assert(!std::isnan(F[0])); // NOLINT
		assert(!std::isnan(F[1])); // NOLINT
		assert(!std::isnan(F[2])); // NOLINT

		x1Flux_(density_index, i) = F[0];
		x1Flux_(x1Momentum_index, i) = F[1];
		x1Flux_(energy_index, i) = F[2];
	}
}

template <typename problem_t>
void HydroSystem<problem_t>::AddSourceTerms(array_t &U,
					    std::pair<int, int> range)
{
	// TODO(ben): to be implemented
}

#endif // HYDRO_SYSTEM_HPP_
