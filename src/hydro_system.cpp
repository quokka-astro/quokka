//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hydro_system.cpp
/// \brief Implements methods for solving the (1d) Euler equations.
///

#include "hydro_system.hpp"

// We must *define* static member variables here, outside of the class
// *declaration*, even though the definitions are trivial.
const HydroSystem::NxType::argument HydroSystem::Nx;
const HydroSystem::LxType::argument HydroSystem::Lx;
const HydroSystem::CFLType::argument HydroSystem::CFL;
const HydroSystem::GammaType::argument HydroSystem::Gamma;

HydroSystem::HydroSystem(NxType const &nx, LxType const &lx,
			 CFLType const &cflNumber, GammaType const &gamma)
    : HyperbolicSystem{nx.get(), lx.get(), cflNumber.get(), 3},
      gamma_(gamma.get())
{
	assert((gamma_ > 1.0)); // NOLINT

	density_.InitWithShallowSlice(consVar_, 2, density_index, 0);
	x1Momentum_.InitWithShallowSlice(consVar_, 2, x1Momentum_index, 0);
	energy_.InitWithShallowSlice(consVar_, 2, energy_index, 0);

	primDensity_.InitWithShallowSlice(primVar_, 2, primDensity_index, 0);
	x1Velocity_.InitWithShallowSlice(primVar_, 2, x1Velocity_index, 0);
	pressure_.InitWithShallowSlice(primVar_, 2, pressure_index, 0);
}

auto HydroSystem::density(const int i) -> double { return density_(i); }

auto HydroSystem::set_density(const int i) -> double & { return density_(i); }

auto HydroSystem::x1Momentum(const int i) -> double { return x1Momentum_(i); }

auto HydroSystem::set_x1Momentum(const int i) -> double &
{
	return x1Momentum_(i);
}

auto HydroSystem::energy(const int i) -> double { return energy_(i); }

auto HydroSystem::set_energy(const int i) -> double & { return energy_(i); }

auto HydroSystem::primDensity(const int i) -> double { return primDensity_(i); }

auto HydroSystem::x1Velocity(const int i) -> double { return x1Velocity_(i); }

auto HydroSystem::pressure(const int i) -> double { return pressure_(i); }

auto HydroSystem::ComputeMass() -> double
{
	double mass = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		mass += density_(i) * dx_;
	}

	return mass;
}

void HydroSystem::ConservedToPrimitive(AthenaArray<double> &cons,
				       const std::pair<int, int> range)
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

void HydroSystem::ComputeTimestep()
{
	double dt = std::numeric_limits<double>::max();

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
}

// TODO(ben): add flux limiter for positivity preservation.
void HydroSystem::ComputeFluxes(const std::pair<int, int> range)
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
		const double E_R = P_L / (gamma_ - 1.0) + ke_R;

		const double H_L = (E_L + P_L) / rho_L; // enthalpy
		const double H_R = (E_R + P_R) / rho_R;

		const double cs_L = std::sqrt(gamma_ * P_L / rho_L);
		const double cs_R = std::sqrt(gamma_ * P_R / rho_R);

		assert(cs_L > 0.0); // NOLINT
		assert(cs_R > 0.0); // NOLINT

		// compute Roe averages

		const double vx_roe =
		    (std::sqrt(rho_L) * vx_L + std::sqrt(rho_R) * vx_L) /
		    (std::sqrt(rho_L) + std::sqrt(rho_R)); // Roe-average vx

		const double vroe_sq = vx_roe * vx_roe; // modify for 3d!!!

		const double H_roe =
		    (std::sqrt(rho_L) * H_L + std::sqrt(rho_R) * H_R) /
		    (std::sqrt(rho_L) + std::sqrt(rho_R)); // Roe-average H

		const double cs_roe =
		    std::sqrt((gamma_ - 1.) * (H_roe - 0.5 * vroe_sq));

		// compute PVRS states (Toro 10.5.2)

		const double rho_bar = 0.5 * (rho_L + rho_R);
		const double cs_bar = 0.5 * (cs_L + cs_R);
		const double P_PVRS =
		    0.5 * (P_L + P_R) + 0.5 * (vx_L - vx_R) * rho_bar * cs_bar;
		const double P_star = std::max(P_PVRS, 0.0);

		const double q_L =
		    (P_star <= P_L)
			? 1.0
			: std::sqrt(1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
					      ((P_star / P_L) - 1.0));

		const double q_R =
		    (P_star <= P_R)
			? 1.0
			: std::sqrt(1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
					      ((P_star / P_R) - 1.0));

		// compute wave speeds

		const double s_L = vx_L - cs_L * q_L;
		const double s_R = vx_R + cs_R * q_R;

		const double s_Lroe =
		    (vx_roe - cs_roe); // Roe-average vx - Roe-average cs
		const double s_Rroe =
		    (vx_roe + cs_roe); // Roe-average vx + Roe-average cs

		const double S_L = std::min(s_L, s_Lroe);
		const double S_R = std::max(s_R, s_Rroe);

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

		const double avisc_coef = 0.1;
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

#if 0
void HydroSystem::FlattenShocks(AthenaArray<double> &q,
				const std::pair<int, int> range)
{
	// N.B.: shock flattening doesn't appear to really do much. Perhaps a
	// bug?

	for (int i = range.first; i < range.second; ++i) {

		const double a_minus = densityXRight_(i);   // a_L,i in C&W
		const double a_plus = densityXLeft_(i + 1); // a_R,i in C&W
		const double a = q(i);			    // a_i in C&W

		auto flatten_f = [q](int j) {
			const double a1 = 0.75;
			const double a2 = 10.;
			const double eps = 0.33;
			double f = 0.;

			const double shock_ratio =
			    (q(j + 1) - q(j - 1)) / (q(j + 2) - q(j - 2));
			const double qa = (q(j + 1) - q(j - 1)) /
					  std::min(q(j + 1), q(j - 1));
			if ((qa > eps) && ((q(j - 1) - q(j + 1)) > 0.)) {
				f = 1.0 - std::max(0., (shock_ratio - a1) * a2);
			}

			return f;
		};

		const double f_i = flatten_f(i);
		double f_s;
		if (q(i + 1) - q(i - 1) < 0.) {
			f_s = flatten_f(i + 1);
		} else {
			f_s = flatten_f(i - 1);
		}

		const double f = std::max(f_i, f_s);

		const double new_a_minus = a * f + a_minus * (1.0 - f);
		const double new_a_plus = a * f + a_plus * (1.0 - f);

		densityXRight_(i) = new_a_minus;
		densityXLeft_(i + 1) = new_a_plus;
	}
}
#endif

void HydroSystem::AddSourceTerms(AthenaArray<double> &source_terms) {}
