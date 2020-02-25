//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.cpp
/// \brief Implements methods for solving the (1d) radiation moment equations.
///

#include "radiation_system.hpp"

// We must *define* static member variables here, outside of the class
// *declaration*, even though the definitions are trivial.
const RadSystem::NxType::argument RadSystem::Nx;
const RadSystem::LxType::argument RadSystem::Lx;
const RadSystem::CFLType::argument RadSystem::CFL;

RadSystem::RadSystem(NxType const &nx, LxType const &lx,
		     CFLType const &cflNumber)
    : HyperbolicSystem{nx.get(), lx.get(), cflNumber.get(), 2}
{
	radEnergy_.InitWithShallowSlice(consVar_, 2, radEnergy_index, 0);
	x1RadFlux_.InitWithShallowSlice(consVar_, 2, x1RadFlux_index, 0);
}

auto RadSystem::radEnergy(const int i) -> double { return radEnergy_(i); }

auto RadSystem::set_radEnergy(const int i) -> double & { return radEnergy_(i); }

auto RadSystem::x1RadFlux(const int i) -> double { return x1RadFlux_(i); }

auto RadSystem::set_x1RadFlux(const int i) -> double & { return x1RadFlux_(i); }

auto RadSystem::ComputeRadEnergy() -> double
{
	double energy = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		energy += radEnergy_(i) * dx_;
	}

	return energy;
}

void RadSystem::ConservedToPrimitive(AthenaArray<double> &cons,
				     const std::pair<int, int> range)
{
	// have to copy to primitive array (even though it's identical)
	for (int n = 0; n < nvars_; ++n) {
		for (int i = range.first; i < range.second; ++i) {
			primVar_(n, i) = cons(n, i);
		}
	}
}

void RadSystem::ComputeTimestep()
{
	double dt = std::numeric_limits<double>::max();

	for (int i = 0; i < dim1_; ++i) {
		const double signal_max = c_hat;
		const double thisDt = cflNumber_ * (dx_ / signal_max);
		dt = std::min(dt, thisDt);
	}

	dt_ = dt;
}

// TODO(ben): add flux limiter for positivity preservation.
void RadSystem::ComputeFluxes(const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = range.first; i < (range.second + 1); ++i) {
		// HLL solver following Toro (1998) and Balsara (2017).
		// Radiation eigenvalues from Skinner & Ostriker (2013).

		// gather left- and right- state variables

		const double erad_L = x1LeftState_(radEnergy_index, i);
		const double erad_R = x1RightState_(radEnergy_index, i);
		const double Fx_L = x1LeftState_(x1RadFlux_index, i);
		const double Fx_R = x1RightState_(x1RadFlux_index, i);

		// compute "reduced flux" == ||F|| / (c * erad)

		const double Fnorm_L = std::abs(Fx_L);
		const double Fnorm_R = std::abs(Fx_R);
		const double f_L = Fnorm_L / (c_light * erad_L);
		const double f_R = Fnorm_R / (c_light * erad_R);

		// compute radiation pressure tensors

		const double chi_L = ();
		const double chi_R = ();

		const double Tdiag_L = ();
		const double Tdiag_R = ();

		const double Tnn_L = ();
		const double Tnn_R = ();

		// angle between interface and radiation flux

		const double mu_L = Fx_L / f_L;
		const double mu_R = Fx_R / f_R;

		// compute min/max eigenvalues (following S&O, Eq. 41a)

		const double f_facL = std::sqrt(4.0 - 3.0 * (f_L * f_L));
		const double f_facR = std::sqrt(4.0 - 3.0 * (f_R * f_R));

		const double u_L = c_hat * (mu_L * f_L) / f_facL;
		const double u_R = c_hat * (mu_R * f_R) / f_facR;

		const double a_L =
		    c_hat *
		    std::sqrt((2. / 3.) * ((f_facL * f_facL) - f_facL) +
			      2.0 * (mu_L * mu_L) *
				  (2.0 - (f_L * f_L) - f_facL)) /
		    f_facL;

		const double a_R =
		    c_hat *
		    std::sqrt((2. / 3.) * ((f_facR * f_facR) - f_facR) +
			      2.0 * (mu_R * mu_R) *
				  (2.0 - (f_R * f_R) - f_facR)) /
		    f_facR;

		// compute min/max wave speeds (following Toro, Eq. 10.48)

		const double S_L = std::min((u_L - a_L), (u_R - a_R));
		const double S_R = std::max((u_R + a_R), (u_L + a_L));

		// compute fluxes

		const std::valarray<double> F_L = {(c_hat / c_light) * Fx_L,
						   c_hat * c_light * Prad_L};

		const std::valarray<double> F_R = {(c_hat / c_light) * Fx_R,
						   c_hat * c_light * Prad_R};

		const std::valarray<double> U_L = {erad_L, Fx_L};
		const std::valarray<double> U_R = {erad_R, Fx_R};

		const std::valarray<double> F_star =
		    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R +
		    (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

		std::valarray<double> F(2);

		// open the Riemann fan

		if (S_L > 0.0) {
			F = F_L;
		} else if (S_R < 0.0) {
			F = F_R;
		} else { // S_L <= 0.0 <= S_R
			F = F_star;
		}

		// check states are valid

		assert(!std::isnan(F[0])); // NOLINT
		assert(!std::isnan(F[1])); // NOLINT

		x1Flux_(radEnergy_index, i) = F[0];
		x1Flux_(x1RadFlux_index, i) = F[1];
	}
}
