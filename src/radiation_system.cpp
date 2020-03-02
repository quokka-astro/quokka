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

	gasEnergy_.NewAthenaArray(nx.get() + 2.0 * nghost_);
	staticGasDensity_.NewAthenaArray(nx.get() + 2.0 * nghost_);
}

auto RadSystem::c_light() -> double { return c_light_; }

auto RadSystem::radiation_constant() -> double { return radiation_constant_; }

auto RadSystem::radEnergy(const int i) -> double { return radEnergy_(i); }

auto RadSystem::set_radEnergy(const int i) -> double & { return radEnergy_(i); }

auto RadSystem::x1RadFlux(const int i) -> double { return x1RadFlux_(i); }

auto RadSystem::set_x1RadFlux(const int i) -> double & { return x1RadFlux_(i); }

auto RadSystem::gasEnergy(const int i) -> double { return gasEnergy_(i); }

auto RadSystem::set_gasEnergy(const int i) -> double & { return gasEnergy_(i); }

auto RadSystem::staticGasDensity(const int i) -> double
{
	return staticGasDensity_(i);
}

auto RadSystem::set_staticGasDensity(const int i) -> double &
{
	return staticGasDensity_(i);
}

auto RadSystem::ComputeRadEnergy() -> double
{
	double energy = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		energy += radEnergy_(i) * dx_;
	}

	return energy;
}

auto RadSystem::ComputeGasEnergy() -> double
{
	double energy = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		energy += gasEnergy_(i) * dx_;
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

void RadSystem::ComputeTimestep(const double dt_max)
{
	// double dt = std::numeric_limits<double>::max();
	double dt = dt_max;

	for (int i = 0; i < dim1_; ++i) {
		const double signal_max = c_hat_;
		const double thisDt = cflNumber_ * (dx_ / signal_max);
		dt = std::min(dt, thisDt);
	}

	dt_ = dt;
}

// TODO(ben): add flux limiter for positivity preservation.
//
void RadSystem::ComputeFluxes(const std::pair<int, int> range)
{
	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	for (int i = range.first; i < (range.second + 1); ++i) {

		// HLL solver following Toro (1998) and Balsara (2017).
		// Radiation eigenvalues from Skinner & Ostriker (2013).

		const double eps = 1e-10;

		// gather left- and right- state variables

		const double erad_L = x1LeftState_(radEnergy_index, i);
		const double erad_R = x1RightState_(radEnergy_index, i);
		const double Fx_L = x1LeftState_(x1RadFlux_index, i);
		const double Fx_R = x1RightState_(x1RadFlux_index, i);
		assert(erad_L > 0.0); // NOLINT
		assert(erad_R > 0.0); // NOLINT

		// Compute "reduced flux" f == ||F|| / (c * erad)
		// NOTE: It must always be the case that 0 <= f <= 1!
		// This means that the radiation energy density *cannot* be
		// zero!

		const double Fnorm_L = std::sqrt(Fx_L * Fx_L); // modify in 3d!
		const double Fnorm_R = std::sqrt(Fx_R * Fx_R);
		double f_L = Fnorm_L / (c_light_ * erad_L);
		double f_R = Fnorm_R / (c_light_ * erad_R);
		assert(f_L <= 1.0 + eps); // NOLINT
		assert(f_R <= 1.0 + eps); // NOLINT
		f_L = std::min(f_L, 1.0);
		f_R = std::min(f_R, 1.0);

		// angle between interface and radiation flux

		double nx_L = Fx_L / Fnorm_L;
		double nx_R = Fx_R / Fnorm_R;

		if (Fnorm_L == 0.0) { // direction is undefined, so just drop
				      // direction-dependent terms
			nx_L = 0.0;
		}
		if (Fnorm_R == 0.0) { // direction is undefined, so just drop
				      // direction-dependent terms
			nx_R = 0.0;
		}

		const double mu_L = (nx_L); // modify in 3d!
		const double mu_R = (nx_R);

		// compute radiation pressure tensors

		const double f_facL = std::sqrt(4.0 - 3.0 * (f_L * f_L));
		const double f_facR = std::sqrt(4.0 - 3.0 * (f_R * f_R));

		// compute Eddington factors
		const double chi_L =
		    (3.0 + 4.0 * (f_L * f_L)) / (5.0 + 2.0 * f_facL);
		const double chi_R =
		    (3.0 + 4.0 * (f_R * f_R)) / (5.0 + 2.0 * f_facR);

		// diagonal term of Eddington tensor
		const double Tdiag_L = (1.0 - chi_L) / 2.0;
		const double Tdiag_R = (1.0 - chi_R) / 2.0;

		// anisotropic term of Eddington tensor (in the direction of the
		// rad. flux)
		const double Txx_L = (3.0 * chi_L - 1.0) / 2.0 * (nx_L * nx_L);
		const double Txx_R = (3.0 * chi_R - 1.0) / 2.0 * (nx_R * nx_R);

		// compute the elements of the total radiation pressure tensor
		const double Pxx_L = (Tdiag_L + Txx_L) * erad_L;
		const double Pxx_R = (Tdiag_R + Txx_R) * erad_R;

		// compute min/max eigenvalues (following S&O, Eq. 41a)

		const double u_L = c_hat_ * (mu_L * f_L) / f_facL;
		const double u_R = c_hat_ * (mu_R * f_R) / f_facR;

		const double a_L =
		    c_hat_ *
		    std::sqrt((2. / 3.) * ((f_facL * f_facL) - f_facL) +
			      2.0 * (mu_L * mu_L) *
				  (2.0 - (f_L * f_L) - f_facL)) /
		    f_facL;

		const double a_R =
		    c_hat_ *
		    std::sqrt((2. / 3.) * ((f_facR * f_facR) - f_facR) +
			      2.0 * (mu_R * mu_R) *
				  (2.0 - (f_R * f_R) - f_facR)) /
		    f_facR;

		// compute min/max wave speeds (following Toro, Eq. 10.48)

		const double S_L = u_L - a_L;
		const double S_R = u_R + a_R;

		// compute fluxes

		const std::valarray<double> F_L = {(c_hat_ / c_light_) * Fx_L,
						   c_hat_ * c_light_ * Pxx_L};

		const std::valarray<double> F_R = {(c_hat_ / c_light_) * Fx_R,
						   c_hat_ * c_light_ * Pxx_R};

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

auto RadSystem::ComputeOpacity(const double rho, const double Temp) -> double
{
	// TODO(ben): interpolate from a table
	return 0.4; // cm^2 g^-1 (Thomson opacity)
}

auto RadSystem::ComputeOpacityTempDerivative(const double rho,
					     const double Temp) -> double
{
	// TODO(ben): interpolate from a table
	return 0.0;
}

void RadSystem::AddSourceTerms(std::pair<int, int> range)
{
	// Lorentz transform the radiation variables into the comoving frame
	// TransformIntoComovingFrame(fluid_velocity);

	// Add source terms

	// 1. Compute gas energy and radiation energy update following Howell &
	// Greenough [Journal of Computational Physics 184 (2003) 53–78].

	for (int i = range.first; i < range.second; ++i) {
		const double dt = dt_;
		const double c = c_light_;
		const double a_rad = radiation_constant_;

		// load fluid properties

		const double mu = mean_molecular_mass_cgs_;
		const double gamma = gamma_;
		const double k_B = boltzmann_constant_cgs_;
		const double c_v = k_B / (mu * (gamma - 1.0));

		const double rho = staticGasDensity_(i);
		const double Egas0 = gasEnergy_(i);

		// load radiation energy
		const double Erad0 = radEnergy_(i);
		const double Etot0 = Egas0 + Erad0;

		// BEGIN NEWTON-RAPHSON LOOP
		double F_G;
		double F_R;
		double rhs;
		double T_gas;
		double kappa;
		double fourPiB;
		double dB_dTgas;
		double dkappa_dTgas;
		double drhs_dEgas;
		double dFG_dEgas;
		double dFG_dErad;
		double dFR_dEgas;
		double dFR_dErad;
		double eta;
		double deltaErad;
		double deltaEgas;

		double Egas_guess = Egas0;
		double Erad_guess = Erad0;
		const double resid_tol = 1e-10;
		const int maxIter = 200;

		for (int n = 1; n <= maxIter; ++n) {

			// compute material temperature, opacity
			T_gas = Egas_guess / (rho * c_v);
			kappa = ComputeOpacity(rho, T_gas);
			fourPiB = c * a_rad * std::pow(T_gas, 4);

			// compute derivatives w/r/t T_gas
			const double dB_dTgas = (4.0 * fourPiB) / T_gas;
			const double dkappa_dTgas =
			    ComputeOpacityTempDerivative(rho, T_gas);

			// compute residuals
			rhs = dt * (rho * kappa) * (fourPiB - c * Erad_guess);
			F_G = (Egas_guess - Egas0) + rhs;
			F_R = (Erad_guess - Erad0) - rhs;

			// check if converged
			if ((std::fabs(std::max(F_G / Egas0, F_G / Etot0)) <
			     resid_tol) &&
			    (std::fabs(std::max(F_R / Erad0, F_R / Etot0)) <
			     resid_tol)) {
				std::cout << "converged after " << n
					  << " iterations.\n";
				break;
			}

			// compute Jacobian elements
			drhs_dEgas =
			    (dt / c_v) *
			    (kappa * dB_dTgas +
			     dkappa_dTgas * (fourPiB - c * Erad_guess));

			dFG_dEgas = 1.0 + drhs_dEgas;
			dFG_dErad = dt * (-(rho * kappa) * c);
			dFR_dEgas = -drhs_dEgas;
			dFR_dErad = 1.0 + dt * ((rho * kappa) * c);

			// Update variables
			eta = -dFR_dEgas / dFG_dEgas;
			// eta = (eta > 0.0) ? eta : 0.0;

			deltaErad =
			    -(F_R + eta * F_G) / (dFR_dErad + eta * dFG_dErad);
			deltaEgas = -(F_G + dFG_dErad * deltaErad) / dFG_dEgas;

			Egas_guess += deltaEgas;
			Erad_guess += deltaErad;

		} // END NEWTON-RAPHSON LOOP

		assert((std::fabs(F_G / Egas0) < resid_tol) && // NOLINT
		       (std::fabs(F_R / Erad0) < resid_tol));  // NOLINT

		// store new radiation energy
		radEnergy_(i) = Erad_guess;
		gasEnergy_(i) = Egas_guess;

		// 2. Compute radiation flux update

		// TODO(ben): write this.

		// Lorentz transform back to 'laboratory' frame
		// TransformIntoComovingFrame(-fluid_velocity);
	}
}