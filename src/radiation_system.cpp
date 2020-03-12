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
    : HyperbolicSystem{nx.get(), lx.get(), cflNumber.get(), 4}
{
	radEnergy_.InitWithShallowSlice(consVar_, 2, radEnergy_index, 0);
	x1RadFlux_.InitWithShallowSlice(consVar_, 2, x1RadFlux_index, 0);

	gasEnergy_.InitWithShallowSlice(consVar_, 2, gasEnergy_index, 0);
	staticGasDensity_.InitWithShallowSlice(consVar_, 2, gasDensity_index,
					       0);

	radEnergySource_.NewAthenaArray(nx.get() + 2.0 * nghost_);
}

auto RadSystem::c_light() -> double { return c_light_; }

void RadSystem::set_c_light(double c_light)
{
	c_light_ = c_light;
	c_hat_ = c_light;
}

auto RadSystem::radiation_constant() -> double { return radiation_constant_; }

void RadSystem::set_radiation_constant(double arad)
{
	radiation_constant_ = arad;
}

void RadSystem::set_lx(const double lx)
{
	assert(lx > 0.0); // NOLINT
	lx_ = lx;
	dx_ = lx / static_cast<double>(nx_);
}

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

auto RadSystem::set_radEnergySource(const int i) -> double &
{
	return radEnergySource_(i);
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

void RadSystem::FillGhostZones(AthenaArray<double> &cons)
{
	// In general, this step will require MPI communication, and interaction
	// with the main AMR code.

	// Su & Olson (1997)* boundary conditions
	// [Reflecting boundary on left, constant on right.]
	// [*Subtle differences* from Marshak boundary condition!]

	// x1 left side boundary (reflecting)
	for (int i = 0; i < nghost_; ++i) {
		cons(radEnergy_index, i) =
		    cons(radEnergy_index, nghost_ + (nghost_ - i));
		cons(x1RadFlux_index, i) =
		    -cons(x1RadFlux_index, nghost_ + (nghost_ - i));
	}

	// x1 right side boundary (constant temperature, extrapolate flux)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) = Erad_floor_;
		cons(x1RadFlux_index, i) = 0.0;
	}
}

bool RadSystem::CheckStatesValid(AthenaArray<double> &cons,
				 const std::pair<int, int> range)
{
	bool all_valid = true;

	for (int i = range.first; i < range.second; ++i) {
		const auto E_r = cons(radEnergy_index, i);
		const auto Fx = cons(x1RadFlux_index, i);
		const auto reducedFluxX1 = Fx / (c_light_ * E_r);

		if (E_r < 0.0) {
			std::cout << "ERROR: Positivity failure at i = " << i
				  << " with energy density = " << E_r
				  << std::endl;
			all_valid = false;
		}

		if (std::abs(reducedFluxX1) > 1.0) {
			std::cout << "ERROR: Flux limiting failure at i = " << i
				  << " with reduced flux = " << reducedFluxX1
				  << std::endl;
			all_valid = false;
		}
	}

	return all_valid;
}

void RadSystem::ConservedToPrimitive(AthenaArray<double> &cons,
				     const std::pair<int, int> range)
{
	// keep radiation energy density as-is
	// convert (Fx,Fy,Fz) into reduced flux components (fx,fy,fx):
	//   F_x -> F_x / (c*E_r)

	for (int i = range.first; i < range.second; ++i) {
		const auto E_r = cons(radEnergy_index, i);
		const auto Fx = cons(x1RadFlux_index, i);
		const auto reducedFluxX1 = Fx / (c_light_ * E_r);

		// check admissibility of states
		assert(E_r > 0.0);			// NOLINT
		assert(std::abs(reducedFluxX1) <= 1.0); // NOLINT

		primVar_(primRadEnergy_index, i) = E_r;
		primVar_(x1ReducedFlux_index, i) = reducedFluxX1;
	}
}

void RadSystem::ComputeTimestep(const double dt_max)
{
	// double dt = std::numeric_limits<double>::max();
	double dt = dt_max;

	// std::cout << "dt_max = " << dt_max << "\n";

	for (int i = 0; i < dim1_; ++i) {
		const double signal_max = c_hat_;
		const double thisDt = cflNumber_ * (dx_ / signal_max);
		dt = std::min(dt, thisDt);
	}

	// std::cout << "Timestep determined to be: " << dt << "\n";

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

		// gather left- and right- state variables

		const double erad_L = x1LeftState_(primRadEnergy_index, i);
		const double erad_R = x1RightState_(primRadEnergy_index, i);

		const double fx_L = x1LeftState_(x1ReducedFlux_index, i);
		const double fx_R = x1RightState_(x1ReducedFlux_index, i);

		// compute scalar reduced flux f
		// [modify in 3d!]
		const double f_L = std::sqrt(fx_L * fx_L);
		const double f_R = std::sqrt(fx_R * fx_R);

		// check that states are physically-admissible
		// (NOTE: It must always be the case that 0 <= f <= 1!
		// 	      This implies that erad != zero!)

		assert(erad_L > 0.0); // NOLINT
		assert(erad_R > 0.0); // NOLINT
		assert(f_L <= 1.0);   // NOLINT
		assert(f_R <= 1.0);   // NOLINT

		// angle between interface and radiation flux \hat{n}
		// If direction is undefined, just drop direction-dependent
		// terms.

		const double nx_L = (f_L > 0.0) ? (fx_L / f_L) : 0.0;
		const double nx_R = (f_R > 0.0) ? (fx_R / f_R) : 0.0;

		// Compute "un-reduced" Fx, ||F||

		const double Fx_L = fx_L * (c_light_ * erad_L);
		const double Fx_R = fx_R * (c_light_ * erad_R);

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

#if 0
		// compute min/max eigenvalues (following S&O, Eq. 41a)
		const double mu_L = (nx_L); // modify in 3d!
		const double mu_R = (nx_R);

		const double u_L = (c_hat_ / f_facL) * (mu_L * f_L);
		const double u_R = (c_hat_ / f_facR) * (mu_R * f_R);

		double a_L =
		    (c_hat_ / f_facL) *
		    std::sqrt((2. / 3.) * ((f_facL * f_facL) - f_facL) +
			      2.0 * (mu_L * mu_L) *
				  (2.0 - (f_L * f_L) - f_facL));

		double a_R =
		    (c_hat_ / f_facL) *
		    std::sqrt((2. / 3.) * ((f_facR * f_facR) - f_facR) +
			      2.0 * (mu_R * mu_R) *
				  (2.0 - (f_R * f_R) - f_facR));

		const double S_L = u_L - a_L;
		const double S_R = u_R + a_R;
#endif

		// frozen Eddington tensor approximation, following Balsara
		// (1999) [JQSRT Vol. 61, No. 5, pp. 617–627, 1999], Eq. 46.

		const double S_L = -c_hat_ * std::sqrt(Tdiag_L + Txx_L);
		const double S_R = c_hat_ * std::sqrt(Tdiag_R + Txx_R);

		assert(std::abs(S_L) <= c_hat_); // NOLINT
		assert(std::abs(S_R) <= c_hat_); // NOLINT

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

		// in the frozen Eddington tensor approximation, we are always
		// in the star region, so F = F_star
		F = F_star;

		// check states are valid
		assert(!std::isnan(F[0])); // NOLINT
		assert(!std::isnan(F[1])); // NOLINT

		// these are computed with the linearized wavespeeds
		x1Flux_(radEnergy_index, i) = F[0];
		x1Flux_(x1RadFlux_index, i) = F[1];
	}
}

auto RadSystem::ComputeOpacity(const double rho, const double Temp) -> double
{
	// TODO(ben): interpolate from a table
	// return 0.4; // cm^2 g^-1 (Thomson opacity)
	return 1.0;
}

auto RadSystem::ComputeOpacityTempDerivative(const double rho,
					     const double Temp) -> double
{
	// TODO(ben): interpolate from a table
	return 0.0;
}

void RadSystem::AddSourceTerms(AthenaArray<double> &cons,
			       std::pair<int, int> range)
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
		// const double c_v = boltzmann_constant_cgs_ /
		// (mean_molecular_mass_cgs_ * (gamma_ - 1.0));

		// Su & Olson (1997) test problem
		const double eps_SuOlson = 1.0;
		const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

		const double rho = cons(gasDensity_index, i);
		const double Egas0 = cons(gasEnergy_index, i);
		const double c_v =
		    alpha_SuOlson *
		    std::pow(Egas0 / (rho * alpha_SuOlson), 3. / 4.);

		// load radiation energy
		const double Erad0 = cons(radEnergy_index, i);

		assert(Egas0 > 0.0); // NOLINT
		assert(Erad0 > 0.0); // NOLINT

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
		double Src;
		double eta;
		double deltaErad;
		double deltaEgas;

		double Egas_guess = Egas0;
		double Erad_guess = Erad0;
		const double T_floor = 1e-10;
		const double resid_tol = 1e-10;
		const int maxIter = 200;
		int n;
		for (n = 1; n <= maxIter; ++n) {

			// compute material temperature
			T_gas = Egas_guess / (rho * c_v);

			// compute opacity, emissivity
			kappa = ComputeOpacity(rho, T_gas);
			fourPiB = c * a_rad * std::pow(T_gas, 4);

			// constant radiation energy source term
			Src = dt * (c * radEnergySource_(i));

			// compute derivatives w/r/t T_gas
			const double dB_dTgas = (4.0 * fourPiB) / T_gas;
			const double dkappa_dTgas =
			    ComputeOpacityTempDerivative(rho, T_gas);

			// compute residuals
			rhs = dt * (rho * kappa) * (fourPiB - c * Erad_guess);
			F_G = (Egas_guess - Egas0) + rhs;
			F_R = (Erad_guess - Erad0) - (rhs + Src);

			// check if converged
			if ((std::abs(F_G / Etot0) < resid_tol) &&
			    (std::abs(F_R / Etot0) < resid_tol)) {
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

		assert(std::abs(F_G / Etot0) < resid_tol); // NOLINT
		assert(std::abs(F_R / Etot0) < resid_tol); // NOLINT

		assert(Erad_guess > 0.0); // NOLINT
		assert(Egas_guess > 0.0); // NOLINT

		// store new radiation energy
		cons(radEnergy_index, i) = Erad_guess;
		cons(gasEnergy_index, i) = Egas_guess;

		// 2. Compute radiation flux update

		const double F_rad = cons(x1RadFlux_index, i);
		const double new_F_rad =
		    (1. / (1.0 + (rho * kappa) * c * dt)) * F_rad;

		cons(x1RadFlux_index, i) = new_F_rad;

		// Lorentz transform back to 'laboratory' frame
		// TransformIntoComovingFrame(-fluid_velocity);
	}
}

void RadSystem::PredictStep(const std::pair<int, int> range)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	for (int i = range.first; i < range.second; ++i) {
		// construct new state: radiation energy density
		const double E_0 = consVar_(radEnergy_index, i);
		const double FE_1 = -1.0 * (dt_ / dx_) *
				    (x1Flux_(radEnergy_index, i + 1) -
				     x1Flux_(radEnergy_index, i));
		const double E_new = E_0 + FE_1;

		// construct new state: X1 radiation flux
		const double x1F_0 = consVar_(x1RadFlux_index, i);
		const double Fx1F_1 = -1.0 * (dt_ / dx_) *
				      (x1Flux_(x1RadFlux_index, i + 1) -
				       x1Flux_(x1RadFlux_index, i));
		const double x1F_new = x1F_0 + Fx1F_1;

		// check validity, fallback to diffusive flux if
		// necessary
		const double x1ReducedFlux_new = x1F_new / (c_light_ * E_new);

		if (std::abs(x1ReducedFlux_new) < 1.0) {
			consVarPredictStep_(radEnergy_index, i) = E_new;
			consVarPredictStep_(x1RadFlux_index, i) = x1F_new;

		} else {
			std::cout
			    << "WARNING: [stage 1] flux limited at i = " << i
			    << " with reduced flux = " << x1ReducedFlux_new
			    << std::endl;

			const double FE_1d =
			    -1.0 * (dt_ / dx_) *
			    (x1FluxDiffusive_(radEnergy_index, i + 1) -
			     x1FluxDiffusive_(radEnergy_index, i));
			const double E_newd = E_0 + FE_1d;

			const double Fx1F_1d =
			    -1.0 * (dt_ / dx_) *
			    (x1FluxDiffusive_(x1RadFlux_index, i + 1) -
			     x1FluxDiffusive_(x1RadFlux_index, i));
			const double x1F_newd = x1F_0 + Fx1F_1d;

			consVarPredictStep_(radEnergy_index, i) = E_newd;
			consVarPredictStep_(x1RadFlux_index, i) = x1F_newd;
		}
	}
}

void RadSystem::AddFluxesRK2(AthenaArray<double> &U0, AthenaArray<double> &U1)
{
	// By convention, the fluxes are defined on the left edge of
	// each zone, i.e. flux_(i) is the flux *into* zone i through
	// the interface on the left of zone i, and -1.0*flux(i+1) is
	// the flux *into* zone i through the interface on the right of
	// zone i.

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		// RK-SSP2 integrator

		// construct new state: radiation energy density
		const double E_0 = U0(radEnergy_index, i);
		const double E_1 = U1(radEnergy_index, i);
		const double FE_1 = -1.0 * (dt_ / dx_) *
				    (x1Flux_(radEnergy_index, i + 1) -
				     x1Flux_(radEnergy_index, i));
		const double E_new = 0.5 * E_0 + 0.5 * E_1 + 0.5 * FE_1;

		// construct new state: X1 radiation flux
		const double x1F_0 = U0(x1RadFlux_index, i);
		const double x1F_1 = U1(x1RadFlux_index, i);
		const double Fx1F_1 = -1.0 * (dt_ / dx_) *
				      (x1Flux_(x1RadFlux_index, i + 1) -
				       x1Flux_(x1RadFlux_index, i));
		const double x1F_new = 0.5 * x1F_0 + 0.5 * x1F_1 + 0.5 * Fx1F_1;

		// check validity, fallback to diffusive flux if
		// necessary
		const double x1ReducedFlux_new = x1F_new / (c_light_ * E_new);

		if (std::abs(x1ReducedFlux_new) < 1.0) {
			U0(radEnergy_index, i) = E_new;
			U0(x1RadFlux_index, i) = x1F_new;

		} else {
			std::cout
			    << "WARNING: [stage 2] flux limited at i = " << i
			    << " with reduced flux = " << x1ReducedFlux_new
			    << std::endl;

			const double FE_1d =
			    -1.0 * (dt_ / dx_) *
			    (x1FluxDiffusive_(radEnergy_index, i + 1) -
			     x1FluxDiffusive_(radEnergy_index, i));
			const double E_newd =
			    0.5 * E_0 + 0.5 * E_1 + 0.5 * FE_1d;

			const double Fx1F_1d =
			    -1.0 * (dt_ / dx_) *
			    (x1FluxDiffusive_(x1RadFlux_index, i + 1) -
			     x1FluxDiffusive_(x1RadFlux_index, i));
			const double x1F_newd =
			    0.5 * x1F_0 + 0.5 * x1F_1 + 0.5 * Fx1F_1d;

			U0(radEnergy_index, i) = E_newd;
			U0(x1RadFlux_index, i) = x1F_newd;
		}
	}
}

void RadSystem::AddFluxesSDC(AthenaArray<double> &U_new,
			     AthenaArray<double> &U_0)
{
	// By convention, the fluxes are defined on the left edge of
	// each zone, i.e. flux_(i) is the flux *into* zone i through
	// the interface on the left of zone i, and -1.0*flux(i+1) is
	// the flux *into* zone i through the interface on the right of
	// zone i.

	// Perform flux limiting on intercell fluxes
	// (This may be necessary within SDC correction iteration loop.)
	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		const double Fp = x1Flux_(radEnergy_index, i + 1);
		const double Fm = x1Flux_(radEnergy_index, i);

		const double dE = -1.0 * (dt_ / dx_) * (Fp - Fm);
		const double E0 = U_0(radEnergy_index, i);

		// prevent negative energy density
		const double R = (-1.0 * dE) / (E0 - Erad_floor_);
		// const double lambda_R =
		//    std::pow(1.0 + (R * R * R * R), -1.0 / 4.0);
		const double lambda_R = 1.0 / R;

		if (R > 0.0) {
			const double wp = (-Fp < 0.0) ? lambda_R : 1.0;
			const double wm = (Fm < 0.0) ? lambda_R : 1.0;

			const double new_dE =
			    -1.0 * (dt_ / dx_) * (wp * Fp - wm * Fm);

			const double eps = 1e-10;
			assert((1.0 - ((E0 + new_dE) / Erad_floor_)) <
			       eps); // NOLINT

			for (int n = 0; n < nvars_; ++n) {
				x1Flux_(n, i + 1) *= wp;
				x1Flux_(n, i) *= wm;
			}
		}
	}

	// Do normal forward Euler step
	for (int n = 0; n < nvars_; ++n) {
		for (int i = nghost_; i < nx_ + nghost_; ++i) {
			const double Fp = x1Flux_(n, i + 1);
			const double Fm = x1Flux_(n, i);
			const double FU = -1.0 * (dt_ / dx_) * (Fp - Fm);
			U_new(n, i) = U_0(n, i) + FU;
		}
	}
}