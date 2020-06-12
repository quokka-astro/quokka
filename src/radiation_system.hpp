#ifndef RADIATION_SYSTEM_HPP_ // NOLINT
#define RADIATION_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.hpp
/// \brief Defines a class for solving the (1d) radiation moment equations.
///

// c++ headers
#include <cassert>
#include <cmath>
#include <iostream>
#include <valarray>

// library headers
#include <fmt/format.h>

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
template <typename problem_t>
class RadSystem : public HyperbolicSystem<problem_t>
{
	// See
	// https://stackoverflow.com/questions/4010281/accessing-protected-members-of-superclass-in-c-with-templates
	// for why this is necessary.
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
	using HyperbolicSystem<problem_t>::consVar_;
	using HyperbolicSystem<problem_t>::consVarPredictStep_;

      public:
	enum consVarIndex {
		radEnergy_index = 0,
		x1RadFlux_index = 1,
		gasEnergy_index = 2,
		gasDensity_index = 3,
		x1GasMomentum_index = 4,
	};

	enum primVarIndex {
		primRadEnergy_index = 0,
		x1ReducedFlux_index = 1,
	};

	double c_light_ = 2.99792458e10;	 // cgs
	double c_hat_ = c_light_;		 // for now
	double radiation_constant_ = 7.5646e-15; // cgs

	const double mean_molecular_mass_ = (0.5) * 1.6726231e-24; // cgs
	const double boltzmann_constant_ = 1.380658e-16;	   // cgs
	const double gamma_ = (5. / 3.);

	double Erad_floor_ = 0.0;

	struct RadSystemArgs {
		int nx;
		double lx;
		double cflNumber;
	};

	explicit RadSystem(RadSystemArgs args);

	void FillGhostZones(array_t &cons) override;
	void ConservedToPrimitive(array_t &cons,
				  std::pair<int, int> range) override;
	void AddSourceTerms(array_t &cons, std::pair<int, int> range) override;
	auto CheckStatesValid(array_t &cons, std::pair<int, int> range) const
	    -> bool override;

	// static functions

	static auto ComputeOpacity(double rho, double Tgas) -> double;
	static auto ComputeOpacityTempDerivative(double rho, double Tgas)
	    -> double;
	static auto ComputeTgasFromEgas(double Egas) -> double;
	static auto ComputeEgasFromTgas(double Tgas) -> double;
	auto ComputeEgasTempDerivative(double rho, double Tgas) -> double;

	// setter functions:

	void set_cflNumber(double cflNumber);
	void set_lx(double lx);
	auto set_radEnergy(int i) -> double &;
	auto set_x1RadFlux(int i) -> double &;
	auto set_gasEnergy(int i) -> double &;
	auto set_staticGasDensity(int i) -> double &;
	auto set_x1GasMomentum(int i) -> double &;
	auto set_radEnergySource(int i) -> double &;
	void set_c_light(double c_light);
	void set_radiation_constant(double arad);

	// accessor functions:

	[[nodiscard]] auto radEnergy(int i) const -> double;
	[[nodiscard]] auto x1RadFlux(int i) const -> double;
	[[nodiscard]] auto gasEnergy(int i) const -> double;
	[[nodiscard]] auto staticGasDensity(int i) const -> double;
	[[nodiscard]] auto x1GasMomentum(int i) const -> double;
	[[nodiscard]] auto ComputeRadEnergy() const -> double;
	[[nodiscard]] auto ComputeGasEnergy() const -> double;
	[[nodiscard]] auto c_light() const -> double;
	[[nodiscard]] auto radiation_constant() const -> double;

      protected:
	array_t radEnergy_;
	array_t x1RadFlux_;
	array_t gasEnergy_;
	array_t staticGasDensity_;
	array_t x1GasMomentum_;
	array_t radEnergySource_;

	// virtual function overrides

	void PredictStep(std::pair<int, int> range) override;
	void AddFluxesSDC(array_t &U_new, array_t &U_0) override;
	void AddFluxesRK2(array_t &U0, array_t &U1) override;
	void ComputeFluxes(std::pair<int, int> range) override;
	void ComputeTimestep(double dt_max) override;
};

template <typename problem_t>
RadSystem<problem_t>::RadSystem(RadSystemArgs args)
    : HyperbolicSystem<problem_t>{args.nx, args.lx, args.cflNumber, 5}
{
	radEnergy_.InitWithShallowSlice(consVar_, 2, radEnergy_index, 0);
	x1RadFlux_.InitWithShallowSlice(consVar_, 2, x1RadFlux_index, 0);

	gasEnergy_.InitWithShallowSlice(consVar_, 2, gasEnergy_index, 0);
	staticGasDensity_.InitWithShallowSlice(consVar_, 2, gasDensity_index,
					       0);
	x1GasMomentum_.InitWithShallowSlice(consVar_, 2, x1GasMomentum_index, 0);

	radEnergySource_.NewAthenaArray(args.nx + 2 * nghost_);
}

template <typename problem_t>
auto RadSystem<problem_t>::c_light() const -> double
{
	return c_light_;
}

template <typename problem_t>
void RadSystem<problem_t>::set_c_light(double c_light)
{
	c_light_ = c_light;
	c_hat_ = c_light;
}

template <typename problem_t>
auto RadSystem<problem_t>::radiation_constant() const -> double
{
	return radiation_constant_;
}

template <typename problem_t>
void RadSystem<problem_t>::set_radiation_constant(double arad)
{
	radiation_constant_ = arad;
}

template <typename problem_t> void RadSystem<problem_t>::set_lx(const double lx)
{
	assert(lx > 0.0); // NOLINT
	lx_ = lx;
	dx_ = lx / static_cast<double>(nx_);
}

template <typename problem_t>
auto RadSystem<problem_t>::radEnergy(const int i) const -> double
{
	return radEnergy_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::set_radEnergy(const int i) -> double &
{
	return radEnergy_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::x1RadFlux(const int i) const -> double
{
	return x1RadFlux_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::set_x1RadFlux(const int i) -> double &
{
	return x1RadFlux_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::gasEnergy(const int i) const -> double
{
	return gasEnergy_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::set_gasEnergy(const int i) -> double &
{
	return gasEnergy_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::staticGasDensity(const int i) const -> double
{
	return staticGasDensity_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::set_staticGasDensity(const int i) -> double &
{
	return staticGasDensity_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::x1GasMomentum(const int i) const -> double
{
	return x1GasMomentum_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::set_x1GasMomentum(const int i) -> double &
{
	return x1GasMomentum_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::set_radEnergySource(const int i) -> double &
{
	return radEnergySource_(i);
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeRadEnergy() const -> double
{
	double energy = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		energy += radEnergy_(i) * dx_;
	}

	return energy;
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeGasEnergy() const -> double
{
	double energy = 0.0;

	for (int i = nghost_; i < nx_ + nghost_; ++i) {
		energy += gasEnergy_(i) * dx_;
	}

	return energy;
}

template <typename problem_t>
void RadSystem<problem_t>::FillGhostZones(array_t &cons)
{
	// In general, this step will require MPI communication, and interaction
	// with the main AMR code.

	// Su & Olson (1997)* boundary conditions
	// [Reflecting boundary on left, constant on right.]
	// [*Subtle differences* from Marshak boundary condition!]

	// x1 left side boundary (reflecting)
	for (int i = 0; i < nghost_; ++i) {
		cons(radEnergy_index, i) =
		    cons(radEnergy_index, nghost_ + (nghost_ - i - 1));
		cons(x1RadFlux_index, i) =
		    -1.0 * cons(x1RadFlux_index, nghost_ + (nghost_ - i - 1));
	}

	// x1 right side boundary (reflecting)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) = cons(
		    radEnergy_index, (nghost_ + nx_) - (i - nx_ - nghost_ + 1));
		cons(x1RadFlux_index, i) =
		    -1.0 * cons(x1RadFlux_index,
				(nghost_ + nx_) - (i - nx_ - nghost_ + 1));
	}

#if 0
	// x1 right side boundary (constant temperature, extrapolate flux)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) = Erad_floor_;
		cons(x1RadFlux_index, i) = 0.0;
	}
#endif
}

template <typename problem_t>
auto RadSystem<problem_t>::CheckStatesValid(
    array_t &cons, const std::pair<int, int> range) const -> bool
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

template <typename problem_t>
void RadSystem<problem_t>::ConservedToPrimitive(array_t &cons,
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

template <typename problem_t>
void RadSystem<problem_t>::ComputeTimestep(const double dt_max)
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

template <typename problem_t>
void RadSystem<problem_t>::ComputeFluxes(const std::pair<int, int> range)
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

		// compute Levermore (1984) closure [Eq. 25]
		const double f_facL = std::sqrt(4.0 - 3.0 * (f_L * f_L));
		const double f_facR = std::sqrt(4.0 - 3.0 * (f_R * f_R));

		const double chi_L =
		    (3.0 + 4.0 * (f_L * f_L)) / (5.0 + 2.0 * f_facL);
		const double chi_R =
		    (3.0 + 4.0 * (f_R * f_R)) / (5.0 + 2.0 * f_facR);

#if 0
		// compute Minerbo (1978) closure [piecewise approximation]
		// (For unknown reasons, this closure tends to work better
		// than the Levermore/Lorentz closure on the Su & Olson 1997 test.)
		const double chi_L =
		    (f_L < 1. / 3.) ? (1. / 3.) : (0.5 - f_L + 1.5 * f_L * f_L);
		const double chi_R =
		    (f_R < 1. / 3.) ? (1. / 3.) : (0.5 - f_R + 1.5 * f_R * f_R);
#endif

		assert((chi_L >= 1. / 3.) && (chi_L <= 1.0)); // NOLINT
		assert((chi_R >= 1. / 3.) && (chi_R <= 1.0)); // NOLINT

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

		x1Flux_(radEnergy_index, i) = F[0];
		x1Flux_(x1RadFlux_index, i) = F[1];
	}
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeOpacity(const double rho, const double Tgas)
    -> double
{
	// TODO(ben): interpolate from a table
	// return 0.4; // cm^2 g^-1 (Thomson opacity)
	return 1.0;
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeOpacityTempDerivative(const double rho,
							const double Tgas)
    -> double
{
	// TODO(ben): interpolate from a table
	return 0.0;
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeEgasTempDerivative(const double rho,
						     const double Tgas)
    -> double
{
	const double c_v =
	    boltzmann_constant_ / (mean_molecular_mass_ * (gamma_ - 1.0));
	return (rho * c_v);
}

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &cons,
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
		const double rho = cons(gasDensity_index, i);
		const double Egas0 = cons(gasEnergy_index, i);

		// load radiation energy
		const double Erad0 = cons(radEnergy_index, i);

		assert(Egas0 > 0.0); // NOLINT
		assert(Erad0 > 0.0); // NOLINT

		const double Etot0 = Egas0 + Erad0;

		// BEGIN NEWTON-RAPHSON LOOP
		double F_G = NAN;
		double F_R = NAN;
		double rhs = NAN;
		double T_gas = NAN;
		double kappa = NAN;
		double fourPiB = NAN;
		double dB_dTgas = NAN;
		double dkappa_dTgas = NAN;
		double drhs_dEgas = NAN;
		double dFG_dEgas = NAN;
		double dFG_dErad = NAN;
		double dFR_dEgas = NAN;
		double dFR_dErad = NAN;
		double Src = NAN;
		double eta = NAN;
		double deltaErad = NAN;
		double deltaEgas = NAN;

		double Egas_guess = Egas0;
		double Erad_guess = Erad0;
		const double T_floor = 1e-10;
		const double resid_tol = 1e-10;
		const int maxIter = 200;
		int n = 0;
		for (n = 0; n < maxIter; ++n) {

			// compute material temperature
			T_gas = RadSystem<problem_t>::ComputeTgasFromEgas(
			    Egas_guess);

			// compute opacity, emissivity
			kappa =
			    RadSystem<problem_t>::ComputeOpacity(rho, T_gas);
			fourPiB = c * a_rad * std::pow(T_gas, 4);

			// constant radiation energy source term
			Src = dt * (c * radEnergySource_(i));

			// compute derivatives w/r/t T_gas
			const double dB_dTgas = (4.0 * fourPiB) / T_gas;
			const double dkappa_dTgas =
			    RadSystem<problem_t>::ComputeOpacityTempDerivative(
				rho, T_gas);

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
			const double c_v =
			    ComputeEgasTempDerivative(rho, T_gas);

			drhs_dEgas =
			    (rho * dt / c_v) *
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

		const double Frad_x = cons(x1RadFlux_index, i);
		const double new_Frad_x =
		    (1. / (1.0 + (rho * kappa) * c * dt)) * Frad_x;

		cons(x1RadFlux_index, i) = new_Frad_x;

		// 3. Compute conservative gas momentum update
		//	[N.B. should this step happen after the Lorentz transform?]

		const double dF_x = new_Frad_x - Frad_x;
		const double dx1Momentum = -dF_x / (c*c);

		cons(x1GasMomentum_index, i) += dx1Momentum;

		// Lorentz transform back to 'laboratory' frame
		// TransformIntoComovingFrame(-fluid_velocity);
	}
}

template <typename problem_t>
void RadSystem<problem_t>::PredictStep(const std::pair<int, int> range)
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

template <typename problem_t>
void RadSystem<problem_t>::AddFluxesRK2(array_t &U0, array_t &U1)
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

template <typename problem_t>
void RadSystem<problem_t>::AddFluxesSDC(array_t &U_new, array_t &U_0)
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

#endif // RADIATION_SYSTEM_HPP_
