#ifndef RAD_SOURCE_TERMS_BASE_HPP_
#define RAD_SOURCE_TERMS_BASE_HPP_

#include "radiation/radiation_base.hpp"

// Compute radiation energy fractions for each photon group from a Planck function, given nGroups, radBoundaries, and temperature
// This function enforces that the total fraction is 1.0, no matter what are the group boundaries
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries,
									      amrex::Real temperature) -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<amrex::Real, nGroups_> radEnergyFractions{};
	if constexpr (nGroups_ == 1) {
		radEnergyFractions[0] = 1.0;
		return radEnergyFractions;
	} else {
		amrex::Real const energy_unit_over_kT = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * temperature);
		amrex::Real y = NAN;
		amrex::Real previous = 0.0;
		for (int g = 0; g < nGroups_ - 1; ++g) {
			const amrex::Real x = boundaries[g + 1] * energy_unit_over_kT;
			if (x >= 100.) { // 100. is the upper limit of x in the table
				y = 1.0;
			} else {
				y = integrate_planck_from_0_to_x(x);
			}
			radEnergyFractions[g] = y - previous;
			previous = y;
		}
		// last group, enforcing the total fraction to be 1.0
		y = 1.0;
		radEnergyFractions[nGroups_ - 1] = y - previous;
		AMREX_ASSERT(std::abs(sum(radEnergyFractions) - 1.0) < 1.0e-10);

		return radEnergyFractions;
	}
}

// define ComputeThermalRadiation for single-group, returns the thermal radiation power = a_r * T^4
template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> Real
{
	double power = radiation_constant_ * std::pow(temperature, 4);
	// set floor
	if (power < Erad_floor_) {
		power = Erad_floor_;
	}
	return power;
}

// define ComputeThermalRadiationMultiGroup, returns the thermal radiation power for each photon group. = a_r * T^4 * radEnergyFractions
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto
RadSystem<problem_t>::ComputeThermalRadiationMultiGroup(amrex::Real temperature,
							amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>
{
	const double power = radiation_constant_ * std::pow(temperature, 4);
	const auto radEnergyFractions = ComputePlanckEnergyFractions(boundaries, temperature);
	auto Erad_g = power * radEnergyFractions;
	// set floor
	for (int g = 0; g < nGroups_; ++g) {
		if (Erad_g[g] < Erad_floor_) {
			Erad_g[g] = Erad_floor_;
		}
	}
	return Erad_g;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real
{
	// by default, d emission/dT = 4 emission / T
	return 4. * radiation_constant_ * std::pow(temperature, 3);
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeMultiGroup(
    amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>
{
	// by default, d emission/dT = 4 emission / T
	auto radEnergyFractions = ComputePlanckEnergyFractions(boundaries, temperature);
	double d_power_dt = 4. * radiation_constant_ * std::pow(temperature, 3);
	return d_power_dt * radEnergyFractions;
}

// Linear equation solver for matrix with non-zeros at the first row, first column, and diagonal only.
// solve the linear system
//   [a00 a0i] [x0] = [y0]
//   [ai0 aii] [xi]   [yi]
// for x0 and xi, where a0i = (a01, a02, a03, ...); ai0 = (a10, a20, a30, ...); aii = (a11, a22, a33, ...), xi = (x1, x2, x3, ...), yi = (y1, y2, y3, ...)
template <typename problem_t>
AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqs(const double a00, const quokka::valarray<double, nGroups_> &a0i,
								const quokka::valarray<double, nGroups_> &ai0, const quokka::valarray<double, nGroups_> &aii,
								const double &y0, const quokka::valarray<double, nGroups_> &yi, double &x0,
								quokka::valarray<double, nGroups_> &xi)
{
	auto ratios = a0i / aii;
	x0 = (-sum(ratios * yi) + y0) / (-sum(ratios * ai0) + a00);
	xi = (yi - ai0 * x0) / aii;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::Solve3x3matrix(const double C00, const double C01, const double C02, const double C10, const double C11,
								const double C12, const double C20, const double C21, const double C22, const double Y0,
								const double Y1, const double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>
{
	// Solve the 3x3 matrix equation: C * X = Y under the assumption that only the diagonal terms
	// are guaranteed to be non-zero and are thus allowed to be divided by.

	auto E11 = C11 - C01 * C10 / C00;
	auto E12 = C12 - C02 * C10 / C00;
	auto E21 = C21 - C01 * C20 / C00;
	auto E22 = C22 - C02 * C20 / C00;
	auto Z1 = Y1 - Y0 * C10 / C00;
	auto Z2 = Y2 - Y0 * C20 / C00;
	auto X2 = (Z2 - Z1 * E21 / E11) / (E22 - E12 * E21 / E11);
	auto X1 = (Z1 - E12 * X2) / E11;
	auto X0 = (Y0 - C01 * X1 - C02 * X2) / C00;

	return std::make_tuple(X0, X1, X2);
}

template <typename problem_t>
void RadSystem<problem_t>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time)
{
	// do nothing -- user implemented
}


template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEddingtonFactor(double f_in) -> double
{
	// f is the reduced flux == |F|/cE.
	// compute Levermore (1984) closure [Eq. 25]
	// the is the M1 closure that is derived from Lorentz invariance
	const double f = clamp(f_in, 0., 1.); // restrict f to be within [0, 1]
	const double f_fac = std::sqrt(4.0 - 3.0 * (f * f));
	const double chi = (3.0 + 4.0 * (f * f)) / (5.0 + 2.0 * f_fac);

#if 0 // NOLINT
      // compute Minerbo (1978) closure [piecewise approximation]
      // (For unknown reasons, this closure tends to work better
      // than the Levermore/Lorentz closure on the Su & Olson 1997 test.)
	const double chi = (f < 1. / 3.) ? (1. / 3.) : (0.5 - f + 1.5 * f*f);
#endif

	return chi;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeEddingtonTensor(const double fx, const double fy, const double fz) -> std::array<std::array<double, 3>, 3>
{
	// Compute the radiation pressure tensor

	// AMREX_ASSERT(f < 1.0); // there is sometimes a small (<1%) flux
	// limiting violation when using P1 AMREX_ASSERT(f_R < 1.0);

	auto f = std::sqrt(fx * fx + fy * fy + fz * fz);
	std::array<amrex::Real, 3> fvec = {fx, fy, fz};

	// angle between interface and radiation flux \hat{n}
	// If direction is undefined, just drop direction-dependent
	// terms.
	std::array<amrex::Real, 3> n{};

	for (int ii = 0; ii < 3; ++ii) {
		n[ii] = (f > 0.) ? (fvec[ii] / f) : 0.;
	}

	// compute radiation pressure tensors
	const double chi = RadSystem<problem_t>::ComputeEddingtonFactor(f);

	AMREX_ASSERT((chi >= 1. / 3.) && (chi <= 1.0)); // NOLINT

	// diagonal term of Eddington tensor
	const double Tdiag = (1.0 - chi) / 2.0;

	// anisotropic term of Eddington tensor (in the direction of the
	// rad. flux)
	const double Tf = (3.0 * chi - 1.0) / 2.0;

	// assemble Eddington tensor
	std::array<std::array<double, 3>, 3> T{};

	for (int ii = 0; ii < 3; ++ii) {
		for (int jj = 0; jj < 3; ++jj) {
			const double delta_ij = (ii == jj) ? 1 : 0;
			T[ii][jj] = Tdiag * delta_ij + Tf * (n[ii] * n[jj]);
		}
	}

	return T;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeRadPressure(const double erad, const double Fx, const double Fy, const double Fz, const double fx,
							       const double fy, const double fz) -> RadPressureResult
{
	// Compute the radiation pressure tensor and the maximum signal speed and return them as a struct.

	// check that states are physically admissible
	AMREX_ASSERT(erad > 0.0);

	// Compute the Eddington tensor
	auto T = ComputeEddingtonTensor(fx, fy, fz);

	// frozen Eddington tensor approximation, following Balsara
	// (1999) [JQSRT Vol. 61, No. 5, pp. 617–627, 1999], Eq. 46.
	double Tnormal = NAN;
	if constexpr (DIR == FluxDir::X1) {
		Tnormal = T[0][0];
	} else if constexpr (DIR == FluxDir::X2) {
		Tnormal = T[1][1];
	} else if constexpr (DIR == FluxDir::X3) {
		Tnormal = T[2][2];
	}

	// compute fluxes F_L, F_R
	// T_nx, T_ny, T_nz indicate components where 'n' is the direction of the
	// face normal. F_n is the radiation flux component in the direction of the
	// face normal
	double Fn = NAN;
	double Tnx = NAN;
	double Tny = NAN;
	double Tnz = NAN;

	if constexpr (DIR == FluxDir::X1) {
		Fn = Fx;

		Tnx = T[0][0];
		Tny = T[0][1];
		Tnz = T[0][2];
	} else if constexpr (DIR == FluxDir::X2) {
		Fn = Fy;

		Tnx = T[1][0];
		Tny = T[1][1];
		Tnz = T[1][2];
	} else if constexpr (DIR == FluxDir::X3) {
		Fn = Fz;

		Tnx = T[2][0];
		Tny = T[2][1];
		Tnz = T[2][2];
	}

	AMREX_ASSERT(Fn != NAN);
	AMREX_ASSERT(Tnx != NAN);
	AMREX_ASSERT(Tny != NAN);
	AMREX_ASSERT(Tnz != NAN);

	RadPressureResult result{};
	result.F = {Fn, Tnx * erad, Tny * erad, Tnz * erad};
	// It might be possible to remove this 0.1 floor without affecting the code. I tried and only the 3D RadForce failed (causing S_L = S_R = 0.0 and F[0] =
	// NAN). Read more on https://github.com/quokka-astro/quokka/pull/582 .
	result.S = std::max(0.1, std::sqrt(Tnormal));

	return result;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> Real
{
	return NAN;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEnergyMeanOpacity(const double rho, const double Tgas) -> Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto
RadSystem<problem_t>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							   const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = NAN;
		exponents_and_values[1][g] = NAN;
	}
	return exponents_and_values;
}

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> amrex::GpuArray<double, nGroups_>
{
	// Compute the exponents for the radiation energy density, radiation flux, radiation pressure, or Planck function.

	// Note: Could save some memory by using bin_center_previous and bin_center_current
	amrex::GpuArray<double, nGroups_> bin_center{};
	amrex::GpuArray<double, nGroups_> quant_mean{};
	amrex::GpuArray<double, nGroups_ - 1> logslopes{};
	amrex::GpuArray<double, nGroups_> exponents{};
	for (int g = 0; g < nGroups_; ++g) {
		bin_center[g] = std::sqrt(boundaries[g] * boundaries[g + 1]);
		quant_mean[g] = quant[g] / (boundaries[g + 1] - boundaries[g]);
		if (g > 0) {
			AMREX_ASSERT(bin_center[g] > bin_center[g - 1]);
			if (quant_mean[g] == 0.0 && quant_mean[g - 1] == 0.0) {
				logslopes[g - 1] = 0.0;
			} else if (quant_mean[g - 1] * quant_mean[g] <= 0.0) {
				if (quant_mean[g] > quant_mean[g - 1]) {
					logslopes[g - 1] = inf;
				} else {
					logslopes[g - 1] = -inf;
				}
			} else {
				logslopes[g - 1] = std::log(std::abs(quant_mean[g] / quant_mean[g - 1])) / std::log(bin_center[g] / bin_center[g - 1]);
			}
			AMREX_ASSERT(!std::isnan(logslopes[g - 1]));
		}
	}

	for (int g = 0; g < nGroups_; ++g) {
		if (g == 0) {
			if constexpr (!special_edge_bin_slopes) {
				exponents[g] = -1.0;
			} else {
				exponents[g] = 2.0;
			}
		} else if (g == nGroups_ - 1) {
			if constexpr (!special_edge_bin_slopes) {
				exponents[g] = -1.0;
			} else {
				exponents[g] = -4.0;
			}
		} else {
			exponents[g] = minmod_func(logslopes[g - 1], logslopes[g]);
		}
		AMREX_ASSERT(!std::isnan(exponents[g]));
	}

	if constexpr (PPL_free_slope_st_total) {
		int peak_idx = 0; // index of the peak of logslopes
		for (; peak_idx < nGroups_; ++peak_idx) {
			if (peak_idx == nGroups_ - 1) {
				peak_idx += 0;
				break;
			}
			if (exponents[peak_idx] >= 0.0 && exponents[peak_idx + 1] < 0.0) {
				break;
			}
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(peak_idx < nGroups_ - 1,
						 "Peak index not found. Here peak_index is the index at which the exponent changes its sign.");
		double quant_sum = 0.0;
		double part_sum = 0.0;
		for (int g = 0; g < nGroups_; ++g) {
			quant_sum += quant[g];
			if (g == peak_idx) {
				continue;
			}
			part_sum += exponents[g] * quant[g];
		}
		if (quant[peak_idx] > 0.0 && quant_sum > 0.0) {
			exponents[peak_idx] = (-quant_sum - part_sum) / quant[peak_idx];
			AMREX_ASSERT(!std::isnan(exponents[peak_idx]));
		}
	}
	return exponents;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto
RadSystem<problem_t>::ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
					      amrex::GpuArray<double, nGroups_> const &radBoundaryRatios,
					      amrex::GpuArray<double, nGroups_> const &alpha_quant) -> quokka::valarray<double, nGroups_>
{
	amrex::GpuArray<double, nGroups_ + 1> const &alpha_kappa = kappa_expo_and_lower_value[0];
	amrex::GpuArray<double, nGroups_ + 1> const &kappa_lower = kappa_expo_and_lower_value[1];

	quokka::valarray<double, nGroups_> kappa{};
	for (int g = 0; g < nGroups_; ++g) {
		double alpha = alpha_quant[g] + 1.0;
		if (alpha > 100.) {
			kappa[g] = kappa_lower[g] * std::pow(radBoundaryRatios[g], kappa_expo_and_lower_value[0][g]);
			continue;
		}
		if (alpha < -100.) {
			kappa[g] = kappa_lower[g];
			continue;
		}
		double part1 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part1 = std::log(radBoundaryRatios[g]);
		} else {
			part1 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		alpha += alpha_kappa[g];
		double part2 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part2 = std::log(radBoundaryRatios[g]);
		} else {
			part2 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		kappa[g] = kappa_lower[g] / part1 * part2;
		AMREX_ASSERT(!std::isnan(kappa[g]));
	}
	return kappa;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::PlanckFunction(const double nu, const double T) -> double
{
	// returns 4 pi B(nu) / c
	double const coeff = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * T);
	double const x = coeff * nu;
	if (x > 100.) {
		return 0.0;
	}
	double planck_integral = NAN;
	if (x <= 1.0e-10) {
		// Taylor series
		planck_integral = x * x - x * x * x / 2.;
	} else {
		planck_integral = std::pow(x, 3) / (std::exp(x) - 1.0);
	}
	return coeff / (std::pow(PI, 4) / 15.0) * (radiation_constant_ * std::pow(T, 4)) * planck_integral;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeDiffusionFluxMeanOpacity(
    const quokka::valarray<double, nGroups_> kappaPVec, const quokka::valarray<double, nGroups_> kappaEVec,
    const quokka::valarray<double, nGroups_> fourPiBoverC, const amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
    const amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge, const amrex::GpuArray<double, nGroups_ + 1> kappa_slope) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaF{};
	for (int g = 0; g < nGroups_; ++g) {
		// kappaF[g] = 4. / 3. * kappaPVec[g] * fourPiBoverC[g] + 1. / 3. * kappa_slope[g] * kappaPVec[g] * fourPiBoverC[g] - 1. / 3. *
		// delta_nu_kappa_B_at_edge[g];
		kappaF[g] = (kappaPVec[g] + 1. / 3. * kappaEVec[g]) * fourPiBoverC[g] +
			    1. / 3. * (kappa_slope[g] * kappaEVec[g] * fourPiBoverC[g] - delta_nu_kappa_B_at_edge[g]);
		auto const denom = 4. / 3. * fourPiBoverC[g] - 1. / 3. * delta_nu_B_at_edge[g];
		if (denom <= 0.0) {
			AMREX_ASSERT(kappaF[g] == 0.0);
			kappaF[g] = 0.0;
		} else {
			kappaF[g] /= denom;
		}
	}
	return kappaF;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
									 amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa_center{};
	for (int g = 0; g < nGroups_; ++g) {
		kappa_center[g] =
		    kappa_expo_and_lower_value[1][g] * std::pow(rad_boundaries[g + 1] / rad_boundaries[g], 0.5 * kappa_expo_and_lower_value[0][g]);
	}
	return kappa_center;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxInDiffusionLimit(const amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, const double T,
									     const double vel) -> amrex::GpuArray<double, nGroups_>
{
	double const coeff = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * T);
	amrex::GpuArray<double, nGroups_ + 1> edge_values{};
	amrex::GpuArray<double, nGroups_> flux{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		auto x = coeff * rad_boundaries[g];
		edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x) - 1. / 3. * x * (std::pow(x, 3) / (std::exp(x) - 1.0)) / gInf;
		// test: reproduce the Planck function
		// edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x);
	}
	for (int g = 0; g < nGroups_; ++g) {
		flux[g] = vel * radiation_constant_ * std::pow(T, 4) * (edge_values[g + 1] - edge_values[g]);
	}
	return flux;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeDustTemperature(double const T_gas, double const T_d_init, double const rho,
									quokka::valarray<double, nGroups_> const &Erad, double dustGasCoeff,
									amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
									amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios) -> double
{
	quokka::valarray<double, nGroups_> kappaPVec{};
	quokka::valarray<double, nGroups_> kappaEVec{};

	const double num_density = rho / mean_molecular_mass_;
	const double Lambda_compare = dustGasCoeff * num_density * num_density * std::sqrt(T_gas) * T_gas;

	amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
	for (int g = 0; g < nGroups_; ++g) {
		alpha_quant_minus_one[g] = -1.0;
	}

	// solve for dust temperature T_d using Newton iteration
	double T_d = T_d_init;
	const double lambda_rel_tol = 1.0e-8;
	const int max_ite_td = 100;
	int ite_td = 0;
	for (; ite_td < max_ite_td; ++ite_td) {
		quokka::valarray<double, nGroups_> fourPiBoverC{};

		if constexpr (nGroups_ == 1) {
			fourPiBoverC[0] = ComputeThermalRadiationSingleGroup(T_d);
		} else {
			fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, rad_boundaries);
		}

		if constexpr (opacity_model_ == OpacityModel::single_group) {
			kappaPVec[0] = ComputePlanckOpacity(rho, T_d);
			kappaEVec[0] = ComputeEnergyMeanOpacity(rho, T_d);
		} else {
			const auto kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T_d);
			if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
				for (int g = 0; g < nGroups_; ++g) {
					kappaPVec[g] = kappa_expo_and_lower_value[1][g];
					kappaEVec[g] = kappa_expo_and_lower_value[1][g];
				}
			} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) {
				kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, alpha_quant_minus_one);
				kappaEVec = kappaPVec;
			} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
				const auto alpha_B = ComputeRadQuantityExponents(fourPiBoverC, rad_boundaries);
				const auto alpha_E = ComputeRadQuantityExponents(Erad, rad_boundaries);
				kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, alpha_B);
				kappaEVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, alpha_E);
			}
		}
		AMREX_ASSERT(!kappaPVec.hasnan());
		AMREX_ASSERT(!kappaEVec.hasnan());

		const double LHS = c_light_ * rho * sum(kappaEVec * Erad - kappaPVec * fourPiBoverC) +
				   dustGasCoeff * num_density * num_density * std::sqrt(T_gas) * (T_gas - T_d);

		if (std::abs(LHS) < lambda_rel_tol * std::abs(Lambda_compare)) {
			break;
		}

		quokka::valarray<double, nGroups_> d_fourpib_over_c_d_t{};
		if constexpr (nGroups_ == 1) {
			d_fourpib_over_c_d_t[0] = ComputeThermalRadiationTempDerivativeSingleGroup(T_d);
		} else {
			d_fourpib_over_c_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, rad_boundaries);
		}
		const double dLHS_dTd = -c_light_ * rho * sum(kappaPVec * d_fourpib_over_c_d_t) - dustGasCoeff * num_density * num_density * std::sqrt(T_gas);
		const double delta_T_d = LHS / dLHS_dTd;
		T_d -= delta_T_d;

		if (ite_td > 0) {
			if (std::abs(delta_T_d) < lambda_rel_tol * std::abs(T_d)) {
				break;
			}
		}
	}

	AMREX_ASSERT_WITH_MESSAGE(ite_td < max_ite_td, "Newton iteration for dust temperature failed to converge.");
	if (ite_td >= max_ite_td) {
		T_d = -1.0;
	}
	return T_d;
}

#endif // RAD_SOURCE_TERMS_BASE_HPP_