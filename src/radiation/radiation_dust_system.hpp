// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RADIATION_DUST_SYSTEM_HPP_
#define RADIATION_DUST_SYSTEM_HPP_

#include <algorithm>

#include "radiation/radiation_system.hpp"

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> amrex::Real
{
	return 0.0;
}

// Compute the Jacobian of energy update equations for the gas-dust-radiation system. The result is a struct containing the following elements:
// J00: (0, 0) component of the Jacobian matrix. = d F0 / d Egas
// F0: (0) component of the residual. = Egas residual
// Fg_abs_sum: sum of the absolute values of the each component of Fg that has tau(g) > 0
// J0g: (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d F0 / d R_g
// Jg0: (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d Egas
// Jgg: (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d R_g
// Fg: (g) components of the residual, g = 1, 2, ..., nGroups. = Erad residual
template <typename problem_t>
AMREX_GPU_DEVICE auto
RadSystem<problem_t>::ComputeJacobianForGasAndDust(double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff,
						   quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src,
						   const double Q_dust, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v,
						   double /*lambda_gd_time_dt*/, quokka::valarray<double, nGroups_> const &kappaPoverE,
						   quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, const double num_den, const double dt)
    -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	const double cscale = c_light_ / c_hat_;

	// compute cooling/heating terms
	const auto cooling = DefineNetCoolingRate(T_gas, num_den) * dt;
	const auto cooling_derivative = DefineNetCoolingRateTempDerivative(T_gas, num_den) * dt;
	const double CR_heating = DefineCosmicRayHeatingRate(num_den) * dt;

	result.F0 = Egas_diff + cscale * sum(Rvec) + sum(cooling) - CR_heating - Q_dust;
	result.Fg = Erad_diff - (Rvec + Src);
	if constexpr (add_line_cooling_to_radiation_in_jac) {
		result.Fg -= (1.0 / cscale) * cooling;
	}
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		} else {
			result.Fg_abs_sum += std::abs(result.Fg[g] + Rvec[g]);
		}
	}

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;

	result.J00 = 1.0 + sum(cooling_derivative) / c_v;
	result.J0g.fillin(cscale);
	const double d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
	dEg_dT *= d_Td_d_T;
	const double dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));
	const auto rg = kappaPoverE * d_fourpiboverc_d_t * dTd_dRg;
	result.Jg0 = 1.0 / c_v * dEg_dT - (1 / cscale) * cooling_derivative - 1.0 / cscale * rg * result.J00;
	// Note that Fg is modified here, but it does not change Fg_abs_sum, which is used to check the convergence.
	result.Fg = result.Fg - 1.0 / cscale * rg * result.F0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
		}
	}

	return result;
}

// Compute the Jacobian of energy update equations for the gas-dust-radiation system with gas and dust decoupled. The result is a struct containing the
// following elements: J00: (0, 0) component of the Jacobian matrix. = d F0 / d T_d F0: (0) component of the residual. = sum_g R_g - lambda_gd_time_dt
// Fg_abs_sum: sum of the absolute values of the each component of Fg that has tau(g) > 0
// J0g: (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d F0 / d R_g
// Jg0: (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d T_d
// Jgg: (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d R_g
// Fg: (g) components of the residual, g = 1, 2, ..., nGroups. = Erad residual
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustDecoupled(
    double /*T_gas*/, double /*T_d*/, double /*Egas_diff*/, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec,
    quokka::valarray<double, nGroups_> const &Src, const double Q_dust, double /*coeff_n*/, quokka::valarray<double, nGroups_> const &tau, double /*c_v*/,
    double lambda_gd_time_dt, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t)
    -> JacobianResult<problem_t>
{
	constexpr double c_hat_over_c = RadSystem_Traits<problem_t>::c_hat_over_c;
	JacobianResult<problem_t> result;

	// lambda_gd_time_dt and Rvec are c_hat/c scaled. So, we are solving
	// sum(R_g) - lambda_gd_time_dt - c_hat / c * Q_dust = 0. We could have equivalently moved the c_hat / c factor to the other terms.
	result.F0 = -lambda_gd_time_dt + sum(Rvec) - c_hat_over_c * Q_dust;
	result.Fg = Erad_diff - (Rvec + Src);
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		}
	}

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;

	result.J00 = 0.0;
	result.J0g.fillin(1.0);
	result.Jg0 = dEg_dT;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
		}
	}

	return result;
}

// Compute the Jacobian of energy update equations for the gas-dust-radiation system. The result is a struct containing the following elements:
// J00: (0, 0) component of the Jacobian matrix. = d F0 / d Egas
// F0: (0) component of the residual. = Egas residual
// Fg_abs_sum: sum of the absolute values of the each component of Fg that has tau(g) > 0
// J0g: (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d F0 / d R_g
// Jg0: (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d Egas
// Jgg: (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d R_g
// Fg: (g) components of the residual, g = 1, 2, ..., nGroups. = Erad residual
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustWithPE(
    double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad, quokka::valarray<double, nGroups_> const &Erad0,
    double PE_heating_energy_derivative, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, const double Q_dust,
    const double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double /*lambda_gd_time_dt*/,
    quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double const num_den, double const dt)
    -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	const double cscale = c_light_ / c_hat_;

	// compute cooling/heating terms
	const auto cooling = DefineNetCoolingRate(T_gas, num_den) * dt;
	const auto cooling_derivative = DefineNetCoolingRateTempDerivative(T_gas, num_den) * dt;
	const double CR_heating = DefineCosmicRayHeatingRate(num_den) * dt;

	result.F0 = Egas_diff + cscale * sum(Rvec) + sum(cooling) - PE_heating_energy_derivative * Erad[nGroups_ - 1] - CR_heating - Q_dust;
	result.Fg = Erad - Erad0 - (Rvec + Src);
	if constexpr (add_line_cooling_to_radiation_in_jac) {
		result.Fg -= (1.0 / cscale) * cooling;
	}
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		} else {
			result.Fg_abs_sum += std::abs(result.Fg[g] + Rvec[g]);
		}
	}

	// const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, radBoundaries_g_copy);
	AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto d_Eg_d_Rg = -1.0 * kappaPoverE;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			d_Eg_d_Rg[g] = -LARGE;
		} else {
			d_Eg_d_Rg[g] /= tau[g];
		}
	}

	result.J00 = 1.0 + sum(cooling_derivative) / c_v;
	result.J0g.fillin(cscale);
	result.J0g[nGroups_ - 1] -= PE_heating_energy_derivative * d_Eg_d_Rg[nGroups_ - 1];
	const double d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
	const auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t * d_Td_d_T;
	const double dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));
	const auto rg = kappaPoverE * d_fourpiboverc_d_t * dTd_dRg;
	result.Jg0 = 1.0 / c_v * dEg_dT - (1 / cscale) * cooling_derivative - 1.0 / cscale * rg * result.J00;
	// Note that Fg is modified here, but it does not change Fg_abs_sum, which is used to check the convergence.
	result.Fg = result.Fg - 1.0 / cscale * rg * result.F0;
	result.Jgg = d_Eg_d_Rg + (-1.0);
	result.Jgg[nGroups_ - 1] += rg[nGroups_ - 1] - (rg[nGroups_ - 1] / cscale) * PE_heating_energy_derivative * d_Eg_d_Rg[nGroups_ - 1];
	result.Jg1 = rg - 1.0 / cscale * rg * result.J0g[nGroups_ - 1]; // note that this is the (nGroups_ - 1)th column, except for the (nGroups_ - 1)th row

	return result;
}

// Linear equation solver for matrix with non-zeros at the first row, first column, and diagonal only.
// solve the linear system
//   [a00 a0i] [x0] = [y0]
//   [ai0 aii] [xi]   [yi]
// for x0 and xi, where a0i = (a01, a02, a03, ...); ai0 = (a10, a20, a30, ...); aii = (a11, a22, a33, ...), xi = (x1, x2, x3, ...), yi = (y1, y2, y3, ...)
template <typename problem_t>
AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqsWithLastColumn(JacobianResult<problem_t> const &jacobian, double &x0,
									      quokka::valarray<double, nGroups_> &xi)
{
	// Note that the following routine only works when the FUV group is the last group, i.e., pe_index = nGroups_ - 1

	// note that jacobian.Jgg[pe_index] is the right value for a[pe_index][pe_index], while jacobian.Jg1[pe_index] is NOT.
	const int pe_index = nGroups_ - 1;
	const auto ratios = jacobian.J0g / jacobian.Jgg;

	const auto a00_new = jacobian.J00 - sum(ratios * jacobian.Jg0);
	const auto y0_new = jacobian.F0 - sum(ratios * jacobian.Fg);
	auto a01_new = jacobian.J0g[pe_index] - sum(ratios * jacobian.Jg1);
	// re-accounting for the pe_index'th element of jacobian.Jg1
	a01_new = a01_new + ratios[pe_index] * jacobian.Jg1[pe_index] - ratios[pe_index] * jacobian.Jgg[pe_index];
	const auto a10 = jacobian.Jg0[pe_index];
	const auto a11 = jacobian.Jgg[pe_index];
	const auto y1 = jacobian.Fg[pe_index];
	// solve linear equations [[a00_new, a01_new], [a10, a11]] [[x0], [xi[pe_index]]] = [y0_new, y1]
	x0 = (y0_new - a01_new / a11 * y1) / (a00_new - a01_new / a11 * a10);
	const auto x1 = (y1 - a10 * x0) / a11;
	xi[pe_index] = x1;
	// xi = (jacobian.Fg - jacobian.Jg0 * x0) / jacobian.Jgg;
	for (int g = 0; g < pe_index; ++g) {
		xi[g] = (jacobian.Fg[g] - jacobian.Jg0[g] * x0 - jacobian.Jg1[g] * x1) / jacobian.Jgg[g];
	}
	x0 *= -1.0;
	xi *= -1.0;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasDustRadiationEnergyExchange(
    double const Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double const rho, double const coeff_n, double const dt,
    amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work,
    quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src, const double Q_dust,
    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, double resid_tol, double rel_change_tol, double /*tempFloor*/, int *p_iteration_counter,
    int *p_iteration_failure_counter, NewtonOuterState<problem_t> &outer_state)
    -> NewtonIterationResult<problem_t> // NOSONAR: High cognitive complexity is expected for this numerical solver
{
	// 1. Compute energy exchange

	// BEGIN NEWTON-RAPHSON LOOP
	// Define the source term: S = dt chat gamma rho (kappa_P B - kappa_E E) + dt chat c^-2 gamma rho kappa_F v * F_i, where gamma =
	// 1 / sqrt(1 - v^2 / c^2) is the Lorentz factor. Solve for the new radiation energy and gas internal energy using a
	// Newton-Raphson method using the base variables (Egas, D_0, D_1,
	// ...), where D_i = R_i / tau_i^(t) and tau_i^(t) = dt * chat * gamma * rho * kappa_{P,i}^(t) is the optical depth across chat
	// * dt for group i at time t. Compared with the old base (Egas, Erad_0, Erad_1, ...), this new base is more stable and
	// converges faster. Furthermore, the PlanckOpacityTempDerivative term is not needed anymore since we assume d/dT (kappa_P /
	// kappa_E) = 0 in the calculation of the Jacobian. Note that this assumption only affects the convergence rate of the
	// Newton-Raphson iteration and does not affect the result at all once the iteration is converged.
	//
	// The Jacobian of F(E_g, D_i) is
	//
	// dF_G / dE_g = 1
	// dF_G / dD_i = c / chat * tau0_i
	// dF_{D,i} / dE_g = 1 / (chat * C_v) * (kappa_{P,i} / kappa_{E,i}) * d/dT (4 \pi B_i)
	// dF_{D,i} / dD_i = - (1 / (chat * dt * rho * kappa_{E,i}) + 1) * tau0_i = - ((1 / tau_i)(kappa_Pi / kappa_Ei) + 1) * tau0_i

	const double c = c_light_; // make a copy of c_light_ to avoid compiler error "undefined in device code"
	const double chat = c_hat_;
	const double cscale = c / chat;
	constexpr double c_hat_over_c = RadSystem_Traits<problem_t>::c_hat_over_c;
	// Only the thermal groups take part in the Newton solve. The chemical bands have zero thermal-coupling opacity
	// (enforced by DefineThermalCouplingOpacityExponentsAndLowerValues), so tau = 0 for them, their residual is zero
	// and so is their step: they are left out of every per-group Newton loop, keep the zero-initialised
	// tau/Rvec/work_local entries, and get their radiation energy from the tau = 0 branch after the loop.
	constexpr int nGT = nGroupsThermal_;

	int dust_model = 1;
	double T_d0 = NAN;
	double lambda_gd_times_dt = NAN;
	const double T_gas0 = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas0, massScalars);
	AMREX_ASSERT(T_gas0 >= 0.);
	double Etot0 = NAN;
	if (n_outer_iter == 0) {
		// rad-08: everything in this block depends only on the old state (Egas0, Erad0Vec, Src, Q_dust), not on the
		// work term, so it is computed in the first outer pass and reused by the later ones.
		T_d0 = ComputeDustTemperatureBateKeto(T_gas0, T_gas0, rho, Erad0Vec, coeff_n, dt, NAN, 0, Q_dust, rad_boundaries);
		AMREX_ASSERT_WITH_MESSAGE(T_d0 >= 0., "Dust temperature is negative!");
		if (T_d0 < 0.0) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
		}

		const double max_Gamma_gd = coeff_n * std::max(std::sqrt(T_gas0) * T_gas0, std::sqrt(T_d0) * T_d0);
		// A zero collisional coupling coefficient means the gas and the dust exchange no energy at all, which is
		// exactly what the decoupled model describes, so select it on that ground alone. The comparison below
		// cannot be relied on to do it: it flips sense once Egas0 is negative, and the coupled branch it then
		// selects divides by coeff_n, filling the state with NaNs that surface later as a spurious
		// "Newton-Raphson iteration failed to converge".
		if (!(coeff_n > 0.0) || (cscale * max_Gamma_gd < ISM_Traits<problem_t>::gas_dust_coupling_threshold * Egas0)) {
			dust_model = 2;
			lambda_gd_times_dt = coeff_n * std::sqrt(T_gas0) * (T_gas0 - T_d0);
		}

		// const double Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src)) + Q_dust;
		if (dust_model == 1) {
			Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src)) + Q_dust;
		} else {
			// Only the radiation and collisional exchange supply energy to decoupled dust. Its equilibrium
			// Planck energy can greatly exceed the actual radiation energy in a thin cell; using that as a
			// residual scale can accept spurious emission and corrupt the subsequent radiation force.
			Etot0 = std::abs(lambda_gd_times_dt) + (sum(Erad0Vec) + sum(Src)) + Q_dust / cscale;
		}
		outer_state.dust_model = dust_model;
		outer_state.T_d0 = T_d0;
		outer_state.lambda_gd_times_dt = lambda_gd_times_dt;
		outer_state.Etot0 = Etot0;
		outer_state.have_Egas_post = false;
	} else {
		dust_model = outer_state.dust_model;
		T_d0 = outer_state.T_d0;
		lambda_gd_times_dt = outer_state.lambda_gd_times_dt;
		Etot0 = outer_state.Etot0;
	}

	double T_gas = NAN;
	double T_d = NAN;
	double delta_x = NAN;
	quokka::valarray<double, nGroups_> Rvec{};
	quokka::valarray<double, nGroups_> tau0{};	 // optical depth across c * dt at old state
	quokka::valarray<double, nGroups_> tau{};	 // optical depth across c * dt at new state
	quokka::valarray<double, nGroups_> work_local{}; // work term used in the Newton-Raphson iteration of the current outer iteration
	quokka::valarray<double, nGroups_> fourPiBoverC{};
	quokka::valarray<double, nGroups_> d_fourpiboverc_d_t{};
	double T_d_planck = NAN; // temperature at which fourPiBoverC and the opacity table were last evaluated
	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
	OpacityTerms<problem_t> opacity_terms{};

	// fill kappa_expo_and_lower_value with NAN to get warned when there are uninitialized values
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < nGroups_ + 1; ++j) {
			kappa_expo_and_lower_value[i][j] = NAN;
		}
	}

	if constexpr (!(opacity_model_ == OpacityModel::piecewise_constant_opacity)) {
		for (int g = 0; g < nGroups_; ++g) {
			rad_boundary_ratios[g] = rad_boundaries[g + 1] / rad_boundaries[g];
		}
	}

	// define a list of alpha_quant for the model PPL_opacity_fixed_slope_spectrum
	amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
	if constexpr ((opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) ||
		      (gamma_ == 1.0 && opacity_model_ == OpacityModel::PPL_opacity_full_spectrum)) {
		if constexpr (!special_edge_bin_slopes) {
			for (int g = 0; g < nGroups_; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		} else {
			alpha_quant_minus_one[0] = 2.0;
			alpha_quant_minus_one[nGroups_ - 1] = -4.0;
			for (int g = 1; g < nGroups_ - 1; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		}
	}

	// rad-08 warm start: a later outer pass starts from the previous pass's converged Newton state (Egas, Erad, T_d)
	// instead of the old state. The work term of the current pass enters through Rvec at n == 0 as before.
	bool warm_start = (n_outer_iter > 0) && outer_state.warm;
	int n_warm_iters = 0; // iterations spent in a warm-started attempt that had to be restarted cold

	double Egas_guess = warm_start ? outer_state.Egas : Egas0;
	auto EradVec_guess = warm_start ? outer_state.EradVec : Erad0Vec;

	double Egas_guess_prev = Egas_guess;
	auto EradVec_guess_prev = EradVec_guess;

	const double H_num_den = ComputeNumberDensityH(rho, massScalars);

	T_gas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
	AMREX_ASSERT(T_gas >= 0.);

	const int maxIter = 100;
	// Rebasing an optically thin group onto Erad_g (see newton_erad_base_tau_threshold) is applied only in
	// the decoupled branch. With dust_model == 1 the dust temperature is itself a function of sum(Rvec), and
	// ComputeJacobianForGasAndDust has already eliminated that coupling in the R_g unknowns, so the columns
	// are no longer a plain change of variable away from the Erad_g ones.
	const bool rebase_thin = (dust_model == 2) && !use_D_as_base;
	// Adaptive damping state. In the decoupled-dust branch the Newton iteration can oscillate with growing
	// amplitude rather than converge, so the step is shortened whenever the radiation residual fails to
	// decrease, and allowed to grow back towards a full step when it does. See newton_damping_* below.
	double relax = 1.0;
	double Fg_abs_sum_prev = std::numeric_limits<double>::max();
	double delta_x_prev = 0.0;
	// worsen_streak counts consecutive iterations that fail to reduce the residual, so a cut fires only once it
	// reaches newton_damping_patience; cooldown_remaining then blocks further cuts or growth for a few iterations.
	int worsen_streak = 0;
	int cooldown_remaining = 0;
	int n = 0;
	// rad-08: if a warm-started attempt fails (no convergence within maxIter, or a negative dust temperature), restart
	// the pass from the old state with a fresh iteration budget, i.e. exactly as the solver ran before warm starts.
	auto restart_cold = [&]() {
		n_warm_iters += n + 1;
		warm_start = false;
		n = -1; // incremented to 0 by the loop
		Egas_guess = Egas0;
		EradVec_guess = Erad0Vec;
		Egas_guess_prev = Egas_guess;
		EradVec_guess_prev = EradVec_guess;
		T_gas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
		relax = 1.0;
		Fg_abs_sum_prev = std::numeric_limits<double>::max();
		delta_x_prev = 0.0;
		worsen_streak = 0;
		cooldown_remaining = 0;
	};
	for (; n < maxIter; ++n) { // NOSONAR
		// if relative change is within tol, break
		if (rel_change_tol > 0.0 && n > 0) {
			const double Erad_tot_guess_prev = sum(EradVec_guess_prev);
			const auto Erad_rel_diff = abs(EradVec_guess - EradVec_guess_prev);
			const auto Egas_rel_diff = std::abs(Egas_guess - Egas_guess_prev);

			if ((sum(Erad_rel_diff) <= rel_change_tol * Erad_tot_guess_prev) && (Egas_rel_diff <= rel_change_tol * Egas_guess_prev)) {
				break;
			}
		}

		Egas_guess_prev = Egas_guess;
		EradVec_guess_prev = EradVec_guess;

		// 1. Compute dust temperature
		// If the dust model is turned off, ComputeDustTemperature should be a function that returns T_gas.

		if (n > 0) {
			T_gas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
			AMREX_ASSERT(T_gas >= 0.);
		}

		if (dust_model == 1) {
			if (n == 0) {
				T_d = warm_start ? outer_state.T_d : T_d0;
			} else {
				double R_sum = 0.0;
				for (int g = 0; g < nGT; ++g) {
					R_sum += Rvec[g];
				}
				T_d = T_gas - (R_sum - Q_dust / cscale) / (coeff_n * std::sqrt(T_gas));
				AMREX_ASSERT_WITH_MESSAGE(warm_start || T_d >= 0.,
							  "Dust temperature is negative! Consider increasing ISM_Traits::gas_dust_coupling_threshold");
			}
		} else {
			if (n == 0) {
				T_d = warm_start ? outer_state.T_d : T_d0;
			}
		}
		if (T_d < 0.0) {
			if (warm_start) {
				restart_cold();
				continue;
			}
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
		}

		// 2. Compute kappaP and kappaE at dust temperature

		// one fused pass over the group edges gives both 4 pi B / c and its temperature derivative
		ComputeThermalRadiationAndTempDerivativeMultiGroup(T_d, rad_boundaries, fourPiBoverC, d_fourpiboverc_d_t);
		T_d_planck = T_d;

		opacity_terms = ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios, fourPiBoverC, EradVec_guess, n,
								     opacity_terms.alpha_E, opacity_terms.alpha_P);

		if (n == 0) {
			// Compute kappaF and the delta_nu_kappa_B term. kappaF is used to compute the work term.
			// Will update opacity_terms in place
			ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms); // update opacity_terms in place
		}

		// 3. In the first loop, calculate kappaF, work, tau0, R

		if (n == 0) {

			if constexpr ((beta_order_ == 1) && (include_work_term_in_source)) {
				// compute the work term at the old state
				// const double gamma = 1.0 / sqrt(1.0 - vsqr / (c * c));
				if (n_outer_iter == 0) {
					for (int g = 0; g < nGT; ++g) {
						if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
							work_local[g] = vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * dt;
						} else {
							kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T_d);
							work_local[g] = vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * dt *
									(1.0 + kappa_expo_and_lower_value[0][g]);
						}
					}
				} else {
					// If n_outer_iter > 0, use the work term from the previous outer iteration, which is passed as the parameter 'work'
					work_local = work;
				}
			} else {
				work_local.fillin(0.0);
			}

			// With a warm start, this is the R implied by the previous pass's (T_d, Erad) and the current work term (rad-08; thermal groups only, rad-10).
			for (int g = 0; g < nGT; ++g) {
				tau0[g] = dt * rho * opacity_terms.kappaP[g] * chat;
				tau[g] = tau0[g];
				Rvec[g] = (fourPiBoverC[g] - EradVec_guess[g] / opacity_terms.kappaPoverE[g]) * tau0[g] + work_local[g];
			}
			if constexpr (use_D_as_base) {
				// tau0 is used as a scaling factor for Rvec
				for (int g = 0; g < nGT; ++g) {
					if (tau0[g] <= 1.0) {
						tau0[g] = 1.0;
					}
				}
			}
		} else { // in the second and later loops, calculate tau, then recover whichever of E and R is not
			 // the unknown for that group (see newton_erad_base_tau_threshold)
			for (int g = 0; g < nGT; ++g) {
				tau[g] = dt * rho * opacity_terms.kappaP[g] * chat;
			}
			for (int g = 0; g < nGT; ++g) {
				// If tau = 0.0, Erad_guess shouldn't change
				if (tau[g] > 0.0) {
					if (rebase_thin && (tau[g] < newton_erad_base_tau_threshold)) {
						// Erad_g is the unknown for this thin group; R_g follows from it without
						// cancellation
						Rvec[g] = (fourPiBoverC[g] - EradVec_guess[g] / opacity_terms.kappaPoverE[g]) * tau[g] + work_local[g];
					} else {
						EradVec_guess[g] = opacity_terms.kappaPoverE[g] * (fourPiBoverC[g] - (Rvec[g] - work_local[g]) / tau[g]);
					}
					if constexpr (force_rad_floor_in_iteration) {
						if (EradVec_guess[g] < 0.0) {
							Egas_guess -= cscale * (Erad_floor_ - EradVec_guess[g]);
							EradVec_guess[g] = Erad_floor_;
						}
					}
				}
			}
		}

		AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());
		const double c_v = ::quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

		const auto Egas_diff = Egas_guess - Egas0;

		// Matrix-free arrowhead Newton system. The Jacobian has non-zeros only in its first row (J0g), first column
		// (Jg0) and diagonal (Jgg), so the linear solve (SolveLinearEqs) needs only two sums over the groups,
		// sum_g J0g Fg / Jgg and sum_g J0g Jg0 / Jgg. Instead of materialising the JacobianResult arrays, the
		// per-group entries are formed by arrow_row(g) inside the loop that accumulates those sums, and formed again
		// in the loop that computes and applies the step of each group. This is the same algebra as
		// ComputeJacobianForGasAndDust / ComputeJacobianForGasAndDustDecoupled, the use_D_as_base scaling,
		// RebaseThinGroupsOntoErad and SolveLinearEqs, evaluated one group at a time.
		double F0 = NAN;	// (0) component of the residual
		double J00_rows = NAN;	// (0, 0) component of the Jacobian before the thin-group rebase (enters Jg0 when dust_model == 1)
		double d_Td_d_T = NAN;	// dust_model == 1 only
		double dTd_dRg = NAN;	// dust_model == 1 only
		quokka::valarray<double, nGroups_> cooling{};
		quokka::valarray<double, nGroups_> cooling_derivative{};
		if (dust_model == 1) {
			// compute cooling/heating terms
			cooling = DefineNetCoolingRate(T_gas, H_num_den) * dt;
			cooling_derivative = DefineNetCoolingRateTempDerivative(T_gas, H_num_den) * dt;
			const double CR_heating = DefineCosmicRayHeatingRate(H_num_den) * dt;
			F0 = Egas_diff + cscale * sum(Rvec) + sum(cooling) - CR_heating - Q_dust;
			J00_rows = 1.0 + sum(cooling_derivative) / c_v;
			d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
			dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));
		} else {
			// lambda_gd_times_dt and Rvec are c_hat/c scaled. So, we are solving
			// sum(R_g) - lambda_gd_time_dt - c_hat / c * Q_dust = 0.
			F0 = -lambda_gd_times_dt + sum(Rvec) - c_hat_over_c * Q_dust;
			J00_rows = 0.0;
		}

		struct ArrowRow {
			double J0g;	  // (0, g) component of the Jacobian
			double Jg0;	  // (g, 0) component of the Jacobian
			double Jgg;	  // (g, g) component of the Jacobian
			double Fg;	  // (g) component of the residual (as modified for dust_model == 1)
			double abs_resid; // contribution of group g to Fg_abs_sum
			double dJ00;	  // change of J00 from rebasing group g onto Erad_g
		};
		auto arrow_row = [&](int g) -> ArrowRow {
			ArrowRow r{};
			r.dJ00 = 0.0;
			const double Erad_diff_g = EradVec_guess[g] - Erad0Vec[g];
			double Fg_raw = Erad_diff_g - (Rvec[g] + Src[g]);
			// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this
			// assumption only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged
			// solution at all.
			const double dEg_dT_g = opacity_terms.kappaPoverE[g] * d_fourpiboverc_d_t[g];
			if (dust_model == 1) {
				if constexpr (add_line_cooling_to_radiation_in_jac) {
					Fg_raw -= (1.0 / cscale) * cooling[g];
				}
				r.abs_resid = (tau[g] > 0.0) ? std::abs(Fg_raw) : std::abs(Fg_raw + Rvec[g]);
				const double rg = dEg_dT_g * dTd_dRg;
				r.J0g = cscale;
				r.Jg0 = 1.0 / c_v * (dEg_dT_g * d_Td_d_T) - (1 / cscale) * cooling_derivative[g] - 1.0 / cscale * rg * J00_rows;
				// Note that Fg is modified here, but it does not change Fg_abs_sum, which is used to check the convergence.
				r.Fg = Fg_raw - 1.0 / cscale * rg * F0;
			} else {
				r.abs_resid = (tau[g] > 0.0) ? std::abs(Fg_raw) : 0.0;
				r.J0g = 1.0;
				r.Jg0 = dEg_dT_g;
				r.Fg = Fg_raw;
			}
			if (tau[g] <= 0.0) {
				r.Jgg = -std::numeric_limits<double>::infinity();
			} else {
				r.Jgg = -1.0 * opacity_terms.kappaPoverE[g] / tau[g] - 1.0;
			}
			if constexpr (use_D_as_base) {
				r.J0g = r.J0g * tau0[g];
				r.Jgg = r.Jgg * tau0[g];
			} else {
				if (rebase_thin && (tau[g] > 0.0) && (tau[g] < newton_erad_base_tau_threshold)) {
					// see RebaseThinGroupsOntoErad
					const double scale = -tau[g] / opacity_terms.kappaPoverE[g]; // dR_g / dErad_g
					r.dJ00 = r.J0g * scale * r.Jg0;
					r.Jgg *= scale;
					r.J0g *= scale;
					r.Jg0 *= scale;
				}
			}
			return r;
		};

		// First pass over the groups: residual norm and the two sums of the arrowhead solve.
		double J00 = J00_rows;
		double Fg_abs_sum = 0.0;
		double sum_ratio_Fg = 0.0;
		double sum_ratio_Jg0 = 0.0;
		for (int g = 0; g < nGT; ++g) {
			const ArrowRow r = arrow_row(g);
			Fg_abs_sum += r.abs_resid;
			J00 -= r.dJ00;
			const double ratio = r.J0g / r.Jgg;
			sum_ratio_Fg += ratio * r.Fg;
			sum_ratio_Jg0 += ratio * r.Jg0;
		}

		// Round-off floor on the radiation residual, as in SolveGasRadiationEnergyExchange: a group whose
		// unknown is R_g has its energy recovered from a cancelling difference, so |Fg| cannot fall below
		// the double-precision round-off of the larger operand and a purely relative test is unreachable.
		double Fg_roundoff = 0.0;
		for (int g = 0; g < nGT; ++g) {
			if (tau[g] > 0.0) {
				// A group rebased onto Erad_g reaches its residual without that cancellation, so its floor
				// is just the round-off of the terms of the residual itself.
				const double operand =
				    (rebase_thin && (tau[g] < newton_erad_base_tau_threshold))
					? std::max(std::max(std::abs(EradVec_guess[g]), std::abs(Rvec[g])), std::max(std::abs(Erad0Vec[g]), std::abs(Src[g])))
					: std::max(fourPiBoverC[g], EradVec_guess[g]);
				Fg_roundoff += std::numeric_limits<double>::epsilon() * operand;
			}
		}

		// check relative convergence of the residuals, or that the radiation residual has bottomed out at
		// the round-off floor and cannot be reduced any further
		if ((std::abs(F0 / Etot0) < resid_tol) &&
		    ((cscale * Fg_abs_sum / Etot0 < resid_tol) || (Fg_abs_sum < newton_resid_roundoff_factor * Fg_roundoff))) {
			break;
		}

		// update variables: x0 of the arrowhead solve (SolveLinearEqs)
		delta_x = (sum_ratio_Fg - F0) / (-sum_ratio_Jg0 + J00);
		AMREX_ASSERT(!std::isnan(delta_x));

		// Update independent variables (Egas_guess, Rvec). The step of group g, xi_g = (-Fg - Jg0 x0) / Jgg, is
		// computed and applied in the second pass over the groups below.
		// enable_dE_constrain is used to prevent the gas temperature from dropping/increasing below/above the radiation
		// temperature
		if (dust_model == 2) {
			if (n > 0) {
				// The bound is copied into a local first: std::max binds its argument by reference, and
				// that would ODR-use the namespace-scope constexpr constant, which nvcc does not make
				// available in device code.
				const double damping_min = newton_damping_min;
				const bool worsened = Fg_abs_sum > Fg_abs_sum_prev;
				worsen_streak = worsened ? worsen_streak + 1 : 0;
				if (cooldown_remaining > 0) {
					--cooldown_remaining;
				} else if (worsen_streak >= newton_damping_patience) {
					// Patience-gated cut: a lone bad iteration (which an oscillating -- not diverging --
					// residual produces regularly) does not by itself shorten the step, only a sustained
					// run of non-improving iterations does, whatever the period of the oscillation.
					relax *= newton_damping_down;
					worsen_streak = 0;
					cooldown_remaining = newton_damping_cooldown;
				} else if (!worsened) {
					relax *= newton_damping_up;
				}
				relax = std::min(std::max(relax, damping_min), 1.0);
			}
			Fg_abs_sum_prev = Fg_abs_sum;
			// Oscillation catch. When the iteration cycles about the root rather than approaching it, the
			// steps alternate in sign and each one overshoots past the root. For a clean period-two cycle
			// x_a <-> x_b the root is the midpoint, which is reached by half of the current step, so take
			// that instead of the raw Newton step, and let the usual convergence test below decide -- this
			// damps the cycle without bypassing the criterion. (Not the mean of the current and previous
			// steps: the previous one has already been applied, so that re-applies half of it -- a 2-cycle's
			// steps +d, -d average to 0 and the iterate stalls.)
			const bool halve = (n > 0 && delta_x * delta_x_prev < 0.0);
			const double step_x = halve ? 0.5 * delta_x : delta_x;
			delta_x_prev = delta_x;
			for (int g = 0; g < nGT; ++g) {
				const ArrowRow r = arrow_row(g);
				const double delta_R_g = (-1.0 * r.Fg - r.Jg0 * delta_x) / r.Jgg;
				AMREX_ASSERT(!std::isnan(delta_R_g));
				const double step_R_g = halve ? 0.5 * delta_R_g : delta_R_g;
				if (rebase_thin && (tau[g] > 0.0) && (tau[g] < newton_erad_base_tau_threshold)) {
					// step_R holds the Erad_g step for a rebased group. Rvec is advanced to first order
					// as well, so that it stays usable if the group leaves the thin regime; when it does
					// not, Rvec is recovered exactly at the top of the next iteration.
					EradVec_guess[g] += relax * step_R_g;
					Rvec[g] += -tau[g] / opacity_terms.kappaPoverE[g] * relax * step_R_g;
				} else {
					Rvec[g] += relax * step_R_g;
				}
			}
			T_d += relax * step_x;
		} else {
			const double T_rad = std::sqrt(std::sqrt(sum(EradVec_guess) / radiation_constant_));
			if (enable_dE_constrain && delta_x / c_v > std::max(T_gas, T_rad)) {
				Egas_guess = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_rad);
				// Rvec.fillin(0.0);
			} else {
				for (int g = 0; g < nGT; ++g) {
					const ArrowRow r = arrow_row(g);
					const double delta_R_g = (-1.0 * r.Fg - r.Jg0 * delta_x) / r.Jgg;
					AMREX_ASSERT(!std::isnan(delta_R_g));
					if constexpr (use_D_as_base) {
						Rvec[g] += tau0[g] * delta_R_g;
					} else {
						Rvec[g] += delta_R_g;
					}
				}
				Egas_guess += delta_x;
			}
		}

		// check relative and absolute convergence of E_r
		// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
		// 	break;
		// }

		if (warm_start && n == maxIter - 1) {
			restart_cold(); // the warm-started attempt did not converge: rerun the pass cold
		}
	} // END NEWTON-RAPHSON LOOP

	// rad-08: keep the raw Newton state (before the post-Newton additions below) as the next pass's initial guess
	outer_state.warm = (n < maxIter);
	outer_state.Egas = Egas_guess;
	outer_state.T_d = T_d;
	outer_state.EradVec = EradVec_guess;

	// Inject the source directly into transparent groups (tau ~ 0). The Newton solve above excludes such
	// groups from its residual and Jacobian (Fg_abs_sum and Jgg skip tau <= 0) and leaves their radiation
	// energy at Erad0, so an injected source in a transparent group would otherwise be silently dropped
	// while UpdateFlux still applies the matching flux source, leaving |F| > c E. This mirrors the loop in
	// the gas-only solver (source_terms_multi_group.hpp) and the single-group negligible-optical-depth
	// branch; Src is already counted in Etot0, so it is energy-consistent. Groups with tau > 0 (the usual
	// case) and groups without a source are unaffected.
	for (int g = 0; g < nGroups_; ++g) {
		if (!(tau[g] > 0.0)) {
			EradVec_guess[g] = Erad0Vec[g] + Src[g];
		}
	}

	const auto cooling_tend = DefineNetCoolingRate(T_gas, H_num_den) * dt;
	if (dust_model == 2) {
		// include line cooling/heating, cosmic ray heating terms; implicitly update Egas_guess

		// rad-08: in the decoupled branch the Newton iteration leaves Egas at Egas0 (unless the radiation floor is
		// forced inside the iteration), so this implicit gas update depends only on the old state and lambda_gd_times_dt,
		// not on the work term. Solve it in the first pass that reaches it and reuse the result in later passes.
		bool reuse_Egas_post = false;
		if constexpr (!force_rad_floor_in_iteration) {
			reuse_Egas_post = outer_state.have_Egas_post;
		}
		if (reuse_Egas_post) {
			Egas_guess = outer_state.Egas_post;
		} else {
			const double CR_heating = DefineCosmicRayHeatingRate(H_num_den) * dt;

			const double compare = Egas_guess + std::abs(cscale * lambda_gd_times_dt) + sum(abs(cooling_tend)) + std::abs(CR_heating);

			// RHS of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling)
			auto rhs = [=](double Egas_) -> double {
				const double T_gas_ = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
				const auto cooling_ = DefineNetCoolingRate(T_gas_, H_num_den) * dt;
				return Egas_ - Egas0 + cscale * lambda_gd_times_dt + sum(cooling_) - CR_heating;
			};

			// {RHS, Jacobian} of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling), sharing T_gas_
			auto rhs_jac = [=](double Egas_) -> amrex::GpuArray<double, 2> {
				const double T_gas_ = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
				const auto cooling_ = DefineNetCoolingRate(T_gas_, H_num_den) * dt;
				const auto d_cooling_d_Tgas_ = DefineNetCoolingRateTempDerivative(T_gas_, H_num_den) * dt;
				const double c_v_ = ::quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas_, massScalars); // Egas = c_v * T
				// The residual is a function of Egas_, so convert dCooling/dT to dCooling/dEgas via the chain rule (dT/dEgas = 1 / c_v).
				return amrex::GpuArray<double, 2>{Egas_ - Egas0 + cscale * lambda_gd_times_dt + sum(cooling_) - CR_heating,
								  1.0 + sum(d_cooling_d_Tgas_) / c_v_};
			};

			Egas_guess = BackwardEulerOneVariable(rhs, rhs_jac, Egas0, compare);
			if constexpr (!force_rad_floor_in_iteration) {
				outer_state.have_Egas_post = true;
				outer_state.Egas_post = Egas_guess;
			}
		}
	}

	if constexpr (!add_line_cooling_to_radiation_in_jac) {
		AMREX_ASSERT_WITH_MESSAGE(min(cooling_tend) >= 0., "add_line_cooling_to_radiation has to be enabled when there is negative cooling rate!");
		// TODO(CCH): potential GPU-related issue here.
		EradVec_guess += (1 / cscale) * cooling_tend;
	}

	AMREX_ASSERT(Egas_guess > 0.0);
	AMREX_ASSERT(min(EradVec_guess) >= 0.0);

	AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration for matter-radiation coupling failed to converge!");
	if (n >= maxIter) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
	}

	if (p_iteration_counter != nullptr) { // null unless print_iteration_counts is set (rad-02); rad-08: warm-start iterations counted too
		amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);			// total number of radiation updates. NOLINT
		amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n_warm_iters + n + 1); // total number of Newton-Raphson iterations. NOLINT
		amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n_warm_iters + n + 1); // maximum number of Newton-Raphson iterations. NOLINT
		if (dust_model == 2) {
			amrex::Gpu::Atomic::Add(&p_iteration_counter[3], 1); // total number of decoupled gas-dust iterations. NOLINT
		}
	}

	NewtonIterationResult<problem_t> result;

	if (n > 0) {
		// calculate kappaF since the temperature has changed
		// Will update opacity_terms in place
		ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms); // update opacity_terms in place
	}

	result.Egas = Egas_guess;
	result.EradVec = EradVec_guess;
	result.work = work_local;
	result.T_gas = T_gas;
	result.T_d = T_d;
	// UpdateFlux reuses fourPiBoverC and the opacity table of the final iteration. Both were evaluated at T_d_planck; if T_d moved
	// since (decoupled-dust update followed by a relative-change or maxIter exit), re-evaluate at the returned T_d, as UpdateFlux did.
	if (T_d != T_d_planck) {
		fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, rad_boundaries);
		opacity_terms.kappa_expo_and_lower_value = DefineThermalCouplingOpacityExponentsAndLowerValues(rad_boundaries, rho, T_d);
	}
	result.fourPiBoverC = fourPiBoverC;
	result.opacity_terms = opacity_terms;
	result.gas_decoupled = (dust_model == 2);
	return result;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasDustRadiationEnergyExchangeWithPE(
    double const Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double const rho, double const coeff_n, double const dt,
    amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work,
    quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src, const double Q_dust,
    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, double resid_tol, double rel_change_tol, double /*tempFloor*/, int *p_iteration_counter,
    int *p_iteration_failure_counter, NewtonOuterState<problem_t> &outer_state)
    -> NewtonIterationResult<problem_t> // NOSONAR: Numerical solver with inherently high complexity
{
	// 1. Compute energy exchange

	// BEGIN NEWTON-RAPHSON LOOP
	// Define the source term: S = dt chat gamma rho (kappa_P B - kappa_E E) + dt chat c^-2 gamma rho kappa_F v * F_i, where gamma =
	// 1 / sqrt(1 - v^2 / c^2) is the Lorentz factor. Solve for the new radiation energy and gas internal energy using a
	// Newton-Raphson method using the base variables (Egas, D_0, D_1,
	// ...), where D_i = R_i / tau_i^(t) and tau_i^(t) = dt * chat * gamma * rho * kappa_{P,i}^(t) is the optical depth across chat
	// * dt for group i at time t. Compared with the old base (Egas, Erad_0, Erad_1, ...), this new base is more stable and
	// converges faster. Furthermore, the PlanckOpacityTempDerivative term is not needed anymore since we assume d/dT (kappa_P /
	// kappa_E) = 0 in the calculation of the Jacobian. Note that this assumption only affects the convergence rate of the
	// Newton-Raphson iteration and does not affect the result at all once the iteration is converged.
	//
	// The Jacobian of F(E_g, D_i) is
	//
	// dF_G / dE_g = 1
	// dF_G / dD_i = c / chat * tau0_i
	// dF_{D,i} / dE_g = 1 / (chat * C_v) * (kappa_{P,i} / kappa_{E,i}) * d/dT (4 \pi B_i)
	// dF_{D,i} / dD_i = - (1 / (chat * dt * rho * kappa_{E,i}) + 1) * tau0_i = - ((1 / tau_i)(kappa_Pi / kappa_Ei) + 1) * tau0_i

	const double c = c_light_; // make a copy of c_light_ to avoid compiler error "undefined in device code"
	const double chat = c_hat_;
	const double cscale = c / chat;

	int dust_model = 1;
	double T_d0 = NAN;
	double lambda_gd_times_dt = NAN;
	const double T_gas0 = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas0, massScalars);
	AMREX_ASSERT(T_gas0 >= 0.);
	double Etot0 = NAN;
	if (n_outer_iter == 0) {
		// rad-08: invariant across outer passes; computed in the first pass and reused by the later ones.
		T_d0 = ComputeDustTemperatureBateKeto(T_gas0, T_gas0, rho, Erad0Vec, coeff_n, dt, NAN, 0, Q_dust, rad_boundaries);
		AMREX_ASSERT_WITH_MESSAGE(T_d0 >= 0., "Dust temperature is negative!");
		if (T_d0 < 0.0) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
		}

		const double max_Gamma_gd = coeff_n * std::max(std::sqrt(T_gas0) * T_gas0, std::sqrt(T_d0) * T_d0);
		// A zero collisional coupling coefficient means the gas and the dust exchange no energy at all, which is
		// exactly what the decoupled model describes, so select it on that ground alone. The comparison below
		// cannot be relied on to do it: it flips sense once Egas0 is negative, and the coupled branch it then
		// selects divides by coeff_n, filling the state with NaNs that surface later as a spurious
		// "Newton-Raphson iteration failed to converge".
		if (!(coeff_n > 0.0) || (cscale * max_Gamma_gd < ISM_Traits<problem_t>::gas_dust_coupling_threshold * Egas0)) {
			dust_model = 2;
			lambda_gd_times_dt = coeff_n * std::sqrt(T_gas0) * (T_gas0 - T_d0);
		}

		// const double Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src)) + Q_dust;
		if (dust_model == 1) {
			Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src)) + Q_dust;
		} else {
			// for dust_model == 2 (decoupled gas and dust), Egas0 is not involved in the iteration
			Etot0 = std::abs(lambda_gd_times_dt) + (sum(Erad0Vec) + sum(Src)) + Q_dust / cscale;
		}
		outer_state.dust_model = dust_model;
		outer_state.T_d0 = T_d0;
		outer_state.lambda_gd_times_dt = lambda_gd_times_dt;
		outer_state.Etot0 = Etot0;
	} else {
		dust_model = outer_state.dust_model;
		T_d0 = outer_state.T_d0;
		lambda_gd_times_dt = outer_state.lambda_gd_times_dt;
		Etot0 = outer_state.Etot0;
	}

	double T_gas = NAN;
	double T_d = NAN;
	double delta_x = NAN;
	quokka::valarray<double, nGroups_> delta_R{};
	quokka::valarray<double, nGroups_> Rvec{};
	quokka::valarray<double, nGroups_> tau0{};	 // optical depth across c * dt at old state
	quokka::valarray<double, nGroups_> tau{};	 // optical depth across c * dt at new state
	quokka::valarray<double, nGroups_> work_local{}; // work term used in the Newton-Raphson iteration of the current outer iteration
	quokka::valarray<double, nGroups_> fourPiBoverC{};
	quokka::valarray<double, nGroups_> d_fourpiboverc_d_t{};
	double T_d_planck = NAN; // temperature at which fourPiBoverC and the opacity table were last evaluated
	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
	OpacityTerms<problem_t> opacity_terms{};

	// fill kappa_expo_and_lower_value with NAN to avoid mistakes caused by uninitialized values
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < nGroups_ + 1; ++j) {
			kappa_expo_and_lower_value[i][j] = NAN;
		}
	}

	if constexpr (!(opacity_model_ == OpacityModel::piecewise_constant_opacity)) {
		for (int g = 0; g < nGroups_; ++g) {
			rad_boundary_ratios[g] = rad_boundaries[g + 1] / rad_boundaries[g];
		}
	}

	// define a list of alpha_quant for the model PPL_opacity_fixed_slope_spectrum
	amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
	if constexpr ((opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) ||
		      (gamma_ == 1.0 && opacity_model_ == OpacityModel::PPL_opacity_full_spectrum)) {
		if constexpr (!special_edge_bin_slopes) {
			for (int g = 0; g < nGroups_; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		} else {
			alpha_quant_minus_one[0] = 2.0;
			alpha_quant_minus_one[nGroups_ - 1] = -4.0;
			for (int g = 1; g < nGroups_ - 1; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		}
	}

	// rad-08 warm start from the previous pass's converged Newton state (see SolveGasDustRadiationEnergyExchange)
	bool warm_start = (n_outer_iter > 0) && outer_state.warm;
	int n_warm_iters = 0;

	double Egas_guess = warm_start ? outer_state.Egas : Egas0;
	auto EradVec_guess = warm_start ? outer_state.EradVec : Erad0Vec;

	double Egas_guess_prev = Egas_guess;
	auto EradVec_guess_prev = EradVec_guess;

	T_gas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
	AMREX_ASSERT(T_gas >= 0.);

	// phtoelectric heating
	// (evaluated at the old gas temperature T_gas0, which is what the cold start used, so the equations solved are unchanged)
	const double H_num_den = ComputeNumberDensityH(rho, massScalars);
	const double PE_heating_energy_derivative = dt * DefinePhotoelectricHeatingE1Derivative(T_gas0, H_num_den);

	const int maxIter = 100;
	int n = 0;
	auto restart_cold = [&]() {
		n_warm_iters += n + 1;
		warm_start = false;
		n = -1; // incremented to 0 by the loop
		Egas_guess = Egas0;
		EradVec_guess = Erad0Vec;
		Egas_guess_prev = Egas_guess;
		EradVec_guess_prev = EradVec_guess;
		T_gas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
	};
	for (; n < maxIter; ++n) { // NOSONAR
		// if relative change is within tol, break
		if (rel_change_tol > 0.0 && n > 0) {
			const double Erad_tot_guess_prev = sum(EradVec_guess_prev);
			const auto Erad_rel_diff = abs(EradVec_guess - EradVec_guess_prev);
			const auto Egas_rel_diff = std::abs(Egas_guess - Egas_guess_prev);

			if ((sum(Erad_rel_diff) <= rel_change_tol * Erad_tot_guess_prev) && (Egas_rel_diff <= rel_change_tol * Egas_guess_prev)) {
				break;
			}
		}

		Egas_guess_prev = Egas_guess;
		EradVec_guess_prev = EradVec_guess;

		// 1. Compute dust temperature
		// If the dust model is turned off, ComputeDustTemperature should be a function that returns T_gas.

		if (n > 0) {
			T_gas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
			AMREX_ASSERT(T_gas >= 0.);
		}

		if (dust_model == 1) {
			if (n == 0) {
				T_d = warm_start ? outer_state.T_d : T_d0;
			} else {
				T_d = T_gas - (sum(Rvec) - Q_dust / cscale) / (coeff_n * std::sqrt(T_gas));
			}
		} else {
			if (n == 0) {
				T_d = warm_start ? outer_state.T_d : T_d0;
			}
		}
		AMREX_ASSERT_WITH_MESSAGE(warm_start || T_d >= 0.,
					  "Dust temperature is negative! Consider increasing ISM_Traits::gas_dust_coupling_threshold");
		if (T_d < 0.0) {
			if (warm_start) {
				restart_cold();
				continue;
			}
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
		}

		// 2. Compute kappaP and kappaE at dust temperature

		// one fused pass over the group edges gives both 4 pi B / c and its temperature derivative
		ComputeThermalRadiationAndTempDerivativeMultiGroup(T_d, rad_boundaries, fourPiBoverC, d_fourpiboverc_d_t);
		T_d_planck = T_d;

		opacity_terms = ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios, fourPiBoverC, EradVec_guess, n,
								     opacity_terms.alpha_E, opacity_terms.alpha_P);

		if (n == 0) {
			// Compute kappaF and the delta_nu_kappa_B term. kappaF is used to compute the work term.
			// Only the last two arguments (kappaFVec, delta_nu_kappa_B_at_edge) are modified in this function.
			ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms); // update opacity_terms in place
		}

		// 3. In the first loop, calculate kappaF, work, tau0, R

		if (n == 0) {

			if constexpr ((beta_order_ == 1) && (include_work_term_in_source)) {
				// compute the work term at the old state
				// const double gamma = 1.0 / sqrt(1.0 - vsqr / (c * c));
				if (n_outer_iter == 0) {
					for (int g = 0; g < nGroups_; ++g) {
						if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
							work_local[g] = vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * dt;
						} else {
							kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T_d);
							work_local[g] = vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * dt *
									(1.0 + kappa_expo_and_lower_value[0][g]);
						}
					}
				} else {
					// If n_outer_iter > 0, use the work term from the previous outer iteration, which is passed as the parameter 'work'
					work_local = work;
				}
			} else {
				work_local.fillin(0.0);
			}

			tau0 = dt * rho * opacity_terms.kappaP * chat;
			tau = tau0;
			Rvec = (fourPiBoverC - EradVec_guess / opacity_terms.kappaPoverE) * tau0 + work_local;
			if constexpr (use_D_as_base) {
				// tau0 is used as a scaling factor for Rvec
				for (int g = 0; g < nGroups_; ++g) {
					if (tau0[g] <= 1.0) {
						tau0[g] = 1.0;
					}
				}
			}
		} else { // in the second and later loops, calculate tau and E (given R)
			tau = dt * rho * opacity_terms.kappaP * chat;
			for (int g = 0; g < nGroups_; ++g) {
				// If tau = 0.0, Erad_guess shouldn't change
				if (tau[g] > 0.0) {
					EradVec_guess[g] = opacity_terms.kappaPoverE[g] * (fourPiBoverC[g] - (Rvec[g] - work_local[g]) / tau[g]);
					if constexpr (force_rad_floor_in_iteration) {
						if (EradVec_guess[g] < 0.0) {
							Egas_guess -= cscale * (Erad_floor_ - EradVec_guess[g]);
							EradVec_guess[g] = Erad_floor_;
						}
					}
				}
			}
		}

		AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());
		const double c_v = ::quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

		const auto Egas_diff = Egas_guess - Egas0;
		const auto Erad_diff = EradVec_guess - Erad0Vec;

		JacobianResult<problem_t> jacobian;

		if (dust_model == 1) {
			jacobian = ComputeJacobianForGasAndDustWithPE(T_gas, T_d, Egas_diff, EradVec_guess, Erad0Vec, PE_heating_energy_derivative, Rvec, Src,
								      Q_dust, coeff_n, tau, c_v, lambda_gd_times_dt, opacity_terms.kappaPoverE,
								      d_fourpiboverc_d_t, H_num_den, dt);
		} else {
			jacobian = ComputeJacobianForGasAndDustDecoupled(T_gas, T_d, Egas_diff, Erad_diff, Rvec, Src, Q_dust, coeff_n, tau, c_v,
									 lambda_gd_times_dt, opacity_terms.kappaPoverE, d_fourpiboverc_d_t);
		}

		if constexpr (use_D_as_base) {
			jacobian.J0g = jacobian.J0g * tau0;
			jacobian.Jgg = jacobian.Jgg * tau0;
		}

		// Round-off floor on the radiation residual, as in SolveGasRadiationEnergyExchange: an optically
		// thin group sitting far below its local blackbody has its energy reconstructed by a catastrophic
		// cancellation, so |Fg| cannot fall below the double-precision round-off of the larger operand and a
		// purely relative test is unreachable.
		double Fg_roundoff = 0.0;
		for (int g = 0; g < nGroups_; ++g) {
			if (tau[g] > 0.0) {
				Fg_roundoff += std::numeric_limits<double>::epsilon() * std::max(fourPiBoverC[g], EradVec_guess[g]);
			}
		}

		// check relative convergence of the residuals, or that the radiation residual has bottomed out at
		// the round-off floor and cannot be reduced any further
		if ((std::abs(jacobian.F0 / Etot0) < resid_tol) &&
		    ((cscale * jacobian.Fg_abs_sum / Etot0 < resid_tol) || (jacobian.Fg_abs_sum < newton_resid_roundoff_factor * Fg_roundoff))) {
			break;
		}

#if 0
		// For debugging: print (Egas0, Erad0Vec, tau0), which defines the initial condition for a Newton-Raphson iteration
		if (n == 0) {
			std::cout << "Egas0 = " << Egas0 << ", Erad0Vec = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << Erad0Vec[g] << ", ";
			}
			std::cout << "], tau0 = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << tau0[g] << ", ";
			}
			std::cout << "]";
			std::cout << "; C_V = " << c_v << ", a_rad = " << radiation_constant_ << ", coeff_n = " << coeff_n << "\n";
		} else if (n >= 0) {
			std::cout << "n = " << n << ", Egas_guess = " << Egas_guess << ", EradVec_guess = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << EradVec_guess[g] << ", ";
			}
			std::cout << "], tau = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << tau[g] << ", ";
			}
			std::cout << "]";
			std::cout << ", F_G = " << jacobian.F0 << ", F_D_abs_sum = " << jacobian.Fg_abs_sum << ", Etot0 = " << Etot0 << "\n";
		}
#endif

		// update variables
		RadSystem<problem_t>::SolveLinearEqsWithLastColumn(jacobian, delta_x, delta_R); // This is modify delta_x and delta_R in place
		AMREX_ASSERT(!std::isnan(delta_x));
		AMREX_ASSERT(!delta_R.hasnan());

		// Update independent variables (Egas_guess, Rvec)
		// enable_dE_constrain is used to prevent the gas temperature from dropping/increasing below/above the radiation
		// temperature
		if (dust_model == 2) {
			T_d += delta_x;
			Rvec += delta_R;
		} else {
			const double T_rad = std::sqrt(std::sqrt(sum(EradVec_guess) / radiation_constant_));
			if (enable_dE_constrain && delta_x / c_v > std::max(T_gas, T_rad)) {
				Egas_guess = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_rad);
				// Rvec.fillin(0.0);
			} else {
				Egas_guess += delta_x;
				if constexpr (use_D_as_base) {
					Rvec += tau0 * delta_R;
				} else {
					Rvec += delta_R;
				}
			}
		}

		// check relative and absolute convergence of E_r
		// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
		// 	break;
		// }

		if (warm_start && n == maxIter - 1) {
			restart_cold(); // the warm-started attempt did not converge: rerun the pass cold
		}
	} // END NEWTON-RAPHSON LOOP

	// rad-08: keep the raw Newton state as the next pass's initial guess
	outer_state.warm = (n < maxIter);
	outer_state.Egas = Egas_guess;
	outer_state.T_d = T_d;
	outer_state.EradVec = EradVec_guess;

	// Inject the source directly into transparent groups (tau ~ 0). The Newton solve above excludes such
	// groups from its residual and Jacobian (Fg_abs_sum and Jgg skip tau <= 0) and leaves their radiation
	// energy at Erad0, so an injected source in a transparent group would otherwise be silently dropped
	// while UpdateFlux still applies the matching flux source, leaving |F| > c E. This mirrors the loop in
	// the gas-only solver (source_terms_multi_group.hpp) and the single-group negligible-optical-depth
	// branch; Src is already counted in Etot0, so it is energy-consistent. Groups with tau > 0 (the usual
	// case) and groups without a source are unaffected.
	for (int g = 0; g < nGroups_; ++g) {
		if (!(tau[g] > 0.0)) {
			EradVec_guess[g] = Erad0Vec[g] + Src[g];
		}
	}

	const auto cooling_tend = DefineNetCoolingRate(T_gas, H_num_den) * dt;
	if (dust_model == 2) {
		// compute cooling/heating terms; implicitly update Egas_guess

		const double CR_heating = DefineCosmicRayHeatingRate(H_num_den) * dt;
		const double compare = Egas_guess + std::abs(cscale * lambda_gd_times_dt) + sum(abs(cooling_tend)) + std::abs(CR_heating);

		// RHS of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling) - PE_heating_energy_derivative * EradVec_guess[nGroups_ -
		// 1];
		auto rhs = [=](double Egas_) -> double {
			const double T_gas_ = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
			const auto cooling_ = DefineNetCoolingRate(T_gas_, H_num_den) * dt;
			return Egas_ - Egas0 + cscale * lambda_gd_times_dt + sum(cooling_) - PE_heating_energy_derivative * EradVec_guess[nGroups_ - 1] -
			       CR_heating;
		};

		// Jacobian of the RHS of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling) + PE_heating_energy_derivative *
		// EradVec_guess[nGroups_ - 1];
		// (returned together with the RHS, sharing T_gas_; rad-08)
		auto rhs_jac = [=](double Egas_) -> amrex::GpuArray<double, 2> {
			const double T_gas_ = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
			const auto cooling_ = DefineNetCoolingRate(T_gas_, H_num_den) * dt;
			const auto d_cooling_d_Tgas_ = DefineNetCoolingRateTempDerivative(T_gas_, H_num_den) * dt;
			const double c_v_ = ::quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas_, massScalars); // Egas = c_v * T
			// The residual is a function of Egas_, so convert dCooling/dT to dCooling/dEgas via the chain rule (dT/dEgas = 1 / c_v).
			return amrex::GpuArray<double, 2>{Egas_ - Egas0 + cscale * lambda_gd_times_dt + sum(cooling_) -
							      PE_heating_energy_derivative * EradVec_guess[nGroups_ - 1] - CR_heating,
							  1.0 + sum(d_cooling_d_Tgas_) / c_v_};
		};

		Egas_guess = BackwardEulerOneVariable(rhs, rhs_jac, Egas0, compare);
	}

	if constexpr (!add_line_cooling_to_radiation_in_jac) {
		EradVec_guess += (1 / cscale) * cooling_tend;
	}

	AMREX_ASSERT(Egas_guess > 0.0);
	AMREX_ASSERT(min(EradVec_guess) >= 0.0);

	AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration failed to converge!");
	if (n >= maxIter) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
	}

	if (p_iteration_counter != nullptr) { // null unless print_iteration_counts is set (rad-02); rad-08: warm-start iterations counted too
		amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);			// total number of radiation updates. NOLINT
		amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n_warm_iters + n + 1); // total number of Newton-Raphson iterations. NOLINT
		amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n_warm_iters + n + 1); // maximum number of Newton-Raphson iterations. NOLINT
		if (dust_model == 2) {
			amrex::Gpu::Atomic::Add(&p_iteration_counter[3], 1); // total number of decoupled gas-dust iterations. NOLINT
		}
	}

	NewtonIterationResult<problem_t> result;

	if (n > 0) {
		// calculate kappaF since the temperature has changed
		// Will update opacity_terms in place
		ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms); // update opacity_terms in place
	}

	result.Egas = Egas_guess;
	result.EradVec = EradVec_guess;
	result.work = work_local;
	result.T_gas = T_gas;
	result.T_d = T_d;
	// UpdateFlux reuses fourPiBoverC and the opacity table of the final iteration. Both were evaluated at T_d_planck; if T_d moved
	// since (decoupled-dust update followed by a relative-change or maxIter exit), re-evaluate at the returned T_d, as UpdateFlux did.
	if (T_d != T_d_planck) {
		fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, rad_boundaries);
		opacity_terms.kappa_expo_and_lower_value = DefineThermalCouplingOpacityExponentsAndLowerValues(rad_boundaries, rho, T_d);
	}
	result.fourPiBoverC = fourPiBoverC;
	result.opacity_terms = opacity_terms;
	result.gas_decoupled = (dust_model == 2);
	return result;
}


#endif // RADIATION_DUST_SYSTEM_HPP_
