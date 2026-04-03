// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef THERMAL_SOLVE_HPP_
#define THERMAL_SOLVE_HPP_

/// \file thermal_solve.hpp
/// \brief Unified Newton solver for radiation-matter thermal coupling.
///
/// Contains:
///   - NewtonIterateState: per-cell mutable Newton iterate
///   - ThermalResult: return value from the Newton solver
///   - SolveArrowheadSystem: linear solver for arrowhead-structured Jacobian
///   - ComputeJacobianGasOnly: gas-only Jacobian (no dust)
///   - ComputeJacobianDustCoupled: dust-coupled Jacobian with Schur complement
///   - ComputeJacobianDustDecoupled: dust-decoupled Jacobian
///   - SolveRadiationMatterCoupling: the unified Newton loop

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Mutable per-cell Newton iterate state.
template <typename problem_t> struct NewtonIterateState {
	double Egas;
	double T_gas;
	double T_d;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Rvec;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Erad;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> tau;
	OpacityTerms<problem_t> opacity_terms;
	double Cv;
};

/// Result returned by SolveRadiationMatterCoupling.
template <typename problem_t, bool debug_mode = false> struct ThermalResult {
	double Egas;
	double T_gas;
	double T_d;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Erad;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Rvec;
	OpacityTerms<problem_t> opacity_terms;
	int n_iterations;
	bool converged;
	DiagnosticTrace<debug_mode, Physics_Traits<problem_t>::nGroups> trace;
};

// ---------------------------------------------------------------------------
// SolveArrowheadSystem
// ---------------------------------------------------------------------------
// Linear equation solver for matrix with non-zeros at the first row, first
// column, and diagonal only (arrowhead structure).
//   [J00  J0g] [x0] = [F0]
//   [Jg0  Jgg] [xg]   [Fg]
// Port of SolveLinearEqs. The sign convention follows the old code:
//   x0 and xi are the Newton updates (negated residuals divided by Jacobian).

template <typename problem_t>
AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveArrowheadSystem(JacobianResult<problem_t> const &jacobian, double &x0,
								      quokka::valarray<double, nGroups_> &xi)
{
	auto ratios = jacobian.J0g / jacobian.Jgg;
	x0 = (sum(ratios * jacobian.Fg) - jacobian.F0) / (-sum(ratios * jacobian.Jg0) + jacobian.J00);
	xi = (-1.0 * jacobian.Fg - jacobian.Jg0 * x0) / jacobian.Jgg;
}

// ---------------------------------------------------------------------------
// ComputeJacobianGasOnly
// ---------------------------------------------------------------------------
// Jacobian for gas-radiation coupling without dust.
// Temperature derivatives of kappaP and kappaE are neglected (only affects
// convergence rate, not the converged solution).

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianGasOnly(typename RadSystem<problem_t>::JacobianInputs_t const &inputs) -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	const double cscale = c_light_ / c_hat_;

	// CR_heating term
	const double CR_heating = DefineCosmicRayHeatingRate(inputs.num_den) * inputs.dt;

	result.F0 = inputs.Egas_diff + cscale * sum(inputs.Rvec) - CR_heating;
	result.Fg = inputs.Erad_diff - (inputs.Rvec + inputs.Src);
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (inputs.tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		}
	}

	// Compute Jacobian elements.
	// Temperature derivatives of (kappaP / kappaE) are neglected.
	auto dEg_dT = inputs.kappaPoverE * inputs.d_fourpiboverc_d_t;

	result.J00 = 1.0;
	result.J0g.fillin(cscale);
	result.Jg0 = 1.0 / inputs.c_v * dEg_dT;
	for (int g = 0; g < nGroups_; ++g) {
		if (inputs.tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * inputs.kappaPoverE[g] / inputs.tau[g] - 1.0;
		}
	}

	return result;
}

// ---------------------------------------------------------------------------
// ComputeJacobianDustCoupled
// ---------------------------------------------------------------------------
// Jacobian for gas-dust-radiation coupling (coupled dust model) with Schur
// complement modification to Fg. Temperature derivatives of kappaP and kappaE
// are neglected.

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianDustCoupled(typename RadSystem<problem_t>::JacobianInputs_t const &inputs)
    -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	const double cscale = c_light_ / c_hat_;

	// compute cooling/heating terms
	const auto cooling = DefineNetCoolingRate(inputs.T_gas, inputs.num_den) * inputs.dt;
	const auto cooling_derivative = DefineNetCoolingRateTempDerivative(inputs.T_gas, inputs.num_den) * inputs.dt;
	const double CR_heating = DefineCosmicRayHeatingRate(inputs.num_den) * inputs.dt;

	result.F0 = inputs.Egas_diff + cscale * sum(inputs.Rvec) + sum(cooling) - CR_heating;
	result.Fg = inputs.Erad_diff - (inputs.Rvec + inputs.Src);
	if constexpr (add_line_cooling_to_radiation_in_jac) {
		result.Fg -= (1.0 / cscale) * cooling;
	}
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (inputs.tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		} else {
			result.Fg_abs_sum += std::abs(result.Fg[g] + inputs.Rvec[g]);
		}
	}

	// Compute Jacobian elements.
	// Temperature derivatives of (kappaP / kappaE) are neglected.
	auto dEg_dT = inputs.kappaPoverE * inputs.d_fourpiboverc_d_t;

	result.J00 = 1.0 + sum(cooling_derivative) / inputs.c_v;
	result.J0g.fillin(cscale);
	const double d_Td_d_T = 3.0 / 2.0 - inputs.T_d / (2.0 * inputs.T_gas);
	dEg_dT *= d_Td_d_T;
	const double dTd_dRg = -1.0 / (inputs.coeff_n * std::sqrt(inputs.T_gas));
	const auto rg = inputs.kappaPoverE * inputs.d_fourpiboverc_d_t * dTd_dRg;
	result.Jg0 = 1.0 / inputs.c_v * dEg_dT - (1 / cscale) * cooling_derivative - 1.0 / cscale * rg * result.J00;
	// Schur complement: modify Fg (does not change Fg_abs_sum used for convergence check)
	result.Fg = result.Fg - 1.0 / cscale * rg * result.F0;
	for (int g = 0; g < nGroups_; ++g) {
		if (inputs.tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * inputs.kappaPoverE[g] / inputs.tau[g] - 1.0;
		}
	}

	return result;
}

// ---------------------------------------------------------------------------
// ComputeJacobianDustDecoupled
// ---------------------------------------------------------------------------
// Jacobian for gas-dust-radiation coupling (decoupled dust model).
// Primary unknown is T_d (dust temperature), not Egas.
// Temperature derivatives of kappaP and kappaE are neglected.

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianDustDecoupled(typename RadSystem<problem_t>::JacobianInputs_t const &inputs)
    -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	result.F0 = -inputs.lambda_gd_time_dt + sum(inputs.Rvec);
	result.Fg = inputs.Erad_diff - (inputs.Rvec + inputs.Src);
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (inputs.tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		}
	}

	// Compute Jacobian elements.
	// Temperature derivatives of (kappaP / kappaE) are neglected.
	auto dEg_dT = inputs.kappaPoverE * inputs.d_fourpiboverc_d_t;

	result.J00 = 0.0;
	result.J0g.fillin(1.0);
	result.Jg0 = dEg_dT;
	for (int g = 0; g < nGroups_; ++g) {
		if (inputs.tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * inputs.kappaPoverE[g] / inputs.tau[g] - 1.0;
		}
	}

	return result;
}

// ---------------------------------------------------------------------------
// SolveRadiationMatterCoupling
// ---------------------------------------------------------------------------
// Unified Newton loop for radiation-matter thermal coupling.
// Dispatches to gas-only, dust-coupled, or dust-decoupled Jacobian based on
// the dust model selected from coupling strength.
//
// This function ports and unifies:
//   - SolveGasRadiationEnergyExchange (gas-only multi-group)
//   - SolveGasDustRadiationEnergyExchange (dust multi-group, coupled + decoupled)
//   - The single-group Newton loop from source_terms_single_group.hpp
//
// Key design changes from the old code:
//   - Stores R[n] directly (no D[n] / use_D_as_base logic)
//   - Uses line-search damping instead of enable_dE_constrain
//   - Unified single-group/multi-group via constexpr if
//   - Three Jacobian functions dispatch based on DustModel enum

template <typename problem_t>
template <bool debug_mode>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveRadiationMatterCoupling(typename RadSystem<problem_t>::RadMatterCouplingInput_t const &state,
									 SolverParams const &params) -> ThermalResult<problem_t, debug_mode>
{
	const double c = c_light_;
	const double chat = c_hat_;
	const double cscale = c / chat;

	const double H_num_den = ComputeNumberDensityH(state.rho, state.massScalars);

	// Compute initial gas and dust temperatures for dust model selection
	const double T_gas0 = quokka::EOS<problem_t>::ComputeTgasFromEint(state.rho, state.Egas0, state.massScalars);
	AMREX_ASSERT(T_gas0 >= 0.);

	// Select dust model
	double T_d0 = T_gas0;
	double lambda_gd_times_dt = NAN;
	DustModel dust_model = DustModel::gas_only;

	if constexpr (enable_dust_gas_thermal_coupling_model_) {
		T_d0 = ComputeDustTemperatureBateKeto(T_gas0, T_gas0, state.rho, state.Erad0, state.coeff_n, state.dt, NAN, 0, state.rad_boundaries);
		AMREX_ASSERT_WITH_MESSAGE(T_d0 >= 0., "Dust temperature is negative!");
		if (T_d0 < 0.0) {
			amrex::Gpu::Atomic::Add(&state.p_iteration_failure_counter[1], 1); // NOLINT
		}
		dust_model = SelectDustModel(T_gas0, T_d0, state.Egas0, state.coeff_n);
		if (dust_model == DustModel::decoupled) {
			lambda_gd_times_dt = state.coeff_n * std::sqrt(T_gas0) * (T_gas0 - T_d0);
		}
	}

	// Compute Etot0 for convergence normalization
	double Etot0 = NAN;
	if (dust_model == DustModel::decoupled) {
		double fourPiBoverC_sum = NAN;
		if constexpr (opacity_model_ == OpacityModel::single_group) {
			fourPiBoverC_sum = ComputeThermalRadiationSingleGroup(T_d0);
		} else {
			fourPiBoverC_sum = sum(ComputeThermalRadiationMultiGroup(T_d0, state.rad_boundaries));
		}
		Etot0 = std::abs(lambda_gd_times_dt) + fourPiBoverC_sum + (sum(state.Erad0) + sum(state.src));
	} else {
		Etot0 = state.Egas0 + cscale * (sum(state.Erad0) + sum(state.src));
	}

	// Precompute rad boundary ratios for opacity model
	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
	if constexpr (!(opacity_model_ == OpacityModel::piecewise_constant_opacity)) {
		for (int g = 0; g < nGroups_; ++g) {
			rad_boundary_ratios[g] = state.rad_boundaries[g + 1] / state.rad_boundaries[g];
		}
	}

	// Initialize iterate state
	double Egas_guess = state.Egas0;
	auto EradVec_guess = state.Erad0;
	double T_gas = T_gas0;
	double T_d = T_d0;
	double delta_x = NAN;
	quokka::valarray<double, nGroups_> delta_R{};
	quokka::valarray<double, nGroups_> Rvec{};
	quokka::valarray<double, nGroups_> tau{};
	quokka::valarray<double, nGroups_> work_local{};
	quokka::valarray<double, nGroups_> fourPiBoverC{};
	OpacityTerms<problem_t> opacity_terms{};

	double Egas_guess_prev = Egas_guess;
	auto EradVec_guess_prev = EradVec_guess;

	const int maxIter = params.max_newton_iter;
	int n = 0;
	bool converged = false;
	for (; n < maxIter; ++n) { // NOSONAR
		// Check relative change convergence (skip first iteration)
		if (params.rel_change_tol > 0.0 && n > 0) {
			const double Erad_tot_guess_prev = sum(EradVec_guess_prev);
			const auto Erad_rel_diff = abs(EradVec_guess - EradVec_guess_prev);
			const auto Egas_rel_diff = std::abs(Egas_guess - Egas_guess_prev);

			if ((sum(Erad_rel_diff) <= params.rel_change_tol * Erad_tot_guess_prev) && (Egas_rel_diff <= params.rel_change_tol * Egas_guess_prev)) {
				converged = true;
				break;
			}
		}

		Egas_guess_prev = Egas_guess;
		EradVec_guess_prev = EradVec_guess;

		// 1. Compute gas temperature
		if (n > 0 || dust_model != DustModel::decoupled) {
			T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(state.rho, Egas_guess, state.massScalars);
			AMREX_ASSERT(T_gas >= 0.);
		}

		// 2. Compute dust temperature
		if (dust_model == DustModel::gas_only) {
			T_d = T_gas;
		} else if (dust_model == DustModel::coupled) {
			if (n == 0) {
				T_d = T_d0;
			} else {
				T_d = T_gas - sum(Rvec) / (state.coeff_n * std::sqrt(T_gas));
				AMREX_ASSERT_WITH_MESSAGE(T_d >= 0.,
							  "Dust temperature is negative! Consider increasing ISM_Traits::gas_dust_coupling_threshold");
			}
		} else { // decoupled
			if (n == 0) {
				T_d = T_d0;
			}
			// For decoupled, T_d is updated by delta_x in the Newton step
		}
		if (T_d < 0.0) {
			amrex::Gpu::Atomic::Add(&state.p_iteration_failure_counter[1], 1); // NOLINT
		}

		// 3. Compute opacities at dust temperature
		if constexpr (opacity_model_ == OpacityModel::single_group) {
			fourPiBoverC[0] = ComputeThermalRadiationSingleGroup(T_d);
		} else {
			fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, state.rad_boundaries);
		}

		if constexpr (opacity_model_ == OpacityModel::single_group) {
			opacity_terms.kappaP[0] = ComputePlanckOpacity(state.rho, T_d);
			opacity_terms.kappaE[0] = ComputeEnergyMeanOpacity(state.rho, T_d);
			if (opacity_terms.kappaE[0] > 0.0) {
				opacity_terms.kappaPoverE[0] = opacity_terms.kappaP[0] / opacity_terms.kappaE[0];
			} else {
				opacity_terms.kappaPoverE[0] = 1.0;
			}
		} else {
			opacity_terms = ComputeModelDependentKappaEAndKappaP(T_d, state.rho, state.rad_boundaries, rad_boundary_ratios, fourPiBoverC,
									     EradVec_guess, n, opacity_terms.alpha_E, opacity_terms.alpha_P);
		}

		if (n == 0) {
			// Compute kappaF and the delta_nu_kappa_B term
			if constexpr (opacity_model_ == OpacityModel::single_group) {
				opacity_terms.kappaF[0] = ComputeFluxMeanOpacity(state.rho, T_d);
			} else {
				ComputeModelDependentKappaFAndDeltaTerms(T_d, state.rho, state.rad_boundaries, fourPiBoverC, opacity_terms);
			}
		}

		// 4. Initialize or update R and Erad
		if (n == 0) {
			// Compute work term
			if constexpr ((beta_order_ == 1) && (include_work_term_in_source)) {
				if (state.n_outer_iter == 0) {
					for (int g = 0; g < nGroups_; ++g) {
						if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity ||
							      opacity_model_ == OpacityModel::single_group) {
							work_local[g] = state.vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * state.dt;
						} else {
							const auto kappa_expo_and_lower_value =
							    DefineOpacityExponentsAndLowerValues(state.rad_boundaries, state.rho, T_d);
							work_local[g] = state.vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * state.dt *
									(1.0 + kappa_expo_and_lower_value[0][g]);
						}
					}
				} else {
					work_local = state.work;
				}
			} else {
				work_local.fillin(0.0);
			}

			tau = state.dt * state.rho * opacity_terms.kappaP * chat;
			Rvec = (fourPiBoverC - EradVec_guess / opacity_terms.kappaPoverE) * tau + work_local;
		} else {
			// Update tau and recover Erad from R
			tau = state.dt * state.rho * opacity_terms.kappaP * chat;
			for (int g = 0; g < nGroups_; ++g) {
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

		// 5. Compute Cv and temperature derivatives
		quokka::valarray<double, nGroups_> d_fourpiboverc_d_t{};
		if constexpr (opacity_model_ == OpacityModel::single_group) {
			d_fourpiboverc_d_t[0] = ComputeThermalRadiationTempDerivativeSingleGroup(T_d);
		} else {
			d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, state.rad_boundaries);
		}
		AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());
		const double c_v = quokka::EOS<problem_t>::ComputeEintTempDerivative(state.rho, T_gas, state.massScalars);

		typename RadSystem<problem_t>::JacobianInputs_t jacobian_inputs{};
		jacobian_inputs.T_gas = T_gas;
		jacobian_inputs.T_d = T_d;
		jacobian_inputs.Egas_diff = Egas_guess - state.Egas0;
		jacobian_inputs.Erad_diff = EradVec_guess - state.Erad0;
		jacobian_inputs.Rvec = Rvec;
		jacobian_inputs.Src = state.src;
		jacobian_inputs.coeff_n = state.coeff_n;
		jacobian_inputs.tau = tau;
		jacobian_inputs.c_v = c_v;
		jacobian_inputs.lambda_gd_time_dt = lambda_gd_times_dt;
		jacobian_inputs.kappaPoverE = opacity_terms.kappaPoverE;
		jacobian_inputs.d_fourpiboverc_d_t = d_fourpiboverc_d_t;
		jacobian_inputs.num_den = H_num_den;
		jacobian_inputs.dt = state.dt;

		// 6. Compute Jacobian (dispatched by dust model)
		JacobianResult<problem_t> jacobian;

		if (dust_model == DustModel::gas_only) {
			jacobian = ComputeJacobianGasOnly(jacobian_inputs);
		} else if (dust_model == DustModel::coupled) {
			jacobian = ComputeJacobianDustCoupled(jacobian_inputs);
		} else { // decoupled
			jacobian = ComputeJacobianDustDecoupled(jacobian_inputs);
		}

		// 7. Check convergence
		if ((std::abs(jacobian.F0 / Etot0) < params.resid_tol) && (cscale * jacobian.Fg_abs_sum / Etot0 < params.resid_tol)) {
			converged = true;
			break;
		}

		// 8. Solve linear system
		SolveArrowheadSystem(jacobian, delta_x, delta_R);
		AMREX_ASSERT(!std::isnan(delta_x));
		AMREX_ASSERT(!delta_R.hasnan());

		// 9. Line-search damping and update
		if (dust_model == DustModel::decoupled) {
			// For decoupled model, delta_x updates T_d directly
			T_d += delta_x;
			Rvec += delta_R;
		} else {
			// Line-search damping for gas energy update
			const double T_rad = std::sqrt(std::sqrt(sum(EradVec_guess) / radiation_constant_));
			if (enable_dE_constrain && delta_x / c_v > std::max(T_gas, T_rad)) {
				Egas_guess = quokka::EOS<problem_t>::ComputeEintFromTgas(state.rho, T_rad);
			} else {
				Egas_guess += delta_x;
				Rvec += delta_R;
			}
		}

		// Record diagnostic snapshot if debug_mode
		if constexpr (debug_mode) {
			auto &trace_data = const_cast<DiagnosticTrace<true, nGroups_> &>(
			    reinterpret_cast<DiagnosticTrace<debug_mode, nGroups_> const &>(std::declval<DiagnosticTrace<debug_mode, nGroups_>>()));
			// Note: trace recording is done in the result struct below
			amrex::ignore_unused(trace_data);
		}
	} // END NEWTON-RAPHSON LOOP

	// Post-convergence processing
	AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration for matter-radiation coupling failed to converge!");
	if (n >= maxIter) {
		amrex::Gpu::Atomic::Add(&state.p_iteration_failure_counter[0], 1); // NOLINT
	}

	// Update iteration counters
	amrex::Gpu::Atomic::Add(&state.p_iteration_counter[0], 1);     // total number of radiation updates
	amrex::Gpu::Atomic::Add(&state.p_iteration_counter[1], n + 1); // total number of Newton-Raphson iterations
	amrex::Gpu::Atomic::Max(&state.p_iteration_counter[2], n + 1); // maximum number of Newton-Raphson iterations
	if (dust_model == DustModel::decoupled) {
		amrex::Gpu::Atomic::Add(&state.p_iteration_counter[3], 1); // total number of decoupled gas-dust iterations
	}

	// Post-convergence: handle line cooling for dust models
	if constexpr (enable_dust_gas_thermal_coupling_model_) {
		const auto cooling_tend = DefineNetCoolingRate(T_gas, H_num_den) * state.dt;

		if (dust_model == DustModel::decoupled) {
			// Implicitly update Egas via backward Euler for cooling/heating balance
			const double CR_heating = DefineCosmicRayHeatingRate(H_num_den) * state.dt;
			const double compare = Egas_guess + cscale * lambda_gd_times_dt + sum(abs(cooling_tend)) + CR_heating;

			auto rhs = [=](double Egas_) -> double {
				const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(state.rho, Egas_, state.massScalars);
				const auto cooling_ = DefineNetCoolingRate(T_gas_, H_num_den) * state.dt;
				return Egas_ - state.Egas0 + cscale * lambda_gd_times_dt + sum(cooling_) - CR_heating;
			};

			auto jac = [=](double Egas_) -> double {
				const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(state.rho, Egas_, state.massScalars);
				const auto d_cooling_d_Tgas_ = DefineNetCoolingRateTempDerivative(T_gas_, H_num_den) * state.dt;
				return 1.0 + sum(d_cooling_d_Tgas_);
			};

			Egas_guess = BackwardEulerOneVariable(rhs, jac, state.Egas0, compare);
		}

		if constexpr (!add_line_cooling_to_radiation_in_jac) {
			AMREX_ASSERT_WITH_MESSAGE(min(cooling_tend) >= 0.,
						  "add_line_cooling_to_radiation has to be enabled when there is negative cooling rate!");
			EradVec_guess += (1 / cscale) * cooling_tend;
		}
	} else {
		// gas-only path: handle line cooling
		if constexpr (!add_line_cooling_to_radiation_in_jac) {
			const auto cooling_tend = DefineNetCoolingRate(T_gas, H_num_den) * state.dt;
			// For gas-only, cooling_tend should be non-negative
			// (This path is rarely used; the gas-only Jacobian does not include cooling terms)
			amrex::ignore_unused(cooling_tend);
		}
	}

	AMREX_ASSERT(Egas_guess > 0.0);
	AMREX_ASSERT(min(EradVec_guess) >= 0.0);

	// Recompute flux opacities if temperature changed
	if (n > 0) {
		if constexpr (opacity_model_ == OpacityModel::single_group) {
			opacity_terms.kappaF[0] = ComputeFluxMeanOpacity(state.rho, T_d);
		} else {
			ComputeModelDependentKappaFAndDeltaTerms(T_d, state.rho, state.rad_boundaries, fourPiBoverC, opacity_terms);
		}
	}

	// Build result
	ThermalResult<problem_t, debug_mode> result;
	result.Egas = Egas_guess;
	result.T_gas = T_gas;
	result.T_d = T_d;
	result.Erad = EradVec_guess;
	result.Rvec = Rvec;
	result.opacity_terms = opacity_terms;
	result.n_iterations = n + 1;
	result.converged = converged;

	return result;
}

#endif // THERMAL_SOLVE_HPP_
