// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RAD_SOURCE_TERMS_MULTI_GROUP_HPP_ // NOLINT
#define RAD_SOURCE_TERMS_MULTI_GROUP_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

template <typename problem_t>
auto RadSystem<problem_t>::ComputeJacobianForPureGas(double /*T_gas*/, double /*T_d*/, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double /*coeff_n*/, quokka::valarray<double, nGroups_> const &tau, double c_v, double cscale, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t) -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	result.F0 = Egas_diff;
	result.Fg = Erad_diff - (Rvec + Src);
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
			result.F0 += cscale * Rvec[g];
		}
	}

	// const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, radBoundaries_g_copy);
	AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;

	result.J00 = 1.0;
	result.J0g.fillin(cscale);
	result.Jg0 = 1.0 / c_v * dEg_dT;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
		}
	}

	return result;
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeJacobianForGasAndDust(double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double cscale, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t) -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	result.F0 = Egas_diff;
	result.Fg = Erad_diff - (Rvec + Src);
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
			result.F0 += cscale * Rvec[g];
		}
	}

	// const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, radBoundaries_g_copy);
	AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;

	result.J00 = 1.0;
	result.J0g.fillin(cscale);
	const double d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
	// const double coeff_n = dt * dustGasCoeff_local * num_den * num_den / cscale;
	dEg_dT *= d_Td_d_T;
	const double dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));
	const auto rg = kappaPoverE * d_fourpiboverc_d_t * dTd_dRg;
	result.Jg0 = 1.0 / c_v * dEg_dT - 1.0 / cscale * rg * result.J00;
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

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_radiation, const int stage, double dustGasCoeff, int *p_iteration_counter, int *p_num_failed_coupling, int *p_num_failed_dust, int *p_num_failed_outer_ite)
{
	static_assert(beta_order_ == 0 || beta_order_ == 1);

	arrayconst_t &consPrev = consVar; // make read-only
	array_t &consNew = consVar;
	auto dt = dt_radiation;
	if (stage == 2) {
		dt = (1.0 - IMEX_a32) * dt_radiation;
	}

	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	// Add source terms

	// 1. Compute gas energy and radiation energy update following Howell &
	// Greenough [Journal of Computational Physics 184 (2003) 53–78].

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// make a local reference of p_num_failed
		auto p_num_failed_coupling_local = p_num_failed_coupling;
		auto p_num_failed_dust_local = p_num_failed_dust;
		auto p_num_failed_outer_local = p_num_failed_outer_ite;
		auto p_iteration_counter_local = p_iteration_counter;

		const double c = c_light_;
		const double chat = c_hat_;
		const double dustGasCoeff_local = dustGasCoeff;

		// load fluid properties
		const double rho = consPrev(i, j, k, gasDensity_index);
		const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
		const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
		const std::array<double, 3> gasMtm0 = {x1GasMom0, x2GasMom0, x3GasMom0};
		const double Egastot0 = consPrev(i, j, k, gasEnergy_index);
		auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consPrev, i, j, k);

		// load radiation energy
		quokka::valarray<double, nGroups_> Erad0Vec;
		for (int g = 0; g < nGroups_; ++g) {
			Erad0Vec[g] = consPrev(i, j, k, radEnergy_index + numRadVars_ * g);
		}
		AMREX_ASSERT(min(Erad0Vec) > 0.0);
		const double Erad0 = sum(Erad0Vec);

		// load radiation energy source term
		// plus advection source term (for well-balanced/SDC integrators)
		quokka::valarray<double, nGroups_> Src;
		for (int g = 0; g < nGroups_; ++g) {
			Src[g] = dt * (chat * radEnergySource(i, j, k, g));
		}

		double Egas0 = NAN;
		double Ekin0 = NAN;
		double Etot0 = NAN;
		double Egas_guess = NAN;
		double T_gas = NAN;
		double T_d = NAN;
		quokka::valarray<double, nGroups_> fourPiBoverC{};
		quokka::valarray<double, nGroups_> EradVec_guess{};
		quokka::valarray<double, nGroups_> kappaPVec{};
		quokka::valarray<double, nGroups_> kappaEVec{};
		quokka::valarray<double, nGroups_> kappaFVec{};
		amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
		amrex::GpuArray<double, nGroups_> alpha_B{};
		amrex::GpuArray<double, nGroups_> alpha_E{};
		quokka::valarray<double, nGroups_> kappaPoverE{};
		quokka::valarray<double, nGroups_> tau0{}; // optical depth across c * dt at old state
		quokka::valarray<double, nGroups_> tau{};  // optical depth across c * dt at new state
		quokka::valarray<double, nGroups_> work{};
		quokka::valarray<double, nGroups_> work_prev{};
		amrex::GpuArray<amrex::Real, 3> dMomentum{};
		amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> Frad_t1{};
		amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge{};
		amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge{};

		work.fillin(0.0);
		work_prev.fillin(0.0);

		if constexpr (gamma_ != 1.0) {
			Egas0 = ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0);
			Etot0 = Egas0 + (c / chat) * (Erad0 + sum(Src));
		}

		// make a copy of radBoundaries_g
		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
		amrex::GpuArray<double, nGroups_> radBoundaryRatios_copy{};
		for (int g = 0; g < nGroups_ + 1; ++g) {
			radBoundaries_g_copy[g] = radBoundaries_g[g];
		}
		for (int g = 0; g < nGroups_; ++g) {
			radBoundaryRatios_copy[g] = radBoundaries_g_copy[g + 1] / radBoundaries_g_copy[g];
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

		amrex::Real gas_update_factor = 1.0;
		if (stage == 1) {
			gas_update_factor = IMEX_a32;
		}

		const double num_den = rho / mean_molecular_mass_;

		const int max_ite = 5;
		int ite = 0;
		for (; ite < max_ite; ++ite) {
			quokka::valarray<double, nGroups_> Rvec{};

			EradVec_guess = Erad0Vec;

			if constexpr (gamma_ != 1.0) {
				Egas_guess = Egas0;
				Ekin0 = Egastot0 - Egas0;

				AMREX_ASSERT(min(Src) >= 0.0);
				AMREX_ASSERT(Egas0 > 0.0);

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

				double deltaEgas = NAN;
				quokka::valarray<double, nGroups_> deltaD{};
				quokka::valarray<double, nGroups_> F_D{};

				const double resid_tol = 1.0e-11; // 1.0e-15;
				const int maxIter = enable_dust_gas_thermal_coupling_model_ ? 100 : 50;
				int n = 0;
				for (; n < maxIter; ++n) {
					T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
					AMREX_ASSERT(T_gas >= 0.);

					// dust temperature
					if constexpr (!enable_dust_gas_thermal_coupling_model_) {
						T_d = T_gas;
					} else {
						if (n == 0) {
							T_d = ComputeDustTemperature(T_gas, T_gas, rho, EradVec_guess, dustGasCoeff_local, radBoundaries_g_copy,
										     radBoundaryRatios_copy);
						} else {
							const auto Lambda_gd = sum(Rvec) / (dt * chat / c);
							T_d = T_gas - Lambda_gd / (dustGasCoeff_local * num_den * num_den * std::sqrt(T_gas));
						}
						AMREX_ASSERT_WITH_MESSAGE(T_d >= 0., "Dust temperature is negative!");
						if (T_d < 0.0) {
							amrex::Gpu::Atomic::Add(p_num_failed_dust_local, 1);
						}
					}

					fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, radBoundaries_g_copy);

					kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, T_d);
					if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
						for (int g = 0; g < nGroups_; ++g) {
							kappaPVec[g] = kappa_expo_and_lower_value[1][g];
							kappaEVec[g] = kappa_expo_and_lower_value[1][g];
						}
					} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) {
						kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_quant_minus_one);
						kappaEVec = kappaPVec;
					} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
						if (n < max_ite_to_update_alpha_E) {
							alpha_B = ComputeRadQuantityExponents(fourPiBoverC, radBoundaries_g_copy);
							alpha_E = ComputeRadQuantityExponents(EradVec_guess, radBoundaries_g_copy);
						}
						kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_B);
						kappaEVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_E);
					}
					AMREX_ASSERT(!kappaPVec.hasnan());
					AMREX_ASSERT(!kappaEVec.hasnan());
					for (int g = 0; g < nGroups_; ++g) {
						if (kappaEVec[g] > 0.0) {
							kappaPoverE[g] = kappaPVec[g] / kappaEVec[g];
						} else {
							kappaPoverE[g] = 1.0;
						}
					}

					// In the first loop, calculate kappaF, work, tau0, R
					if (n == 0) {
						for (int g = 0; g < nGroups_; ++g) {
							auto const nu_L = radBoundaries_g_copy[g];
							auto const nu_R = radBoundaries_g_copy[g + 1];
							auto const B_L = PlanckFunction(nu_L, T_d); // 4 pi B(nu) / c
							auto const B_R = PlanckFunction(nu_R, T_d); // 4 pi B(nu) / c
							auto const kappa_L = kappa_expo_and_lower_value[1][g];
							auto const kappa_R = kappa_L * std::pow(nu_R / nu_L, kappa_expo_and_lower_value[0][g]);
							delta_nu_kappa_B_at_edge[g] = nu_R * kappa_R * B_R - nu_L * kappa_L * B_L;
							delta_nu_B_at_edge[g] = nu_R * B_R - nu_L * B_L;
						}
						if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
							kappaFVec = kappaPVec;
						} else {
							if constexpr (use_diffuse_flux_mean_opacity) {
								kappaFVec = ComputeDiffusionFluxMeanOpacity(kappaPVec, kappaEVec, fourPiBoverC,
													    delta_nu_kappa_B_at_edge, delta_nu_B_at_edge,
													    kappa_expo_and_lower_value[0]);
							} else {
								// for simplicity, I assume kappaF = kappaE when opacity_model_ ==
								// OpacityModel::PPL_opacity_full_spectrum, if !use_diffuse_flux_mean_opacity. We won't
								// use this option anyway.
								kappaFVec = kappaEVec;
							}
						}
						AMREX_ASSERT(!kappaFVec.hasnan());

						if constexpr ((beta_order_ == 1) && (include_work_term_in_source)) {
							// compute the work term at the old state
							// const double gamma = 1.0 / sqrt(1.0 - vsqr / (c * c));
							if (ite == 0) {
								for (int g = 0; g < nGroups_; ++g) {
									const double frad0 = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
									const double frad1 = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
									const double frad2 = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
									// work = v * F * chi
									if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
										work[g] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2) *
											  kappaFVec[g] * chat / (c * c) * dt;
									} else {
										work[g] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2) *
											  (1.0 + kappa_expo_and_lower_value[0][g]) * kappaFVec[g] * chat /
											  (c * c) * dt;
									}
								}
							}
						}

						tau0 = dt * rho * kappaPVec * chat;
						tau = tau0;
						Rvec = (fourPiBoverC - EradVec_guess / kappaPoverE) * tau0 + work;
						if constexpr (use_D_as_base) {
							// tau0 is used as a scaling factor for Rvec
							for (int g = 0; g < nGroups_; ++g) {
								if (tau0[g] <= 1.0) {
									tau0[g] = 1.0;
								}
							}
						}
					} else { // in the second and later loops, calculate tau and E (given R)
						tau = dt * rho * kappaPVec * chat;
						for (int g = 0; g < nGroups_; ++g) {
							// If tau = 0.0, Erad_guess shouldn't change
							if (tau[g] > 0.0) {
								EradVec_guess[g] = kappaPoverE[g] * (fourPiBoverC[g] - (Rvec[g] - work[g]) / tau[g]);
								if constexpr (force_rad_floor_in_iteration) {
									if (EradVec_guess[g] < 0.0) {
										Egas_guess -= (c_light_ / c_hat_) * (Erad_floor_ - EradVec_guess[g]);
										EradVec_guess[g] = Erad_floor_;
									}
								}
							}
						}
					}

					const double cscale = c / chat;
					const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, radBoundaries_g_copy);
					AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());
					const double c_v = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

					const auto Egas_diff = Egas_guess - Egas0;
					const auto Erad_diff = EradVec_guess - Erad0Vec;
					JacobianResult<problem_t> jacobian;

					if (enable_dust_gas_thermal_coupling_model_) {
						const double coeff_n = dt * dustGasCoeff_local * num_den * num_den / cscale;
						jacobian = ComputeJacobianForGasAndDust(T_gas, T_d, Egas_diff, Erad_diff, Rvec, Src, coeff_n, 
										tau, c_v, cscale, kappaPoverE, d_fourpiboverc_d_t);
					} else {
						jacobian = ComputeJacobianForPureGas(NAN, NAN, Egas_diff, Erad_diff, Rvec, Src, NAN, 
										tau, c_v, cscale, kappaPoverE, d_fourpiboverc_d_t);
					}

					if constexpr (use_D_as_base) {
						jacobian.J0g = jacobian.J0g * tau0;
						jacobian.Jgg = jacobian.Jgg * tau0;
					}

					// check relative convergence of the residuals
					if ((std::abs(jacobian.F0 / Etot0) < resid_tol) && ((c / chat) * jacobian.Fg_abs_sum / Etot0 < resid_tol)) {
						break;
					}

#if 0
					// For debugging: print (Egas0, Erad0Vec, tau0), which defines the initial condition for a Newton-Raphson iteration
					if (n == 0) {
						std::cout << "Egas0 = " << Egas0 << ", Erad0Vec = " << Erad0Vec[0] << ", tau0 = " << tau0[0]
							  << "; C_V = " << c_v << ", a_rad = " << radiation_constant_ << std::endl;
					} else if (n >= 0) {
						std::cout << "n = " << n << ", Egas_guess = " << Egas_guess << ", EradVec_guess = " << EradVec_guess[0]
							  << ", tau = " << tau[0];
						std::cout << ", F_G = " << jacobian.F0 << ", F_D_abs_sum = " << jacobian.Fg_abs_sum << ", Etot0 = " << Etot0 << std::endl;
					}
#endif

					// update variables
					RadSystem<problem_t>::SolveLinearEqs(jacobian, deltaEgas, deltaD);
					AMREX_ASSERT(!std::isnan(deltaEgas));
					AMREX_ASSERT(!deltaD.hasnan());

					// Update independent variables (Egas_guess, Rvec)
					// enable_dE_constrain is used to prevent the gas temperature from dropping/increasing below/above the radiation temperature
					if (!enable_dE_constrain) {
						Egas_guess += deltaEgas;
						if constexpr (use_D_as_base) {
							Rvec += tau0 * deltaD;
						} else {
							Rvec += deltaD;
						}
					} else {
						const double T_rad = std::pow(sum(EradVec_guess) / radiation_constant_, 0.25);
						if (deltaEgas / c_v > std::max(T_gas, T_rad)) {
							Egas_guess = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_rad);
							Rvec.fillin(0.0);
						} else {
							Egas_guess += deltaEgas;
							if constexpr (use_D_as_base) {
								Rvec += tau0 * deltaD;
							} else {
								Rvec += deltaD;
							}
						}
					}

					// check relative and absolute convergence of E_r
					// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
					// 	break;
					// }
				} // END NEWTON-RAPHSON LOOP

				AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration failed to converge!");
				if (n >= maxIter) {
					amrex::Gpu::Atomic::Add(p_num_failed_coupling_local, 1);
				}

				// update iteration counter: (+1, +ite, max(self, ite))
				amrex::Gpu::Atomic::Add(&p_iteration_counter_local[0], 1);     // total number of radiation updates
				amrex::Gpu::Atomic::Add(&p_iteration_counter_local[1], n + 1); // total number of Newton-Raphson iterations
				amrex::Gpu::Atomic::Max(&p_iteration_counter_local[2], n + 1); // maximum number of Newton-Raphson iterations

				// std::cout << "Newton-Raphson converged after " << n << " it." << std::endl;
				AMREX_ASSERT(Egas_guess > 0.0);
				AMREX_ASSERT(min(EradVec_guess) >= 0.0);

				if (n > 0) {
					// calculate kappaF since the temperature has changed
					for (int g = 0; g < nGroups_; ++g) {
						auto const nu_L = radBoundaries_g_copy[g];
						auto const nu_R = radBoundaries_g_copy[g + 1];
						auto const B_L = PlanckFunction(nu_L, T_d); // 4 pi B(nu) / c
						auto const B_R = PlanckFunction(nu_R, T_d); // 4 pi B(nu) / c
						auto const kappa_L = kappa_expo_and_lower_value[1][g];
						auto const kappa_R = kappa_L * std::pow(nu_R / nu_L, kappa_expo_and_lower_value[0][g]);
						delta_nu_kappa_B_at_edge[g] = nu_R * kappa_R * B_R - nu_L * kappa_L * B_L;
						delta_nu_B_at_edge[g] = nu_R * B_R - nu_L * B_L;
					}
					if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
						kappaFVec = kappaPVec;
					} else {
						if constexpr (use_diffuse_flux_mean_opacity) {
							kappaFVec =
							    ComputeDiffusionFluxMeanOpacity(kappaPVec, kappaEVec, fourPiBoverC, delta_nu_kappa_B_at_edge,
											    delta_nu_B_at_edge, kappa_expo_and_lower_value[0]);
						} else {
							// for simplicity, I assume kappaF = kappaE when opacity_model_ ==
							// OpacityModel::PPL_opacity_full_spectrum, if !use_diffuse_flux_mean_opacity. We won't use this
							// option anyway.
							kappaFVec = kappaEVec;
						}
					}
				}
			} else { // if constexpr gamma_ == 1.0
				T_d = T_gas;
				if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
					kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, T_d);
					for (int g = 0; g < nGroups_; ++g) {
						kappaFVec[g] = kappa_expo_and_lower_value[1][g];
					}
				} else {
					kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, T_d);
					kappaFVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_quant_minus_one);
				}
			}

			// Erad_guess is the new radiation energy (excluding work term)
			// Egas_guess is the new gas internal energy

			// 2. Compute radiation flux update

			amrex::GpuArray<amrex::Real, 3> Frad_t0{};
			dMomentum = {0., 0., 0.};

			for (int g = 0; g < nGroups_; ++g) {
				Frad_t0[0] = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
				Frad_t0[1] = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
				Frad_t0[2] = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);

				if constexpr ((gamma_ != 1.0) && (beta_order_ == 1)) {
					const auto erad = EradVec_guess[g];
					std::array<double, 3> v_terms{};

					auto fx = Frad_t0[0] / (c_light_ * erad);
					auto fy = Frad_t0[1] / (c_light_ * erad);
					auto fz = Frad_t0[2] / (c_light_ * erad);
					double F_coeff = chat * rho * kappaFVec[g] * dt;
					auto Tedd = ComputeEddingtonTensor(fx, fy, fz);

					for (int n = 0; n < 3; ++n) {
						// compute thermal radiation term
						double Planck_term = NAN;

						if constexpr (include_delta_B) {
							Planck_term = kappaPVec[g] * fourPiBoverC[g] - 1.0 / 3.0 * delta_nu_kappa_B_at_edge[g];
						} else {
							Planck_term = kappaPVec[g] * fourPiBoverC[g];
						}

						Planck_term *= chat * dt * gasMtm0[n];

						// compute radiation pressure
						double pressure_term = 0.0;
						for (int z = 0; z < 3; ++z) {
							pressure_term += gasMtm0[z] * Tedd[n][z] * erad;
						}
						// Simplification: assuming Eddington tensors are the same for all groups, we have kappaP = kappaE
						if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
							pressure_term *= chat * dt * kappaEVec[g];
						} else {
							pressure_term *= chat * dt * (1.0 + kappa_expo_and_lower_value[0][g]) * kappaEVec[g];
						}

						v_terms[n] = Planck_term + pressure_term;
					}

					for (int n = 0; n < 3; ++n) {
						// Compute flux update
						Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

						// Compute conservative gas momentum update
						dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
					}
				} else { // NOT if constexpr (gamma_ != 1.0 && beta_order_ == 1), i.e. gamma_ == 1.0 or beta_order_ == 0
					for (int n = 0; n < 3; ++n) {
						Frad_t1[n][g] = Frad_t0[n] / (1.0 + rho * kappaFVec[g] * chat * dt);
						// Compute conservative gas momentum update
						dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
					}
				}
			} // end loop over radiation groups for flux update

			amrex::Real const x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0];
			amrex::Real const x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1];
			amrex::Real const x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2];

			// 3. Deal with the work term.
			if constexpr ((gamma_ != 1.0) && (beta_order_ == 1)) {
				// compute difference in gas kinetic energy before and after momentum update
				amrex::Real const Egastot1 = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas_guess);
				amrex::Real const Ekin1 = Egastot1 - Egas_guess;
				amrex::Real const dEkin_work = Ekin1 - Ekin0;

				if constexpr (include_work_term_in_source) {
					// New scheme: the work term is included in the source terms. The work done by radiation went to internal energy, but it
					// should go to the kinetic energy. Remove the work term from internal energy.
					Egas_guess -= dEkin_work;
				} else {
					// Old scheme: since the source term does not include work term, add the work term to radiation energy.

					// compute loss of radiation energy to gas kinetic energy
					auto dErad_work = -(c_hat_ / c_light_) * dEkin_work;

					// apportion dErad_work according to kappaF_i * (v * F_i)
					quokka::valarray<double, nGroups_> energyLossFractions{};
					if constexpr (nGroups_ == 1) {
						energyLossFractions[0] = 1.0;
					} else {
						// compute energyLossFractions
						for (int g = 0; g < nGroups_; ++g) {
							energyLossFractions[g] =
							    kappaFVec[g] * (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]);
						}
						auto energyLossFractionsTot = sum(energyLossFractions);
						if (energyLossFractionsTot != 0.0) {
							energyLossFractions /= energyLossFractionsTot;
						} else {
							energyLossFractions.fillin(0.0);
						}
					}
					for (int g = 0; g < nGroups_; ++g) {
						auto radEnergyNew = EradVec_guess[g] + dErad_work * energyLossFractions[g];
						// AMREX_ASSERT(radEnergyNew > 0.0);
						if (radEnergyNew < Erad_floor_) {
							// return energy to Egas_guess
							Egas_guess -= (Erad_floor_ - radEnergyNew) * (c / chat);
							radEnergyNew = Erad_floor_;
						}
						EradVec_guess[g] = radEnergyNew;
					}
				}
			} // End of step 3

			if constexpr ((beta_order_ == 0) || (gamma_ == 1.0) || (!include_work_term_in_source)) {
				break;
			} else {
				// If you are here, then you are using the new scheme. Step 3 is skipped. The work term is included in the source term, but it
				// is lagged. The work term is updated in the next step.
				for (int g = 0; g < nGroups_; ++g) {
					// copy work to work_prev
					work_prev[g] = work[g];
					// compute new work term from the updated radiation flux and velocity
					// work = v * F * chi
					if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
						work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) * kappaFVec[g] *
							  chat / (c * c) * dt;
					} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum ||
							     opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
						work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) *
							  (1.0 + kappa_expo_and_lower_value[0][g]) * kappaFVec[g] * chat / (c * c) * dt;
					}
				}

				// Check for convergence of the work term: if the relative change in the work term is less than 1e-13, then break the loop
				const double lag_tol = 1.0e-13;
				if ((sum(abs(work)) == 0.0) || ((c / chat) * sum(abs(work - work_prev)) < lag_tol * Etot0) ||
				    (sum(abs(work - work_prev)) <= lag_tol * sum(Rvec)) || (sum(abs(work - work_prev)) <= 1.0e-8 * sum(abs(work)))) {
					break;
				}
			}
		} // end full-step iteration

		AMREX_ASSERT_WITH_MESSAGE(ite < max_ite, "AddSourceTerms iteration failed to converge!");
		if (ite >= max_ite) {
			amrex::Gpu::Atomic::Add(p_num_failed_outer_local, 1);
		}

		// 4b. Store new radiation energy, gas energy
		// In the first stage of the IMEX scheme, the hydro quantities are updated by a fraction (defined by
		// gas_update_factor) of the time step.
		const auto x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0] * gas_update_factor;
		const auto x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1] * gas_update_factor;
		const auto x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2] * gas_update_factor;
		consNew(i, j, k, x1GasMomentum_index) = x1GasMom1;
		consNew(i, j, k, x2GasMomentum_index) = x2GasMom1;
		consNew(i, j, k, x3GasMomentum_index) = x3GasMom1;
		if constexpr (gamma_ != 1.0) {
			Egas_guess = Egas0 + (Egas_guess - Egas0) * gas_update_factor;
			consNew(i, j, k, gasInternalEnergy_index) = Egas_guess;
			consNew(i, j, k, gasEnergy_index) = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas_guess);
		} else {
			amrex::ignore_unused(EradVec_guess);
			amrex::ignore_unused(Egas_guess);
			amrex::ignore_unused(Egas0);
			amrex::ignore_unused(Etot0);
			amrex::ignore_unused(work);
			amrex::ignore_unused(work_prev);
			amrex::ignore_unused(kappaPVec);
			amrex::ignore_unused(kappaEVec);
			amrex::ignore_unused(kappaPoverE);
			amrex::ignore_unused(fourPiBoverC);
		}
		for (int g = 0; g < nGroups_; ++g) {
			if constexpr (gamma_ != 1.0) {
				consNew(i, j, k, radEnergy_index + numRadVars_ * g) = EradVec_guess[g];
			}
			consNew(i, j, k, x1RadFlux_index + numRadVars_ * g) = Frad_t1[0][g];
			consNew(i, j, k, x2RadFlux_index + numRadVars_ * g) = Frad_t1[1][g];
			consNew(i, j, k, x3RadFlux_index + numRadVars_ * g) = Frad_t1[2][g];
		}
	});
}

#endif // RAD_SOURCE_TERMS_MULTI_GROUP_HPP_