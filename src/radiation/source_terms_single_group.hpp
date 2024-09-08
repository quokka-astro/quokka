
template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_radiation,
					  const int stage, double dustGasCoeff)
{
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
		double lorentz_factor = NAN;
		double lorentz_factor_v = NAN;
		double lorentz_factor_v_v = NAN;
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
		if constexpr (nGroups_ > 1) {
			if constexpr (static_cast<int>(opacity_model_) > 0) {
				for (int g = 0; g < nGroups_; ++g) {
					radBoundaryRatios_copy[g] = radBoundaries_g_copy[g + 1] / radBoundaries_g_copy[g];
				}
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

				const double betaSqr = (x1GasMom0 * x1GasMom0 + x2GasMom0 * x2GasMom0 + x3GasMom0 * x3GasMom0) / (rho * rho * c * c);

				static_assert(beta_order_ <= 3);
				if constexpr ((beta_order_ == 0) || (beta_order_ == 1)) {
					lorentz_factor = 1.0;
					lorentz_factor_v = 1.0;
				} else if constexpr (beta_order_ == 2) {
					lorentz_factor = 1.0 + 0.5 * betaSqr;
					lorentz_factor_v = 1.0;
					lorentz_factor_v_v = 1.0;
				} else if constexpr (beta_order_ == 3) {
					lorentz_factor = 1.0 + 0.5 * betaSqr;
					lorentz_factor_v = 1.0 + 0.5 * betaSqr;
					lorentz_factor_v_v = 1.0;
				} else {
					lorentz_factor = 1.0 / sqrt(1.0 - betaSqr);
					lorentz_factor_v = lorentz_factor;
					lorentz_factor_v_v = lorentz_factor;
				}

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

				double F_G = NAN;
				double F_G_previous = NAN;
				double deltaEgas = NAN;
				quokka::valarray<double, nGroups_> deltaD{};
				quokka::valarray<double, nGroups_> F_D{};
				quokka::valarray<double, nGroups_> F_D_previous{};

				const double resid_tol = 1.0e-11; // 1.0e-15;
				const double resid_tol_weak = 1.0e-11;
				const int maxIter = 400;
				const int maxIter_weak = 100;
				int n = 0;
				// bool good_iteration = true;
				double F_sq = NAN;
				double F_sq_previous = 0.0;
				const double line_search_scale_factor = 0.9;
				quokka::valarray<amrex::Real, nGroups_ + 1> n_flips{};
				n_flips.fillin(0);
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
							if (T_d < 0.) {
								Egas_guess = Egas_guess - (1. - line_search_scale_factor) * deltaEgas;
								Rvec = Rvec - (1. - line_search_scale_factor) * deltaD;
								deltaEgas *= line_search_scale_factor;
								deltaD = deltaD * line_search_scale_factor;
								continue;
							}
						}
					}
					// AMREX_ASSERT(T_d >= 0.);


					fourPiBoverC = ComputeThermalRadiation(T_d, radBoundaries_g_copy);

					if constexpr (opacity_model_ == OpacityModel::single_group) {
						kappaPVec[0] = ComputePlanckOpacity(rho, T_d);
						kappaEVec[0] = ComputeEnergyMeanOpacity(rho, T_d);
					} else {
						kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, T_d);
						if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
							for (int g = 0; g < nGroups_; ++g) {
								kappaPVec[g] = kappa_expo_and_lower_value[1][g];
								kappaEVec[g] = kappa_expo_and_lower_value[1][g];
							}
						} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) {
							kappaPVec =
							    ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_quant_minus_one);
							kappaEVec = kappaPVec;
						} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
							if (n < max_ite_to_update_alpha_E) {
								alpha_B = ComputeRadQuantityExponents(fourPiBoverC, radBoundaries_g_copy);
								alpha_E = ComputeRadQuantityExponents(EradVec_guess, radBoundaries_g_copy);
							}
							kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_B);
							kappaEVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_E);
						}
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
						if constexpr (opacity_model_ == OpacityModel::single_group) {
							kappaFVec[0] = ComputeFluxMeanOpacity(rho, T_d);
						} else {
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
									kappaFVec = ComputeDiffusionFluxMeanOpacity(
									    kappaPVec, kappaEVec, fourPiBoverC, delta_nu_kappa_B_at_edge, delta_nu_B_at_edge,
									    kappa_expo_and_lower_value[0]);
								} else {
									// for simplicity, I assume kappaF = kappaE when opacity_model_ ==
									// OpacityModel::PPL_opacity_full_spectrum, if !use_diffuse_flux_mean_opacity. We won't
									// use this option anyway.
									kappaFVec = kappaEVec;
								}
							}
						}
						AMREX_ASSERT(!kappaFVec.hasnan());

						if constexpr ((beta_order_ != 0) && (include_work_term_in_source)) {
							// compute the work term at the old state
							// const double gamma = 1.0 / sqrt(1.0 - vsqr / (c * c));
							if (ite == 0) {
								if constexpr (opacity_model_ == OpacityModel::single_group) {
									const double frad0 = consPrev(i, j, k, x1RadFlux_index);
									const double frad1 = consPrev(i, j, k, x2RadFlux_index);
									const double frad2 = consPrev(i, j, k, x3RadFlux_index);
									// work = v * F * chi
									work[0] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2) *
										  (2.0 * kappaEVec[0] - kappaFVec[0]) * chat / (c * c) * lorentz_factor_v * dt;
								} else if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
									for (int g = 0; g < nGroups_; ++g) {
										const double frad0 = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
										const double frad1 = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
										const double frad2 = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
										// work = v * F * chi
										work[g] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2) *
											  kappaFVec[g] * chat / (c * c) * dt;
									}
								} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum ||
										     opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
									for (int g = 0; g < nGroups_; ++g) {
										const double frad0 = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
										const double frad1 = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
										const double frad2 = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
										// work = v * F * chi
										work[g] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2) *
											  (1.0 + kappa_expo_and_lower_value[0][g]) * kappaFVec[g] * chat /
											  (c * c) * dt;
									}
								}
							}
						}

						tau0 = dt * rho * kappaPVec * chat * lorentz_factor;
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
						tau = dt * rho * kappaPVec * chat * lorentz_factor;

						// NOTE: The commented out code below is an attempt to solve the netagive radiation energy problem by 
						// reverting Rvec to the previous state and updating Egas only. This is replaced by the `if (Egas_guess + deltaEgas <= 0.0)`
						// clause below. I keep this code here for future reference.
						// Given it some thought, I think EradVec_guess should be allowed to be negative in the Newton-Raphson iteration.

						// const bool good_iteration_previous = good_iteration;
						// good_iteration = true;
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
								// if (EradVec_guess[g] < 0.0) {
								// 	good_iteration = false;
								// 	Rvec[g] -= deltaD[g];
								// }
							}
						}
						// if (!good_iteration) {
						// 	AMREX_ASSERT_WITH_MESSAGE(good_iteration_previous, "Two consecutive bad iterations. Aborting.");
						// 	continue;
						// }
					}

					const double cscale = c / chat;
					F_G = Egas_guess - Egas0;
					F_D = EradVec_guess - Erad0Vec - (Rvec + Src);
					if constexpr (enable_line_cooling_) {
						// F_G += cscale * dt * sum(ComputeLineCooling(rho, T_gas));
						// F_D -= dt * ComputeLineCooling(rho, T_gas);
					}
					if constexpr (enable_photoelectric_heating_) {
						F_G -= dt * DefinePhotoelectricHeatingE1Derivative(T_gas, num_den) * EradVec_guess[nGroups_ - 1];
					}
					F_sq = F_G * F_G;
					double F_D_abs_sum = 0.0;
					for (int g = 0; g < nGroups_; ++g) {
						if (tau[g] > 0.0) {
							F_G += (c / chat) * Rvec[g];
							F_D_abs_sum += std::abs(F_D[g]);
							F_sq += (c / chat) * Rvec[g] * (c / chat) * Rvec[g];
							F_sq += F_D[g] * F_D[g];
						}
					}

					const double local_resid_tol = (n < maxIter_weak) ? resid_tol : resid_tol_weak;
					if (n > 0) {
						if (std::abs(F_G) > local_resid_tol * Etot0 && F_G * F_G_previous <= 0.0) {
							n_flips[nGroups_] += 1;
						} else {
							n_flips[nGroups_] = 0;
						}
						if ((c / chat) * F_D_abs_sum > local_resid_tol * Etot0) {
							for (int g = 0; g < nGroups_; ++g) {
								if (tau[g] > 0.0) {
									if (F_D[g] * F_D_previous[g] <= 0.0) {
										n_flips[g] += 1;
									} else {
										n_flips[g] = 0;
									}
								} else {
									n_flips[g] = 0;
								}
							}
						} else {
							for (int g = 0; g < nGroups_; ++g) {
								n_flips[g] = 0;
							}
						}
					}
					F_G_previous = F_G;
					F_D_previous = F_D;

					if constexpr (use_backward_line_search) {
						// if (n > 0 && F_sq >= F_sq_previous) {
						const double flipping_line_search_factor = 0.5;
						if (max(n_flips) >= 5) {
							n_flips.fillin(0);
							Egas_guess = Egas_guess - (1. - flipping_line_search_factor) * deltaEgas;
							Rvec = Rvec - (1. - flipping_line_search_factor) * deltaD;
							deltaEgas *= flipping_line_search_factor;
							deltaD = deltaD * flipping_line_search_factor;
							continue;
						}

						F_sq_previous = F_sq;
					}

					if (n > 200) {
#ifndef NDEBUG
						std::cout << "n = " << n << ", F_G = " << F_G << ", F_D_abs_sum = " << F_D_abs_sum
							  << ", F_D_abs_sum / Etot0 = " << F_D_abs_sum / Etot0 << std::endl;
#endif
					}

					// check relative convergence of the residuals
					if ((std::abs(F_G / Etot0) < local_resid_tol) && ((c / chat) * F_D_abs_sum / Etot0 < local_resid_tol)) {
						break;
					}

					const double c_v = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

					const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivative(T_d, radBoundaries_g_copy);
					AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());

					// compute Jacobian elements
					// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
					// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

					const double y0 = -F_G;
					auto yg = -1. * F_D;

					quokka::valarray<double, nGroups_> dF0_dXg{};
					quokka::valarray<double, nGroups_> dFg_dX0{};
					quokka::valarray<double, nGroups_> dFg_dXg{};
					quokka::valarray<double, nGroups_> dFg_dX1{};

					// Photoelectric heating from the p'th group. Rearrange it to the last (N_g'th) group in the Jacobian matrix.
					// TODO(CCH): do the reordering to make FUV the last group

					// M_00
					const double dF0_dX0 = 1.0;
					if constexpr (enable_line_cooling_) {
						// add d Lambda_g / d T
						// dF0_dX0 = dF0_dX0 + cscale / c_v * sum(dLambda_g_dT);
					}
					// M_0g
					dF0_dXg.fillin(cscale);
					double photoheating = NAN;
					if constexpr (enable_photoelectric_heating_) {
						photoheating = dt * DefinePhotoelectricHeatingE1Derivative(T_gas, num_den);
						if (tau[nGroups_ - 1] <= 0.0) {
							// dF0_dXg[nGroups_ - 1] = std::numeric_limits<double>::infinity();
							dF0_dXg[nGroups_ - 1] = LARGE;
						} else {
							dF0_dXg[nGroups_ - 1] += photoheating * kappaPoverE[nGroups_ - 1] / tau[nGroups_ - 1];
						}
					}
					// M_gg, same for dust and dust-free cases; same with or without cooling
					for (int g = 0; g < nGroups_; ++g) {
						if (tau[g] <= 0.0) {
							// dFg_dXg[g] = -std::numeric_limits<double>::infinity();
							dFg_dXg[g] = - LARGE;
						} else {
							dFg_dXg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
						}
					}
					// M_g0
					if constexpr (!enable_dust_gas_thermal_coupling_model_) {
						dFg_dX0 = 1.0 / c_v * kappaPoverE * d_fourpiboverc_d_t;
						if constexpr (enable_line_cooling_) {
							// add d Lambda_g / d T
							// dFg_dX0 = dFg_dX0 - 1.0 / (c_v * cscale) * dt * dLambda_g_dT;
						}
					} else {
						const double d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
						const double coeff_n = dt * dustGasCoeff_local * num_den * num_den / cscale;
						const double dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));
						const auto rg = kappaPoverE * d_fourpiboverc_d_t * dTd_dRg;
						dFg_dX0 = 1.0 / c_v * kappaPoverE * d_fourpiboverc_d_t * d_Td_d_T;
						if constexpr (enable_line_cooling_) {
							// add d Lambda_g / d T
							// dFg_dX0 = dFg_dX0 - 1.0 / (c_v * cscale) * dt * dLambda_g_dT;
						}
						dFg_dX0 = dFg_dX0 - 1.0 / cscale * rg * dF0_dX0;
						yg = yg - 1.0 / cscale * rg * y0;
						if constexpr (enable_photoelectric_heating_) {
							dFg_dX1 = rg - 1.0 / cscale * rg * dF0_dXg[nGroups_ - 1]; // note that this is the (nGroups_ - 1)th column, but with a wrong value for for the (nGroups_ - 1)th row
							// this is the (nGroups_ - 1)th row of the (nGroups_ - 1)th column
							if (tau[nGroups_ - 1] <= 0.0) {
								// dFg_dXg[nGroups_ - 1] = -std::numeric_limits<double>::infinity();
								dFg_dXg[nGroups_ - 1] = - LARGE;
							} else {
								dFg_dXg[nGroups_ - 1] -= rg[nGroups_ - 1] / cscale * photoheating * kappaPoverE[nGroups_ - 1] / tau[nGroups_ - 1];
							}
						}
					}

					if constexpr (use_D_as_base) {
						dF0_dXg = dF0_dXg * tau0;
						dFg_dXg = dFg_dXg * tau0;
					}

					// update variables
					if constexpr (enable_photoelectric_heating_ && enable_dust_gas_thermal_coupling_model_) {
						RadSystem<problem_t>::SolveLinearEqsWithLastColumn(dF0_dX0, dF0_dXg, dFg_dX0, dFg_dXg, dFg_dX1, y0, yg, deltaEgas, deltaD);
					} else {
						RadSystem<problem_t>::SolveLinearEqs(dF0_dX0, dF0_dXg, dFg_dX0, dFg_dXg, y0, yg, deltaEgas, deltaD);
					}

					AMREX_ASSERT(!std::isnan(deltaEgas));
					AMREX_ASSERT(!deltaD.hasnan());

					if (Egas_guess + deltaEgas <= 0.0) {
						// Egas_guess + deltaEgas < 0 usually happens when Egas_guess = 0, and yg is horizontal at R = R0.
						// If this happens, take a better guess at the intersection of R = R0 and Egas - Egas0 + R0 = 0, which gives 
						// Egas_guess = Egas0 - sum(R0).
						deltaEgas = - sum(Rvec);
						deltaD.fillin(0.0);
						Egas_guess += deltaEgas;
					} else {
						Egas_guess += deltaEgas;
						if constexpr (use_D_as_base) {
							deltaD = tau0 * deltaD;
						}
						Rvec += deltaD;
					}
					CUSTOM_ASSERT_WITH_MESSAGE(Egas_guess > 0.0, "Egas_guess <= 0 after step ", n);

					// check relative and absolute convergence of E_r
					// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
					// 	break;
					// }
				} // END NEWTON-RAPHSON LOOP

				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration failed to converge!");
				// std::cout << "Newton-Raphson converged after " << n << " it." << std::endl;
				AMREX_ALWAYS_ASSERT(Egas_guess > 0.0);
				AMREX_ALWAYS_ASSERT(min(EradVec_guess) >= 0.0);

				if (n > 0) {
					// calculate kappaF since the temperature has changed
					if constexpr (opacity_model_ == OpacityModel::single_group) {
						kappaFVec[0] = ComputeFluxMeanOpacity(rho, T_d);
					} else {
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
								// OpacityModel::PPL_opacity_full_spectrum, if !use_diffuse_flux_mean_opacity. We won't use this
								// option anyway.
								kappaFVec = kappaEVec;
							}
						}
					}
				}
			} else { // if constexpr gamma_ == 1.0
				T_d = T_gas;
				if constexpr (opacity_model_ == OpacityModel::single_group) {
					kappaFVec[0] = ComputeFluxMeanOpacity(rho, T_d);
				} else if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
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

				if constexpr ((gamma_ != 1.0) && (beta_order_ != 0)) {
					auto erad = EradVec_guess[g];
					std::array<double, 3> gasVel{};
					std::array<double, 3> v_terms{};

					auto fx = Frad_t0[0] / (c_light_ * erad);
					auto fy = Frad_t0[1] / (c_light_ * erad);
					auto fz = Frad_t0[2] / (c_light_ * erad);
					double F_coeff = chat * rho * kappaFVec[g] * dt * lorentz_factor;
					auto Tedd = ComputeEddingtonTensor(fx, fy, fz);

					for (int n = 0; n < 3; ++n) {
						// compute thermal radiation term
						double Planck_term = NAN;
						if constexpr (opacity_model_ == OpacityModel::single_group) {
							Planck_term = kappaPVec[g] * fourPiBoverC[g] * lorentz_factor_v;
							// compute (kappa_F - kappa_E) term
							if (kappaFVec[g] != kappaEVec[g]) {
								Planck_term += (kappaFVec[g] - kappaEVec[g]) * erad * std::pow(lorentz_factor_v, 3);
							}
						} else {
							if constexpr (include_delta_B) {
								Planck_term = kappaPVec[g] * fourPiBoverC[g] - 1.0 / 3.0 * delta_nu_kappa_B_at_edge[g];
							} else {
								Planck_term = kappaPVec[g] * fourPiBoverC[g];
							}
						}
						Planck_term *= chat * dt * gasMtm0[n];

						// compute radiation pressure
						double pressure_term = 0.0;
						for (int z = 0; z < 3; ++z) {
							pressure_term += gasMtm0[z] * Tedd[n][z] * erad;
						}
						if constexpr (opacity_model_ == OpacityModel::single_group) {
							pressure_term *= chat * dt * kappaFVec[g] * lorentz_factor_v;
						} else {
							// Simplification: assuming Eddington tensors are the same for all groups, we have kappaP = kappaE
							if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
								pressure_term *= chat * dt * kappaEVec[g];
							} else {
								pressure_term *= chat * dt * (1.0 + kappa_expo_and_lower_value[0][g]) * kappaEVec[g];
							}
						}

						v_terms[n] = Planck_term + pressure_term;
					}

					if constexpr (beta_order_ == 1) {
						for (int n = 0; n < 3; ++n) {
							// Compute flux update
							Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

							// Compute conservative gas momentum update
							dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
						}
					} else {
						if (kappaFVec[g] == kappaEVec[g]) {
							for (int n = 0; n < 3; ++n) {
								// Compute flux update
								Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

								// Compute conservative gas momentum update
								dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
							}
						} else {
							const double K0 =
							    2.0 * rho * chat * dt * (kappaFVec[g] - kappaEVec[g]) / c / c * std::pow(lorentz_factor_v_v, 3);

							// A test to see if this routine reduces to the correct result when ignoring the beta^2 terms
							// const double X0 = 1.0 + rho * chat * dt * (kappaFVec[g]);
							// const double K0 = 0.0;

							// Solve 3x3 matrix equation A * x = B, where A[i][j] = delta_ij * X0 + K0 * v_i * v_j and B[i] =
							// O_beta_tau_terms[i] + Frad_t0[i]
							const double A00 = 1.0 + F_coeff + K0 * gasVel[0] * gasVel[0];
							const double A01 = K0 * gasVel[0] * gasVel[1];
							const double A02 = K0 * gasVel[0] * gasVel[2];

							const double A10 = K0 * gasVel[1] * gasVel[0];
							const double A11 = 1.0 + F_coeff + K0 * gasVel[1] * gasVel[1];
							const double A12 = K0 * gasVel[1] * gasVel[2];

							const double A20 = K0 * gasVel[2] * gasVel[0];
							const double A21 = K0 * gasVel[2] * gasVel[1];
							const double A22 = 1.0 + F_coeff + K0 * gasVel[2] * gasVel[2];

							const double B0 = v_terms[0] + Frad_t0[0];
							const double B1 = v_terms[1] + Frad_t0[1];
							const double B2 = v_terms[2] + Frad_t0[2];

							auto [sol0, sol1, sol2] = Solve3x3matrix(A00, A01, A02, A10, A11, A12, A20, A21, A22, B0, B1, B2);
							Frad_t1[0][g] = sol0;
							Frad_t1[1][g] = sol1;
							Frad_t1[2][g] = sol2;
							for (int n = 0; n < 3; ++n) {
								dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
							}
						}
					}
				} else { // if constexpr (gamma_ == 1.0 || beta_order_ == 0)
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
			if constexpr ((gamma_ != 1.0) && (beta_order_ != 0)) {
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
					if constexpr (opacity_model_ == OpacityModel::single_group) {
						work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) * chat / (c * c) *
							  lorentz_factor_v * (2.0 * kappaEVec[g] - kappaFVec[g]) * dt;
					} else if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
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
				if ((sum(abs(work)) == 0.0) || ((c / chat) * sum(abs(work - work_prev)) / Etot0 < lag_tol) ||
				    (sum(abs(work - work_prev)) <= lag_tol * sum(Rvec)) ||
				    (sum(abs(work)) > 0.0 && sum(abs(work - work_prev)) <= 1.0e-8 * sum(abs(work)))) {
					break;
				}
			}
		} // end full-step iteration

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ite < max_ite, "AddSourceTerms iteration failed to converge!");

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

#endif // RADIATION_SYSTEM_HPP_