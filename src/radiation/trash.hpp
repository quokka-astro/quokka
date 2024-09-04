
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
						const double R_sum = n == 0 ? NAN : sum(Rvec);
						T_d = ComputeDustTemperature(T_gas, T_gas, rho, EradVec_guess, coeff_n, dt, R_sum, n,
												radBoundaries_g_copy, radBoundaryRatios_copy);
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

					const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, radBoundaries_g_copy);
					AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());
					const double c_v = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

					const auto Egas_diff = Egas_guess - Egas0;
					const auto Erad_diff = EradVec_guess - Erad0Vec;
					JacobianResult<problem_t> jacobian;

					if (enable_dust_gas_thermal_coupling_model_) {
						jacobian = ComputeJacobianForGasAndDust(T_gas, T_d, Egas_diff, Erad_diff, Rvec, Src, coeff_n, tau, c_v, cscale,
											kappaPoverE, d_fourpiboverc_d_t);
					} else {
						jacobian = ComputeJacobianForPureGas(NAN, NAN, Egas_diff, Erad_diff, Rvec, Src, NAN, tau, c_v, cscale,
										     kappaPoverE, d_fourpiboverc_d_t);
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
					// enable_dE_constrain is used to prevent the gas temperature from dropping/increasing below/above the radiation
					// temperature
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



				// // Step 1.4: If the temperature has changed, update kappaF

				// if (updated_energy.n > 0) {
				// 	// calculate kappaF since the temperature has changed
				// 	for (int g = 0; g < nGroups_; ++g) {
				// 		auto const nu_L = radBoundaries_g_copy[g];
				// 		auto const nu_R = radBoundaries_g_copy[g + 1];
				// 		auto const B_L = PlanckFunction(nu_L, T_d); // 4 pi B(nu) / c
				// 		auto const B_R = PlanckFunction(nu_R, T_d); // 4 pi B(nu) / c
				// 		auto const kappa_L = kappa_expo_and_lower_value[1][g];
				// 		auto const kappa_R = kappa_L * std::pow(nu_R / nu_L, kappa_expo_and_lower_value[0][g]);
				// 		delta_nu_kappa_B_at_edge[g] = nu_R * kappa_R * B_R - nu_L * kappa_L * B_L;
				// 		delta_nu_B_at_edge[g] = nu_R * B_R - nu_L * B_L;
				// 	}
				// 	if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
				// 		kappaFVec = kappaPVec;
				// 	} else {
				// 		if constexpr (use_diffuse_flux_mean_opacity) {
				// 			kappaFVec =
				// 			    ComputeDiffusionFluxMeanOpacity(kappaPVec, kappaEVec, fourPiBoverC, delta_nu_kappa_B_at_edge,
				// 							    delta_nu_B_at_edge, kappa_expo_and_lower_value[0]);
				// 		} else {
				// 			// for simplicity, I assume kappaF = kappaE when opacity_model_ ==
				// 			// OpacityModel::PPL_opacity_full_spectrum, if !use_diffuse_flux_mean_opacity. We won't use this
				// 			// option anyway.
				// 			kappaFVec = kappaEVec;
				// 		}
				// 	}
				// }