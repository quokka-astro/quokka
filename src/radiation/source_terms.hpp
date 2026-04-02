// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RAD_SOURCE_TERMS_HPP_ // NOLINT
#define RAD_SOURCE_TERMS_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

#define LARGE 1.0e100

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, Real dt_implicit,
						     double gas_update_factor_in, double dustGasCoeff, double tol_h, double /*tol_rel_h*/, double /*tempFloor*/,
						     int *p_iteration_counter, int *p_iteration_failure_counter)
{
	arrayconst_t &consPrev = consVar; // make read-only
	array_t &consNew = consVar;
	auto dt = dt_implicit;

	// don't need radBoundaries_g for single-group

	// Add source terms

	// 1. Compute gas energy and radiation energy update following the scheme of Howell &
	// Greenough [Journal of Computational Physics 184 (2003) 53–78], which was later modified by
	// He, Wibking, & Krumholz (2024)

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		auto p_iteration_counter_local = p_iteration_counter;		      // NOLINT
		auto p_iteration_failure_counter_local = p_iteration_failure_counter; // NOLINT

		const double c = c_light_;
		const double chat = c_hat_;
		const double dustGasCoeff_ = dustGasCoeff;
		const double resid_tol = tol_h;

		// load fluid properties
		const double rho = consPrev(i, j, k, gasDensity_index);
		const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
		const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
		const std::array<double, 3> gasMtm0 = {x1GasMom0, x2GasMom0, x3GasMom0};
		const double Egastot0 = consPrev(i, j, k, gasEnergy_index);
		auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consPrev, i, j, k);

		// load radiation energy
		const double Erad0 = consPrev(i, j, k, radEnergy_index);
		AMREX_ASSERT(Erad0 > 0.0);

		const double cscale = c / chat;

		// load radiation energy source term
		// plus advection source term (for well-balanced/SDC integrators)
		// Note that radEnergySource should contain the luminosity volume density, L / V; unit: erg s^-1 cm^-3
		const double Src = radEnergySource(i, j, k, 0) * dt / cscale;
		if constexpr (gamma_ != 1.0) {
			AMREX_ASSERT(Src >= 0.0);
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
		double fourPiBoverC = NAN;
		double Erad_guess = NAN;
		double kappaP = NAN;
		double kappaE = NAN;
		double kappaF = NAN;
		double kappaPoverE = NAN;
		double work = 0.0;
		double work_prev = 0.0;
		amrex::GpuArray<Real, 3> dMomentum{};
		amrex::GpuArray<Real, 3> Frad_t1{};

		if constexpr (gamma_ != 1.0) {
			Egas0 = ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0);
			Etot0 = Egas0 + cscale * (Erad0 + Src);
			AMREX_ASSERT(Egas0 > 0.0);
		}

		const Real gas_update_factor = gas_update_factor_in;

		double coeff_n = NAN;
		const double H_num_den = ComputeNumberDensityH(rho, massScalars);
		if constexpr (enable_dust_gas_thermal_coupling_model_) {
			coeff_n = dt * dustGasCoeff_ * H_num_den * H_num_den / cscale;
		} else {
			amrex::ignore_unused(coeff_n);
			amrex::ignore_unused(H_num_den);
			amrex::ignore_unused(dustGasCoeff_);
		}

		const int max_ite = 5;
		int ite = 0;
		for (; ite < max_ite; ++ite) {
			double R = NAN;

			Erad_guess = Erad0;

			if constexpr (gamma_ != 1.0) {
				double tau0 = NAN;
				double tau = NAN;

				Egas_guess = Egas0;
				Ekin0 = Egastot0 - Egas0;

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

				// BEGIN NEWTON-RAPHSON LOOP (this is written for multi-group, but it's valid for single-group if we set i == 0)
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
				double deltaEgas = NAN;
				double deltaR = NAN;
				double F_D = NAN;

				const int maxIter = 100;
				int n = 0;
				for (; n < maxIter; ++n) {
					T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
					AMREX_ASSERT(T_gas >= 0.);

					// dust temperature
					if constexpr (!enable_dust_gas_thermal_coupling_model_) {
						T_d = T_gas;
					} else {
						const quokka::valarray<double, 1> Erad_guess_vec{Erad_guess};
						T_d = ComputeDustTemperatureBateKeto(T_gas, T_gas, rho, Erad_guess_vec, coeff_n, dt, R, n);
						AMREX_ASSERT_WITH_MESSAGE(T_d >= 0., "Dust temperature is negative!");
						if (T_d < 0.0) {
							amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[1], 1); // NOLINT
						}
					}

					fourPiBoverC = ComputeThermalRadiationSingleGroup(T_d);

					kappaP = ComputePlanckOpacity(rho, T_d);
					kappaE = ComputeEnergyMeanOpacity(rho, T_d);
					AMREX_ASSERT(!std::isnan(kappaP));
					AMREX_ASSERT(!std::isnan(kappaE));
					AMREX_ASSERT(kappaP >= 0.0);
					AMREX_ASSERT(kappaE >= 0.0);
					if (kappaE > 0.0) {
						kappaPoverE = kappaP / kappaE;
					} else {
						kappaPoverE = 1.0;
					}

					// In the first loop, calculate kappaF, work, tau0, R
					if (n == 0) {
						kappaF = ComputeFluxMeanOpacity(rho, T_d);
						AMREX_ASSERT(!std::isnan(kappaF));

						if constexpr ((beta_order_ != 0) && (include_work_term_in_source)) {
							// compute the work term at the old state
							if (ite == 0) {
								const double frad0 = consPrev(i, j, k, x1RadFlux_index);
								const double frad1 = consPrev(i, j, k, x2RadFlux_index);
								const double frad2 = consPrev(i, j, k, x3RadFlux_index);
								// work = v * F * chi
								work = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2) * (2.0 * kappaE - kappaF) *
								       chat / (c * c) * lorentz_factor_v * dt;
							}
						}

						tau0 = dt * rho * kappaP * chat * lorentz_factor;
						tau = tau0;
						R = (fourPiBoverC - Erad_guess / kappaPoverE) * tau0 + work;
						// tau0 is used as a scaling factor for Rvec
						tau0 = std::max(tau0, 1.0);
					} else { // in the second and later loops, calculate tau and E (given R)
						tau = dt * rho * kappaP * chat * lorentz_factor;
						if (tau > 0.0) {
							Erad_guess = kappaPoverE * (fourPiBoverC - (R - work) / tau);
							if constexpr (force_rad_floor_in_iteration) {
								if (Erad_guess <= 0.0) {
									Egas_guess -= (c / chat) * (Erad_floor_ - Erad_guess);
									Erad_guess = Erad_floor_;
								}
							}
							// In general, Erad_guess is not guaranteed to be positive during the iteration steps.
							// This is fine as long as it is positive at the end of the iteration. However, if Erad_guess
							// is negative in the last iteration, we need to turn on force_rad_floor_in_iteration to
							// correct it. This is a backup safety measure. For all the test problems I have tried, this
							// is not necessary.
						}
					}

					double cooling = 0.0;
					double cooling_derivative = 0.0;
					const double CR_heating = DefineCosmicRayHeatingRate(H_num_den) * dt;
					if constexpr (enable_dust_gas_thermal_coupling_model_) {
						cooling = DefineNetCoolingRate(T_gas, H_num_den)[0];
						cooling_derivative = DefineNetCoolingRateTempDerivative(T_gas, H_num_den)[0];
					}

					// Check for convergence. We need to take care of a special situation when tau is very small, in which case the source
					// term won't be able to cancel the residual no matter how many iterations we try. This could happen when Src is
					// non-zero or when the opacity is a sharp function of temperature. We set the criterion to be that: tau *
					// std::max(a_rad * T_gas^4, E_tot0) < resid_tol * Etot0.

					F_G = Egas_guess - Egas0 + cscale * R + cooling * dt - CR_heating;
					F_D = Erad_guess - Erad0 - (R + Src);
					double F_D_abs = 0.0;
					if (tau * std::max(radiation_constant_ * std::pow(T_gas, 4), Etot0) < resid_tol * Etot0) {
						Erad_guess = Erad0 + Src;
						F_D = 0.0;
						F_D_abs = 0.0;
					} else {
						F_D_abs = std::abs(F_D);
					}
					if constexpr (add_line_cooling_to_radiation_in_jac) {
						F_D -= (1.0 / cscale) * cooling * dt;
					}

					// check relative convergence of the residuals
					if ((std::abs(F_G) < resid_tol * Etot0) && (cscale * F_D_abs < resid_tol * Etot0)) {
						break;
					}

					const double c_v = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

					const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeSingleGroup(T_d);
					AMREX_ASSERT(!std::isnan(d_fourpiboverc_d_t));

					// compute Jacobian elements
					// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
					// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

					auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;

					double J00 = NAN;
					double J01 = NAN;
					double J10 = NAN;
					double J11 = NAN;

					if constexpr (!enable_dust_gas_thermal_coupling_model_) {
						J00 = 1.0 + cooling_derivative * dt / c_v;
						J01 = cscale;
						J10 = 1.0 / c_v * dEg_dT - (1 / cscale) * cooling_derivative * dt;
						if (tau <= 0.0) {
							J11 = -std::numeric_limits<double>::infinity();
						} else {
							J11 = -1.0 * kappaPoverE / tau - 1.0;
						}
					} else {
						const double d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
						dEg_dT *= d_Td_d_T;
						const double dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));

						J00 = 1.0;
						J01 = cscale;
						J10 = 1.0 / c_v * dEg_dT;
						if (tau <= 0.0) {
							J11 = -LARGE;
						} else {
							J11 = kappaPoverE * d_fourpiboverc_d_t * dTd_dRg - kappaPoverE / tau - 1.0;
						}
					}

					AMREX_ASSERT(!std::isnan(J10));
					AMREX_ASSERT(!std::isnan(J11));

					const double y0 = -F_G;
					const auto y1 = -1. * F_D;

					// solve the linear system
					const double det = J00 * J11 - J01 * J10;
					AMREX_ASSERT(det != 0.0);
					deltaEgas = (J11 * y0 - J01 * y1) / det;
					deltaR = (J00 * y1 - J10 * y0) / det;

					if (!enable_dE_constrain) {
						Egas_guess += deltaEgas;
						R += deltaR;
					} else {
						double T_rad = NAN;
						AMREX_ASSERT(Erad_guess >= 0.0);
						T_rad = std::sqrt(std::sqrt(Erad_guess / radiation_constant_));
						if (deltaEgas / c_v > std::max(T_gas, T_rad)) {
							Egas_guess = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_rad);
							// R = 0.0;
						} else {
							Egas_guess += deltaEgas;
							R += deltaR;
						}
					}

				} // END NEWTON-RAPHSON LOOP

				AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration failed to converge!");
				if (n >= maxIter) {
					amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[0], 1); // NOLINT
				}

				// update iteration counter: (+1, +ite, max(self, ite))
				amrex::Gpu::Atomic::Add(&p_iteration_counter_local[0], 1);     // total number of radiation updates. NOLINT
				amrex::Gpu::Atomic::Add(&p_iteration_counter_local[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
				amrex::Gpu::Atomic::Max(&p_iteration_counter_local[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT

				AMREX_ASSERT(Egas_guess > 0.0);
				AMREX_ASSERT(Erad_guess >= 0.0);

				if constexpr (!add_line_cooling_to_radiation_in_jac) {
					const auto cooling_tend = DefineNetCoolingRate(T_gas, H_num_den)[0] * dt;
					AMREX_ASSERT_WITH_MESSAGE(cooling_tend >= 0.,
								  "add_line_cooling_to_radiation has to be enabled when there is negative cooling rate!");
					// TODO(CCH): potential GPU-related issue here.
					Erad_guess += (1 / cscale) * cooling_tend;
				}

				if (n > 0) {
					// calculate kappaF since the temperature has changed
					kappaF = ComputeFluxMeanOpacity(rho, T_d);
				}
			} else { // if constexpr gamma_ == 1.0
				T_d = T_gas;
				kappaF = ComputeFluxMeanOpacity(rho, T_d);

				amrex::ignore_unused(p_iteration_counter_local);
				amrex::ignore_unused(Ekin0);
				amrex::ignore_unused(lorentz_factor);
				amrex::ignore_unused(lorentz_factor_v);
				amrex::ignore_unused(lorentz_factor_v_v);
				amrex::ignore_unused(work_prev);
				amrex::ignore_unused(R);
			}

			// Egas_guess is the new gas internal energy
			// Erad_guess is the new radiation energy (excluding work term)

			// 2. Compute radiation flux update

			amrex::GpuArray<amrex::Real, 3> Frad_t0{};
			dMomentum = {0., 0., 0.};

			Frad_t0[0] = consPrev(i, j, k, x1RadFlux_index);
			Frad_t0[1] = consPrev(i, j, k, x2RadFlux_index);
			Frad_t0[2] = consPrev(i, j, k, x3RadFlux_index);

			if constexpr ((gamma_ != 1.0) && (beta_order_ != 0)) {
				auto erad = Erad_guess;
				std::array<double, 3> gasVel{};
				std::array<double, 3> v_terms{};

				auto fx = Frad_t0[0] / (c_light_ * erad);
				auto fy = Frad_t0[1] / (c_light_ * erad);
				auto fz = Frad_t0[2] / (c_light_ * erad);
				const double F_coeff = chat * rho * kappaF * dt * lorentz_factor;
				auto Tedd = ComputeEddingtonTensor(fx, fy, fz);

				for (int n = 0; n < 3; ++n) {
					// compute thermal radiation term
					double Planck_term = kappaP * fourPiBoverC * lorentz_factor_v;
					// compute (kappa_F - kappa_E) term
					if (kappaF != kappaE) {
						Planck_term += (kappaF - kappaE) * erad * std::pow(lorentz_factor_v, 3);
					}
					Planck_term *= chat * dt * gasMtm0[n];

					// compute radiation pressure
					double pressure_term = 0.0;
					for (int z = 0; z < 3; ++z) {
						pressure_term += gasMtm0[z] * Tedd[n][z] * erad;
					}
					pressure_term *= chat * dt * kappaF * lorentz_factor_v;

					v_terms[n] = Planck_term + pressure_term;
				}

				if constexpr (beta_order_ == 1) {
					for (int n = 0; n < 3; ++n) {
						// Compute flux update
						Frad_t1[n] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

						// Compute conservative gas momentum update
						dMomentum[n] += -(Frad_t1[n] - Frad_t0[n]) / (c * chat);
					}
				} else {
					if (kappaF == kappaE) {
						for (int n = 0; n < 3; ++n) {
							// Compute flux update
							Frad_t1[n] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

							// Compute conservative gas momentum update
							dMomentum[n] += -(Frad_t1[n] - Frad_t0[n]) / (c * chat);
						}
					} else {
						const double K0 = 2.0 * rho * chat * dt * (kappaF - kappaE) / c / c * std::pow(lorentz_factor_v_v, 3);

						// A test to see if this routine reduces to the correct result when ignoring the beta^2 terms
						// const double X0 = 1.0 + rho * chat * dt * (kappaF);
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
						Frad_t1[0] = sol0;
						Frad_t1[1] = sol1;
						Frad_t1[2] = sol2;
						for (int n = 0; n < 3; ++n) {
							dMomentum[n] += -(Frad_t1[n] - Frad_t0[n]) / (c * chat);
						}
					}
				}
			} else { // if constexpr (gamma_ == 1.0 || beta_order_ == 0)
				for (int n = 0; n < 3; ++n) {
					Frad_t1[n] = Frad_t0[n] / (1.0 + rho * kappaF * chat * dt);
					// Compute conservative gas momentum update
					dMomentum[n] += -(Frad_t1[n] - Frad_t0[n]) / (c * chat);
				}
			}

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

					auto radEnergyNew = Erad_guess + dErad_work;
					// AMREX_ASSERT(radEnergyNew > 0.0);
					if (radEnergyNew < Erad_floor_) {
						// return energy to Egas_guess
						Egas_guess -= (Erad_floor_ - radEnergyNew) * (c / chat);
						radEnergyNew = Erad_floor_;
					}
					Erad_guess = radEnergyNew;
				}
			} // End of step 3

			if constexpr ((beta_order_ == 0) || (gamma_ == 1.0) || (!include_work_term_in_source)) {
				break;
			} else {
				// If you are here, then you are using the new scheme. Step 3 is skipped. The work term is included in the source term, but it
				// is lagged. The work term is updated in the next step.
				work_prev = work;
				// compute new work term from the updated radiation flux and velocity
				// work = v * F * chi
				work = (x1GasMom1 * Frad_t1[0] + x2GasMom1 * Frad_t1[1] + x3GasMom1 * Frad_t1[2]) * chat / (c * c) * lorentz_factor_v *
				       (2.0 * kappaE - kappaF) * dt;

				// Check for convergence of the work term: if the relative change in the work term is less than 1e-13, then break the loop
				const double lag_tol = 1.0e-13;
				if ((std::abs(work) == 0.0) || (cscale * std::abs(work - work_prev) < lag_tol * Etot0) ||
				    (std::abs(work - work_prev) <= lag_tol * R) || (std::abs(work - work_prev) <= 1.0e-8 * std::abs(work))) {
					break;
				}
			}
		} // end full-step iteration

		AMREX_ASSERT_WITH_MESSAGE(ite < max_ite, "AddSourceTerms outer iteration failed to converge!");
		if (ite >= max_ite) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[2], 1); // NOLINT
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
			consNew(i, j, k, radEnergy_index) = Erad_guess;
		} else {
			amrex::ignore_unused(Erad_guess);
			amrex::ignore_unused(Egas_guess);
			amrex::ignore_unused(Egas0);
			amrex::ignore_unused(Etot0);
			amrex::ignore_unused(work);
			amrex::ignore_unused(work_prev);
			amrex::ignore_unused(kappaP);
			amrex::ignore_unused(kappaE);
			amrex::ignore_unused(kappaPoverE);
			amrex::ignore_unused(fourPiBoverC);
		}
		consNew(i, j, k, x1RadFlux_index) = Frad_t1[0];
		consNew(i, j, k, x2RadFlux_index) = Frad_t1[1];
		consNew(i, j, k, x3RadFlux_index) = Frad_t1[2];
	});
}

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_implicit,
					  double gas_update_factor_in, double dustGasCoeff, double const tol_h, double const tol_rel_h,
					  double const tempFloor_local, int *p_iteration_counter, int *p_iteration_failure_counter)
{
	static_assert(beta_order_ == 0 || beta_order_ == 1);

	// For single-group problems, dispatch to the dedicated single-group implementation.
	// This avoids calling SolveRadiationMatterCoupling (and PlanckFunction via
	// ComputeModelDependentKappaFAndDeltaTerms) which requires energy_unit in RadSystem_Traits.
	// The entire multi-group body below is guarded by if constexpr to prevent instantiation
	// of SolveRadiationMatterCoupling for single-group problem types.
	if constexpr (opacity_model_ == OpacityModel::single_group) {
		AddSourceTermsSingleGroup(consVar, radEnergySource, indexRange, dt_implicit, gas_update_factor_in, dustGasCoeff, tol_h, tol_rel_h,
					  tempFloor_local, p_iteration_counter, p_iteration_failure_counter);
	} else {

	arrayconst_t &consPrev = consVar; // make read-only
	array_t &consNew = consVar;
	auto dt = dt_implicit;

	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;
	const double tempFloor_h = tempFloor_local;

	// Add source terms

	// 1. Compute gas energy and radiation energy update following Howell &
	// Greenough [Journal of Computational Physics 184 (2003) 53-78].

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// make a local reference
		auto p_iteration_counter_local = p_iteration_counter;		      // NOLINT
		auto p_iteration_failure_counter_local = p_iteration_failure_counter; // NOLINT

		const double tol = tol_h;
		const double tol_rel = tol_rel_h;
		const double tempFloor = tempFloor_h;

		const double c = c_light_;
		const double chat = c_hat_;
		const double dustGasCoeff_local = dustGasCoeff;

		// load fluid properties
		const double rho = consPrev(i, j, k, gasDensity_index);
		const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
		const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
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
		// Note that radEnergySource should contain the luminosity volume density, L / V; unit: erg s^-1 cm^-3
		quokka::valarray<double, nGroups_> Src;
		for (int g = 0; g < nGroups_; ++g) {
			Src[g] = dt * (chat / c * radEnergySource(i, j, k, g));
		}

		double Egas0 = NAN;
		double Ekin0 = NAN;
		double Etot0 = NAN;
		double Egas_guess = NAN;
		quokka::valarray<double, nGroups_> work{};
		quokka::valarray<double, nGroups_> work_prev{};

		if constexpr (gamma_ != 1.0) {
			Egas0 = ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0);
			Etot0 = Egas0 + (c / chat) * (Erad0 + sum(Src));
			Ekin0 = Egastot0 - Egas0;
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

		amrex::Real gas_update_factor = gas_update_factor_in;

		const double H_num_den = ComputeNumberDensityH(rho, massScalars);
		const double cscale = c / chat;
		double coeff_n = NAN;
		if constexpr (enable_dust_gas_thermal_coupling_model_) {
			coeff_n = dt * dustGasCoeff_local * H_num_den * H_num_den / cscale;
		}

		// Outer iteration loop to update the work term until it converges
		const int max_iter = 5;
		int iter = 0;
		for (; iter < max_iter; ++iter) {
			amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
			NewtonIterationResult<problem_t> updated_energy;

			// 1. Compute matter-radiation energy exchange for non-isothermal gas

			if constexpr (gamma_ != 1.0) {

				// 1.2. Compute a term required to calculate the work. This is only required in the first outer loop.

				quokka::valarray<double, nGroups_> vel_times_F{};
				if constexpr (include_work_term_in_source) {
					if (iter == 0) {
						for (int g = 0; g < nGroups_; ++g) {
							// Compute vel_times_F[g] = sum(vel * F_g)
							const double frad0 = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
							const double frad1 = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
							const double frad2 = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
							vel_times_F[g] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2);
						}
					}
				}

				// 1.3. Compute the gas and radiation energy update.
				// For single-group (piecewise_constant_opacity), use the old solver to avoid calling
				// PlanckFunction which requires energy_unit in RadSystem_Traits.
				// For multi-group, use the unified Newton solver (SolveRadiationMatterCoupling).

				// multi-group path: use unified Newton solver
				// (single_group dispatches to AddSourceTermsSingleGroup above and returns early)
				auto thermal_result = SolveRadiationMatterCoupling(Egas0, Erad0Vec, rho, coeff_n, dt, massScalars, iter, work,
										   vel_times_F, Src, radBoundaries_g_copy, tol, tol_rel, tempFloor,
										   p_iteration_counter_local, p_iteration_failure_counter_local);

				// Convert ThermalResult to NewtonIterationResult for UpdateFluxAndMomentum
				updated_energy.Egas = thermal_result.Egas;
				updated_energy.T_gas = thermal_result.T_gas;
				updated_energy.T_d = thermal_result.T_d;
				updated_energy.EradVec = thermal_result.Erad;
				updated_energy.work = work; // work is updated by UpdateFluxAndMomentum
				updated_energy.opacity_terms = thermal_result.opacity_terms;

				Egas_guess = updated_energy.Egas;

				// copy work to work_prev (before UpdateFluxAndMomentum may update it)
				for (int g = 0; g < nGroups_; ++g) {
					work_prev[g] = work[g];
				}

				kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, updated_energy.T_d);
			} else { // constexpr (gamma_ == 1.0)
				kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, NAN);
				if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
					for (int g = 0; g < nGroups_; ++g) {
						updated_energy.opacity_terms.kappaF[g] = kappa_expo_and_lower_value[1][g];
					}
				} else {
					updated_energy.opacity_terms.kappaF =
					    ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_quant_minus_one);
				}
			}

			// Erad_guess is the new radiation energy (excluding work term)
			// Egas_guess is the new gas internal energy

			// 2. Compute radiation flux update

			// 2.1. Update flux and gas momentum
			auto updated_flux = UpdateFluxAndMomentum(i, j, k, consPrev, updated_energy, dt, gas_update_factor, Ekin0);

			// 2.2. Check for convergence of the work term
			bool work_converged = true;
			if constexpr ((beta_order_ == 0) || (gamma_ == 1.0) || (!include_work_term_in_source)) {
				// pass
			} else {
				work = updated_energy.work;

				// Check for convergence of the work term
				auto const Egastot1 =
				    ComputeEgasFromEint(rho, updated_flux.gasMomentum[0], updated_flux.gasMomentum[1], updated_flux.gasMomentum[2], Egas_guess);
				const double rel_lag_tol = 1.0e-8;
				const double lag_tol = 1.0e-13;
				double ref_work = rel_lag_tol * sum(abs(work));
				ref_work = std::max(ref_work, lag_tol * Egastot1 / (c_light_ / c_hat_));
				// ref_work = std::max(ref_work, lag_tol * sum(Rvec)); // comment out because Rvec is not accessible here
				if (sum(abs(work - work_prev)) > ref_work) {
					work_converged = false;
				}
			}

			// 3. If converged, store new radiation energy, gas energy
			if (work_converged) {
				consNew(i, j, k, x1GasMomentum_index) = updated_flux.gasMomentum[0];
				consNew(i, j, k, x2GasMomentum_index) = updated_flux.gasMomentum[1];
				consNew(i, j, k, x3GasMomentum_index) = updated_flux.gasMomentum[2];
				for (int g = 0; g < nGroups_; ++g) {
					consNew(i, j, k, radEnergy_index + numRadVars_ * g) = updated_flux.Erad[g];
					consNew(i, j, k, x1RadFlux_index + numRadVars_ * g) = updated_flux.Frad[0][g];
					consNew(i, j, k, x2RadFlux_index + numRadVars_ * g) = updated_flux.Frad[1][g];
					consNew(i, j, k, x3RadFlux_index + numRadVars_ * g) = updated_flux.Frad[2][g];
				}
				if constexpr (gamma_ != 1.0) {
					Egas_guess = updated_energy.Egas;
				}
				break;
			}
		} // end full-step iteration

		AMREX_ASSERT_WITH_MESSAGE(iter < max_iter, "AddSourceTerms iteration failed to converge!");
		if (iter >= max_iter) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[2], 1); // NOLINT
		}

		// 4b. Store new radiation energy, gas energy
		// In the first stage of the IMEX scheme, the hydro quantities are updated by a fraction (defined by
		// gas_update_factor) of the time step.
		const auto x1GasMom1 = consNew(i, j, k, x1GasMomentum_index);
		const auto x2GasMom1 = consNew(i, j, k, x2GasMomentum_index);
		const auto x3GasMom1 = consNew(i, j, k, x3GasMomentum_index);

		if constexpr (gamma_ != 1.0) {
			Egas_guess = Egas0 + (Egas_guess - Egas0) * gas_update_factor;
			consNew(i, j, k, gasInternalEnergy_index) = Egas_guess;
			consNew(i, j, k, gasEnergy_index) = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas_guess);
		} else {
			amrex::ignore_unused(Egas_guess);
			amrex::ignore_unused(Egas0);
			amrex::ignore_unused(Etot0);
			amrex::ignore_unused(work);
			amrex::ignore_unused(work_prev);
		}
	});

	} // end if constexpr (opacity_model_ != single_group)
}

#endif // RAD_SOURCE_TERMS_HPP_
