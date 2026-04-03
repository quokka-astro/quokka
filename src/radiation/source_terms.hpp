// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RAD_SOURCE_TERMS_HPP_ // NOLINT
#define RAD_SOURCE_TERMS_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// Unified AddSourceTerms handles both single-group and multi-group opacity models.
// Single-group support is enabled via constexpr-if branches in SolveRadiationMatterCoupling
// and UpdateFluxAndMomentum that dispatch to simpler opacity functions (ComputePlanckOpacity,
// ComputeEnergyMeanOpacity, ComputeFluxMeanOpacity) instead of the multi-group PlanckFunction-based
// opacity evaluation that requires energy_unit in RadSystem_Traits.

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_implicit,
					  double gas_update_factor_in, double dustGasCoeff, double const tol_h, double const tol_rel_h,
					  double const tempFloor_local, int *p_iteration_counter, int *p_iteration_failure_counter)
{
	static_assert(beta_order_ == 0 || beta_order_ == 1, "beta_order >= 2 is not supported by the unified source term solver");

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

				// 1.3. Compute the gas and radiation energy update via the unified Newton solver.
				auto thermal_result = SolveRadiationMatterCoupling(Egas0, Erad0Vec, rho, coeff_n, dt, massScalars, iter, work, vel_times_F, Src,
										   radBoundaries_g_copy, tol, tol_rel, tempFloor, p_iteration_counter_local,
										   p_iteration_failure_counter_local);

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

				if constexpr (opacity_model_ != OpacityModel::single_group) {
					kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, updated_energy.T_d);
				}
			} else { // constexpr (gamma_ == 1.0)
				if constexpr (opacity_model_ == OpacityModel::single_group) {
					updated_energy.opacity_terms.kappaF[0] = ComputeFluxMeanOpacity(rho, NAN);
				} else {
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
}

#endif // RAD_SOURCE_TERMS_HPP_
