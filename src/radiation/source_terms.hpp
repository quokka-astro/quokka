// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RAD_SOURCE_TERMS_HPP_ // NOLINT
#define RAD_SOURCE_TERMS_HPP_
/// \file source_terms.hpp
/// \brief The implicit radiation source-term update for one IMEX stage: the opacity helpers, the closed-form update of
/// dust-absorption bands, the flux and momentum update, and the cell kernel AddSourceTerms that drives the coupling
/// solve of radiation_coupling.hpp for single-group and multigroup radiation alike.

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// Compute kappaE and kappaP based on the opacity model. Returns them with alpha_P and alpha_E (the latter two are set by
// PPL_opacity_full_spectrum only, which fits alpha_E to Erad and alpha_P to fourPiBoverC).
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeModelDependentKappaEAndKappaP(double const T, double const rho,
										 amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
										 amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
										 quokka::valarray<double, nGroups_> const &fourPiBoverC,
										 quokka::valarray<double, nGroups_> const &Erad) -> OpacityTerms<problem_t>
{
	OpacityTerms<problem_t> result;

	const auto kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T);

	if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
		for (int g = 0; g < nGroups_; ++g) {
			result.kappaP[g] = kappa_expo_and_lower_value[1][g];
			result.kappaE[g] = kappa_expo_and_lower_value[1][g];
		}
	} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) {
		amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
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
		result.kappaP = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, alpha_quant_minus_one);
		result.kappaE = result.kappaP;
	} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
		result.alpha_E = ComputeRadQuantityExponents(Erad, rad_boundaries);
		result.alpha_P = ComputeRadQuantityExponents(fourPiBoverC, rad_boundaries);
		result.kappaE = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, result.alpha_E);
		result.kappaP = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, result.alpha_P);
	}
	AMREX_ASSERT(!result.kappaP.hasnan());
	AMREX_ASSERT(!result.kappaE.hasnan());

	return result;
}

// Compute kappaF and the delta_nu_kappa_B_at_edge term. kappaF is used to compute the work term and the delta_nu_kappa_B_at_edge term is used to compute the
// transport between groups in the momentum function. Only the last two arguments (kappaFVec, delta_nu_kappa_B_at_edge) are modified in this function.
template <typename problem_t>
AMREX_GPU_DEVICE void
RadSystem<problem_t>::ComputeModelDependentKappaFAndDeltaTerms(double const T, double const rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
							       quokka::valarray<double, nGroups_> const &fourPiBoverC, OpacityTerms<problem_t> &opacity_terms)
{
	amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge{};
	const auto kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T);
	// A band that does not emit has no thermal emission for the Doppler shift to act on, so its
	// momentum-of-emission and group-coupling terms are zero by construction; both delta terms are left at
	// zero here. Computing them anyway from the Planck function, as the code used to, injects a spurious
	// velocity-proportional momentum source into a band that radiates nothing. See issue #2309.
	//
	// Note what restricting the sum costs. sum_g Delta_g(nu kappa B) telescopes to the value of
	// nu kappa B at the two ends of whatever range it is summed over, so dropping the non-emitting bands
	// makes it terminate at the top of the emitting sub-grid instead of at the top of the whole grid. That
	// is harmless only where nu kappa B is already negligible, which is the existing requirement on
	// radBoundaries: the emitting bands must span the blackbody.
	for (int g = 0; g < nGroups_; ++g) {
		if (g >= nGroupsEmitting_) {
			opacity_terms.delta_nu_kappa_B_at_edge[g] = 0.0;
			delta_nu_B_at_edge[g] = 0.0;
			continue;
		}
		auto const nu_L = rad_boundaries[g];
		auto const nu_R = rad_boundaries[g + 1];
		auto const B_L = PlanckFunction(nu_L, T); // 4 pi B(nu) / c
		auto const B_R = PlanckFunction(nu_R, T); // 4 pi B(nu) / c
		auto const kappa_L = kappa_expo_and_lower_value[1][g];
		auto const kappa_R = kappa_L * std::pow(nu_R / nu_L, kappa_expo_and_lower_value[0][g]);
		opacity_terms.delta_nu_kappa_B_at_edge[g] = nu_R * kappa_R * B_R - nu_L * kappa_L * B_L;
		delta_nu_B_at_edge[g] = nu_R * B_R - nu_L * B_L;
	}
	if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
		opacity_terms.kappaF = opacity_terms.kappaP;
	} else {
		if constexpr (use_diffuse_flux_mean_opacity) {
			opacity_terms.kappaF =
			    ComputeDiffusionFluxMeanOpacity(opacity_terms.kappaP, opacity_terms.kappaE, fourPiBoverC, opacity_terms.delta_nu_kappa_B_at_edge,
							    delta_nu_B_at_edge, kappa_expo_and_lower_value[0]);
		} else {
			// for simplicity, I assume kappaF = kappaE when opacity_model_ ==
			// OpacityModel::PPL_opacity_full_spectrum, if !use_diffuse_flux_mean_opacity. We won't use this
			// option anyway.
			opacity_terms.kappaF = opacity_terms.kappaE;
		}
	}
	// A band that does not emit has no Planck weight to average a flux-mean opacity over.
	// ComputeDiffusionFluxMeanOpacity divides by (4/3) 4piB/c - (1/3) Delta_g(nu B), which is exactly zero
	// for such a band, so its guard against a non-positive denominator would return kappaF = 0. That is
	// not a harmless default here: kappaF carries the radiation force, the attenuation of the flux, and
	// the work term, so a dust-absorption band would lose energy through kappaE while exerting no force
	// on the gas at all. Fall back to the energy-mean opacity, which is the right flux mean when the
	// spectrum is not the Planck function. This is a no-op under piecewise_constant_opacity, where
	// kappaF, kappaP and kappaE are already equal.
	for (int g = nGroupsEmitting_; g < nGroups_; ++g) {
		opacity_terms.kappaF[g] = opacity_terms.kappaE[g];
	}
}

// Lorentz factors {gamma, gamma_v, gamma_vv} of the single-group beta_order >= 2 path; all one otherwise. gamma scales
// the optical depth of the step, gamma_v the velocity-dependent terms of the flux update and the work term.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeLorentzFactors(double const rho, std::array<double, 3> const &gasMtm) -> amrex::GpuArray<double, 3>
{
	amrex::GpuArray<double, 3> lorentz{1.0, 1.0, 1.0};
	if constexpr (nGroups_ == 1 && beta_order_ >= 2) {
		const double betaSqr = (gasMtm[0] * gasMtm[0] + gasMtm[1] * gasMtm[1] + gasMtm[2] * gasMtm[2]) / (rho * rho * c_light_ * c_light_);
		if constexpr (beta_order_ == 2) {
			lorentz = {1.0 + 0.5 * betaSqr, 1.0, 1.0};
		} else if constexpr (beta_order_ == 3) {
			lorentz = {1.0 + 0.5 * betaSqr, 1.0 + 0.5 * betaSqr, 1.0};
		} else {
			const double g = 1.0 / std::sqrt(1.0 - betaSqr);
			lorentz = {g, g, g};
		}
	} else {
		amrex::ignore_unused(rho, gasMtm);
	}
	return lorentz;
}

// Every opacity the flux update and the work term need, at the temperature the radiation couples at: kappaP, kappaE,
// kappaF and the Delta(nu kappa B) edge terms. Erad enters only the alpha_E fit of PPL_opacity_full_spectrum.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeOpacityTermsAt(CouplingCell<problem_t> const &cell, double const T,
								  quokka::valarray<double, nGroups_> const &Erad) -> OpacityTerms<problem_t>
{
	OpacityTerms<problem_t> terms{};
	if constexpr (nGroups_ == 1) {
		amrex::ignore_unused(Erad);
		terms.kappaP[0] = ComputePlanckOpacity(cell.rho, T);
		terms.kappaE[0] = ComputeEnergyMeanOpacity(cell.rho, T);
		terms.kappaF[0] = ComputeFluxMeanOpacity(cell.rho, T);
		AMREX_ASSERT(!std::isnan(terms.kappaF[0]));
	} else {
		const auto fourPiBoverC = ComputeThermalRadiationMultiGroup(T, cell.rad_boundaries);
		terms = ComputeModelDependentKappaEAndKappaP(T, cell.rho, cell.rad_boundaries, cell.rad_boundary_ratios, fourPiBoverC, Erad);
		ComputeModelDependentKappaFAndDeltaTerms(T, cell.rho, cell.rad_boundaries, fourPiBoverC, terms);
	}
	return terms;
}

// The work term of each group over the step, on the radiation side: (v . F_g) chi_g chat / c^2 dt, with chi the
// single-group (2 kappa_E - kappa_F) gamma_v, the piecewise-constant kappa_F, or the PPL (1 + alpha_g) kappa_F.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeWorkTerm(CouplingCell<problem_t> const &cell, double const T, OpacityTerms<problem_t> const &opacity,
							    quokka::valarray<double, nGroups_> const &vel_times_F, double const lorentz_v)
    -> quokka::valarray<double, nGroups_>
{
	const double c = c_light_;
	const double chat = c_hat_;
	quokka::valarray<double, nGroups_> work{};
	if constexpr (nGroups_ == 1) {
		amrex::ignore_unused(T);
		work[0] = vel_times_F[0] * (2.0 * opacity.kappaE[0] - opacity.kappaF[0]) * chat / (c * c) * lorentz_v * cell.dt;
	} else if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
		amrex::ignore_unused(T, lorentz_v);
		for (int g = 0; g < nGroups_; ++g) {
			work[g] = vel_times_F[g] * opacity.kappaF[g] * chat / (c * c) * cell.dt;
		}
	} else {
		amrex::ignore_unused(lorentz_v);
		const auto kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(cell.rad_boundaries, cell.rho, T);
		for (int g = 0; g < nGroups_; ++g) {
			work[g] = vel_times_F[g] * opacity.kappaF[g] * chat / (c * c) * cell.dt * (1.0 + kappa_expo_and_lower_value[0][g]);
		}
	}
	return work;
}

// Solve the energy exchange for dust-absorption-only bands. These bands emit nothing and give none of
// the absorbed energy to the gas, so nothing here depends on the gas temperature: the opacity is a
// function of (rho, T) alone and the radiation equation of each group is linear in its own E_g. The
// coupling solve of SolveEnergyExchange therefore has nothing to solve, and each group reduces to a
// backward-Euler absorption sink,
//
//     E_g <- (E_g^0 + S_g + W_g) / (1 + chat * rho * kappa_{E,g} * dt) ,
//
// where W_g is the work term. The work term is the only quantity that reaches the gas; it is lagged
// across the outer iteration of AddSourceTerms, which is the only iteration left in this path. The
// absorbed energy c * kappa_{E,g} * E_g leaves the simulation: it heats the dust, which re-radiates it
// in the infrared, and neither the dust temperature nor that emission is followed. Total energy is
// therefore not conserved in this mode, by construction.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveDustAbsorptionBands(CouplingCell<problem_t> const &cell, int *p_iteration_counter)
    -> EnergyExchangeResult<problem_t>
{
	const double cscale = c_light_ / c_hat_;

	// The gas temperature is used only to evaluate the opacity, which does not depend on the radiation
	// field, so the old-state value is all that is needed.
	//
	// IMPORTANT: this is the start-of-step temperature, and it is never revised. The outer iteration in
	// AddSourceTerms re-enters this function with the same Egas0 each time (it converges the work term,
	// not the gas temperature), so the opacity is always evaluated at T(Egas0). Without photoelectric
	// heating that lag is harmless, because only the work term moves the gas energy and it is tiny.
	// With photoelectric heating the gas temperature can change materially within a step,
	// so THIS PATH ASSUMES THE OPACITY DOES NOT DEPEND ON THE GAS TEMPERATURE. That holds for the
	// intended use -- ultraviolet dust opacity is a property of the grains, not of the gas -- and it is
	// exact whenever DefineOpacityExponentsAndLowerValues ignores its Tgas argument. If a problem does
	// make kappa depend on Tgas, the opacity lags the photoelectric heating by one step and the error is
	// first order in dt.
	const double T_gas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(cell.rho, cell.Egas0, cell.massScalars);
	AMREX_ASSERT(T_gas >= 0.);
	const auto rad0 = cell.Erad0 + cell.Src + cell.work;
	const auto opacity_terms = ComputeOpacityTermsAt(cell, T_gas, rad0);

	// Backward-Euler absorption, group by group. Chemical bands, if any, are treated the same way here:
	// they too emit nothing and pass their absorbed energy to photochemistry rather than to the gas. The
	// caller has already removed their source from Src and injects it after this solve.
	//
	// The photoelectric heating is accumulated alongside, as Gamma_PE * dt with
	// Gamma_PE = sum_g epsilon_g * pe_heating_rate_coeff_ * n_H * E_g. It is a direct physical heating rate
	// on the gas, so it is added to the gas energy without the cscale factor that converts radiation-side
	// energy to the gas side. Note what it does not depend on: neither kappa nor chat. The grain physics
	// lives in the empirical coefficient, so a band heats the gas whether or not it is being absorbed, and
	// that energy is not taken from the radiation.
	// A static constexpr member has no device storage, so it cannot be indexed with a runtime group
	// number inside device code. Copy it to a local first, as AddSourceTerms does with radBoundaries_.
	const amrex::GpuArray<double, nGroups_> pe_efficiency = pe_heating_efficiency_;
	const double n_H = ComputeNumberDensityH(cell.rho, cell.massScalars);
	quokka::valarray<double, nGroups_> EradVec_guess{};
	double PE_heating = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		const double tau = cell.tau_scale * cell.rho * opacity_terms.kappaE[g];
		EradVec_guess[g] = rad0[g] / (1.0 + tau);
		if constexpr (enable_dust_pe_heating_) {
			PE_heating += pe_efficiency[g] * pe_heating_rate_coeff_ * n_H * EradVec_guess[g] * cell.dt;
		}
	}
	if constexpr (!enable_dust_pe_heating_) {
		amrex::ignore_unused(pe_efficiency, n_H);
	}
	AMREX_ASSERT(min(EradVec_guess) >= 0.0);

	// The gas receives the work term and the photoelectric heating -- and nothing else. In particular it
	// does not receive the rest of the absorbed radiation energy, which is what distinguishes this path
	// from the thermal solve. Note that EradVec_guess does not depend on the gas energy, so this is a
	// closed-form update and not a fixed point: the photoelectric heating adds no iteration.
	const double Egas_guess = cell.Egas0 - cscale * sum(cell.work) + PE_heating;
	AMREX_ASSERT(Egas_guess > 0.0);

	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1); // total number of radiation updates. NOLINT
	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], 1); // total number of (here, trivial) evaluations. NOLINT
	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], 1); // NOLINT

	EnergyExchangeResult<problem_t> result{};
	result.Egas = Egas_guess;
	result.EradVec = EradVec_guess;
	result.work = cell.work;
	result.T_gas = T_gas;
	result.T_d = T_gas;
	result.opacity_terms = opacity_terms;
	return result;
}

// The energy exchange of one cell over the step: the bracketed solve of radiation_coupling.hpp in the form the dust
// model selects, the iteration counters, and the opacities at the temperature the radiation coupled at, which the flux
// update needs. An unconverged cell keeps its old state and is counted; the driver aborts on the count.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveEnergyExchange(CouplingCell<problem_t> const &cell, double const tol, int *p_iteration_counter,
								int *p_iteration_failure_counter) -> EnergyExchangeResult<problem_t>
{
	CouplingSolution<problem_t> sol{};
	if constexpr (enable_dust_gas_thermal_coupling_model_) {
		sol = SolveDustCoupling(cell, tol);
	} else {
		sol = SolveGasCoupling(cell, tol);
	}
	AMREX_ASSERT_WITH_MESSAGE(sol.converged, "matter-radiation coupling failed to converge!");
	if (!sol.converged) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
	}
	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);	      // total number of coupling solves. NOLINT
	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], sol.nevals); // total number of residual evaluations. NOLINT
	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], sol.nevals); // maximum number of residual evaluations. NOLINT
	AMREX_ASSERT(sol.Egas > 0.0);
	AMREX_ASSERT(min(sol.Erad) >= 0.0);

	EnergyExchangeResult<problem_t> result{};
	result.Egas = sol.Egas;
	result.T_gas = sol.T_gas;
	result.T_d = sol.T_d;
	result.EradVec = sol.Erad;
	result.work = cell.work;
	result.opacity_terms = ComputeOpacityTermsAt(cell, sol.T_d, sol.Erad);
	return result;
}

// Update radiation flux and gas momentum. Returns FluxUpdateResult struct. The function also updates energy.Egas and energy.work.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::UpdateFlux(int const i, int const j, int const k, arrayconst_t const &consPrev,
						       EnergyExchangeResult<problem_t> &energy, CouplingCell<problem_t> const &cell,
						       double const gas_update_factor, double const Ekin0,
						       amrex::GpuArray<quokka::valarray<double, nGroups_>, 3> const &Src_flux, double Emag,
						       amrex::GpuArray<double, 3> const &lorentz) -> FluxUpdateResult<problem_t>
{
	amrex::GpuArray<amrex::Real, 3> Frad_t0{};
	amrex::GpuArray<amrex::Real, 3> dMomentum{0., 0., 0.};
	amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> Frad_t1{};

	double const rho = consPrev(i, j, k, gasDensity_index);
	const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
	const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
	const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
	const std::array<double, 3> gasMtm0 = {x1GasMom0, x2GasMom0, x3GasMom0};

	quokka::valarray<double, nGroups_> fourPiBoverC{};
	if constexpr (nGroups_ == 1) {
		fourPiBoverC[0] = (nGroupsThermal_ == 0) ? 0.0 : ComputeThermalRadiationSingleGroup(energy.T_d);
	} else {
		fourPiBoverC = ComputeThermalRadiationMultiGroup(energy.T_d, cell.rad_boundaries);
	}
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
	if constexpr (nGroups_ > 1) {
		kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(cell.rad_boundaries, rho, energy.T_d);
	}

	const double chat = c_hat_;

	for (int g = 0; g < nGroups_; ++g) {
		// The user-defined flux source is added to the old-time flux, so the implicit absorption below acts on
		// the freshly injected radiation as well.
		Frad_t0[0] = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g) + Src_flux[0][g];
		Frad_t0[1] = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g) + Src_flux[1][g];
		Frad_t0[2] = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g) + Src_flux[2][g];

		if constexpr ((gamma_ == 1.0) || (beta_order_ == 0)) {
			for (int n = 0; n < 3; ++n) {
				Frad_t1[n][g] = Frad_t0[n] / (1.0 + rho * energy.opacity_terms.kappaF[g] * chat * cell.dt);
				// Compute conservative gas momentum update
				dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c_light_ * chat);
			}
		} else {
			const auto erad = energy.EradVec[g];
			std::array<double, 3> v_terms{};

			auto fx = Frad_t0[0] / (c_light_ * erad);
			auto fy = Frad_t0[1] / (c_light_ * erad);
			auto fz = Frad_t0[2] / (c_light_ * erad);
			double F_coeff = chat * rho * energy.opacity_terms.kappaF[g] * cell.dt * lorentz[0];
			auto Tedd = ComputeEddingtonTensor(fx, fy, fz);

			for (int n = 0; n < 3; ++n) {
				double Planck_term = NAN;
				double pressure_term = 0.0;
				for (int z = 0; z < 3; ++z) {
					pressure_term += gasMtm0[z] * Tedd[n][z] * erad;
				}
				if constexpr (nGroups_ == 1) {
					// single group: the (kappa_F - kappa_E) term and the Lorentz factors of beta_order >= 2
					Planck_term = energy.opacity_terms.kappaP[g] * fourPiBoverC[g] * lorentz[1];
					if (energy.opacity_terms.kappaF[g] != energy.opacity_terms.kappaE[g]) {
						Planck_term +=
						    (energy.opacity_terms.kappaF[g] - energy.opacity_terms.kappaE[g]) * erad * std::pow(lorentz[1], 3);
					}
					pressure_term *= chat * cell.dt * energy.opacity_terms.kappaF[g] * lorentz[1];
				} else {
					if constexpr (include_delta_B) {
						Planck_term = energy.opacity_terms.kappaP[g] * fourPiBoverC[g] -
							      1.0 / 3.0 * energy.opacity_terms.delta_nu_kappa_B_at_edge[g];
					} else {
						Planck_term = energy.opacity_terms.kappaP[g] * fourPiBoverC[g];
					}
					// Simplification: assuming Eddington tensors are the same for all groups, we have kappaP = kappaE
					if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
						pressure_term *= chat * cell.dt * energy.opacity_terms.kappaE[g];
					} else {
						pressure_term *= chat * cell.dt * (1.0 + kappa_expo_and_lower_value[0][g]) * energy.opacity_terms.kappaE[g];
					}
				}
				Planck_term *= chat * cell.dt * gasMtm0[n];
				v_terms[n] = Planck_term + pressure_term;
			}

			// Compute flux update. The single-group beta_order >= 2 case with kappa_F != kappa_E couples the three flux
			// components through the O(beta^2) term and solves a 3x3 system instead.
			bool solved_3x3 = false;
			if constexpr (nGroups_ == 1 && beta_order_ >= 2) {
				if (energy.opacity_terms.kappaF[g] != energy.opacity_terms.kappaE[g]) {
					const double kappaF = energy.opacity_terms.kappaF[g];
					const double kappaE = energy.opacity_terms.kappaE[g];
					const double lorentz_factor_v_v = lorentz[2];
					const double c = c_light_;
					// Moved as it was from the old single-group driver: gasVel is never assigned, so the K0 terms are zero.
					std::array<double, 3> gasVel{};
					const double K0 = 2.0 * rho * chat * cell.dt * (kappaF - kappaE) / c / c * std::pow(lorentz_factor_v_v, 3);

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
					Frad_t1[0][g] = sol0;
					Frad_t1[1][g] = sol1;
					Frad_t1[2][g] = sol2;
					solved_3x3 = true;
				}
			}
			if (!solved_3x3) {
				for (int n = 0; n < 3; ++n) {
					Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);
				}
			}

			for (int n = 0; n < 3; ++n) {
				// Compute conservative gas momentum update
				dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c_light_ * chat);
			}
		}
	}

	amrex::Real x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0];
	amrex::Real x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1];
	amrex::Real x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2];

	FluxUpdateResult<problem_t> updated_flux;

	for (int g = 0; g < nGroups_; ++g) {
		updated_flux.Erad[g] = energy.EradVec[g];
	}

	// 3. Deal with the work term.
	if constexpr ((gamma_ != 1.0) && (beta_order_ != 0)) {
		// compute difference in gas kinetic energy before and after momentum update
		amrex::Real const Egastot1 = ::quokka::EOS<problem_t>::ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, energy.Egas, Emag);
		amrex::Real const Ekin1 = Egastot1 - energy.Egas;
		amrex::Real const dEkin_work = Ekin1 - Ekin0;

		if constexpr (include_work_term_in_source) {
			// New scheme: the work term is included in the source terms. The work done by radiation went to internal energy, but it
			// should go to the kinetic energy. Remove the work term from internal energy.
			// Cap the transfer at the internal energy actually available. In a cold cell that the beam is
			// driving hard, dEkin_work can exceed Egas -- the momentum deposited over one radiation step buys
			// more kinetic energy than the gas holds internally, a mismatch the reduced speed of light widens
			// because the momentum and energy exchanges carry different powers of chat/c. Subtracting it
			// unclamped leaves a negative internal energy, which the EOS rejects outright (debug) and which
			// otherwise flows on silently and NaNs the coupling solve.
			// The cap is not energy conserving. It is a known limitation of the lagged work term, not a
			// timestep problem -- the capped fraction of cell-updates does not fall as dt is refined -- but
			// it can only bind where the transfer already exceeds the internal energy available, i.e. in
			// cold cells the radiation has evacuated, where Egas and hence the discarded energy are
			// minuscule. See work_term_min_eint_fraction and issue #2173.
			const double max_eint_transfer = (1.0 - work_term_min_eint_fraction) * energy.Egas;
			energy.Egas -= std::min(dEkin_work, max_eint_transfer);
			// The work term is included in the source term, but it is lagged. We update the work term here, from the
			// updated radiation flux and velocity.
			quokka::valarray<double, nGroups_> vel_times_F1{};
			for (int g = 0; g < nGroups_; ++g) {
				vel_times_F1[g] = x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g];
			}
			energy.work = ComputeWorkTerm(cell, energy.T_d, energy.opacity_terms, vel_times_F1, lorentz[1]);
		} else {
			// Old scheme: the source term does not include the work term, so we add the work term to the Erad.

			// compute loss of radiation energy to gas kinetic energy
			auto dErad_work = -(c_hat_ / c_light_) * dEkin_work;

			// apportion dErad_work according to kappaF_i * (v * F_i)
			quokka::valarray<double, nGroups_> energyLossFractions{};
			if constexpr (nGroups_ == 1) {
				energyLossFractions[0] = 1.0;
			} else {
				// compute energyLossFractions
				for (int g = 0; g < nGroups_; ++g) {
					energyLossFractions[g] = energy.opacity_terms.kappaF[g] *
								 (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]);
				}
				auto energyLossFractionsTot = sum(energyLossFractions);
				if (energyLossFractionsTot != 0.0) {
					energyLossFractions /= energyLossFractionsTot;
				} else {
					energyLossFractions.fillin(0.0);
				}
			}
			for (int g = 0; g < nGroups_; ++g) {
				auto radEnergyNew = energy.EradVec[g] + dErad_work * energyLossFractions[g];
				// AMREX_ASSERT(radEnergyNew > 0.0);
				if (radEnergyNew < Erad_floor_) {
					// return energy to Egas_guess
					energy.Egas -= (Erad_floor_ - radEnergyNew) * (c_light_ / c_hat_);
					radEnergyNew = Erad_floor_;
				}
				updated_flux.Erad[g] = radEnergyNew;
			}
		}
	}

	x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0] * gas_update_factor;
	x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1] * gas_update_factor;
	x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2] * gas_update_factor;
	updated_flux.gasMomentum = {x1GasMom1, x2GasMom1, x3GasMom1};
	updated_flux.Frad = Frad_t1;

	return updated_flux;
}

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource, arrayconst_t &radFluxSource, amrex::Box const &indexRange,
					  amrex::Real dt_implicit, double gas_update_factor_in, double dustGasCoeff, double const tol_h,
					  double const tempFloor_h, int *p_iteration_counter, int *p_iteration_failure_counter,
					  std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc)
{
	static_assert(nGroups_ == 1 || beta_order_ <= 1, "beta_order > 1 is implemented for single-group radiation only");

	arrayconst_t &consPrev = consVar; // make read-only
	array_t &consNew = consVar;
	const auto dt = dt_implicit;
	const amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		auto p_iteration_counter_local = p_iteration_counter;		      // NOLINT
		auto p_iteration_failure_counter_local = p_iteration_failure_counter; // NOLINT
		const double tol = tol_h;
		const double tempFloor = tempFloor_h;
		const double c = c_light_;
		const double chat = c_hat_;
		const double cscale = c / chat;
		const double dustGasCoeff_local = dustGasCoeff;
		const amrex::Real gas_update_factor = gas_update_factor_in;

		// load fluid properties
		const double rho = consPrev(i, j, k, gasDensity_index);
		const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
		const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
		const std::array<double, 3> gasMtm0 = {x1GasMom0, x2GasMom0, x3GasMom0};
		const double Egastot0 = consPrev(i, j, k, gasEnergy_index);
		const auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consPrev, i, j, k);
		const double Emag = ComputeCellCenteredMagneticEnergy<problem_t>(i, j, k, cons_fc);

		// load radiation energy
		quokka::valarray<double, nGroups_> Erad0Vec{};
		for (int g = 0; g < nGroups_; ++g) {
			Erad0Vec[g] = consPrev(i, j, k, radEnergy_index + numRadVars_ * g);
		}
		AMREX_ASSERT(min(Erad0Vec) > 0.0);

		// load the radiation energy and flux sources, scaled to the radiation side: a thermal group's source carries
		// chat/c, a chemical (ionizing) band's does not. radEnergySource is the luminosity volume density L / V.
		quokka::valarray<double, nGroups_> Src{};
		amrex::GpuArray<quokka::valarray<double, nGroups_>, 3> Src_flux{};
		for (int g = 0; g < nGroups_; ++g) {
			// Avoid if constexpr here: NVCC rejects first-captures inside constexpr-if in device lambdas.
			const double src_scale = (RadSystem_NChemBands<problem_t>::value > 0 && g >= nGroupsThermal_) ? dt : dt * (chat / c);
			Src[g] = src_scale * radEnergySource(i, j, k, g);
			for (int n = 0; n < 3; ++n) {
				Src_flux[n][g] = src_scale * radFluxSource(i, j, k, 3 * g + n);
			}
		}
		// Chemical bands are decoupled from the thermal exchange: their source is injected after the solve, so that the
		// thermal opacity coupling neither absorbs it nor lets it into the gas energy balance.
		quokka::valarray<double, nGroups_> Src_chem{};
		for (int g = nGroupsThermal_; g < nGroups_; ++g) {
			Src_chem[g] = Src[g];
			Src[g] = 0.0;
		}

		double Egas0 = NAN;
		double Ekin0 = NAN;
		double Egas_guess = NAN;
		double T_start = NAN;
		if constexpr (gamma_ != 1.0) {
			Egas0 = ::quokka::EOS<problem_t>::ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0, Emag);
			Ekin0 = Egastot0 - Egas0;
			AMREX_ASSERT(Egas0 > 0.0);
			T_start = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas0, massScalars);
		}

		// the cell the coupling step is solved from; its work term is filled per outer iteration
		const auto lorentz = ComputeLorentzFactors(rho, gasMtm0);
		CouplingCell<problem_t> cell{};
		cell.Egas0 = Egas0;
		cell.Erad0 = Erad0Vec;
		cell.Src = Src;
		cell.rho = rho;
		cell.dt = dt;
		cell.tau_scale = dt * chat * lorentz[0];
		cell.Tfloor = tempFloor;
		cell.massScalars = massScalars;
		cell.rad_boundaries = radBoundaries_g;
		for (int g = 0; g < nGroups_; ++g) {
			cell.rad_boundary_ratios[g] = radBoundaries_g[g + 1] / radBoundaries_g[g];
		}
		if constexpr (gamma_ != 1.0) {
			cell.Emin = ::quokka::EOS<problem_t>::ComputeEintFromTgas(rho, tempFloor, massScalars);
		}
		if constexpr (enable_dust_gas_thermal_coupling_model_) {
			const double n_H = ComputeNumberDensityH(rho, massScalars);
			cell.dtK = dt * dustGasCoeff_local * n_H * n_H;
		} else {
			amrex::ignore_unused(dustGasCoeff_local);
		}

		// the work term at the old state, from the old-state flux and the opacity at the start-of-step temperature
		quokka::valarray<double, nGroups_> work{};
		quokka::valarray<double, nGroups_> work_prev{};
		if constexpr ((gamma_ != 1.0) && (beta_order_ != 0) && include_work_term_in_source) {
			quokka::valarray<double, nGroups_> vel_times_F{};
			for (int g = 0; g < nGroups_; ++g) {
				const double frad0 = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
				const double frad1 = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
				const double frad2 = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
				vel_times_F[g] = x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2;
			}
			const auto opacity0 = ComputeOpacityTermsAt(cell, T_start, Erad0Vec + Src);
			work = ComputeWorkTerm(cell, T_start, opacity0, vel_times_F, lorentz[1]);
		}

		// Outer iteration: the work term is lagged, and updated from the new flux and momentum until it stops changing.
		const int max_iter = 5;
		int iter = 0;
		for (; iter < max_iter; ++iter) {
			EnergyExchangeResult<problem_t> updated_energy{};
			cell.work = work;

			// 1. the energy exchange
			if constexpr (gamma_ != 1.0) {
				if constexpr (dust_absorption_only_) {
					updated_energy = SolveDustAbsorptionBands(cell, p_iteration_counter_local);
				} else {
					updated_energy = SolveEnergyExchange(cell, tol, p_iteration_counter_local, p_iteration_failure_counter_local);
				}
				Egas_guess = updated_energy.Egas;
				work_prev = work;
			} else {
				// isothermal gas: no energy exchange; the radiation keeps its energy, source included, and only the
				// flux is updated. The opacity of such a problem does not depend on the temperature, so kappaF is
				// built directly from the opacity hooks at an undefined temperature, exactly as the old multigroup
				// driver did (the Planck-weighted flux mean of ComputeOpacityTermsAt would be NaN there).
				updated_energy.EradVec = Erad0Vec + Src;
				updated_energy.T_gas = NAN;
				updated_energy.T_d = NAN;
				updated_energy.work = work;
				if constexpr (nGroups_ == 1) {
					updated_energy.opacity_terms.kappaF[0] = ComputeFluxMeanOpacity(rho, NAN);
				} else {
					const auto kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g, rho, NAN);
					if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
						for (int g = 0; g < nGroups_; ++g) {
							updated_energy.opacity_terms.kappaF[g] = kappa_expo_and_lower_value[1][g];
						}
					} else {
						amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
						for (int g = 0; g < nGroups_; ++g) {
							alpha_quant_minus_one[g] = -1.0;
						}
						if constexpr (special_edge_bin_slopes) {
							alpha_quant_minus_one[0] = 2.0;
							alpha_quant_minus_one[nGroups_ - 1] = -4.0;
						}
						updated_energy.opacity_terms.kappaF =
						    ComputeGroupMeanOpacity(kappa_expo_and_lower_value, cell.rad_boundary_ratios, alpha_quant_minus_one);
					}
				}
			}

			// 2. the flux and momentum update
			auto updated_flux = UpdateFlux(i, j, k, consPrev, updated_energy, cell, gas_update_factor, Ekin0, Src_flux, Emag, lorentz);

			// 3. convergence of the work term
			bool work_converged = true;
			if constexpr ((gamma_ != 1.0) && (beta_order_ != 0) && include_work_term_in_source) {
				work = updated_energy.work;
				auto const Egastot1 = ::quokka::EOS<problem_t>::ComputeEgasFromEint(
				    rho, updated_flux.gasMomentum[0], updated_flux.gasMomentum[1], updated_flux.gasMomentum[2], Egas_guess, Emag);
				const double rel_lag_tol = 1.0e-8;
				const double lag_tol = 1.0e-13;
				double ref_work = rel_lag_tol * sum(abs(work));
				ref_work = std::max(ref_work, lag_tol * Egastot1 / cscale);
				if (sum(abs(work - work_prev)) > ref_work) {
					work_converged = false;
				}
			}

			// 4. if converged, store the new state
			if (work_converged) {
				consNew(i, j, k, x1GasMomentum_index) = updated_flux.gasMomentum[0];
				consNew(i, j, k, x2GasMomentum_index) = updated_flux.gasMomentum[1];
				consNew(i, j, k, x3GasMomentum_index) = updated_flux.gasMomentum[2];
				for (int g = 0; g < nGroups_; ++g) {
					consNew(i, j, k, radEnergy_index + numRadVars_ * g) = updated_flux.Erad[g] + Src_chem[g];
					consNew(i, j, k, x1RadFlux_index + numRadVars_ * g) = updated_flux.Frad[0][g];
					consNew(i, j, k, x2RadFlux_index + numRadVars_ * g) = updated_flux.Frad[1][g];
					consNew(i, j, k, x3RadFlux_index + numRadVars_ * g) = updated_flux.Frad[2][g];
				}
				if constexpr (gamma_ != 1.0) {
					Egas_guess = updated_energy.Egas;
				}
				break;
			}
		}

		AMREX_ASSERT_WITH_MESSAGE(iter < max_iter, "AddSourceTerms iteration failed to converge!");
		if (iter >= max_iter) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[1], 1); // NOLINT
		}

		// 5. store the gas energy. In the first stage of the IMEX scheme the hydro quantities are updated by a fraction
		// (gas_update_factor) of the time step.
		if constexpr (gamma_ != 1.0) {
			const auto x1GasMom1 = consNew(i, j, k, x1GasMomentum_index);
			const auto x2GasMom1 = consNew(i, j, k, x2GasMomentum_index);
			const auto x3GasMom1 = consNew(i, j, k, x3GasMomentum_index);
			Egas_guess = Egas0 + (Egas_guess - Egas0) * gas_update_factor;
			consNew(i, j, k, gasInternalEnergy_index) = Egas_guess;
			consNew(i, j, k, gasEnergy_index) =
			    ::quokka::EOS<problem_t>::ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas_guess, Emag);
		} else {
			amrex::ignore_unused(Egas_guess, Egas0, Ekin0, T_start, work_prev);
		}
	});
}

#endif // RAD_SOURCE_TERMS_HPP_
