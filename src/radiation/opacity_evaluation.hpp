// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef OPACITY_EVALUATION_HPP_
#define OPACITY_EVALUATION_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// Compute kappaE and kappaP based on the opacity model. The result is stored in the last five arguments: alpha_P, alpha_E, kappaP, kappaE, and kappaPoverE.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeModelDependentKappaEAndKappaP(
    double const T, double const rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
    quokka::valarray<double, nGroups_> const &fourPiBoverC, quokka::valarray<double, nGroups_> const &Erad, int const n_iter,
    amrex::GpuArray<double, nGroups_> const &alpha_E, amrex::GpuArray<double, nGroups_> const &alpha_P) -> OpacityTerms<problem_t>
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
		if (n_iter < max_iter_to_update_alpha_E) {
			result.alpha_E = ComputeRadQuantityExponents(Erad, rad_boundaries);
			result.alpha_P = ComputeRadQuantityExponents(fourPiBoverC, rad_boundaries);
		} else {
			result.alpha_E = alpha_E;
			result.alpha_P = alpha_P;
		}
		result.kappaE = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, result.alpha_E);
		result.kappaP = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, result.alpha_P);
	}
	AMREX_ASSERT(!result.kappaP.hasnan());
	AMREX_ASSERT(!result.kappaE.hasnan());
	for (int g = 0; g < nGroups_; ++g) {
		if (result.kappaE[g] > 0.0) {
			result.kappaPoverE[g] = result.kappaP[g] / result.kappaE[g];
		} else {
			result.kappaPoverE[g] = 1.0;
		}
	}

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
	for (int g = 0; g < nGroups_; ++g) {
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
}

// EvaluateOpacities: wraps ComputeModelDependentKappaEAndKappaP
// This is the opacity interface for the new thermal solver.
// All opacity-model branching is internal.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::EvaluateOpacities(double T_d, double rho, quokka::valarray<double, nGroups_> const &Erad, int iteration_number,
							      amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
							      amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
							      quokka::valarray<double, nGroups_> const &fourPiBoverC,
							      OpacityTerms<problem_t> const &prev_opacity) -> OpacityTerms<problem_t>
{
	return ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios, fourPiBoverC, Erad, iteration_number, prev_opacity.alpha_E,
						    prev_opacity.alpha_P);
}

// EvaluateFluxOpacities: wraps ComputeModelDependentKappaFAndDeltaTerms
template <typename problem_t>
AMREX_GPU_DEVICE void RadSystem<problem_t>::EvaluateFluxOpacities(double T_d, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
								  quokka::valarray<double, nGroups_> const &fourPiBoverC,
								  OpacityTerms<problem_t> &opacity_terms)
{
	ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms);
}

#endif // OPACITY_EVALUATION_HPP_
