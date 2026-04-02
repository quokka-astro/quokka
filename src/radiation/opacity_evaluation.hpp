// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef OPACITY_EVALUATION_HPP_
#define OPACITY_EVALUATION_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// EvaluateOpacities: wraps ComputeModelDependentKappaEAndKappaP
// This is the opacity interface for the new thermal solver.
// All opacity-model branching is internal.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::EvaluateOpacities(
    double T_d, double rho,
    quokka::valarray<double, nGroups_> const &Erad,
    int iteration_number,
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
AMREX_GPU_DEVICE void RadSystem<problem_t>::EvaluateFluxOpacities(double T_d, double rho,
								  amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
								  quokka::valarray<double, nGroups_> const &fourPiBoverC,
								  OpacityTerms<problem_t> &opacity_terms)
{
	ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms);
}

#endif // OPACITY_EVALUATION_HPP_
