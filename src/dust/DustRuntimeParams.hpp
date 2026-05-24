#ifndef DUST_RUNTIME_PARAMS_HPP_
#define DUST_RUNTIME_PARAMS_HPP_

#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Vector.H"
#include <cmath>
#include <string>

namespace quokka::dust
{

template <unsigned int nDustGroups>
void queryPositiveArray(amrex::ParmParse const &pp, char const *name, amrex::GpuArray<amrex::Real, nDustGroups> &values)
{
	static_assert(nDustGroups > 0);

	amrex::Vector<amrex::Real> parsed_values;
	if (pp.queryarr(name, parsed_values) != 0) {
		if (parsed_values.size() != nDustGroups) {
			amrex::Abort(std::string("dust.") + name + " must contain exactly " + std::to_string(nDustGroups) + " value(s).");
		}

		for (unsigned int g = 0; g < nDustGroups; ++g) {
			values[g] = parsed_values[g];
		}
	}

	for (unsigned int g = 0; g < nDustGroups; ++g) {
		if (!std::isfinite(static_cast<double>(values[g])) || values[g] <= 0.0) {
			amrex::Abort(std::string("dust.") + name + " values must be finite and positive.");
		}
	}
}

// read optional problem-specific grain parameters used by Kwok stopping-time dust problems
template <unsigned int nDustGroups>
void readDustGrainParams(amrex::GpuArray<amrex::Real, nDustGroups> &grain_radius, amrex::GpuArray<amrex::Real, nDustGroups> &grain_density)
{
	amrex::ParmParse const pp("dust");
	queryPositiveArray(pp, "grain_radius", grain_radius);
	queryPositiveArray(pp, "grain_density", grain_density);
}

} // namespace quokka::dust

#endif // DUST_RUNTIME_PARAMS_HPP_
