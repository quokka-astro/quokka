#ifndef DUST_RUNTIME_PARAMS_HPP_
#define DUST_RUNTIME_PARAMS_HPP_

#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Vector.H"
#include <cmath>
#include <string>

namespace quokka::dust
{

enum class ResolvedRkScheme { TP2025, GL4, Midpoint };

struct CoefficientIterationConfig {
	bool enabled = false;
	amrex::Real alphaRelativeTolerance = 1.0e-6;
	amrex::Real chargeAbsoluteTolerance = 1.0e-12;
	amrex::Real chargeRelativeTolerance = 1.0e-6;
	int maxIterations = 20;
};

inline auto resolvedRkSchemeName(ResolvedRkScheme scheme) -> char const *
{
	switch (scheme) {
		case ResolvedRkScheme::TP2025:
			return "TP2025";
		case ResolvedRkScheme::GL4:
			return "GL4";
		case ResolvedRkScheme::Midpoint:
			return "Midpoint";
	}

	return "unknown";
}

inline auto parseResolvedRkScheme(std::string const &scheme_name) -> ResolvedRkScheme
{
	if (scheme_name == "TP2025") {
		return ResolvedRkScheme::TP2025;
	}
	if (scheme_name == "GL4") {
		return ResolvedRkScheme::GL4;
	}
	if (scheme_name == "Midpoint") {
		return ResolvedRkScheme::Midpoint;
	}

	amrex::Abort("dust.resolved_rk_scheme must be one of: TP2025, GL4, Midpoint.");
	return ResolvedRkScheme::GL4;
}

template <unsigned int nDustGroups> void queryPositiveArray(amrex::ParmParse const &pp, char const *name, amrex::GpuArray<amrex::Real, nDustGroups> &values)
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
