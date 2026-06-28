#ifndef EOS_TABULATED_REGISTRY_HPP_
#define EOS_TABULATED_REGISTRY_HPP_

#include <array>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "math/FastMath.hpp"
#include "math/root_finding.hpp"
#include "util/DataTable.hpp"

namespace quokka::ResampledCooling
{

using Real = amrex::Real;

struct resampledGpuConstTables {
	quokka::DataTableGpuConst<2, 1> cooling_rates;
	quokka::DataTableGpuConst<2, 1> temperatures;
	quokka::DataTableGpuConst<2, 1> sound_speeds;
	quokka::DataTableGpuConst<2, 1> pressures;
	quokka::DataTableGpuConst<2, 1> entropies;

	amrex::Real rho_min;
	amrex::Real rho_max;
	amrex::Real eint_min;
	amrex::Real eint_max;
	amrex::Real cloudy_H_mass_fraction;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	if (rho <= 0.0) {
		return 0.0;
	}
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};
	const Real Tgas = tables.temperatures.interpolate_single(point);
	return Tgas;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(Real const rho, Real const Tgas, resampledGpuConstTables const &tables) -> Real
{
	if (rho <= 0.0) {
		return 0.0;
	}
	const Real Eint_min = rho * tables.eint_min;
	const Real Eint_max = rho * tables.eint_max;

	const Real Tmin = ComputeTgasFromEgas(rho, Eint_min, tables);
	if (Tgas <= Tmin) {
		return Eint_min;
	}

	const Real Tmax = ComputeTgasFromEgas(rho, Eint_max, tables);
	if (Tgas >= Tmax) {
		return Eint_max;
	}

	auto f = [=](Real Eint) -> Real { return ComputeTgasFromEgas(rho, Eint, tables) - Tgas; };

	int max_iter = 32;
	auto tol = quokka::math::eps_tolerance<Real>{};
	auto const [Eint_lo, Eint_hi] = quokka::math::toms748_solve(f, Eint_min, Eint_max, Tmin - Tgas, Tmax - Tgas, tol, max_iter);

	return Eint_hi;
}

struct EOSTabulatedRegistry {
	bool active = false;
	resampledGpuConstTables host;
	resampledGpuConstTables device;
};

extern AMREX_GPU_MANAGED EOSTabulatedRegistry *g_eos_tabulated_registry;

inline AMREX_GPU_HOST_DEVICE auto getEOSTabulatedRegistry() -> EOSTabulatedRegistry * { return g_eos_tabulated_registry; }

void registerEOSTabulated(resampledGpuConstTables host_tables, resampledGpuConstTables device_tables);

} // namespace quokka::ResampledCooling

#endif // EOS_TABULATED_REGISTRY_HPP_
