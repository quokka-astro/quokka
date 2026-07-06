#ifndef EOS_TABULATED_REGISTRY_HPP_
#define EOS_TABULATED_REGISTRY_HPP_

#include <array>
#include <cmath>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "math/root_finding.hpp"
#include "util/DataTable.hpp"

namespace quokka::ResampledCooling
{

using Real = amrex::Real;

// Output indices into the DataTable<2, 5> for the five cooling quantities
constexpr int COOLING_RATE_IDX = 0;
constexpr int TEMPERATURE_IDX = 1;
constexpr int SOUND_SPEED_IDX = 2;
constexpr int PRESSURE_IDX = 3;
constexpr int ENTROPY_IDX = 4;

struct resampledGpuConstTables {
	// Single GPU-friendly table holding all 5 cooling outputs
	quokka::DataTableGpuConst<2, 5> all_tables;

	// Hydrogen mass fraction (used to convert rho -> nH)
	amrex::Real cloudy_H_mass_fraction;

	// Physical specific-energy bounds (erg/g) -- used as root-finding bounds in ComputeEgasFromTgas
	amrex::Real eint_min;
	amrex::Real eint_max;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	if (rho <= 0.0) {
		AMREX_ASSERT(rho > 0.0);
		return static_cast<amrex::Real>(NAN);
	}
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {rho, eint};
	return tables.all_tables.interpolate_single(point, TEMPERATURE_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(Real const rho, Real const Tgas, resampledGpuConstTables const &tables) -> Real
{
	if (rho <= 0.0) {
		AMREX_ASSERT(rho > 0.0);
		return static_cast<amrex::Real>(NAN);
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEntropyFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	if (rho <= 0.0) {
		AMREX_ASSERT(rho > 0.0);
		return static_cast<amrex::Real>(NAN);
	}
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {rho, eint};
	return tables.all_tables.interpolate_single(point, ENTROPY_IDX);
}

struct EOSTabulatedRegistry {
	resampledGpuConstTables host;
	resampledGpuConstTables device;
};

extern AMREX_GPU_MANAGED EOSTabulatedRegistry *g_eos_tabulated_registry;

inline AMREX_GPU_HOST_DEVICE auto getEOSTabulatedRegistry() -> EOSTabulatedRegistry * { return g_eos_tabulated_registry; }

void registerEOSTabulated(resampledGpuConstTables host_tables, resampledGpuConstTables device_tables);

} // namespace quokka::ResampledCooling

#endif // EOS_TABULATED_REGISTRY_HPP_
