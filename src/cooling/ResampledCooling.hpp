// ABOUTME: Header for resampled cooling tables that interpolate on (rho, e_int) grid
// ABOUTME: Uses HDF5-format tables produced by extern/cooling/resample_cooling_tables.py
#ifndef RESAMPLEDCOOLING_HPP_ // NOLINT
#define RESAMPLEDCOOLING_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ResampledCooling.hpp
/// \brief Defines methods for interpolating cooling rates from resampled tables.
///

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_iMultiFab.H"

#include "fmt/core.h"
#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "math/ODEIntegrate.hpp"
#include "radiation/radiation_system.hpp"
#include "util/DataTable.hpp"

namespace quokka::ResampledCooling
{

struct resampledGpuConstTables {
	// GPU-friendly const table access
	quokka::DataTableGpuConst<2, 1> cooling_rates;
	quokka::DataTableGpuConst<2, 1> temperatures;
	quokka::DataTableGpuConst<2, 1> sound_speeds;
	quokka::DataTableGpuConst<2, 1> pressures;
	quokka::DataTableGpuConst<2, 1> entropies;

	// density range
	amrex::Real rho_min;
	amrex::Real rho_max;

	// specific internal energy range
	amrex::Real eint_min;
	amrex::Real eint_max;

	// hydrogen mass fraction
	amrex::Real cloudy_H_mass_fraction;
};

class resampled_tables
{
      public:
	quokka::DataTable<2, 1> cooling_rates;
	quokka::DataTable<2, 1> temperatures;
	quokka::DataTable<2, 1> sound_speeds;
	quokka::DataTable<2, 1> pressures;
	quokka::DataTable<2, 1> entropies;

	amrex::Real rho_min;
	amrex::Real rho_max;
	amrex::Real eint_min;
	amrex::Real eint_max;
	amrex::Real cloudy_H_mass_fraction;

	[[nodiscard]] auto const_tables() const -> resampledGpuConstTables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto resampled_cooling_function(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	// Convert Eint (energy density) to eint (specific energy) and then to fast log scale for interpolation
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};

	// Interpolate cooling rate from data tables
	const Real Edot_over_rhosq = tables.cooling_rates.interpolate_single(point);
	// unused computation of the numeric derivative, just to check if it compiles and runs
	// const Real d_Edot_over_d_rhosq = tables.cooling_rates.numeric_derivative(fast_log_rho_val, fast_log_eint_val)[0]; // NOLINT
	const Real Edot = Edot_over_rhosq * (rho * rho);
	return Edot;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	// Convert Eint (energy density) to eint (specific energy) and then to fast log scale for interpolation
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};

	// Interpolate temperature from data tables
	const Real Tgas = tables.temperatures.interpolate_single(point);

	return Tgas;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeCoolingLength(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	// Compute cooling length l_cool = c_s * t_cool
	// Convert Eint (energy density) to eint (specific energy) and then to fast log scale for interpolation
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};

	// Interpolate sound speed from data tables
	const Real cs = tables.sound_speeds.interpolate_single(point);

	const Real Edot = resampled_cooling_function(rho, Eint, tables);
	const Real t_cool = (Edot != 0.0) ? std::abs(Eint / Edot) : std::numeric_limits<Real>::max();

	return cs * t_cool;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputePressureFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	// Convert Eint (energy density) to eint (specific energy) and then to fast log scale for interpolation
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};

	// Interpolate pressure from data tables
	const Real P = tables.pressures.interpolate_single(point);

	return P;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEntropyFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	// Convert Eint (energy density) to eint (specific energy) and then to fast log scale for interpolation
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};

	// Interpolate entropy from data tables
	const Real K = tables.entropies.interpolate_single(point);

	return K;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeSoundSpeedFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	// Convert Eint (energy density) to eint (specific energy) and then to fast log scale for interpolation
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};

	// Interpolate sound speed from data tables
	const Real cs = tables.sound_speeds.interpolate_single(point);

	return cs;
}

struct ResampledCoolingFunctor {
	Real rho;
	resampledGpuConstTables tables;

	AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(Real rho_in, resampledGpuConstTables const &tables_in) : rho(rho_in), tables(tables_in) {}

	AMREX_GPU_HOST_DEVICE ~ResampledCoolingFunctor() = default;
	AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(ResampledCoolingFunctor const &) = default;
	AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(ResampledCoolingFunctor &&) = default;
	AMREX_GPU_HOST_DEVICE auto operator=(ResampledCoolingFunctor const &) -> ResampledCoolingFunctor & = default;
	AMREX_GPU_HOST_DEVICE auto operator=(ResampledCoolingFunctor &&) -> ResampledCoolingFunctor & = default;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs) const -> int
	{
		// compute temperature and cooling rate
		const Real Eint = y_data[0];
		y_rhs[0] = resampled_cooling_function(rho, Eint, tables);
		return 0; // success
	}
};

template <typename problem_t> auto computeCooling(amrex::MultiFab &mf, const Real dt_in, resampled_tables &resampledTables, const Real E_floor) -> bool
{
	const BL_PROFILE("quokka::ResampledCooling::computeCooling()");

	const Real dt = dt_in;
	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4; // not recommended to change this

	auto tables = resampledTables.const_tables();

	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab nsubstepsMF(ba, dmap, 1, 0);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real x1Mom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const Real x2Mom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const Real x3Mom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<problem_t>::energy_index);

			const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			const ResampledCoolingFunctor user_rhs(rho, tables);
			quokka::valarray<Real, 1> y = {Eint};
			quokka::valarray<Real, 1> const abstol = {reltol_floor * E_floor};

			// do integration with RK2 (Heun's method)
			int nsteps = 0;
			rk_adaptive_integrate(user_rhs, 0, y, dt, rtol, abstol, nsteps);
			nsubsteps(i, j, k) = nsteps;

			// check if integration failed
			if (nsteps >= maxStepsODEIntegrate) {
				Real const Edot = resampled_cooling_function(rho, Eint, tables);
				Real const t_cool = Eint / Edot;
				printf("max substeps exceeded! rho = %.17e, Eint = %.17e, cooling " // NOLINT
				       "time = %g, dt = %.17e\n",
				       rho, Eint, t_cool, dt);
			}
			const Real Eint_new = y[0];
			const Real dEint = Eint_new - Eint;

			state(i, j, k, HydroSystem<problem_t>::energy_index) += dEint;
			state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += dEint;
		});
	}

	const int nmax = nsubstepsMF.max(0);
	const Real navg = static_cast<Real>(nsubstepsMF.sum(0)) / static_cast<Real>(nsubstepsMF.boxArray().numPts());
	amrex::Print() << fmt::format("\tcooling substeps (per cell): avg {}, max {}\n", navg, nmax);

	// check if integration succeeded
	if (nmax >= maxStepsODEIntegrate) {
		amrex::Print() << "\t[ResampledCooling] Reaction ODE failure! Retrying hydro update...\n";
		return false;
	}
	return true; // success
}

auto readResampledData(std::string const &hdf5_file, resampled_tables &resampledTables) -> bool;

} // namespace quokka::ResampledCooling

#endif // RESAMPLEDCOOLING_HPP_
