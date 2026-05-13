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

#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "math/ODEIntegrate.hpp"
#include "math/root_finding.hpp"
#include "util/DataTable.hpp"
#include <format>

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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(Real const rho, Real const Tgas, resampledGpuConstTables const &tables) -> Real
{
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

	// Temperature is monotonic in the tabulated specific internal energy, so root-finding on T(Eint) - Tgas converges.
	auto f = [=](Real Eint) -> Real { return ComputeTgasFromEgas(rho, Eint, tables) - Tgas; };

	int max_iter = 32;
	auto tol = quokka::math::eps_tolerance<Real>{};
	auto const [Eint_lo, Eint_hi] = quokka::math::toms748_solve(f, Eint_min, Eint_max, Tmin - Tgas, Tmax - Tgas, tol, max_iter);

	return Eint_hi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeCoolingLength(Real const rho, Real const Eint, resampledGpuConstTables const &tables,
								   Real const_heating_rate = 0.0) -> Real
{
	// Compute cooling length l_cool = c_s * t_cool
	// Convert Eint (energy density) to eint (specific energy) and then to fast log scale for interpolation
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};

	// Interpolate sound speed from data tables
	const Real cs = tables.sound_speeds.interpolate_single(point);

	const Real Edot = resampled_cooling_function(rho, Eint, tables) + const_heating_rate;
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
	Real const_heating_rate;

	AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(Real rho_in, resampledGpuConstTables const &tables_in, Real const_heating_rate_in)
	    : rho(rho_in), tables(tables_in), const_heating_rate(const_heating_rate_in)
	{
	}

	AMREX_GPU_HOST_DEVICE ~ResampledCoolingFunctor() = default;
	AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(ResampledCoolingFunctor const &) = default;
	AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(ResampledCoolingFunctor &&) = default;
	AMREX_GPU_HOST_DEVICE auto operator=(ResampledCoolingFunctor const &) -> ResampledCoolingFunctor & = default;
	AMREX_GPU_HOST_DEVICE auto operator=(ResampledCoolingFunctor &&) -> ResampledCoolingFunctor & = default;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs) const -> int
	{
		// compute temperature and cooling rate
		const Real Eint = y_data[0];
		y_rhs[0] = resampled_cooling_function(rho, Eint, tables) + const_heating_rate;
		return 0; // success
	}
};

// const_heating_rate_per_H: unit erg/s/H
template <typename problem_t>
auto computeCooling(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &mf_fc, const Real dt_in, resampled_tables &resampledTables,
		    const Real temp_floor, const Real const_heating_rate_per_H) -> bool
{
	const BL_PROFILE("quokka::ResampledCooling::computeCooling()");

	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		amrex::Abort("Resampled cooling requires a non-isothermal EOS with positive gas internal energy.");
	}

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

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> state_fc{};
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				state_fc[dir] = mf_fc[dir].const_array(iter);
			}
		}

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    		// cooling function
    		const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
    		const Real nH = rho * tables.cloudy_H_mass_fraction / C::m_p; // unit: cm^-3
    		const ResampledCoolingFunctor user_rhs(rho, tables, const_heating_rate_per_H * nH); // unit: erg/cm^3/s

			// state vector
			const Real Eint = HydroSystem<problem_t>::ComputeInternalEnergy(state, i, j, k, &state_fc);
			quokka::valarray<Real, 1> y = {Eint};

			// integration tolerance
			const Real Eint_floor = (temp_floor > 0.0) ? ComputeEgasFromTgas(rho, temp_floor, tables) : 0.0;
			const Real abstol_floor = amrex::max(Eint_floor, std::numeric_limits<Real>::min());
			quokka::valarray<Real, 1> const abstol = {reltol_floor * abstol_floor};

			// do integration with RK2 (Heun's method)
			int nsteps = 0;
			rk_adaptive_integrate(user_rhs, 0, y, dt, rtol, abstol, nsteps);
			nsubsteps(i, j, k) = nsteps;

			// check if integration failed
			if (nsteps >= maxStepsODEIntegrate) {
				Real const Edot = resampled_cooling_function(rho, Eint, tables) + const_heating_rate_per_H * nH; // unit: erg/cm^3/s
				Real const t_cool = (Edot != 0.0) ? std::abs(Eint / Edot) : std::numeric_limits<Real>::max();
				printf("max substeps exceeded! rho = %.17e, Eint = %.17e, cooling " // NOLINT
				       "time = %g, dt = %.17e\n",
				       rho, Eint, t_cool, dt);
			}
			const Real Eint_new = (temp_floor > 0.0) ? amrex::max(y[0], Eint_floor) : y[0];
			const Real dEint = Eint_new - Eint;

			state(i, j, k, HydroSystem<problem_t>::energy_index) += dEint;
			state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += dEint;
		});
	}

	const int nmax = nsubstepsMF.max(0);
	const Real navg = static_cast<Real>(nsubstepsMF.sum(0)) / static_cast<Real>(nsubstepsMF.boxArray().numPts());
	amrex::Print() << std::format("\tcooling substeps (per cell): avg {}, max {}\n", navg, nmax);

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
