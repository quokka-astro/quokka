// ABOUTME: Header for resampled cooling tables that interpolate on (rho, e_int) grid
// ABOUTME: Uses HDF5-format tables produced by extern/cooling/resample_grackle_cooling_tables.py
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

#include "cooling/EOSTabulatedRegistry.hpp"
#include "hydro/hydro_system.hpp"
#include "math/ODEIntegrate.hpp"
#include "math/root_finding.hpp"
#include "util/DataTable.hpp"
#include <format>

namespace quokka::ResampledCooling
{

// Output indices into the DataTable<2, 5> for the five cooling quantities

class resampled_tables
{
      public:
	quokka::DataTable<2, 5> all_tables;
	amrex::Real cloudy_H_mass_fraction;
	bool include_pe{false};

	[[nodiscard]] auto const_tables() const -> resampledGpuConstTables;
	[[nodiscard]] auto const_tables_host() const -> resampledGpuConstTables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto resampled_cooling_function(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {rho, eint};

	// Interpolate cooling rate (DataTable handles fast_log coordinate transform internally)
	const Real Edot_over_rhosq = tables.all_tables.interpolate_single(point, COOLING_RATE_IDX);
	const Real Edot = Edot_over_rhosq * (rho * rho);
	return Edot;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeCoolingLength(Real const rho, Real const Eint, resampledGpuConstTables const &tables,
								   Real const_heating_rate = 0.0) -> Real
{
	// Compute cooling length l_cool = c_s * t_cool
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {rho, eint};
	const Real cs = tables.all_tables.interpolate_single(point, SOUND_SPEED_IDX);

	const Real Edot = resampled_cooling_function(rho, Eint, tables) + const_heating_rate;
	const Real t_cool = (Edot != 0.0) ? std::abs(Eint / Edot) : std::numeric_limits<Real>::max();

	return cs * t_cool;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputePressureFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {rho, eint};
	return tables.all_tables.interpolate_single(point, PRESSURE_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeSoundSpeedFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real
{
	const Real eint = Eint / rho;
	std::array<amrex::Real, 2> const point = {rho, eint};
	return tables.all_tables.interpolate_single(point, SOUND_SPEED_IDX);
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
			const Real nH = rho * tables.cloudy_H_mass_fraction / C::m_p;			    // unit: cm^-3
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

// const_heating_rate_per_H: unit erg/s/H
template <typename problem_t>
auto computeCooling(amrex::MultiFab &mf, const Real dt_in, resampled_tables &resampledTables, const Real temp_floor, amrex::MultiFab const &n_gamma_mf,
		    const Real const_heating_rate_per_H) -> bool
{
	const BL_PROFILE("quokka::ResampledCooling::computeCooling()");

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf.boxArray() == n_gamma_mf.boxArray(), "mf and n_gamma_mf must have the same BoxArray.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf.DistributionMap() == n_gamma_mf.DistributionMap(), "mf and n_gamma_mf must have the same DistributionMap.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_gamma_mf.nComp() >= 1, "n_gamma_mf must have at least one component.");

	const Real dt = dt_in;
	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4; // not recommended to change this

	auto tables = resampledTables.const_tables();

	constexpr Real alphaB = 2.6e-13;			 // cm^3 / s
	constexpr Real sigma_HI = 6.3e-18;			 // cm^2
	constexpr Real mean_particle_mass_mu = 1.27;		 // dimensionless
	constexpr Real mH = 1.67e-24;				 // g
	constexpr Real photoion_heat_per_abs = 13.6 * C::ev2erg; // erg

	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab nsubstepsMF(ba, dmap, 1, 0);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &n_gamma = n_gamma_mf.const_array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real x1Mom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const Real x2Mom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const Real x3Mom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<problem_t>::energy_index);

			// number density
			const Real nH = rho * tables.cloudy_H_mass_fraction / C::m_p; // unit: cm^-3

			const Real n = (rho > 0.0) ? (rho / (mean_particle_mass_mu * mH)) : 0.0;
			const Real ng = amrex::max(n_gamma(i, j, k, 0), 0.0);

			Real photoion_heating_rate = 0.0; // unit: erg/cm^3/s
			if ((n > 0.0) && (ng > 0.0)) {
				const Real a = alphaB * n;
				const Real b = C::c_light * sigma_HI * ng;
				const Real disc = b * b + 4.0 * a * b;
				Real x = (-b + std::sqrt(disc)) / (2.0 * a);
				x = amrex::min<Real>(1.0, amrex::max<Real>(0.0, x));
				const Real nHI = n * (1.0 - x);
				photoion_heating_rate = C::c_light * sigma_HI * nHI * ng * photoion_heat_per_abs;
			}

			const auto massScalars = RadSystem<problem_t>::ComputeMassScalars(state, i, j, k);
			const Real Eint_floor = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, temp_floor, massScalars);

			const Real Eint = quokka::EOS<problem_t>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas, 0.0);
			const Real heating_rate = const_heating_rate_per_H * nH + photoion_heating_rate;
			const ResampledCoolingFunctor user_rhs(rho, tables, heating_rate); // unit: erg/cm^3/s
			quokka::valarray<Real, 1> y = {Eint};
			quokka::valarray<Real, 1> const abstol = {reltol_floor * Eint_floor};

			// do integration with RK2 (Heun's method)
			int nsteps = 0;
			rk_adaptive_integrate(user_rhs, 0, y, dt, rtol, abstol, nsteps);
			nsubsteps(i, j, k) = nsteps;

			// check if integration failed
			if (nsteps >= maxStepsODEIntegrate) {
				Real const Edot = resampled_cooling_function(rho, Eint, tables) + heating_rate; // unit: erg/cm^3/s
				Real const t_cool = (Edot != 0.0) ? std::abs(Eint / Edot) : std::numeric_limits<Real>::max();
				printf("max substeps exceeded! rho = %.17e, Eint = %.17e, cooling " // NOLINT
				       "time = %g, dt = %.17e\n",
				       rho, Eint, t_cool, dt);
			}

			Real Eint_new = y[0];
			Eint_new = amrex::max(Eint_new, Eint_floor);
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
