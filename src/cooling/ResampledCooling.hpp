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
#include "AMReX_Parser.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Scan.H"
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

//! \brief Spatially varying external heating rate per H atom (erg/s/H), evaluated as f(x, y, z, time, dt).
//!
//! Holds the compiled `heating_rate_external` parser together with the level geometry needed to turn
//! cell indices into coordinates. A default-constructed instance carries no parser and evaluates to zero.
struct ExternalHeatingRate {
	amrex::ParserExecutor<5> parser{};
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo{};
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
	amrex::Real time{0.0};
	amrex::Real dt{0.0};

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, [[maybe_unused]] int j, [[maybe_unused]] int k) const -> amrex::Real
	{
		if (!parser) {
			return 0.0;
		}
		const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + static_cast<amrex::Real>(0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
		const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + static_cast<amrex::Real>(0.5)) * dx[1];
#else
		const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
		const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + static_cast<amrex::Real>(0.5)) * dx[2];
#else
		const amrex::Real z = 0.0;
#endif
		return parser(x, y, z, time, dt);
	}
};

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

// const_heating_rate_per_H: spatially uniform heating rate, unit erg/s/H
// external_heating: optional position-dependent heating rate added to it, unit erg/s/H
template <typename problem_t>
auto computeCooling(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &mf_fc, const Real dt_in, resampled_tables &resampledTables,
		    const Real temp_floor, const Real const_heating_rate_per_H, ExternalHeatingRate const &external_heating = {}) -> bool
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

	// Two-phase integration: phase 1 runs at most nphase1 outer steps per cell. Unfinished cells save
	// their complete integrator state (time, y, dt; step counter as -istep in nsubsteps), are compacted
	// into an index list per box, and phase 2 finishes them over the compacted list.
	static const int nphase1 = []() {
		int n = 16;
		amrex::ParmParse const pp("cooling");
		pp.query("phase1_max_substeps", n);
		return amrex::Clamp(n, 1, maxStepsODEIntegrate);
	}();
	const int nphase1_d = nphase1;
	constexpr int nmaxsteps = maxStepsODEIntegrate;
	amrex::MultiFab resumeMF(ba, dmap, 3, 0); // time, dt, y

	auto cellSetup = [=] AMREX_GPU_DEVICE(amrex::Array4<Real> const &state, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &state_fc,
					      int i, int j, int k, Real &rho, Real &nH, Real &heating_rate_per_H, Real &Eint, Real &Eint_floor,
					      quokka::valarray<Real, 1> &abstol) noexcept {
		rho = state(i, j, k, HydroSystem<problem_t>::density_index);
		nH = rho * tables.cloudy_H_mass_fraction / C::m_p;			      // unit: cm^-3
		heating_rate_per_H = const_heating_rate_per_H + external_heating(i, j, k); // unit: erg/s/H
		Eint = HydroSystem<problem_t>::ComputeInternalEnergy(state, i, j, k, &state_fc);
		Eint_floor = (temp_floor > 0.0) ? ComputeEgasFromTgas(rho, temp_floor, tables) : 0.0;
		const Real abstol_floor = amrex::max(Eint_floor, std::numeric_limits<Real>::min());
		abstol = {reltol_floor * abstol_floor};
	};

	auto cellFinish = [=] AMREX_GPU_DEVICE(amrex::Array4<Real> const &state, amrex::Array4<int> const &nsubsteps, int i, int j, int k, int nsteps,
					       Real rho, Real nH, Real heating_rate_per_H, Real Eint, Real Eint_floor, Real y0) noexcept {
		nsubsteps(i, j, k) = nsteps;

		// check if integration failed
		if (nsteps >= nmaxsteps) {
			Real const Edot = resampled_cooling_function(rho, Eint, tables) + heating_rate_per_H * nH; // unit: erg/cm^3/s
			Real const t_cool = (Edot != 0.0) ? std::abs(Eint / Edot) : std::numeric_limits<Real>::max();
			printf("max substeps exceeded! rho = %.17e, Eint = %.17e, cooling " // NOLINT
			       "time = %g, dt = %.17e\n",
			       rho, Eint, t_cool, dt);
		}
		const Real Eint_new = (temp_floor > 0.0) ? amrex::max(y0, Eint_floor) : y0;
		const Real dEint = Eint_new - Eint;

		state(i, j, k, HydroSystem<problem_t>::energy_index) += dEint;
		state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += dEint;
	};

	// phase 1
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);
		auto const &resume = resumeMF.array(iter);

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> state_fc{};
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				state_fc[dir] = mf_fc[dir].const_array(iter);
			}
		}

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real rho{};
			Real nH{};
			Real heating_rate_per_H{};
			Real Eint{};
			Real Eint_floor{};
			quokka::valarray<Real, 1> abstol{};
			cellSetup(state, state_fc, i, j, k, rho, nH, heating_rate_per_H, Eint, Eint_floor, abstol);
			const ResampledCoolingFunctor user_rhs(rho, tables, heating_rate_per_H * nH); // unit: erg/cm^3/s

			quokka::valarray<Real, 1> y = {Eint};
			Real time = 0;
			Real dtsub = rk_adaptive_initial_dt(user_rhs, 0, y, dt);
			int istep = 0;
			const int status = rk_adaptive_integrate_resume(user_rhs, time, y, dtsub, dt, rtol, abstol, istep, nphase1_d);

			if (status == 0 && istep < nmaxsteps) {
				// unfinished: save integrator state
				resume(i, j, k, 0) = time;
				resume(i, j, k, 1) = dtsub;
				resume(i, j, k, 2) = y[0];
				nsubsteps(i, j, k) = -istep;
				return;
			}
			const int nsteps = (status == 1) ? istep : nmaxsteps;
			cellFinish(state, nsubsteps, i, j, k, nsteps, rho, nH, heating_rate_per_H, Eint, Eint_floor, y[0]);
		});
	}

	// phase 2 (only if any cell on this rank is unfinished)
	if ((nphase1 < maxStepsODEIntegrate) && (nsubstepsMF.min(0, 0, true) < 0)) {
		long long ntotal2 = 0;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &state = mf.array(iter);
			auto const &nsubsteps = nsubstepsMF.array(iter);
			auto const &resume = resumeMF.array(iter);

			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> state_fc{};
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					state_fc[dir] = mf_fc[dir].const_array(iter);
				}
			}

			const auto lo = amrex::lbound(indexRange);
			const auto len = amrex::length(indexRange);
			const int npts = static_cast<int>(indexRange.numPts());

			// compact the unfinished cells into an index list
			amrex::Gpu::DeviceVector<int> cellList(npts);
			int *const cellListPtr = cellList.data();
			auto isUnfinished = [=] AMREX_GPU_DEVICE(int n) noexcept -> int {
				const int ii = lo.x + n % len.x;
				const int jj = lo.y + (n / len.x) % len.y;
				const int kk = lo.z + n / (len.x * len.y);
				return (nsubsteps(ii, jj, kk) < 0) ? 1 : 0;
			};
			const int nunfinished = amrex::Scan::PrefixSum<int>(
			    npts, [=] AMREX_GPU_DEVICE(int n) noexcept -> int { return isUnfinished(n); },
			    [=] AMREX_GPU_DEVICE(int n, int const &s) noexcept {
				    if (isUnfinished(n) != 0) {
					    cellListPtr[s] = n;
				    }
			    },
			    amrex::Scan::Type::exclusive, amrex::Scan::retSum);
			ntotal2 += nunfinished;

			if (nunfinished > 0) {
				amrex::ParallelFor(nunfinished, [=] AMREX_GPU_DEVICE(int m) noexcept {
					const int n = cellListPtr[m];
					const int i = lo.x + n % len.x;
					const int j = lo.y + (n / len.x) % len.y;
					const int k = lo.z + n / (len.x * len.y);

					Real rho{};
					Real nH{};
					Real heating_rate_per_H{};
					Real Eint{};
					Real Eint_floor{};
					quokka::valarray<Real, 1> abstol{};
					cellSetup(state, state_fc, i, j, k, rho, nH, heating_rate_per_H, Eint, Eint_floor, abstol);
					const ResampledCoolingFunctor user_rhs(rho, tables, heating_rate_per_H * nH); // unit: erg/cm^3/s

					Real time = resume(i, j, k, 0);
					Real dtsub = resume(i, j, k, 1);
					quokka::valarray<Real, 1> y = {resume(i, j, k, 2)};
					int istep = -nsubsteps(i, j, k);
					const int status = rk_adaptive_integrate_resume(user_rhs, time, y, dtsub, dt, rtol, abstol, istep, nmaxsteps);

					const int nsteps = (status == 1) ? istep : nmaxsteps;
					cellFinish(state, nsubsteps, i, j, k, nsteps, rho, nH, heating_rate_per_H, Eint, Eint_floor, y[0]);
				});
			}
			amrex::Gpu::streamSynchronize(); // cellList goes out of scope
		}
		amrex::Print(amrex::Print::AllProcs) << std::format("\tcooling phase 2 cells (rank {}): {}\n", amrex::ParallelDescriptor::MyProc(), ntotal2);
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
