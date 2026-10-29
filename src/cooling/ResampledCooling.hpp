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

//! \brief Exact (Townsend 2009-style) integration of the isochoric cooling ODE dE/dt = f(E) = rho^2 Lambda(rho, E/rho) + heating.
//!
//! This integrates exactly the same rate function that ResampledCoolingFunctor gives the RK integrator. At fixed rho the
//! bilinear table interpolant is linear in FastMath::lg(e) along the e axis (rate output is untransformed), and
//! FastMath::lg(e) = 2*(mantissa - 1) + exponent is itself piecewise linear in e with kinks only at powers of two.
//! Between consecutive breakpoints (table nodes in lg(e), and integer lg(e)) the net rate f(E) is therefore affine in E,
//! for any rho (the rho weights are constant) and any heating rate that is constant over the step. On each such piece
//! the time to reach the next breakpoint is the Townsend temporal-evolution-function difference
//!   tau = int dE / f(E) = (dE / f0) * log1p(z) / z,   z = (f_b - f0) / f0,
//! and the solution at time t is E(t) = E0 + f0 * t * expm1(beta t) / (beta t), beta = (f_b - f0) / (E_b - E0).
//! The cumulative Y function cannot be tabulated once per table load: at a density between table rows the rate is a
//! linear blend of the two rows, and the crossing times are nonlinear in it (and depend on the cell's heating rate).
//! So the walk evaluates the closed form only for the pieces actually crossed during the step (usually zero or one).
//!
//! \return the time left over: 0 if the whole interval was integrated exactly; otherwise the trajectory reached the
//! edge of the table (clamped region) or the crossing cap, E holds the state reached and the caller integrates the rest.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto exact_cooling_integrate(Real const rho, Real &E, Real const heating, Real const dt,
								      resampledGpuConstTables const &tables) -> Real
{
	auto const &tab = tables.all_tables;
	const Real x0 = tab.coord_min[1];
	const Real x_end = tab.coord_max[1];
	const Real dx = tab.dcoord[1];
	const Real x_tol = 1.0e-12 * dx;
	constexpr int max_cross = 1024;

	Real t_left = dt;
	const Real e0 = E / rho;
	if (!(e0 > 0.0) || !(rho > 0.0)) {
		return t_left;
	}
	Real x = FastMath::lg(e0);
	Real f = resampled_cooling_function(rho, E, tables) + heating;

	for (int n = 0; n < max_cross; ++n) {
		if (f == 0.0) {
			return 0.0; // exact equilibrium
		}
		if (!(x >= x0) || !(x <= x_end)) {
			return t_left; // outside the table (clamped region): not handled here
		}
		const Real kf = (x - x0) / dx;
		Real xb = NAN;
		if (f < 0.0) {
			int kn = static_cast<int>(std::floor(kf));
			if (x0 + static_cast<Real>(kn) * dx >= x - x_tol) {
				--kn;
			}
			if (kn < 0) {
				return t_left; // would leave the table at the bottom
			}
			const Real xn = x0 + static_cast<Real>(kn) * dx;
			const Real xi = std::ceil(x - x_tol) - 1.0;
			xb = amrex::max(xn, xi);
		} else {
			int kn = static_cast<int>(std::floor(kf)) + 1;
			if (x0 + static_cast<Real>(kn) * dx <= x + x_tol) {
				++kn;
			}
			if (kn > tab.sizes[1] - 1) {
				return t_left; // would leave the table at the top
			}
			const Real xn = x0 + static_cast<Real>(kn) * dx;
			const Real xi = std::floor(x + x_tol) + 1.0;
			xb = amrex::min(xn, xi);
		}
		const Real Eb = rho * FastMath::pow2(xb);
		const Real fb = resampled_cooling_function(rho, Eb, tables) + heating;
		const Real dE = Eb - E; // same sign as f
		const Real z = (fb - f) / f;

		// time to reach the breakpoint (infinite if an equilibrium lies inside this piece)
		Real tb = std::numeric_limits<Real>::max();
		if (z > -1.0) {
			const Real Lz = (z == 0.0) ? 1.0 : std::log1p(z) / z;
			tb = (dE / f) * Lz;
		}

		if (tb >= t_left) {
			const Real beta = (fb - f) / dE;
			const Real bt = beta * t_left;
			const Real phi = (bt == 0.0) ? 1.0 : std::expm1(bt) / bt;
			E = E + f * t_left * phi;
			return 0.0;
		}
		E = Eb;
		f = fb;
		x = xb;
		t_left -= tb;
	}
	return t_left;
}

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
	amrex::iMultiFab exactMF(ba, dmap, 1, 0);

	// The exact integrator relies on the e axis using FastMath::lg spacing and the rate output being untransformed
	// (so that the interpolant is piecewise affine in e). Otherwise every cell uses the adaptive RK integrator.
	const bool use_exact = (tables.all_tables.spacing_types[1] == quokka::TransformType::fast_log) &&
			       (tables.all_tables.output_transforms[COOLING_RATE_IDX] == quokka::TransformType::linear);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);
		auto const &exact = exactMF.array(iter);

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> state_fc{};
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				state_fc[dir] = mf_fc[dir].const_array(iter);
			}
		}

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			// cooling function
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real nH = rho * tables.cloudy_H_mass_fraction / C::m_p;			      // unit: cm^-3
			const Real heating_rate_per_H = const_heating_rate_per_H + external_heating(i, j, k); // unit: erg/s/H
			const ResampledCoolingFunctor user_rhs(rho, tables, heating_rate_per_H * nH);	      // unit: erg/cm^3/s

			// state vector
			const Real Eint = HydroSystem<problem_t>::ComputeInternalEnergy(state, i, j, k, &state_fc);

			// exact integration where it applies (trajectory stays inside the table's e range)
			Real Eint_exact = Eint;
			const Real t_left = use_exact ? exact_cooling_integrate(rho, Eint_exact, heating_rate_per_H * nH, dt, tables) : dt;
			quokka::valarray<Real, 1> y = {Eint_exact};
			exact(i, j, k) = (t_left > 0.0) ? 0 : 1;

			// integration tolerance
			const Real Eint_floor = (temp_floor > 0.0) ? ComputeEgasFromTgas(rho, temp_floor, tables) : 0.0;

			// fallback: integrate the remaining time with RK2 (Heun's method)
			int nsteps = 0;
			if (t_left > 0.0) {
				const Real abstol_floor = amrex::max(Eint_floor, std::numeric_limits<Real>::min());
				quokka::valarray<Real, 1> const abstol = {reltol_floor * abstol_floor};
				rk_adaptive_integrate(user_rhs, 0, y, t_left, rtol, abstol, nsteps);
			}
			nsubsteps(i, j, k) = nsteps;

			// check if integration failed
			if (nsteps >= maxStepsODEIntegrate) {
				Real const Edot = resampled_cooling_function(rho, Eint, tables) + heating_rate_per_H * nH; // unit: erg/cm^3/s
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
	const Real exact_frac = static_cast<Real>(exactMF.sum(0)) / static_cast<Real>(exactMF.boxArray().numPts());
	amrex::Print() << std::format("\tcooling exact-integration fraction: {}\n", exact_frac);

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
