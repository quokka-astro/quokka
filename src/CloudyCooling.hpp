#ifndef CLOUDYCOOLING_HPP_ // NOLINT
#define CLOUDYCOOLING_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CloudyCooling.hpp
/// \brief Defines methods for interpolating cooling rates from Cloudy tables.
///

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "AMReX_ParallelDescriptor.H"
#include "FastMath.hpp"
#include "GrackleDataReader.hpp"
#include "Interpolate2D.hpp"
#include "ODEIntegrate.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "root_finding.hpp"
#include <limits>

namespace quokka::cooling
{
//   Set H mass fraction according to "abundances ism" in Cloudy,
//   which assumes n_He / n_H = 0.098. This gives a value of about 0.72.
//   Using the default value of 0.76 will result in negative electron
//   densities at low temperature.
//   Below, we set X = 1 / (1 + (C::m_p + C::m_e)e * n_He / n_H).

constexpr double cloudy_H_mass_fraction = 1. / (1. + 0.098 * 3.971);

struct cloudyGpuConstTables {
	// these are non-owning, so can make a copy of the whole struct
	amrex::Table1D<const Real> const log_nH;
	amrex::Table1D<const Real> const log_Tgas;

	amrex::Table2D<const Real> const cool;
	amrex::Table2D<const Real> const heat;
	amrex::Table2D<const Real> const meanMolWeight;
};

class cloudy_tables
{
      public:
	std::unique_ptr<amrex::TableData<double, 1>> log_nH;
	std::unique_ptr<amrex::TableData<double, 1>> log_Tgas;

	std::unique_ptr<amrex::TableData<double, 2>> cooling;
	std::unique_ptr<amrex::TableData<double, 2>> heating;
	std::unique_ptr<amrex::TableData<double, 2>> mean_mol_weight;

	[[nodiscard]] auto const_tables() const -> cloudyGpuConstTables;
};

struct ODEUserData {
	Real rho{};
	Real gamma{};
	cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cloudy_cooling_function(Real const rho, Real const T, cloudyGpuConstTables const &tables) -> Real
{
	// interpolate cooling rates from Cloudy tables
	const Real rhoH = rho * cloudy_H_mass_fraction; // mass density of H species
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);
	const Real log_T = std::log10(T);

	const double logCool = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.cool);

	const double logHeat = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.heat);

	const double netLambda = FastMath::pow10(logHeat) - FastMath::pow10(logCool);

	// multiply by the square of H mass density (**NOT number density**)
	const double Edot = (rhoH * rhoH) * netLambda;

	return Edot;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(double rho, double Tgas, double gamma, cloudyGpuConstTables const &tables) -> Real
{
	// convert Egas (internal gas energy) to temperature
	const Real rhoH = rho * cloudy_H_mass_fraction;
	const Real nH = rhoH / (C::m_p + C::m_e);

	// compute mu from mu(T) table
	const Real mu = interpolate2d(std::log10(nH), std::log10(Tgas), tables.log_nH, tables.log_Tgas, tables.meanMolWeight);

	// compute thermal gas energy
	const Real n = rho / ((C::m_p + C::m_e) * mu);
	const Real Pgas = n * C::k_B * Tgas;
	const Real Egas = Pgas / (gamma - 1.);
	return Egas;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(double rho, double Egas, double gamma, cloudyGpuConstTables const &tables) -> Real
{
	// convert Egas (internal gas energy) to temperature

	// check whether temperature is out-of-bounds
	const Real Tmin_table = 10.;
	const Real Tmax_table = 1.0e9;
	const Real Eint_min = ComputeEgasFromTgas(rho, Tmin_table, gamma, tables);
	const Real Eint_max = ComputeEgasFromTgas(rho, Tmax_table, gamma, tables);

	if (Egas <= Eint_min) {
		return Tmin_table;
	} else if (Egas >= Eint_max) {
		return Tmax_table;
	}

	// solve for temperature given Eint (with fixed adiabatic index gamma)
	const Real rhoH = rho * cloudy_H_mass_fraction;
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);

	// mean molecular weight (in Grackle tables) is defined w/r/t
	// (C::m_p + C::m_e)
	const Real C = (gamma - 1.) * Egas / (C::k_B * (rho / (C::m_p + C::m_e)));

	// solve for mu(T)*C == T.
	// (Grackle does this with a fixed-point iteration. We use a more robust
	// method, similar to Brent's method, the TOMS748 method.)
	const Real reltol = 1.0e-5;
	const int maxIterLimit = 100;
	int maxIter = maxIterLimit;

	auto f = [log_nH, C, tables](const Real &T) noexcept {
		// compute new mu from mu(log10 T) table
		Real log_T = clamp(std::log10(T), 1., 9.);
		Real mu = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.meanMolWeight);
		Real fun = C * mu - T;
		return fun;
	};

	// compute temperature bounds using physics
	const Real mu_min = 0.60; // assuming fully ionized (mu ~ 0.6)
	const Real mu_max = 2.33; // assuming neutral fully molecular (mu ~ 2.33)
	const Real T_min = std::clamp(C * mu_min, Tmin_table, Tmax_table);
	const Real T_max = std::clamp(C * mu_max, Tmin_table, Tmax_table);

	// do root-finding
	quokka::math::eps_tolerance<Real> tol(reltol);
	Real T_sol = NAN;

	if (T_min < T_max) {
		auto bounds = quokka::math::toms748_solve(f, T_min, T_max, tol, maxIter);
		T_sol = 0.5 * (bounds.first + bounds.second);

		if ((maxIter >= maxIterLimit) || std::isnan(T_sol)) {
#if 0
			printf("\nTgas iteration failed! rho = %.17g, Eint = %.17g, nH = %e, Tgas "
			       "= %e, "
			       "bounds.first = %e, bounds.second = %e, T_min = %e, T_max = %e, "
			       "maxIter = %d\n",
			       rho, Egas, nH, T_sol, bounds.first, bounds.second, T_min, T_max, maxIter);
#endif
			T_sol = NAN;
		}
	} // else: return NAN

	return T_sol;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeCoolingLength(double rho, double Egas, double gamma, cloudyGpuConstTables const &tables) -> Real
{
	// convert (rho, Egas) to cooling length

	// 1. convert Egas (internal gas energy) to temperature
	const Real Tgas = ComputeTgasFromEgas(rho, Egas, gamma, tables);

	// 2. compute cooling time
	// interpolate cooling rates from Cloudy tables
	const Real rhoH = rho * cloudy_H_mass_fraction; // mass density of H species
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);
	const Real log_T = std::log10(Tgas);
	const double logCool = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.cool);
	const double LambdaCool = FastMath::pow10(logCool);
	const double Edot = (rhoH * rhoH) * LambdaCool;
	// compute cooling time
	const Real t_cool = Egas / Edot;

	// 3. compute cooling length c_s t_cool
	// compute mu from mu(T) table
	const Real mu = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.meanMolWeight);
	const Real c_s = std::sqrt(gamma * C::k_B * Tgas / (mu * (C::m_p + C::m_e)));

	// cooling length
	return c_s * t_cool;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeMMW(double rho, double Egas, double gamma, cloudyGpuConstTables const &tables) -> Real
{
	// convert (rho, Egas) to dimensionless mean molecular weight

	// 1. convert Egas (internal gas energy) to temperature
	const Real Tgas = ComputeTgasFromEgas(rho, Egas, gamma, tables);

	// 2. compute mu from mu(T) table
	const Real rhoH = rho * cloudy_H_mass_fraction; // mass density of H species
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);
	const Real log_T = std::log10(Tgas);
	const Real mu = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.meanMolWeight);
	return mu;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
{
	// unpack user_data
	auto *udata = static_cast<ODEUserData *>(user_data);
	const Real rho = udata->rho;
	const Real gamma = udata->gamma;
	cloudyGpuConstTables const &tables = udata->tables;

	// check whether temperature is out-of-bounds
	const Real Tmin = 10.;
	const Real Tmax = 1.0e9;
	const Real Eint_min = ComputeEgasFromTgas(rho, Tmin, gamma, tables);
	const Real Eint_max = ComputeEgasFromTgas(rho, Tmax, gamma, tables);

	// compute temperature and cooling rate
	const Real Eint = y_data[0];

	if (Eint <= Eint_min) {
		// set cooling to value at Tmin
		y_rhs[0] = cloudy_cooling_function(rho, Tmin, tables);
	} else if (Eint >= Eint_max) {
		// set cooling to value at Tmax
		y_rhs[0] = cloudy_cooling_function(rho, Tmax, tables);
	} else {
		// ok, within tabulated cooling limits
		const Real T = ComputeTgasFromEgas(rho, Eint, gamma, tables);
		if (!std::isnan(T)) { // temp iteration succeeded
			y_rhs[0] = cloudy_cooling_function(rho, T, tables);
		} else { // temp iteration failed
			y_rhs[0] = NAN;
			return 1; // failed
		}
	}

	return 0; // success
}

template <typename problem_t> auto computeCooling(amrex::MultiFab &mf, const Real dt_in, cloudy_tables &cloudyTables, const Real T_floor) -> bool
{
	BL_PROFILE("computeCooling()")

	const Real dt = dt_in;
	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4; // not recommended to change this

	auto tables = cloudyTables.const_tables();

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
			const Real gamma = quokka::EOS_Traits<problem_t>::gamma;
			ODEUserData user_data{rho, gamma, tables};
			quokka::valarray<Real, 1> y = {Eint};
			quokka::valarray<Real, 1> const abstol = {reltol_floor * ComputeEgasFromTgas(rho, T_floor, gamma, tables)};

			// do integration with RK2 (Heun's method)
			int nsteps = 0;
			rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, nsteps);
			nsubsteps(i, j, k) = nsteps;

			if (nsteps >= maxStepsODEIntegrate) {
				Real const T = ComputeTgasFromEgas(rho, Eint, quokka::EOS_Traits<problem_t>::gamma, tables);
				Real const Edot = cloudy_cooling_function(rho, T, tables);
				Real const t_cool = Eint / Edot;
				printf("max substeps exceeded! rho = %.17e, Eint = %.17e, T = %g, cooling "
				       "time = %g, dt = %.17e\n",
				       rho, Eint, T, t_cool, dt);
			}

			const Real Eint_new = y[0];
			const Real dEint = Eint_new - Eint;

			state(i, j, k, HydroSystem<problem_t>::energy_index) += dEint;
			state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += dEint;
		});
	}

	int nmax = nsubstepsMF.max(0);
	Real navg = static_cast<Real>(nsubstepsMF.sum(0)) / static_cast<Real>(nsubstepsMF.boxArray().numPts());
	amrex::Print() << fmt::format("\tcooling substeps (per cell): avg {}, max {}\n", navg, nmax);
	
	// check if integration succeeded
	if (nmax >= maxStepsODEIntegrate) {
		amrex::Print() << "\t[CloudyCooling] Reaction ODE failure. Max steps exceeded in cooling solve!\n";
		return false;
		//amrex::ParallelDescriptor::Abort();
	}
	return true; // success
}

void readCloudyData(std::string &grackle_hdf5_file, cloudy_tables &cloudyTables);

} // namespace quokka::cooling

#endif // CLOUDYCOOLING_HPP_
