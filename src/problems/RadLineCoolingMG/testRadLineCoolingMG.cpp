// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testRadLineCoolingMG.cpp
/// \brief Defines a test problem for line cooling and cosmic-ray heating in a uniform medium.
///

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_dust_system.hpp"
#include "util/OneZoneHarness.hpp"
#include <cmath>
#include <vector>

#include "AMReX_Print.H"
#include "physics_info.hpp"

struct CoolingProblemMG {}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int n_groups_ = 4;
constexpr amrex::GpuArray<double, 5> rad_boundaries_ = {1.00000000e-03, 1.77827941e-02, 3.16227766e-01, 5.62341325e+00, 1.00000000e+02};

constexpr double chat_over_c = 1.0;
constexpr double v0 = 0.0;
constexpr double kappa0 = 0.0;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // mean molecular weight; so that C_V = 1.0
constexpr double C_V = 1.0;

constexpr double nu_unit = 1.0;
constexpr double Erad_bar = a_rad * T0 * T0 * T0 * T0;
constexpr double Erad_floor_ = a_rad * 1e-20;
constexpr double Erad_FUV = Erad_bar; // = 1.0

const double max_time = 10.0;
const int line_index = 0; // last group
const double cooling_rate = 0.1;
const double CR_heating_rate = 0.03;
const double PE_rate = 0.02;

template <> struct SimulationData<CoolingProblemMG> : quokka::onezone::Recorder {};

template <> struct quokka::EOS_Traits<CoolingProblemMG> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<CoolingProblemMG> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr int nGroups = n_groups_;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<CoolingProblemMG> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = nu_unit;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
};

template <> struct ISM_Traits<CoolingProblemMG> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = true;
	static constexpr bool thermal_band_photochemistry = false;   // no photochemistry network in this problem
	static constexpr bool dust_chemical_band_absorption = false; // no chemical bands in this problem
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/,
											       amrex::Real const /*num_density*/) -> amrex::Real
{
	return PE_rate / Erad_FUV;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[line_index] = cooling_rate * temperature;
	return cooling;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[line_index] = cooling_rate;
	return cooling;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
{
	return CR_heating_rate;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<CoolingProblemMG>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
								  const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[1][i] = kappa0;
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<CoolingProblemMG>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const double Egas = quokka::EOS<CoolingProblemMG>::ComputeEintFromTgas(rho0, T0);

	quokka::onezone::Cell<CoolingProblemMG> cell{};
	for (int g = 0; g < n_groups_; ++g) {
		cell[RadSystem<CoolingProblemMG>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = (g == n_groups_ - 1) ? Erad_FUV : Erad_floor_;
	}
	cell[RadSystem<CoolingProblemMG>::gasEnergy_index] = Egas + 0.5 * rho0 * v0 * v0;
	cell[RadSystem<CoolingProblemMG>::gasDensity_index] = rho0;
	cell[RadSystem<CoolingProblemMG>::gasInternalEnergy_index] = Egas;
	cell[RadSystem<CoolingProblemMG>::x1GasMomentum_index] = v0 * rho0;
	quokka::onezone::FillUniform<CoolingProblemMG>(grid_elem, cell);
}

template <> void QuokkaSimulation<CoolingProblemMG>::computeAfterTimestep() { userData_.record(state_new_cc_[0], Geom(0), tNew_[0]); }

auto problem_main() -> int
{
	// This problem is a test of photoelectric heating, line cooling, and cosmic-ray heating in a uniform medium with multigroup radiation. The gas/dust
	// opacity is set to zero, so that the radiation does not interact with matter. The initial conditions are set to a constant temperature and radiation
	// energy density Erad_FUV = 1. The gas cools at a rate of 0.1 per unit time, and is heated by cosmic rays at a rate of 0.03 per unit time. The
	// photoelectric heating rate is 0.02 * Erad_FUV per unit time. The exact solution is given by the following system of equations: dTgas/dt = -0.1 * Tgas
	// + 0.03 + 0.02 * Erad_FUV where Erad_FUV = 1.0 is constant over time.

	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 0.8;

	const double the_dt = 1.0e-2;

	// Problem initialization
	QuokkaSimulation<CoolingProblemMG> sim;

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.maxDt_ = the_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	sim.userData_.addObservable("Tgas", [](quokka::onezone::CellValues const &cell) {
		const amrex::Real rho = cell[RadSystem<CoolingProblemMG>::gasDensity_index];
		static_assert(!Physics_Traits<CoolingProblemMG>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
		const amrex::Real Eint = quokka::EOS<CoolingProblemMG>::ComputeEintFromEgas(
		    rho, cell[RadSystem<CoolingProblemMG>::x1GasMomentum_index], cell[RadSystem<CoolingProblemMG>::x2GasMomentum_index],
		    cell[RadSystem<CoolingProblemMG>::x3GasMomentum_index], cell[RadSystem<CoolingProblemMG>::gasEnergy_index], 0.0);
		return quokka::EOS<CoolingProblemMG>::ComputeTgasFromEint(rho, Eint);
	});
	sim.userData_.addObservable("Erad_line", [](quokka::onezone::CellValues const &cell) {
		return cell[RadSystem<CoolingProblemMG>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * line_index];
	});

	// exact solution
	const double heating_rate = ISM_Traits<CoolingProblemMG>::enable_photoelectric_heating ? PE_rate + CR_heating_rate : CR_heating_rate;
	const auto Egas_exact = [=](double t) {
		return std::exp(-cooling_rate * t) * (cooling_rate * T0 - heating_rate + heating_rate * std::exp(cooling_rate * t)) / cooling_rate;
	};
	const auto Tgas_exact = [=](double t) { return Egas_exact(t) / C_V; };
	const auto Erad_line_exact = [=](double t) { return -(Egas_exact(t) - C_V * T0 - heating_rate * t) * chat_over_c; };

	const std::vector<quokka::onezone::Check> checks = {
	    {.name = "Tgas and Erad_line relative L1 error",
	     .comparisons = {{.observable = "Tgas", .reference = quokka::onezone::AnalyticReference(Tgas_exact)},
			     {.observable = "Erad_line", .reference = quokka::onezone::AnalyticReference(Erad_line_exact)}},
	     .tolerance = 0.0005}};
	return quokka::onezone::Run(sim, checks, {}, "rad_line_cooling_MG.csv");
}
