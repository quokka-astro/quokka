// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testRadLineCooling.cpp
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

struct CoolingProblem {}; // dummy type to allow compile-type polymorphism via template specialization

const double cooling_rate = 0.1;
const double CR_heating_rate = 0.03;

constexpr double c = 1.0;
constexpr double chat = c;
constexpr double v0 = 0.0;
constexpr double kappa0 = 0.0;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // mean molecular weight; so that C_V = 1.0
constexpr double C_V = 1.0;

constexpr double nu_unit = 1.0;
constexpr double erad_floor = a_rad * 1e-20;

const double max_time = 10.0;

template <> struct SimulationData<CoolingProblem> : quokka::onezone::Recorder {};

template <> struct quokka::EOS_Traits<CoolingProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<CoolingProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	// A custom unit system is used here to replicate a dimentionless unit system (c = k_B = a_rad = G = 1), for testing units conversion
	static constexpr UnitSystem unit_system = UnitSystem::CUSTOM;
	static constexpr double unit_length = 1.733039549e-33;
	static constexpr double unit_mass = 2.333695323e-05;
	static constexpr double unit_time = 5.780797690e-44;
	static constexpr double unit_temperature = 1.519155670e+32;
	// Equivalently, set
	// static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	// static constexpr double boltzmann_constant = k_B;
	// static constexpr double gravitational_constant = 1.0;
	// static constexpr double c_light = c;
	// static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<CoolingProblem> {
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = nu_unit;
	static constexpr double c_hat_over_c = chat / c;
};

template <> struct ISM_Traits<CoolingProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr bool thermal_band_photochemistry = false;   // no photochemistry network in this problem
	static constexpr bool dust_chemical_band_absorption = false; // no chemical bands in this problem
	static constexpr bool enable_linear_cooling_heating = true;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[0] = cooling_rate * temperature;
	return cooling;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[0] = cooling_rate;
	return cooling;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
{
	return CR_heating_rate;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<CoolingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<CoolingProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
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

template <> void QuokkaSimulation<CoolingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const double Egas = quokka::EOS<CoolingProblem>::ComputeEintFromTgas(rho0, T0);

	quokka::onezone::Cell<CoolingProblem> cell{};
	cell[RadSystem<CoolingProblem>::radEnergy_index] = erad_floor;
	cell[RadSystem<CoolingProblem>::gasEnergy_index] = Egas + 0.5 * rho0 * v0 * v0;
	cell[RadSystem<CoolingProblem>::gasDensity_index] = rho0;
	cell[RadSystem<CoolingProblem>::gasInternalEnergy_index] = Egas;
	cell[RadSystem<CoolingProblem>::x1GasMomentum_index] = v0 * rho0;
	quokka::onezone::FillUniform<CoolingProblem>(grid_elem, cell);
}

template <> void QuokkaSimulation<CoolingProblem>::computeAfterTimestep() { userData_.record(state_new_cc_[0], Geom(0), tNew_[0]); }

auto problem_main() -> int
{
	// This problem is a test of line cooling and cosmic-ray heating in a uniform medium. The gas/dust opacity is set to zero, so that the radiation does
	// not interact with matter. The initial conditions are set to a constant temperature and zero radiation energy density. The gas cools at a rate of 0.1
	// per unit time, and is heated by cosmic rays at a rate of 0.03 per unit time. The exact solution is given by the following system of equations:
	// dTgas/dt = -0.1 * Tgas + 0.03,

	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 0.8;

	const double the_dt = 1.0e-2;

	// Problem initialization
	QuokkaSimulation<CoolingProblem> sim;

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.maxDt_ = the_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	sim.userData_.addObservable("Tgas", [](quokka::onezone::CellValues const &cell) {
		const amrex::Real rho = cell[RadSystem<CoolingProblem>::gasDensity_index];
		static_assert(!Physics_Traits<CoolingProblem>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
		const amrex::Real Eint = quokka::EOS<CoolingProblem>::ComputeEintFromEgas(
		    rho, cell[RadSystem<CoolingProblem>::x1GasMomentum_index], cell[RadSystem<CoolingProblem>::x2GasMomentum_index],
		    cell[RadSystem<CoolingProblem>::x3GasMomentum_index], cell[RadSystem<CoolingProblem>::gasEnergy_index], 0.0);
		return quokka::EOS<CoolingProblem>::ComputeTgasFromEint(rho, Eint);
	});
	sim.userData_.addObservable("Erad", [](quokka::onezone::CellValues const &cell) { return cell[RadSystem<CoolingProblem>::radEnergy_index]; });

	// exact solution
	const auto Egas_exact = [](double t) {
		return std::exp(-cooling_rate * t) * (cooling_rate * T0 - CR_heating_rate + CR_heating_rate * std::exp(cooling_rate * t)) / cooling_rate;
	};
	const auto Tgas_exact = [=](double t) { return Egas_exact(t) / C_V; };
	const auto Erad_exact = [=](double t) { return -(Egas_exact(t) - C_V * T0 - CR_heating_rate * t) * (chat / c); };

	const std::vector<quokka::onezone::Check> checks = {
	    {.name = "Tgas and Erad relative L1 error",
	     .comparisons = {{.observable = "Tgas", .reference = quokka::onezone::AnalyticReference(Tgas_exact)},
			     {.observable = "Erad", .reference = quokka::onezone::AnalyticReference(Erad_exact)}},
	     .tolerance = 0.0005}};
	const std::vector<quokka::onezone::Figure> figures = {
	    {.filename = "rad_line_cooling_single_group.pdf",
	     .curves = {{.observable = "Tgas", .label = "T_gas", .reference = quokka::onezone::AnalyticReference(Tgas_exact)},
			{.observable = "Erad", .label = "E_rad", .reference = quokka::onezone::AnalyticReference(Erad_exact)}},
	     .xlabel = "t (dimensionless)",
	     .ylabel = "T or Erad (dimensionless)"}};
	return quokka::onezone::Run(sim, checks, {}, "rad_line_cooling.csv", figures);
}
