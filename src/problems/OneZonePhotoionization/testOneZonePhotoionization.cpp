// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testOneZonePhotoionization.cpp
/// \brief Defines a test problem for photoionization.
///

#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "radiation/radiation_system.hpp"
#include "util/OneZoneHarness.hpp"
#include <string>
#include <vector>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct PhotoionizationStreamingProblem {};

constexpr double c = C::c_light;    // speed of light
constexpr double chat = C::c_light; // reduced speed of light
constexpr double kappa0 = 0;	    // opacity

template <> struct quokka::EOS_Traits<PhotoionizationStreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<PhotoionizationStreamingProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
};

template <> struct RadSystem_Traits<PhotoionizationStreamingProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = 0.0;
	static constexpr int beta_order = 0;
	static constexpr auto ChemBandsPowerLawIndex() -> double { return ChemBandsPowerLawIndex_; }
	static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1> { return ChemBandsHeader(); }
};

template <> struct SimulationData<PhotoionizationStreamingProblem> : quokka::onezone::Recorder {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real n_e_init{};
	amrex::Real n_HI_init{};
	amrex::Real n_HII_init{};
	amrex::Real n_photon{};
};

template <> void QuokkaSimulation<PhotoionizationStreamingProblem>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species and temperature
	amrex::ParmParse const pp("photoionization");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e3;
	userData_.n_e_init = 0.0e0_rt;
	userData_.n_HI_init = 1.0e2_rt;
	userData_.n_HII_init = 0.0e0_rt;
	userData_.n_photon = 1.0e5_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("n_e_init", userData_.n_e_init);
	pp.query("n_HI_init", userData_.n_HI_init);
	pp.query("n_HII_init", userData_.n_HII_init);
	pp.query("n_photon", userData_.n_photon);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();

	burn_t state;
	state.T = userData_.temperature;
	Real rhotot = 0.0_rt;
	state.xn[Species::e] = userData_.n_e_init;
	state.xn[Species::H] = userData_.n_HI_init;
	state.xn[Species::H_p] = userData_.n_HII_init;
	rhotot = state.xn[Species::e] * spmasses[Species::e] + state.xn[Species::H] * spmasses[Species::H] + state.xn[Species::H_p] * spmasses[Species::H_p];
	state.rho = rhotot;
	eos(eos_input_rt, state);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PhotoionizationStreamingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PhotoionizationStreamingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<PhotoionizationStreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	numdens[Species::e] = userData_.n_e_init;
	numdens[Species::H] = userData_.n_HI_init;
	numdens[Species::H_p] = userData_.n_HII_init;

	state.T = userData_.temperature;

	// find the density in g/cm^3
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n]; // spmasses contains the masses of all species, defined in EOS
	}
	state.rho = rhotot;
	// call the EOS to set initial internal energy e
	eos(eos_input_rt, state);

	const auto Erad0 = userData_.n_photon * RadSystem<PhotoionizationStreamingProblem>::GetChemBandQuanta(0);
	const auto Egas0 = state.e * rhotot; // initial_Egas;

	quokka::onezone::Cell<PhotoionizationStreamingProblem> cell{};
	for (int g = 0; g < Physics_Traits<PhotoionizationStreamingProblem>::nGroups; ++g) {
		cell[RadSystem<PhotoionizationStreamingProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = Erad0;
	}
	cell[RadSystem<PhotoionizationStreamingProblem>::gasEnergy_index] = Egas0;
	cell[RadSystem<PhotoionizationStreamingProblem>::gasDensity_index] = rhotot;
	cell[RadSystem<PhotoionizationStreamingProblem>::gasInternalEnergy_index] = Egas0;
	for (int nn = 0; nn < NumSpec; ++nn) {
		cell[HydroSystem<PhotoionizationStreamingProblem>::scalar0_index + nn] =
		    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
	}
	quokka::onezone::FillUniform<PhotoionizationStreamingProblem>(grid_elem, cell);
}

template <> void QuokkaSimulation<PhotoionizationStreamingProblem>::computeAfterTimestep() { userData_.record(state_new_cc_[0], Geom(0), tNew_[0]); }

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.3;
	const int max_timesteps = 5000000;
	const double constant_dt = 50.0;

	// Boundary conditions
	constexpr int nvars = RadSystem<PhotoionizationStreamingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<PhotoionizationStreamingProblem> sim(BCs_cc);
	double tend = 1000.0; // s
	amrex::ParmParse const pp("photoionization");
	pp.query("tend", tend);
	sim.stopTime_ = tend;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.constantDt_ = constant_dt;
	sim.plotfileInterval_ = -1;

	using PS = PhotoionizationStreamingProblem;
	using quokka::onezone::CellValues;
	const auto number_density = [](CellValues const &cell, int species) { return cell[HydroSystem<PS>::scalar0_index + species] / spmasses[species]; };
	sim.userData_.addObservable("n_e", [=](CellValues const &cell) { return number_density(cell, Species::e); });
	sim.userData_.addObservable("n_HI", [=](CellValues const &cell) { return number_density(cell, Species::H); });
	sim.userData_.addObservable("n_HII", [=](CellValues const &cell) { return number_density(cell, Species::H_p); });
	sim.userData_.addObservable("n_gamma",
				    [](CellValues const &cell) { return cell[RadSystem<PS>::radEnergy_index] / RadSystem<PS>::GetChemBandQuanta(0); });
	sim.userData_.addObservable("Egas", [](CellValues const &cell) { return cell[RadSystem<PS>::gasEnergy_index]; });
	sim.userData_.addObservable("gas_temp", [=](CellValues const &cell) {
		amrex::GpuArray<amrex::Real, NumSpec> scalars{};
		for (int n = 0; n < NumSpec; ++n) {
			scalars[n] = cell[HydroSystem<PS>::scalar0_index + n];
		}
		const quokka::optional<amrex::GpuArray<amrex::Real, NumSpec>> massScalars = scalars;
		return quokka::EOS<PS>::ComputeTgasFromEint(cell[RadSystem<PS>::gasDensity_index], cell[RadSystem<PS>::gasInternalEnergy_index], massScalars);
	});

	// reference solution: Julia integration of the same network
	int energy_switch = 1;
	amrex::ParmParse const pp1("network");
	pp1.query("energy_switch", energy_switch);
	if (energy_switch != 0 && energy_switch != 1) {
		amrex::Abort("Invalid energy_switch parameter");
	}
	const std::string filename = std::string("../extern/photoionization-julia/") + ((energy_switch == 0) ? "no_energy.csv" : "with_energy.csv");

	const double error_tol = 1e-6;
	std::vector<quokka::onezone::Check> checks;
	const std::vector<std::pair<std::string, int>> columns = {{"n_e", 1}, {"n_HI", 2}, {"n_HII", 3}, {"n_gamma", 4}, {"Egas", 5}};
	for (auto const &[name, column] : columns) {
		if (name == "Egas" && energy_switch == 0) {
			continue; // the gas energy is not evolved
		}
		checks.push_back({.name = name + " mean relative error",
				  .comparisons = {{.observable = name, .reference = quokka::onezone::TableReference(filename, column)}},
				  .tolerance = error_tol,
				  .norm = quokka::onezone::Norm::MeanRelative});
	}

	return quokka::onezone::Run(sim, checks, {}, "photoionization_quokka_output.csv");
}
