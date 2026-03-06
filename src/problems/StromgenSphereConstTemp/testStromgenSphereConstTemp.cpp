//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgenSphereConstTemp.cpp
/// \brief Defines a test problem for the static Stromgen sphere with no temperature dependence.
///

#include "AMReX_GpuComplex.H"
#include "AMReX_ParmParse.H"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"
#include <fmt/format.h>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"


struct StromgenSphereConstTempProblem {
};

template <> struct quokka::EOS_Traits<StromgenSphereConstTempProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StromgenSphereConstTempProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gravitational_constant = C::Gconst;
	static constexpr double c_light = C::c_light;
	static constexpr double radiation_constant = C::a_rad;
};

template <> struct RadSystem_Traits<StromgenSphereConstTempProblem> {
	static constexpr double c_hat_over_c = Physics_Traits<StromgenSphereConstTempProblem>::c_light / C::c_light;
	static constexpr double Erad_floor = 1e-99;
	static constexpr int beta_order = 0;
	static constexpr std::array<double, NumChemActiveRadGroups+1> ChemActiveRadFreqBounds = {3.29e15, 1.50e16}; // Hz
};

template <> void RadSystem<StromgenSphereConstTempProblem>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);
	amrex::Real avg_freq = 0.5_rt * (RadSystem_Traits<StromgenSphereConstTempProblem>::ChemActiveRadFreqBounds[0] + RadSystem_Traits<StromgenSphereConstTempProblem>::ChemActiveRadFreqBounds[1]);
	amrex::Real L_star = Q * C::hplanck * avg_freq;
	amrex::Real volume = dx[0] * dx[1] * dx[2];
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		if ((i == 0) && (j == 0) && (k == 0)) {
			radEnergy(i, j, k) = L_star / volume;
		}
		else {
			radEnergy(i, j, k) = 0.0_rt;
		}
	});
}

template <> struct SimulationData<StromgenSphereConstTempProblem> {
	amrex::Real small_temp;
	amrex::Real small_dens;
	amrex::Real temperature;
	amrex::Real primary_species_1;
	amrex::Real primary_species_2;
	amrex::Real primary_species_3;
	amrex::Real Q;
	std::ofstream output_file_;
	amrex::Real tend;
};

template <> void QuokkaSimulation<StromgenSphereConstTempProblem>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species and temperature
	amrex::ParmParse const pp("stromgen");
	userData_.small_temp = 1e-2;
	pp.query("small_temp", userData_.small_temp);

	userData_.small_dens = 1e-60;
	pp.query("small_dens", userData_.small_dens);

	userData_.temperature = 1.0e4;
	pp.query("temperature", userData_.temperature);

	userData_.tend = 1000.0_rt;
	pp.query("tend", userData_.tend);

	userData_.primary_species_1 = 0.0e0_rt;
	pp.query("primary_species_1", userData_.primary_species_1);
	userData_.primary_species_2 = 1.0e2_rt;
	pp.query("primary_species_2", userData_.primary_species_2);
	userData_.primary_species_3 = 0.0e0_rt;
	pp.query("primary_species_3", userData_.primary_species_3);

	userData_.Q = 1.0e49_rt;
	pp.query("Q", userData_.Q);

	// userData_.output_file_.open("photoionization_output.csv");
	// userData_.output_file_ << "time,Erad,Frad,n_e,n_HI,n_HII,gas_temp,Egas,rho_gas,n_gamma\n";

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgenSphereConstTempProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgenSphereConstTempProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> void QuokkaSimulation<StromgenSphereConstTempProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	for (int n = 1; n <= NumSpec; ++n) {
		switch (n) {
			case 1:
				numdens[n - 1] = userData_.primary_species_1;
				break;
			case 2:
				numdens[n - 1] = userData_.primary_species_2;
				break;
			case 3:
				numdens[n - 1] = userData_.primary_species_3;
				break;
			default:
				amrex::Abort("Cannot initialize number density for chem specie");
				break;
		}
	}

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

	// initial_Egas = state.e * rhotot;
	const auto Egas0 = state.e * rhotot; // initial_Egas;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<StromgenSphereConstTempProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = 1.e-99_rt; // set a very low initial radiation energy density to avoid NaNs in the free-streaming regime
			state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<StromgenSphereConstTempProblem>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<StromgenSphereConstTempProblem>::scalar0_index + nn) = state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<StromgenSphereConstTempProblem>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5); // NOLINT

	// if (amrex::ParallelDescriptor::IOProcessor()) {
	// 	userData_.t_vec_.push_back(tNew_[0]);

	// 	// const amrex::Real Etot_i = values.at(RadSystem<PhotoionizationStreamingProblem>::gasEnergy_index)[0];
	// 	// const amrex::Real x1GasMom = values.at(RadSystem<PhotoionizationStreamingProblem>::x1GasMomentum_index)[0];
	// 	// const amrex::Real x2GasMom = values.at(RadSystem<PhotoionizationStreamingProblem>::x2GasMomentum_index)[0];
	// 	// const amrex::Real x3GasMom = values.at(RadSystem<PhotoionizationStreamingProblem>::x3GasMomentum_index)[0];
	// 	// const amrex::Real rho = values.at(RadSystem<PhotoionizationStreamingProblem>::gasDensity_index)[0];
	// 	// const amrex::Real Egas_i = RadSystem<PhotoionizationStreamingProblem>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);
	// 	const amrex::Real Erad_i = values.at(RadSystem<PhotoionizationStreamingProblem>::radEnergy_index)[0];
	// 	const amrex::Real Flux_i = values.at(RadSystem<PhotoionizationStreamingProblem>::x1RadFlux_index)[0];
	// 	const amrex::Real n_e = values.at(HydroSystem<PhotoionizationStreamingProblem>::scalar0_index)[0] / spmasses[0];
	// 	const amrex::Real n_HI = values.at(HydroSystem<PhotoionizationStreamingProblem>::scalar0_index + 1)[0] / spmasses[1];
	// 	const amrex::Real n_HII = values.at(HydroSystem<PhotoionizationStreamingProblem>::scalar0_index + 2)[0] / spmasses[2];
	// 	const amrex::Real n_gamma = Erad_i / (0.5_rt * (ChemActiveRadFreqBounds[0] + ChemActiveRadFreqBounds[1]) * C::hplanck);
	// 	const amrex::Real rho = values.at(RadSystem<PhotoionizationStreamingProblem>::gasDensity_index)[0];
	// 	const amrex::Real Egas_i = values.at(RadSystem<PhotoionizationStreamingProblem>::gasEnergy_index)[0];
	// 	quokka::optional<amrex::GpuArray<amrex::Real, NumSpec>> massScalars;
	// 	amrex::GpuArray<amrex::Real, NumSpec> scalars{};
	// 	scalars[0] = n_e * spmasses[0]; scalars[1] = n_HI * spmasses[1]; scalars[2] = n_HII * spmasses[2];
	// 	massScalars = scalars;
	// 	const amrex::Real temp = quokka::EOS<PhotoionizationStreamingProblem>::ComputeTgasFromEint(rho, Egas_i, massScalars);

	// 	// std::cout << "computeAfterTimestep() --> temp:" << temp << std::endl;
	// 	// userData_.Trad_vec_.push_back(std::pow(Erad_i / a_rad, 1. / 4.));
	// 	userData_.Erad_vec_.push_back(Erad_i);
	// 	userData_.n_e_vec_.push_back(n_e);
	// 	userData_.n_HI_vec_.push_back(n_HI);
	// 	userData_.n_HII_vec_.push_back(n_HII);
	// 	userData_.gas_temp_vec_.push_back(temp);
	// 	userData_.Egas_vec_.push_back(Egas_i);

	// 	userData_.output_file_ << tNew_[0] << "," << Erad_i << "," << Flux_i << "," << n_e << "," << n_HI << "," << n_HII << "," << temp << "," <<Egas_i << "," << rho << "," << n_gamma << "\n";
	// 	// userData_.Tgas_vec_.push_back(quokka::EOS<PhotoionizationStreamingProblem>::ComputeTgasFromEint(rho, Egas_i));
	// }
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.3;
	const double dt_max = 1e10;
	const int max_timesteps = 5000000;

	// Problem initialization
	QuokkaSimulation<StromgenSphereConstTempProblem> sim;

	// initialize
	sim.setInitialConditions();
	sim.stopTime_ = sim.userData_.tend;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// evolve
	sim.evolve();

	// std::vector<double> &Erad = sim.userData_.Erad_vec_;
	// std::vector<double> &t = sim.userData_.t_vec_;
	// std::vector<double> &n_HI = sim.userData_.n_HI_vec_;
	// std::vector<double> &n_HII = sim.userData_.n_HII_vec_;
	// std::vector<double> &gas_temp = sim.userData_.gas_temp_vec_;
	// std::vector<double> &Egas = sim.userData_.Egas_vec_;

	// read output variables
	// auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	// const int nx = static_cast<int>(position.size());

	// compute error norm
	// std::vector<double> erad(nx);
	// std::vector<double> erad_exact(nx);
	// std::vector<double> xs(nx);
	// for (int i = 0; i < nx; ++i) {
	// 	amrex::Real const x = position[i];
	// 	xs.at(i) = x;
	// 	erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
	// 	double erad_sim = 0.0;
	// 	for (int g = 0; g < Physics_Traits<PhotoionizationStreamingProblem>::nGroups; ++g) {
	// 		erad_sim += values.at(RadSystem<PhotoionizationStreamingProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g)[i];
	// 	}
	// 	erad.at(i) = erad_sim;
	// }

	// double err_norm = 0.;
	// double sol_norm = 0.;
	// for (int i = 0; i < nx; ++i) {
	// 	err_norm += std::abs(erad[i] - erad_exact[i]);
	// 	sol_norm += std::abs(erad_exact[i]);
	// }

	// const double rel_err_norm = err_norm / sol_norm;
	// const double rel_err_tol = 0.01;
	// int status = 1;
	// if (rel_err_norm < rel_err_tol) {
	// 	status = 0;
	// }
	// amrex::Print() << "Relative L1 norm = " << rel_err_norm << '\n';

// #ifdef HAVE_PYTHON
// 	// Plot results
// 	matplotlibcpp::clf();
// 	// matplotlibcpp::ylim(0.0, 1.1);

// 	std::map<std::string, std::string> erad_args;
// 	std::map<std::string, std::string> n_HI_args;
// 	std::map<std::string, std::string> n_HII_args;
// 	std::map<std::string, std::string> temp_args;
// 	std::map<std::string, std::string> Egas_args;
// 	// std::map<std::string, std::string> erad_exact_args;
// 	erad_args["label"] = "Erad";
// 	n_HI_args["label"] = "n_HI";
// 	n_HII_args["label"] = "n_HII";
// 	temp_args["label"] = "gas temp";
// 	Egas_args["label"] = "Egas";
// 	// erad_exact_args["label"] = "exact solution";
// 	// erad_exact_args["linestyle"] = "--";
// 	// matplotlibcpp::plot(xs, erad, erad_args);

// 	matplotlibcpp::plot(t, n_HI, n_HI_args);
// 	matplotlibcpp::plot(t, n_HII, n_HII_args);
// 	// matplotlibcpp::plot(t, Erad, erad_args);
// 	// matplotlibcpp::plot(xs, erad_exact, erad_exact_args);
// 	// std::map<std::string, std::string> Erad_args;
// 	// Erad_args["label"] = "Erad";
// 	// matplotlibcpp::plot(t, Erad, Erad_args);
// 	// matplotlibcpp::plot(t)

// 	matplotlibcpp::legend();
// 	// matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
// 	matplotlibcpp::save("./radiation_streaming_photoionization.pdf");
// 	matplotlibcpp::clf();
// 	matplotlibcpp::plot(t, Erad, erad_args);
// 	matplotlibcpp::legend();
// 	matplotlibcpp::save("./radiation_energy.pdf");
// 	matplotlibcpp::clf();
// 	matplotlibcpp::plot(t, gas_temp, temp_args);
// 	matplotlibcpp::legend();
// 	matplotlibcpp::save("./gas_temperature.pdf");
// 	matplotlibcpp::clf();
// 	matplotlibcpp::plot(t, Egas, Egas_args);
// 	matplotlibcpp::legend();
// 	matplotlibcpp::save("./gas_energy.pdf");
// #endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	// return status;
	return 1;
}
