//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testOneZonePhotoionization.cpp
/// \brief Defines a test problem for photoionization.
///

#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct PhotoionizationStreamingProblem {
};

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
	static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1> { return ChemBandsHeader_; }
};

template <> struct SimulationData<PhotoionizationStreamingProblem> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real tend{};
	amrex::Real n_photon{};
	std::ofstream output_file_;
	std::vector<double> t_vec_;
	std::vector<double> n_e_vec_;
	std::vector<double> n_HI_vec_;
	std::vector<double> n_HII_vec_;
	std::vector<double> n_gamma_vec_;
	std::vector<double> Egas_vec_;
	std::vector<double> temp_vec_;
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
	userData_.primary_species_1 = 0.0e0_rt;
	userData_.primary_species_2 = 1.0e2_rt;
	userData_.primary_species_3 = 0.0e0_rt;
	userData_.tend = 1000.0_rt;
	userData_.n_photon = 1.0e5_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("tend", userData_.tend);
	pp.query("n_photon", userData_.n_photon);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();

	burn_t state;
	state.T = userData_.temperature;
	Real rhotot = 0.0_rt;
	state.xn[0] = userData_.primary_species_1;
	state.xn[1] = userData_.primary_species_2;
	state.xn[2] = userData_.primary_species_3;
	rhotot = state.xn[0] * spmasses[0] + state.xn[1] * spmasses[1] + state.xn[2] * spmasses[2];
	state.rho = rhotot;
	eos(eos_input_rt, state);

	const amrex::Real t = 0.0;
	const amrex::Real n_e = userData_.primary_species_1;
	const amrex::Real n_HI = userData_.primary_species_2;
	const amrex::Real n_HII = userData_.primary_species_3;
	const amrex::Real n_gamma = userData_.n_photon;
	const amrex::Real Egas_i = state.e * rhotot;
	const amrex::Real temp = userData_.temperature;

	userData_.output_file_.open("photoionization_quokka_output.csv");
	userData_.output_file_ << "time,n_e,n_HI,n_HII,n_gamma,Egas,gas_temp\n";
	userData_.output_file_ << std::scientific << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
	userData_.output_file_ << t << "," << n_e << "," << n_HI << "," << n_HII << "," << n_gamma << "," << Egas_i << "," << temp << "\n";
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

	const auto Erad0 = userData_.n_photon * RadSystem<PhotoionizationStreamingProblem>::GetChemBandQuanta(0);
	const auto Egas0 = state.e * rhotot; // initial_Egas;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<PhotoionizationStreamingProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad0;
			state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}
		state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PhotoionizationStreamingProblem>::x3GasMomentum_index) = 0.;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<PhotoionizationStreamingProblem>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<PhotoionizationStreamingProblem>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5); // NOLINT

	if (amrex::ParallelDescriptor::IOProcessor()) {
		const amrex::Real time = tNew_[0];
		const amrex::Real Erad_i = values.at(RadSystem<PhotoionizationStreamingProblem>::radEnergy_index)[0];
		const amrex::Real n_e = values.at(HydroSystem<PhotoionizationStreamingProblem>::scalar0_index)[0] / spmasses[0];
		const amrex::Real n_HI = values.at(HydroSystem<PhotoionizationStreamingProblem>::scalar0_index + 1)[0] / spmasses[1];
		const amrex::Real n_HII = values.at(HydroSystem<PhotoionizationStreamingProblem>::scalar0_index + 2)[0] / spmasses[2];
		const amrex::Real n_gamma = Erad_i / RadSystem<PhotoionizationStreamingProblem>::GetChemBandQuanta(0);
		const amrex::Real rho = values.at(RadSystem<PhotoionizationStreamingProblem>::gasDensity_index)[0];
		const amrex::Real Egas_i = values.at(RadSystem<PhotoionizationStreamingProblem>::gasEnergy_index)[0];
		const amrex::Real Eint_i = values.at(RadSystem<PhotoionizationStreamingProblem>::gasInternalEnergy_index)[0];
		quokka::optional<amrex::GpuArray<amrex::Real, NumSpec>> massScalars;
		amrex::GpuArray<amrex::Real, NumSpec> scalars{};
		scalars[0] = n_e * spmasses[0];
		scalars[1] = n_HI * spmasses[1];
		scalars[2] = n_HII * spmasses[2];
		massScalars = scalars;
		const amrex::Real temp = quokka::EOS<PhotoionizationStreamingProblem>::ComputeTgasFromEint(rho, Eint_i, massScalars);

		userData_.t_vec_.push_back(time);
		userData_.n_e_vec_.push_back(n_e);
		userData_.n_HI_vec_.push_back(n_HI);
		userData_.n_HII_vec_.push_back(n_HII);
		userData_.n_gamma_vec_.push_back(n_gamma);
		userData_.Egas_vec_.push_back(Egas_i);
		userData_.temp_vec_.push_back(temp);

		userData_.output_file_ << time << "," << n_e << "," << n_HI << "," << n_HII << "," << n_gamma << "," << Egas_i << "," << temp << "\n";
	}
}

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

	// initialize
	sim.setInitialConditions();
	sim.stopTime_ = sim.userData_.tend;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.constantDt_ = constant_dt;
	sim.plotfileInterval_ = -1;

	// evolve
	sim.evolve();

	std::vector<double> const &t = sim.userData_.t_vec_;
	std::vector<double> const &n_e = sim.userData_.n_e_vec_;
	std::vector<double> const &n_HI = sim.userData_.n_HI_vec_;
	std::vector<double> const &n_HII = sim.userData_.n_HII_vec_;
	std::vector<double> const &n_gamma = sim.userData_.n_gamma_vec_;
	std::vector<double> const &Egas = sim.userData_.Egas_vec_;
	std::vector<double> const &temp = sim.userData_.temp_vec_;

	int energy_switch = 1;

	amrex::ParmParse const pp1("network");
	pp1.query("energy_switch", energy_switch);

	std::string filename = "../extern/photoionization-julia/";
	if (energy_switch == 0) {
		filename += "no_energy.csv";
	} else if (energy_switch == 1) {
		filename += "with_energy.csv";
	} else {
		amrex::Abort("Invalid energy_switch parameter");
	}

	std::ifstream file(filename);
	if (!file.is_open()) {
		amrex::Abort("Could not open file: " + filename + "\nPlease run this test from the <quokka-root>/tests/ folder.");
	}
	std::string line;
	std::vector<std::vector<double>> data;
	std::getline(file, line);
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string value;
		std::vector<double> row;
		while (std::getline(ss, value, ',')) {
			row.push_back(std::stod(value));
		}
		data.push_back(row);
	}

	std::vector<double> t_julia;
	std::vector<double> n_e_julia;
	std::vector<double> n_HI_julia;
	std::vector<double> n_HII_julia;
	std::vector<double> n_gamma_julia;
	std::vector<double> Egas_julia;
	std::vector<double> temp_julia;
	for (size_t i = 1; i < data.size(); ++i) {
		t_julia.push_back(data[i][0]);
		n_e_julia.push_back(data[i][1]);
		n_HI_julia.push_back(data[i][2]);
		n_HII_julia.push_back(data[i][3]);
		n_gamma_julia.push_back(data[i][4]);
		Egas_julia.push_back(data[i][5]);
		temp_julia.push_back(data[i][6]);
	}

	int errors = 0;
	int status = 0;
	amrex::Real const error_tol = 1e-6;
	amrex::Real species1_error_norm = 0.0;
	amrex::Real species2_error_norm = 0.0;
	amrex::Real species3_error_norm = 0.0;
	amrex::Real photon_error_norm = 0.0;
	amrex::Real energy_error_norm = 0.0;

	for (size_t i = 0; i < t.size(); ++i) {
		species1_error_norm += std::abs(n_e[i] - n_e_julia[i]) / std::abs(n_e_julia[i]);
		species2_error_norm += std::abs(n_HI[i] - n_HI_julia[i]) / std::abs(n_HI_julia[i]);
		species3_error_norm += std::abs(n_HII[i] - n_HII_julia[i]) / std::abs(n_HII_julia[i]);
		photon_error_norm += std::abs(n_gamma[i] - n_gamma_julia[i]) / std::abs(n_gamma_julia[i]);
		if (energy_switch == 1) {
			energy_error_norm += std::abs(Egas[i] - Egas_julia[i]) / std::abs(Egas_julia[i]);
		}
	}
	species1_error_norm /= static_cast<amrex::Real>(t.size());
	species2_error_norm /= static_cast<amrex::Real>(t.size());
	species3_error_norm /= static_cast<amrex::Real>(t.size());
	photon_error_norm /= static_cast<amrex::Real>(t.size());
	energy_error_norm /= static_cast<amrex::Real>(t.size());
	amrex::Print() << "Species 1 L1 error norm = " << species1_error_norm << '\n';
	amrex::Print() << "Species 2 L1 error norm = " << species2_error_norm << '\n';
	amrex::Print() << "Species 3 L1 error norm = " << species3_error_norm << '\n';
	amrex::Print() << "Photon L1 error norm = " << photon_error_norm << '\n';
	amrex::Print() << "Energy L1 error norm = " << energy_error_norm << '\n';
	if (species1_error_norm > error_tol) {
		amrex::Print() << "Species 1 error norm exceeds tolerance!" << '\n';
		errors += 1;
	}
	if (species2_error_norm > error_tol) {
		amrex::Print() << "Species 2 error norm exceeds tolerance!" << '\n';
		errors += 1;
	}
	if (species3_error_norm > error_tol) {
		amrex::Print() << "Species 3 error norm exceeds tolerance!" << '\n';
		errors += 1;
	}
	if (photon_error_norm > error_tol) {
		amrex::Print() << "Photon error norm exceeds tolerance!" << '\n';
		errors += 1;
	}
	if (energy_error_norm > error_tol) {
		amrex::Print() << "Energy error norm exceeds tolerance!" << '\n';
		errors += 1;
	}
	if (errors > 0) {
		status = 1;
	}

#ifdef HAVE_PYTHON
	// Plot n_HI and n_HII
	matplotlibcpp::clf();
	std::map<std::string, std::string> n_HI_args;
	std::map<std::string, std::string> n_HII_args;
	std::map<std::string, std::string> n_HI_julia_args;
	std::map<std::string, std::string> n_HII_julia_args;
	n_HI_args["label"] = "n_HI";
	n_HI_args["color"] = "C0";
	n_HII_args["label"] = "n_HII";
	n_HII_args["color"] = "C1";
	n_HI_julia_args["label"] = "Julia";
	n_HI_julia_args["linestyle"] = "--";
	n_HI_julia_args["color"] = "k";
	n_HII_julia_args["linestyle"] = "--";
	n_HII_julia_args["color"] = "k";
	matplotlibcpp::plot(t, n_HI, n_HI_args);
	matplotlibcpp::plot(t, n_HII, n_HII_args);
	matplotlibcpp::plot(t_julia, n_HI_julia, n_HI_julia_args);
	matplotlibcpp::plot(t_julia, n_HII_julia, n_HII_julia_args);
	matplotlibcpp::yscale("log");
	matplotlibcpp::xlabel("time (s)");
	matplotlibcpp::ylabel("number density (cm^-3)");
	matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	const std::string species_plot_filename =
	    (energy_switch == 0) ? "./photoionization_species_no_energy.pdf" : "./photoionization_species_with_energy.pdf";
	matplotlibcpp::save(species_plot_filename);

	if (energy_switch == 1) {
		// Plot temperature
		matplotlibcpp::clf();
		std::map<std::string, std::string> temp_args;
		std::map<std::string, std::string> temp_julia_args;
		temp_args["label"] = "T";
		temp_args["color"] = "C2";
		temp_julia_args["label"] = "Julia";
		temp_julia_args["linestyle"] = "--";
		temp_julia_args["color"] = "k";
		matplotlibcpp::plot(t, temp, temp_args);
		matplotlibcpp::plot(t_julia, temp_julia, temp_julia_args);
		matplotlibcpp::xlabel("time (s)");
		matplotlibcpp::ylabel("temperature (K)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./photoionization_temperature.pdf");
	}
#endif

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
