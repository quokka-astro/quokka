/// \file test_rad_line_cooling_MG.cpp
/// \brief Defines a test problem for line cooling and cosmic-ray heating in a uniform medium.
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "util/fextract.hpp"
#include "AMReX_Print.H"
#include "physics_info.hpp"
#include "test_rad_line_cooling_MG.hpp"

static constexpr bool export_csv = true;

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int n_groups_ = 4;
constexpr int line_index = 3; // last group
constexpr double CR_heating_rate = 1.0;
constexpr double line_cooling_rate = CR_heating_rate;
constexpr amrex::GpuArray<double, 5> rad_boundaries_ = {1.00000000e-03, 1.77827941e-02, 3.16227766e-01, 5.62341325e+00, 1.00000000e+02};

const double cooling_rate = 1.0e-1;

constexpr double c = 1.0;
constexpr double chat = c;
constexpr double v0 = 0.0;
constexpr double kappa0 = 0.0;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // mean molecular weight; so that C_V = 1.0
constexpr double k_B = 1.0;

constexpr double nu_unit = 1.0;
constexpr double T_equilibrium = 0.768032502191;
constexpr double Erad_bar = a_rad * T0 * T0 * T0 * T0;
constexpr double erad_floor = a_rad * 1e-20;

const double max_time = 1.0;

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = nu_unit;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
};

template <> struct ISM_Traits<PulseProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = true;
	static constexpr bool enable_linear_cooling_heating = true;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const /*num_density*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[0] = cooling_rate * temperature;
	return cooling;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::DefineNetCoolingRateTempDerivative(amrex::Real const temperature, amrex::Real const /*num_density*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[0] = cooling_rate;
	return cooling;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<PulseProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
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

template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, T0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < n_groups_; ++g) {
			state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = erad_floor;
			state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a *linear* radiation diffusion problem, i.e.
	// parameters are chosen such that the radiation and gas temperatures
	// should be near equilibrium, and the opacity is chosen to go as
	// T^3, such that the radiation diffusion equation becomes linear in T.

	// This makes this problem a stringent test of the asymptotic-
	// preserving property of the computational method, since the
	// optical depth per cell at the peak of the temperature profile is
	// of order 10^5.

	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 0.8;

	const double the_dt = 1.0e-1;

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.initDt_ = the_dt;
	sim.maxDt_ = the_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Tgas_exact{};
	std::vector<double> Erad_line{};
	std::vector<double> Erad_line_exact{};
	double Erad_other_groups_error = 0.0;

	const auto t_end = sim.tNew_[0];

	// compute exact solution and error norm
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		double Erad_t = 0.0;
		for (int g = 0; g < n_groups_; ++g) {
			Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
			if (g == line_index) {
				Erad_line.push_back(Erad_t);
				Erad_line_exact.push_back(erad_floor); // TODO
			} else {
				Erad_other_groups_error += std::abs(Erad_t - erad_floor);
			}
		}
		const auto rho_t = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		const auto Egas = values.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho_t, Egas);
		// const double T_exact_solution = T0 - cooling_rate * max_time;
		const double T_exact_solution = std::exp(-cooling_rate * t_end) * T0;
		Tgas_exact.push_back(T_exact_solution);
	}
	// compare temperature with Tgas_exact
	double err_norm_T = 0.;
	double sol_norm_T = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm_T += std::abs(Tgas[i] - Tgas_exact[i]);
		sol_norm_T += std::abs(Tgas_exact[i]);
	}
	const double rel_error_T = err_norm_T / sol_norm_T;
	// compute error norm for Erad_line 
	double err_norm_Erad = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm_Erad += std::abs(Erad_line[i] - Erad_line_exact[i]);
	}
	const double rel_error_Erad = (err_norm_Erad + Erad_other_groups_error) / Erad_bar;
	const double rel_error = std::max(rel_error_T, rel_error_Erad);

	const double error_tol = 1.0e-3;
	amrex::Print() << "Relative L1 error norm for T_gas = " << rel_error << std::endl;
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "T_gas (numerical)";
	Tgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	Tgas_args["label"] = "T_gas (exact)";
	Tgas_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Tgas_exact, Tgas_args);
	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::ylim(0.0, 2.0);
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./rad_line_cooling_MG_temperature.pdf");

	// plot Erad_line
	matplotlibcpp::clf();
	std::map<std::string, std::string> Erad_args;
	std::map<std::string, std::string> Erad_exact_args;
	Erad_args["label"] = "E_rad (numerical)";
	Erad_args["linestyle"] = "-";
	Erad_exact_args["label"] = "E_rad (exact)";
	Erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Erad_line, Erad_args);
	matplotlibcpp::plot(xs, Erad_line_exact, Erad_exact_args);
	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./rad_line_cooling_MG_radiation_energy_density.pdf");
#endif

	if (export_csv) {
		std::ofstream file;
		file.open("rad_line_cooling_temp.csv");
		file << "xs,Tgas,Tgas_exact\n";
		for (size_t i = 0; i < xs.size(); ++i) {
			file << std::scientific << std::setprecision(12) << xs[i] << "," << Tgas[i] << "," << Tgas_exact[i] << "\n";
		}
		file.close();

		file.open("rad_line_cooling_rad_energy_density.csv");
		file << "xs,Erad_line,Erad_line_exact\n";
		for (size_t i = 0; i < xs.size(); ++i) {
			file << std::scientific << std::setprecision(12) << xs[i] << "," << Erad_line[i] << "," << Erad_line_exact[i] << "\n";
		}
		file.close();
	}

	// exit
	return status;
}
