/// \file test_dust_drag.cpp
/// \brief Defines a test problem for dust drag force
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <fmt/format.h>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct StreamingProblem {
};

constexpr double initial_Egas = 1.0e-5;
constexpr double rho = 1.0;
constexpr double v0 = 1.0;

template <> struct quokka::EOS_Traits<StreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StreamingProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Egas0 = initial_Egas;
	const auto vx0 = v0; // initial x velocity

	const auto rho_dust = rho;
	const auto vx_dust = v0;

	// get geometry information for physical coordinates
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = Geom(0).CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = Geom(0).ProbLoArray();

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<StreamingProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::internalEnergy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x1Momentum_index) = rho * vx0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x3Momentum_index) = 0.;

		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		amrex::Real rho_dust_local = (x < 0.5) ? rho_dust : 2.0 * rho_dust;

		state_cc(i, j, k, HydroSystem<StreamingProblem>::dustDensity_index) = rho_dust_local;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x1DustMomentum_index) = rho_dust_local * 0.0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x2DustMomentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x3DustMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double tmax = 1.0;
	const int max_timesteps = 100;

	// Boundary conditions
	constexpr int nvars = HydroSystem<StreamingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<StreamingProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compute error norm for x velocity
	std::vector<double> vx_sim(nx);
	std::vector<double> vx_exact(nx);
	std::vector<double> vx_dust_sim(nx);
	std::vector<double> vx_dust_exact(nx);
	std::vector<double> rho_dust_sim(nx);
	std::vector<double> rho_dust_exact(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;

		// for exact dust density (shifted assuming no interaction and periodic boundaries)
		amrex::Real const t = sim.tNew_[0];
		amrex::Real x_initial = std::fmod(x - v0 * t, Lx);
		if (x_initial < 0.0) x_initial += Lx;
		rho_dust_exact.at(i) = (x_initial < 0.5) ? rho : 2.0 * rho;
		vx_exact.at(i) = v0; // expected x velocity
		vx_dust_exact.at(i) = v0; // expected x velocity of dust

		// compute x velocity from momentum and density
		const double density = values.at(HydroSystem<StreamingProblem>::density_index)[i];
		const double momentum_x = values.at(HydroSystem<StreamingProblem>::x1Momentum_index)[i];
		const double dust_density = values.at(HydroSystem<StreamingProblem>::dustDensity_index)[i];
		const double dust_momentum_x = values.at(HydroSystem<StreamingProblem>::x1DustMomentum_index)[i];
		vx_sim.at(i) = momentum_x / density;
		vx_dust_sim.at(i) = dust_momentum_x / dust_density;
		rho_dust_sim.at(i) = dust_density;
	}
	// compute error norm for gas velocity
	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(vx_sim[i] - vx_exact[i]);
		sol_norm += std::abs(vx_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;

	// compute error norm for dust density
	double err_norm_dust_rho = 0.;
	double sol_norm_dust_rho = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm_dust_rho += std::abs(rho_dust_sim[i] - rho_dust_exact[i]);
		sol_norm_dust_rho += std::abs(rho_dust_exact[i]);
	}

	const double rel_err_norm_dust_rho = err_norm_dust_rho / sol_norm_dust_rho;
	const double rel_err_tol = 0.01;
	int status = 1;
	if ((rel_err_norm < rel_err_tol) && (rel_err_norm_dust_rho < rel_err_tol)) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm for gas x velocity = " << rel_err_norm << '\n';
	amrex::Print() << "Relative L1 norm for dust density = " << rel_err_norm_dust_rho << '\n';

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 1.1);

	std::map<std::string, std::string> vx_args;
	std::map<std::string, std::string> vx_exact_args;
	vx_args["label"] = "numerical solution";
	vx_exact_args["label"] = "exact solution";
	vx_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, vx_sim, vx_args);
	matplotlibcpp::plot(xs, vx_exact, vx_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("gas velocity");
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./dust_drag_gas_velocity.pdf");

	// plot dust velocity
	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 1.1);
	std::map<std::string, std::string> vx_dust_args;
	std::map<std::string, std::string> vx_dust_exact_args;
	vx_dust_args["label"] = "numerical solution";
	vx_dust_args["linestyle"] = "-";
	vx_dust_exact_args["label"] = "exact solution";
	vx_dust_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, vx_dust_sim, vx_dust_args);
	matplotlibcpp::plot(xs, vx_dust_exact, vx_dust_exact_args);
	matplotlibcpp::legend();
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("dust velocity");
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./dust_drag_dust_velocity.pdf");

	// plot dust density
	matplotlibcpp::clf();
	std::map<std::string, std::string> rho_dust_args;
	std::map<std::string, std::string> rho_dust_exact_args;
	rho_dust_args["label"] = "numerical solution";
	rho_dust_args["linestyle"] = "-";
	rho_dust_exact_args["label"] = "exact solution";
	rho_dust_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, rho_dust_sim, rho_dust_args);
	matplotlibcpp::plot(xs, rho_dust_exact, rho_dust_exact_args);
	matplotlibcpp::legend();
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("dust density");
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./dust_drag_dust_density.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
