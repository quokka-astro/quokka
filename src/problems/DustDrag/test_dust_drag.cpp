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

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<StreamingProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::internalEnergy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x1Momentum_index) = rho * vx0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x3Momentum_index) = 0.;

		// state_cc(i, j, k, HydroSystem<StreamingProblem>::dustDensity_index) = rho_dust;
		// state_cc(i, j, k, HydroSystem<StreamingProblem>::x1DustMomentum_index) = rho_dust * vx_dust;
		// state_cc(i, j, k, HydroSystem<StreamingProblem>::x2DustMomentum_index) = 0.;
		// state_cc(i, j, k, HydroSystem<StreamingProblem>::x3DustMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
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
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		vx_exact.at(i) = v0; // expected x velocity
		// compute x velocity from momentum and density
		const double momentum_x = values.at(HydroSystem<StreamingProblem>::x1Momentum_index)[i];
		const double density = values.at(HydroSystem<StreamingProblem>::density_index)[i];
		vx_sim.at(i) = momentum_x / density;
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(vx_sim[i] - vx_exact[i]);
		sol_norm += std::abs(vx_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.01;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm for x velocity = " << rel_err_norm << '\n';

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
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./velocity_test.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
