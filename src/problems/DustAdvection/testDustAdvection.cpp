/// \file testDustAdvection.cpp
/// \brief Defines a test problem for dust transport with drag force
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <format>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct DustAdvection {
};

constexpr double initial_Egas = 1.0e-9;
constexpr double rho = 1.0;
constexpr double v0 = 5.0;
constexpr double dust_v0 = 5.0;

template <> struct quokka::EOS_Traits<DustAdvection> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DustAdvection> {
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

template <> void QuokkaSimulation<DustAdvection>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Egas0 = initial_Egas;
	const auto vx0 = v0;	      // gas velocity
	const auto vx_dust = dust_v0; // dust velocity

	// Gaussian parameters
	const double rho_bg = 1.0;
	const double A = 1.0;	  // amplitude
	const double sigma = 0.1; // width
	const double xc = 0.5;	  // domain center (assuming Lx = 1.0)

	// get geometry information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = Geom(0).CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = Geom(0).ProbLoArray();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

		// Gaussian + background for gas
		amrex::Real const rho_gas_local = rho_bg + A * std::exp(-((x - xc) * (x - xc)) / (2.0 * sigma * sigma));
		state_cc(i, j, k, HydroSystem<DustAdvection>::density_index) = rho_gas_local;
		state_cc(i, j, k, HydroSystem<DustAdvection>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<DustAdvection>::internalEnergy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<DustAdvection>::x1Momentum_index) = rho_gas_local * vx0;
		state_cc(i, j, k, HydroSystem<DustAdvection>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustAdvection>::x3Momentum_index) = 0.;

		// Compute dust values before constexpr-if to ensure proper capture
		// Gaussian + background for dust
		amrex::Real const rho_dust_local = rho_bg + A * std::exp(-((x - xc) * (x - xc)) / (2.0 * sigma * sigma));
		// Reference vx_dust before constexpr-if to ensure proper capture
		amrex::Real const vx_dust_local = vx_dust;

		if constexpr (Physics_Traits<DustAdvection>::is_dust_enabled) {
			state_cc(i, j, k, HydroSystem<DustAdvection>::dustDensity_index) = rho_dust_local;
			state_cc(i, j, k, HydroSystem<DustAdvection>::x1DustMomentum_index) = rho_dust_local * vx_dust_local;
			state_cc(i, j, k, HydroSystem<DustAdvection>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustAdvection>::x3DustMomentum_index) = 0.;
		}
	});
}

auto problem_main() -> int
{
	// problem parameters
	const double Lx = 1.0;
	const double CFL_number = 0.8;

	// Gaussian parameters
	const double rho_bg = 1.0;
	const double A = 1.0;
	const double sigma = 0.1;
	const double xc = 0.5;

	// problem initialization
	QuokkaSimulation<DustAdvection> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	std::vector<double> vx_sim(nx);
	std::vector<double> vx_exact(nx);
	std::vector<double> xs(nx);

	std::vector<double> vx_dust_sim(nx);
	std::vector<double> vx_dust_exact(nx);
	std::vector<double> rho_dust_sim(nx);
	std::vector<double> rho_dust_exact(nx);
	std::vector<double> rho_gas_exact(nx);

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;

		amrex::Real const t = sim.tNew_[0];

		// exact gas density (shifted by v0 * t)
		amrex::Real x_gas_initial = std::fmod(x - v0 * t, Lx);
		if (x_gas_initial < 0.0) {
			x_gas_initial += Lx;
		}
		rho_gas_exact.at(i) = rho_bg + A * std::exp(-((x_gas_initial - xc) * (x_gas_initial - xc)) / (2.0 * sigma * sigma));

		// exact dust density (shifted by dust_v0 * t)
		amrex::Real x_dust_initial = std::fmod(x - dust_v0 * t, Lx);
		if (x_dust_initial < 0.0) {
			x_dust_initial += Lx;
		}
		rho_dust_exact.at(i) = rho_bg + A * std::exp(-((x_dust_initial - xc) * (x_dust_initial - xc)) / (2.0 * sigma * sigma));

		vx_exact.at(i) = v0;
		vx_dust_exact.at(i) = dust_v0;

		// numerical values
		const double density = values.at(HydroSystem<DustAdvection>::density_index)[i];
		const double momentum_x = values.at(HydroSystem<DustAdvection>::x1Momentum_index)[i];
		const double dust_density = values.at(HydroSystem<DustAdvection>::dustDensity_index)[i];
		const double dust_momentum_x = values.at(HydroSystem<DustAdvection>::x1DustMomentum_index)[i];
		vx_sim.at(i) = momentum_x / density;
		vx_dust_sim.at(i) = dust_momentum_x / dust_density;
		rho_dust_sim.at(i) = dust_density;
	}

	// error norm gas velocity
	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(vx_sim[i] - vx_exact[i]);
		sol_norm += std::abs(vx_exact[i]);
	}
	const double rel_err_norm = err_norm / sol_norm;

	// error norm dust density
	double err_norm_dust_rho = 0.;
	double sol_norm_dust_rho = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm_dust_rho += std::abs(rho_dust_sim[i] - rho_dust_exact[i]);
		sol_norm_dust_rho += std::abs(rho_dust_exact[i]);
	}
	const double rel_err_norm_dust_rho = err_norm_dust_rho / sol_norm_dust_rho;

	int status = 1;
	const double rel_err_tol = 0.03;
	if ((rel_err_norm < rel_err_tol) && (rel_err_norm_dust_rho < rel_err_tol)) {
		status = 0;
	}

	amrex::Print() << "Relative L1 norm for gas x velocity = " << rel_err_norm << '\n';
	amrex::Print() << "Relative L1 norm for dust density   = " << rel_err_norm_dust_rho << '\n';

#ifdef HAVE_PYTHON
	// plot density (gas + dust)
	matplotlibcpp::clf();

	std::map<std::string, std::string> rho_gas_args;
	std::map<std::string, std::string> rho_gas_exact_args;
	std::map<std::string, std::string> rho_dust_args;
	std::map<std::string, std::string> rho_dust_exact_args;

	rho_gas_args["label"] = "gas density (numerical)";
	rho_gas_args["color"] = "r";
	rho_gas_args["linestyle"] = "-";

	rho_gas_exact_args["label"] = "gas density (exact)";
	rho_gas_exact_args["color"] = "r";
	rho_gas_exact_args["linestyle"] = "--";

	rho_dust_args["label"] = "dust density (numerical)";
	rho_dust_args["color"] = "b";
	rho_dust_args["linestyle"] = "-.";

	rho_dust_exact_args["label"] = "dust density (exact)";
	rho_dust_exact_args["color"] = "b";
	rho_dust_exact_args["linestyle"] = ":";

	// gas density
	std::vector<double> rho_gas_sim(nx);
	for (int i = 0; i < nx; ++i) {
		rho_gas_sim.at(i) = values.at(HydroSystem<DustAdvection>::density_index)[i];
	}

	matplotlibcpp::plot(xs, rho_gas_sim, rho_gas_args);
	matplotlibcpp::plot(xs, rho_gas_exact, rho_gas_exact_args);
	matplotlibcpp::plot(xs, rho_dust_sim, rho_dust_args);
	matplotlibcpp::plot(xs, rho_dust_exact, rho_dust_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("density");
	matplotlibcpp::title(std::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./dust_drag_density.pdf");

	// plot velocity (gas + dust)
	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 6.0);

	std::map<std::string, std::string> vx_gas_args;
	std::map<std::string, std::string> vx_gas_exact_args;
	std::map<std::string, std::string> vx_dust_args;
	std::map<std::string, std::string> vx_dust_exact_args;

	vx_gas_args["label"] = "gas velocity (numerical)";
	vx_gas_args["color"] = "r";
	vx_gas_args["linestyle"] = "-";

	vx_gas_exact_args["label"] = "gas velocity (exact)";
	vx_gas_exact_args["color"] = "r";
	vx_gas_exact_args["linestyle"] = "--";

	vx_dust_args["label"] = "dust velocity (numerical)";
	vx_dust_args["color"] = "b";
	vx_dust_args["linestyle"] = "-.";

	vx_dust_exact_args["label"] = "dust velocity (exact)";
	vx_dust_exact_args["color"] = "b";
	vx_dust_exact_args["linestyle"] = ":";

	matplotlibcpp::plot(xs, vx_sim, vx_gas_args);
	matplotlibcpp::plot(xs, vx_exact, vx_gas_exact_args);
	matplotlibcpp::plot(xs, vx_dust_sim, vx_dust_args);
	matplotlibcpp::plot(xs, vx_dust_exact, vx_dust_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("velocity");
	matplotlibcpp::title(std::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./dust_drag_velocity.pdf");
#endif // HAVE_PYTHON

	amrex::Print() << "Finished." << '\n';
	return status;
}
