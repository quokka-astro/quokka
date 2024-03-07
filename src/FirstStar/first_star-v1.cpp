/// \file first_star.cpp
/// \brief 
///

#include "first_star.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

struct TheProblem {
};

constexpr double mu = 1.0;
constexpr double k_B = 1.0;
constexpr double gamma_ = 5. / 3.;
constexpr double rho_core = 1.0;
constexpr double rho_bg = 1.0e-3;
constexpr double r1 = 2.0;
constexpr double r_star = 0.5;
constexpr double T0 = 1.0;
constexpr double Cv = 1.0;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = gamma_;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};

template <> void RadhydroSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
		amrex::Real const distxy = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

		double rho = NAN;
		if (r <= r_star) {
			rho = rho_core;
		} else if (r <= r1) {
			rho = rho_core * std::pow(r_star / r, 2);
		} else {
			rho = rho_bg;
		}

		const auto E_int = 1.0 / (gamma_ - 1.0) * rho * Cv * T0;

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = E_int;
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = E_int;
	});
}

// template <> void RadhydroSimulation<SedovProblem>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons)
// {
// 	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
// 	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

// 	// read output variables
// 	// Extract the data at the final time at the center of the y-z plane (center=true) 
// 	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
// 	const int nx = static_cast<int>(position.size());

// 	std::vector<double> xs(nx);
// 	std::vector<double> Tgas(nx);
// 	std::vector<double> Vgas(nx);
// 	std::vector<double> rhogas(nx);

// 	for (int i = 0; i < nx; ++i) {
// 		amrex::Real const x = position[i];
// 		xs.at(i) = x;
// 		const auto rho_t = values.at(RadSystem<TheProblem>::gasDensity_index)[i];
// 		const auto v_t = values.at(RadSystem<TheProblem>::x1GasMomentum_index)[i] / rho_t;
// 		const auto Egas_t = values.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
// 		rhogas.at(i) = rho_t;
// 		Tgas.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas_t);
// 		Vgas.at(i) = v_t;
// 	}

// #ifdef HAVE_PYTHON
// 	// plot temperature
// 	matplotlibcpp::clf();
// 	std::map<std::string, std::string> Tgas_args;
// 	Tgas_args["label"] = "gas temperature";
// 	Tgas_args["linestyle"] = "-";
// 	matplotlibcpp::plot(xs, Tgas, Tgas_args);
// 	matplotlibcpp::xlabel("x");
// 	matplotlibcpp::ylabel("temperature");
// 	// matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	matplotlibcpp::save("./first-star-T.pdf");

// 	// plot gas velocity profile
// 	matplotlibcpp::clf();
// 	std::map<std::string, std::string> vgas_args;
// 	vgas_args["label"] = "gas velocity";
// 	vgas_args["linestyle"] = "-";
// 	matplotlibcpp::plot(xs, Vgas, vgas_args);
// 	matplotlibcpp::xlabel("x");
// 	matplotlibcpp::ylabel("velocity");
// 	// matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	matplotlibcpp::save("./first-star-v.pdf");

// 	// plot density profile
// 	matplotlibcpp::clf();
// 	std::map<std::string, std::string> rhogas_args;
// 	rhogas_args["label"] = "gas density";
// 	rhogas_args["linestyle"] = "-";
// 	matplotlibcpp::plot(xs, rhogas, rhogas_args);
// 	matplotlibcpp::xlabel("x");
// 	matplotlibcpp::ylabel("density");
// 	// matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	matplotlibcpp::save("./first-star-rho.pdf");
// #endif

// }

auto problem_main() -> int
{
	// Problem parameters

	const double max_dt = 1e0;

	// Boundary conditions
	const int ncomp_cc = Physics_Indices<TheProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			// BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadhydroSimulation<TheProblem> sim(BCs_cc);

	sim.doPoissonSolve_ = 1; // enable self-gravity

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.maxDt_ = max_dt;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	// Extract the data at the final time at the center of the y-z plane (center=true) 
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> rhogas(nx);

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		const auto rho_t = values.at(HydroSystem<TheProblem>::density_index)[i];
		const auto v_t = values.at(HydroSystem<TheProblem>::x1momentum_index)[i] / rho_t;
		const auto Egas_t = values.at(HydroSystem<TheProblem>::internalEnergy_index)[i];
		rhogas.at(i) = rho_t;
		Tgas.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas_t);
		Vgas.at(i) = v_t;
	}

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	Tgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("temperature");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./first-star-T.pdf");

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity";
	vgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("velocity");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./first-star-v.pdf");

	// plot density profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> rhogas_args;
	rhogas_args["label"] = "gas density";
	rhogas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, rhogas, rhogas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("density");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./first-star-rho.pdf");
#endif

	// Cleanup and exit
	const int status = 0;
	return status;
}
