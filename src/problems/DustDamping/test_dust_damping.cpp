/// \file test_dust_damping.cpp
/// \brief Defines a test problem for dust drag
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <fmt/format.h>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct StreamingProblem {
};

constexpr double initial_Egas = 1.0 / (1.4 - 1.0) + 0.5 * 1.0 * 1.0 * 1.0;
constexpr double rho = 1.0;
constexpr double rho_dust1 = 10.0;
constexpr double rho_dust2 = 100.0;
constexpr double v0 = 1.0;
constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

template <> struct quokka::EOS_Traits<StreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
	// static constexpr double cs_isothermal = 1.0; // only used when gamma = 1
};

template <> struct Physics_Traits<StreamingProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 2; // number of dust groups
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
	const auto vx0 = v0;		// gas velocity
	const auto vx_dust1 = 2 * v0;	// dust1 velocity
	const auto vx_dust2 = 0.5 * v0; // dust2 velocity

	// get geometry information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = Geom(0).CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = Geom(0).ProbLoArray();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

		// for gas
		state_cc(i, j, k, HydroSystem<StreamingProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::internalEnergy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x1Momentum_index) = rho * vx0;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<StreamingProblem>::x3Momentum_index) = 0.;

		if constexpr (Physics_Traits<StreamingProblem>::is_dust_enabled) {
			// for dust1
			state_cc(i, j, k, HydroSystem<StreamingProblem>::dustDensity_index) = rho_dust1;
			state_cc(i, j, k, HydroSystem<StreamingProblem>::x1DustMomentum_index) = rho_dust1 * vx_dust1;
			state_cc(i, j, k, HydroSystem<StreamingProblem>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<StreamingProblem>::x3DustMomentum_index) = 0.;
			// for dust2
			state_cc(i, j, k, HydroSystem<StreamingProblem>::dustDensity_index + numDustVars) = rho_dust2;
			state_cc(i, j, k, HydroSystem<StreamingProblem>::x1DustMomentum_index + numDustVars) = rho_dust2 * vx_dust2;
			state_cc(i, j, k, HydroSystem<StreamingProblem>::x2DustMomentum_index + numDustVars) = 0.;
			state_cc(i, j, k, HydroSystem<StreamingProblem>::x3DustMomentum_index + numDustVars) = 0.;
		}
	});
}

auto problem_main() -> int
{
	// problem parameters
	const double Lx = 1.0;
	const double CFL_number = 0.4;

	// boundary conditions
	constexpr int nvars = HydroSystem<StreamingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// problem initialization
	QuokkaSimulation<StreamingProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);

	if constexpr (Physics_Traits<StreamingProblem>::is_dust_enabled) {
		// numerical values
		std::vector<double> vx_sim(nx);
		std::vector<double> vx_dust1_sim(nx);
		std::vector<double> vx_dust2_sim(nx);

		std::vector<double> rho_gas_sim(nx);
		std::vector<double> rho_dust1_sim(nx);
		std::vector<double> rho_dust2_sim(nx);

		// exact values
		std::vector<double> vx_exact(nx);
		std::vector<double> vx_dust1_exact(nx);
		std::vector<double> vx_dust2_exact(nx);

		std::vector<double> rho_gas_exact(nx, rho);
		std::vector<double> rho_dust1_exact(nx, rho);
		std::vector<double> rho_dust2_exact(nx, rho);

		for (int i = 0; i < nx; ++i) {
			xs[i] = position[i];

			// velocities
			vx_exact[i] = v0;
			vx_dust1_exact[i] = 2.0 * v0;
			vx_dust2_exact[i] = 0.5 * v0;

			// numerical values
			const double density = values.at(HydroSystem<StreamingProblem>::density_index)[i];
			const double momentum_x = values.at(HydroSystem<StreamingProblem>::x1Momentum_index)[i];
			vx_sim[i] = momentum_x / density;
			rho_gas_sim[i] = density;

			const double dust1_density = values.at(HydroSystem<StreamingProblem>::dustDensity_index)[i];
			const double dust1_momentum_x = values.at(HydroSystem<StreamingProblem>::x1DustMomentum_index)[i];
			vx_dust1_sim[i] = dust1_momentum_x / dust1_density;
			rho_dust1_sim[i] = dust1_density;

			const double dust2_density = values.at(HydroSystem<StreamingProblem>::dustDensity_index + numDustVars)[i];
			const double dust2_momentum_x = values.at(HydroSystem<StreamingProblem>::x1DustMomentum_index + numDustVars)[i];
			vx_dust2_sim[i] = dust2_momentum_x / dust2_density;
			rho_dust2_sim[i] = dust2_density;
		}

		// error norms (check gas + dust1 + dust2)
		auto rel_err = [&](const std::vector<double> &sim, const std::vector<double> &exact) {
			double err = 0.0;
			double sol = 0.0;
			for (int i = 0; i < nx; ++i) {
				err += std::abs(sim[i] - exact[i]);
				sol += std::abs(exact[i]);
			}
			return err / sol;
		};

		double const rel_err_gas_vx = rel_err(vx_sim, vx_exact);
		double const rel_err_dust1_vx = rel_err(vx_dust1_sim, vx_dust1_exact);
		double const rel_err_dust2_vx = rel_err(vx_dust2_sim, vx_dust2_exact);

		amrex::Print() << "Relative L1 norm for gas vx    = " << rel_err_gas_vx << "\n";
		amrex::Print() << "Relative L1 norm for dust1 vx  = " << rel_err_dust1_vx << "\n";
		amrex::Print() << "Relative L1 norm for dust2 vx  = " << rel_err_dust2_vx << "\n";

		int status = 0;
		const double rel_err_tol = 0.01;
		if ((rel_err_gas_vx > rel_err_tol) || (rel_err_dust1_vx > rel_err_tol) || (rel_err_dust2_vx > rel_err_tol)) {
			status = 1;
		}

#ifdef HAVE_PYTHON
		// plot density
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 2.0);
		matplotlibcpp::plot(xs, rho_gas_sim, {{"label", "gas (num)"}, {"color", "r"}, {"linestyle", "-"}});
		matplotlibcpp::plot(xs, rho_gas_exact, {{"label", "gas (exact)"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::plot(xs, rho_dust1_sim, {{"label", "dust1 (num)"}, {"color", "b"}, {"linestyle", "-."}});
		matplotlibcpp::plot(xs, rho_dust1_exact, {{"label", "dust1 (exact)"}, {"color", "b"}, {"linestyle", ":"}});
		matplotlibcpp::plot(xs, rho_dust2_sim, {{"label", "dust2 (num)"}, {"color", "g"}, {"linestyle", "-."}});
		matplotlibcpp::plot(xs, rho_dust2_exact, {{"label", "dust2 (exact)"}, {"color", "g"}, {"linestyle", ":"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_density.pdf");

		// plot velocity
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 2.5 * v0);
		matplotlibcpp::plot(xs, vx_sim, {{"label", "gas vx (num)"}, {"color", "r"}, {"linestyle", "-"}});
		matplotlibcpp::plot(xs, vx_exact, {{"label", "gas vx (exact)"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::plot(xs, vx_dust1_sim, {{"label", "dust1 vx (num)"}, {"color", "b"}, {"linestyle", "-."}});
		matplotlibcpp::plot(xs, vx_dust1_exact, {{"label", "dust1 vx (exact)"}, {"color", "b"}, {"linestyle", ":"}});
		matplotlibcpp::plot(xs, vx_dust2_sim, {{"label", "dust2 vx (num)"}, {"color", "g"}, {"linestyle", "-."}});
		matplotlibcpp::plot(xs, vx_dust2_exact, {{"label", "dust2 vx (exact)"}, {"color", "g"}, {"linestyle", ":"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_velocity.pdf");
#endif
		amrex::Print() << "Finished.\n";
		return status;

	} else {
		// dust disabled case
		std::vector<double> vx_sim(nx);
		std::vector<double> vx_exact(nx);
		std::vector<double> rho_gas_sim(nx, rho);
		std::vector<double> rho_gas_exact(nx, rho);

		for (int i = 0; i < nx; ++i) {
			xs[i] = position[i];
			const double density = values.at(HydroSystem<StreamingProblem>::density_index)[i];
			const double momentum_x = values.at(HydroSystem<StreamingProblem>::x1Momentum_index)[i];
			vx_sim[i] = momentum_x / density;
			rho_gas_sim[i] = density;
			vx_exact[i] = v0;
		}

		double rel_err = 0.0;
		double sol_norm = 0.0;
		for (int i = 0; i < nx; ++i) {
			rel_err += std::abs(vx_sim[i] - vx_exact[i]);
			sol_norm += std::abs(vx_exact[i]);
		}
		const double rel_err_norm = rel_err / sol_norm;

		amrex::Print() << "Relative L1 norm for gas vx = " << rel_err_norm << "\n";

		const int status = (rel_err_norm < 0.01) ? 0 : 1;

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 2.0);
		matplotlibcpp::plot(xs, rho_gas_sim, {{"label", "gas (num)"}, {"color", "r"}, {"linestyle", "-"}});
		matplotlibcpp::plot(xs, rho_gas_exact, {{"label", "gas (exact)"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_density.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 2.0 * v0);
		matplotlibcpp::plot(xs, vx_sim, {{"label", "gas vx (num)"}, {"color", "r"}, {"linestyle", "-"}});
		matplotlibcpp::plot(xs, vx_exact, {{"label", "gas vx (exact)"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_velocity.pdf");
#endif

		amrex::Print() << "Finished.\n";
		return status;
	}
}