/// \file test_dust_damping.cpp
/// \brief Defines a test problem for dust drag
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <fmt/format.h>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

// analytic solution parameters
constexpr double V_COM = 0.63963963963963;
constexpr double LAMBDA1 = -0.52370200744224;
constexpr double LAMBDA2 = -105.976297992557;
constexpr double C_GAS_1 = -0.06458203330249;
constexpr double C_GAS_2 = 0.42494239366285;
constexpr double C_DUST1_1 = 1.36237475791577;
constexpr double C_DUST1_2 = -0.00201439755542;
constexpr double C_DUST2_1 = -0.13559165545855;
constexpr double C_DUST2_2 = -0.00404798418109;

constexpr double RHO_D1 = 10.0;
constexpr double RHO_D2 = 100.0;
constexpr double TS1 = 2.0;
constexpr double TS2 = 1.0;
constexpr double OMEGA = 0.0;
constexpr double P_INITIAL = 1.0;

// analytic solution function declarations
double v_gas_analytic(double t);
double v_dust1_analytic(double t);
double v_dust2_analytic(double t);
double E_gas_analytic(double t);

struct StreamingProblem {
};

template <> struct SimulationData<StreamingProblem> {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_vec_;
	std::vector<double> v_dust1_vec_;
	std::vector<double> v_dust2_vec_;
	std::vector<double> E_gas_vec_;
};

template <> struct quokka::EOS_Traits<StreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
	// static constexpr double cs_isothermal = 1.0; // only used when gamma = 1
};

constexpr double rho = 1.0;
constexpr double rho_dust1 = 10.0;
constexpr double rho_dust2 = 100.0;
constexpr double v0 = 1.0;
constexpr double initial_Egas = P_INITIAL / (quokka::EOS_Traits<StreamingProblem>::gamma - 1.0) + 0.5 * rho * v0 * v0;
constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

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

template <> void QuokkaSimulation<StreamingProblem>::computeBeforeTimestep()
{
	// extract initial physical quantities at t=0
	if (amrex::ParallelDescriptor::IOProcessor() && userData_.t_vec_.empty()) {
		auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

		userData_.t_vec_.push_back(0.0); // initial time t=0

		// extract physical quantities
		const double density = values.at(HydroSystem<StreamingProblem>::density_index)[0];
		const double momentum_x = values.at(HydroSystem<StreamingProblem>::x1Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<StreamingProblem>::energy_index)[0];

		// store gas velocity
		const double v_gas = momentum_x / density;
		userData_.v_gas_vec_.push_back(v_gas);

		// store gas total energy
		userData_.E_gas_vec_.push_back(Egas_total);

		if constexpr (Physics_Traits<StreamingProblem>::is_dust_enabled) {
			// store dust1 velocity
			const double dust1_density = values.at(HydroSystem<StreamingProblem>::dustDensity_index)[0];
			const double dust1_momentum_x = values.at(HydroSystem<StreamingProblem>::x1DustMomentum_index)[0];
			const double v_dust1 = dust1_momentum_x / dust1_density;
			userData_.v_dust1_vec_.push_back(v_dust1);

			// store dust2 velocity
			const double dust2_density = values.at(HydroSystem<StreamingProblem>::dustDensity_index + numDustVars)[0];
			const double dust2_momentum_x = values.at(HydroSystem<StreamingProblem>::x1DustMomentum_index + numDustVars)[0];
			const double v_dust2 = dust2_momentum_x / dust2_density;
			userData_.v_dust2_vec_.push_back(v_dust2);
		}
	}
}

template <> void QuokkaSimulation<StreamingProblem>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]); // store current time

		// extract physical quantities
		const double density = values.at(HydroSystem<StreamingProblem>::density_index)[0];
		const double momentum_x = values.at(HydroSystem<StreamingProblem>::x1Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<StreamingProblem>::energy_index)[0];

		// store gas velocity
		const double v_gas = momentum_x / density;
		userData_.v_gas_vec_.push_back(v_gas);

		// store gas total energy
		userData_.E_gas_vec_.push_back(Egas_total);

		if constexpr (Physics_Traits<StreamingProblem>::is_dust_enabled) {
			// store dust1 velocity
			const double dust1_density = values.at(HydroSystem<StreamingProblem>::dustDensity_index)[0];
			const double dust1_momentum_x = values.at(HydroSystem<StreamingProblem>::x1DustMomentum_index)[0];
			const double v_dust1 = dust1_momentum_x / dust1_density;
			userData_.v_dust1_vec_.push_back(v_dust1);

			// store dust2 velocity
			const double dust2_density = values.at(HydroSystem<StreamingProblem>::dustDensity_index + numDustVars)[0];
			const double dust2_momentum_x = values.at(HydroSystem<StreamingProblem>::x1DustMomentum_index + numDustVars)[0];
			const double v_dust2 = dust2_momentum_x / dust2_density;
			userData_.v_dust2_vec_.push_back(v_dust2);
		}
	}
}

// implementation of analytic solution functions
double analytic_velocity(double t, double c1, double c2) { return V_COM + c1 * std::exp(LAMBDA1 * t) + c2 * std::exp(LAMBDA2 * t); }

double v_gas_analytic(double t) { return analytic_velocity(t, C_GAS_1, C_GAS_2); }

double v_dust1_analytic(double t) { return analytic_velocity(t, C_DUST1_1, C_DUST1_2); }

double v_dust2_analytic(double t) { return analytic_velocity(t, C_DUST2_1, C_DUST2_2); }

// calculate analytic gas energy
double E_gas_analytic(double t)
{
	const int n_points = 1000;
	const double dt = t / n_points;
	double integral = 0.0;

	for (int i = 0; i < n_points; ++i) {
		double t1 = i * dt;
		double t2 = (i + 1) * dt;

		double vg1 = v_gas_analytic(t1);
		double vd1_1 = v_dust1_analytic(t1);
		double vd2_1 = v_dust2_analytic(t1);

		double vg2 = v_gas_analytic(t2);
		double vd1_2 = v_dust1_analytic(t2);
		double vd2_2 = v_dust2_analytic(t2);

		double term1 = (RHO_D1 * (vd1_1 - vg1) / TS1 * vg1 + RHO_D2 * (vd2_1 - vg1) / TS2 * vg1 +
				OMEGA * (RHO_D1 * std::pow(vd1_1 - vg1, 2) / TS1 + RHO_D2 * std::pow(vd2_1 - vg1, 2) / TS2));

		double term2 = (RHO_D1 * (vd1_2 - vg2) / TS1 * vg2 + RHO_D2 * (vd2_2 - vg2) / TS2 * vg2 +
				OMEGA * (RHO_D1 * std::pow(vd1_2 - vg2, 2) / TS1 + RHO_D2 * std::pow(vd2_2 - vg2, 2) / TS2));

		integral += 0.5 * (term1 + term2) * dt;
	}

	const double E_gas_initial = P_INITIAL / (quokka::EOS_Traits<StreamingProblem>::gamma - 1.0) + 0.5 * 1.0 * std::pow(v_gas_analytic(0), 2);
	return E_gas_initial + integral;
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
	sim.constantDt_ = 0.05;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	if constexpr (Physics_Traits<StreamingProblem>::is_dust_enabled) {
		std::vector<double> &t = sim.userData_.t_vec_;
		std::vector<double> &v_gas = sim.userData_.v_gas_vec_;
		std::vector<double> &v_dust1 = sim.userData_.v_dust1_vec_;
		std::vector<double> &v_dust2 = sim.userData_.v_dust2_vec_;
		std::vector<double> &E_gas = sim.userData_.E_gas_vec_;

		// calculate dense analytic solution for plotting
		const size_t n_dense_points = 1000;
		std::vector<double> t_dense(n_dense_points);
		std::vector<double> v_gas_exact_dense(n_dense_points);
		std::vector<double> v_dust1_exact_dense(n_dense_points);
		std::vector<double> v_dust2_exact_dense(n_dense_points);
		std::vector<double> E_gas_exact_dense(n_dense_points);

		double t_max = t.empty() ? 0.0 : t.back();
		for (size_t i = 0; i < n_dense_points; ++i) {
			t_dense[i] = t_max * i / (n_dense_points - 1);
			v_gas_exact_dense[i] = v_gas_analytic(t_dense[i]);
			v_dust1_exact_dense[i] = v_dust1_analytic(t_dense[i]);
			v_dust2_exact_dense[i] = v_dust2_analytic(t_dense[i]);
			E_gas_exact_dense[i] = E_gas_analytic(t_dense[i]);
		}

		// calculate relative L1 norm errors
		std::vector<double> v_gas_exact(t.size());
		std::vector<double> v_dust1_exact(t.size());
		std::vector<double> v_dust2_exact(t.size());
		std::vector<double> E_gas_exact(t.size());

		for (size_t i = 0; i < t.size(); ++i) {
			v_gas_exact[i] = v_gas_analytic(t[i]);
			v_dust1_exact[i] = v_dust1_analytic(t[i]);
			v_dust2_exact[i] = v_dust2_analytic(t[i]);
			E_gas_exact[i] = E_gas_analytic(t[i]);
		}

		auto rel_err = [](const std::vector<double> &sim, const std::vector<double> &exact) {
			double err = 0.0;
			double sol = 0.0;
			for (size_t i = 0; i < sim.size(); ++i) {
				err += std::abs(sim[i] - exact[i]);
				sol += std::abs(exact[i]);
			}
			return err / sol;
		};

		double const rel_err_gas_vx = rel_err(v_gas, v_gas_exact);
		double const rel_err_dust1_vx = rel_err(v_dust1, v_dust1_exact);
		double const rel_err_dust2_vx = rel_err(v_dust2, v_dust2_exact);
		double const rel_err_gas_E = rel_err(E_gas, E_gas_exact);

		amrex::Print() << "Relative L1 norm for gas vx    = " << rel_err_gas_vx << "\n";
		amrex::Print() << "Relative L1 norm for dust1 vx  = " << rel_err_dust1_vx << "\n";
		amrex::Print() << "Relative L1 norm for dust2 vx  = " << rel_err_dust2_vx << "\n";
		amrex::Print() << "Relative L1 norm for gas E     = " << rel_err_gas_E << "\n";

		int status = 0;
		const double rel_err_tol = 0.01;
		if ((rel_err_gas_vx > rel_err_tol) || (rel_err_dust1_vx > rel_err_tol) || (rel_err_dust2_vx > rel_err_tol) || (rel_err_gas_E > rel_err_tol)) {
			status = 1;
		}

#ifdef HAVE_PYTHON
		// plot gas velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_gas, {{"label", "gas vx (num)"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, v_gas_exact_dense, {{"label", "gas vx (exact)"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel("gas velocity");
		matplotlibcpp::title("Gas Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_gas_velocity.pdf");

		// plot dust1 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_dust1, {{"label", "dust1 vx (num)"}, {"color", "b"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, v_dust1_exact_dense, {{"label", "dust1 vx (exact)"}, {"color", "b"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel("dust1 velocity");
		matplotlibcpp::title("Dust1 Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_dust1_velocity.pdf");

		// plot dust2 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_dust2, {{"label", "dust2 vx (num)"}, {"color", "g"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, v_dust2_exact_dense, {{"label", "dust2 vx (exact)"}, {"color", "g"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel("dust2 velocity");
		matplotlibcpp::title("Dust2 Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_dust2_velocity.pdf");

		// plot gas energy
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, E_gas, {{"label", "gas E (num)"}, {"color", "m"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, E_gas_exact_dense, {{"label", "gas E (exact)"}, {"color", "m"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel("gas energy");
		matplotlibcpp::title("Gas Energy Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_gas_energy.pdf");
#endif
		amrex::Print() << "Finished.\n";
		return status;

	} else { // dust disabled case
		// read output variables
		auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
		const int nx = static_cast<int>(position.size());

		std::vector<double> xs(nx);

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