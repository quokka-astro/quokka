/// \file testDustDamping.cpp
/// \brief Defines a test problem for dust drag
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <format>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

// // analytic solution parameters for test A
// constexpr double V_COM = 1.16666666666667;
// constexpr double LAMBDA1 = -0.63397459621556;
// constexpr double LAMBDA2 = -2.36602540378444;
// constexpr double C_GAS_1 = -0.22767090063074;
// constexpr double C_GAS_2 = 0.06100423396407;
// constexpr double C_DUST1_1 = 0.84967936855889;
// constexpr double C_DUST1_2 = -0.01634603522555;
// constexpr double C_DUST2_1 = -0.62200846792815;
// constexpr double C_DUST2_2 = -0.04465819873852;

// constexpr double rho_dust1 = 1.0;
// constexpr double rho_dust2 = 1.0;
// constexpr double TS1 = 2.0;
// constexpr double TS2 = 1.0;
// constexpr double OMEGA = 1.0;
// constexpr double P_INITIAL = 1.0;

// analytic solution parameters for test B
constexpr double V_COM = 1.16666666666667;
constexpr double LAMBDA1 = -141.742430504416;
constexpr double LAMBDA2 = -1058.25756949558;
constexpr double C_GAS_1 = -0.35610569612832;
constexpr double C_GAS_2 = 0.18943902946166;
constexpr double C_DUST1_1 = 0.85310244713865;
constexpr double C_DUST1_2 = -0.01976911380532;
constexpr double C_DUST2_1 = -0.49699675101033;
constexpr double C_DUST2_2 = -0.16966991565634;

constexpr double rho_dust1 = 1.0;
constexpr double rho_dust2 = 1.0;
constexpr double TS1 = 0.01;
constexpr double TS2 = 0.002;
constexpr double OMEGA = 1.0;
constexpr double P_INITIAL = 1.0;

// // analytic solution parameters for test C
// constexpr double V_COM = 0.63963963963963;
// constexpr double LAMBDA1 = -0.52370200744224;
// constexpr double LAMBDA2 = -105.976297992557;
// constexpr double C_GAS_1 = -0.06458203330249;
// constexpr double C_GAS_2 = 0.42494239366285;
// constexpr double C_DUST1_1 = 1.36237475791577;
// constexpr double C_DUST1_2 = -0.00201439755542;
// constexpr double C_DUST2_1 = -0.13559165545855;
// constexpr double C_DUST2_2 = -0.00404798418109;

// constexpr double rho_dust1 = 10.0;
// constexpr double rho_dust2 = 100.0;
// constexpr double TS1 = 2.0;
// constexpr double TS2 = 1.0;
// constexpr double OMEGA = 1.0;
// constexpr double P_INITIAL = 1.0;

// analytic solution function declarations
auto v_gas_analytic(double t) -> double;
auto v_dust1_analytic(double t) -> double;
auto v_dust2_analytic(double t) -> double;
auto E_gas_analytic(double t) -> double;

struct DustDamping {
};

template <> struct SimulationData<DustDamping> {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_vec_;
	std::vector<double> v_dust1_vec_;
	std::vector<double> v_dust2_vec_;
	std::vector<double> E_gas_vec_;
};

template <> struct quokka::EOS_Traits<DustDamping> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
	// static constexpr double cs_isothermal = 1.0; // only used when gamma = 1
};

constexpr double rho = 1.0;
constexpr double v0 = 1.0;
constexpr double Egas0 = P_INITIAL / (quokka::EOS_Traits<DustDamping>::gamma - 1.0) + 0.5 * rho * v0 * v0;
constexpr double Egas0_internal = P_INITIAL / (quokka::EOS_Traits<DustDamping>::gamma - 1.0);
constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

template <> struct Physics_Traits<DustDamping> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 2; // number of dust groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustDamping>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
										   amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 2> alpha{};
	alpha[0] = 1.0 / TS1;
	alpha[1] = 1.0 / TS2;
	return alpha;
}

template <> void QuokkaSimulation<DustDamping>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto vx0 = v0;		// gas velocity
	const auto vx_dust1 = 2 * v0;	// dust1 velocity
	const auto vx_dust2 = 0.5 * v0; // dust2 velocity

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// for gas
		state_cc(i, j, k, HydroSystem<DustDamping>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<DustDamping>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<DustDamping>::internalEnergy_index) = Egas0_internal;
		state_cc(i, j, k, HydroSystem<DustDamping>::x1Momentum_index) = rho * vx0;
		state_cc(i, j, k, HydroSystem<DustDamping>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustDamping>::x3Momentum_index) = 0.;

		// first-capture for CUDA
		const auto vx_dust1_local = vx_dust1;
		const auto vx_dust2_local = vx_dust2;

		if constexpr (Physics_Traits<DustDamping>::is_dust_enabled) {
			// for dust1
			state_cc(i, j, k, HydroSystem<DustDamping>::dustDensity_index) = rho_dust1;
			state_cc(i, j, k, HydroSystem<DustDamping>::x1DustMomentum_index) = rho_dust1 * vx_dust1_local;
			state_cc(i, j, k, HydroSystem<DustDamping>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustDamping>::x3DustMomentum_index) = 0.;
			// for dust2
			state_cc(i, j, k, HydroSystem<DustDamping>::dustDensity_index + numDustVars) = rho_dust2;
			state_cc(i, j, k, HydroSystem<DustDamping>::x1DustMomentum_index + numDustVars) = rho_dust2 * vx_dust2_local;
			state_cc(i, j, k, HydroSystem<DustDamping>::x2DustMomentum_index + numDustVars) = 0.;
			state_cc(i, j, k, HydroSystem<DustDamping>::x3DustMomentum_index + numDustVars) = 0.;
		}
	});
}

template <> void QuokkaSimulation<DustDamping>::computeAfterTimestep()
{
	auto [_, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]); // store current time

		// extract physical quantities
		const double density = values.at(HydroSystem<DustDamping>::density_index)[0];
		const double momentum_x = values.at(HydroSystem<DustDamping>::x1Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<DustDamping>::energy_index)[0];

		// store gas velocity
		const double v_gas = momentum_x / density;
		userData_.v_gas_vec_.push_back(v_gas);

		// store gas total energy
		userData_.E_gas_vec_.push_back(Egas_total);

		if constexpr (Physics_Traits<DustDamping>::is_dust_enabled) {
			// store dust1 velocity
			const double dust1_density = values.at(HydroSystem<DustDamping>::dustDensity_index)[0];
			const double dust1_momentum_x = values.at(HydroSystem<DustDamping>::x1DustMomentum_index)[0];
			const double v_dust1 = dust1_momentum_x / dust1_density;
			userData_.v_dust1_vec_.push_back(v_dust1);

			// store dust2 velocity
			const double dust2_density = values.at(HydroSystem<DustDamping>::dustDensity_index + numDustVars)[0];
			const double dust2_momentum_x = values.at(HydroSystem<DustDamping>::x1DustMomentum_index + numDustVars)[0];
			const double v_dust2 = dust2_momentum_x / dust2_density;
			userData_.v_dust2_vec_.push_back(v_dust2);
		}
	}
}

// implementation of analytic solution functions
auto analytic_velocity(double t, double c1, double c2) -> double { return V_COM + c1 * std::exp(LAMBDA1 * t) + c2 * std::exp(LAMBDA2 * t); }

auto v_gas_analytic(double t) -> double { return analytic_velocity(t, C_GAS_1, C_GAS_2); }

auto v_dust1_analytic(double t) -> double { return analytic_velocity(t, C_DUST1_1, C_DUST1_2); }

auto v_dust2_analytic(double t) -> double { return analytic_velocity(t, C_DUST2_1, C_DUST2_2); }

// calculate analytic gas energy
auto E_gas_analytic(double t) -> double
{
	const int n_points = 1000;
	const double dt = t / n_points;
	double integral = 0.0;

	for (int i = 0; i < n_points; ++i) {
		double const t1 = i * dt;
		double const t2 = (i + 1) * dt;

		double const vg1 = v_gas_analytic(t1);
		double const vd1_1 = v_dust1_analytic(t1);
		double const vd2_1 = v_dust2_analytic(t1);

		double const vg2 = v_gas_analytic(t2);
		double const vd1_2 = v_dust1_analytic(t2);
		double const vd2_2 = v_dust2_analytic(t2);

		double const term1 = (rho_dust1 * (vd1_1 - vg1) / TS1 * vg1 + rho_dust2 * (vd2_1 - vg1) / TS2 * vg1 +
				      OMEGA * (rho_dust1 * std::pow(vd1_1 - vg1, 2) / TS1 + rho_dust2 * std::pow(vd2_1 - vg1, 2) / TS2));

		double const term2 = (rho_dust1 * (vd1_2 - vg2) / TS1 * vg2 + rho_dust2 * (vd2_2 - vg2) / TS2 * vg2 +
				      OMEGA * (rho_dust1 * std::pow(vd1_2 - vg2, 2) / TS1 + rho_dust2 * std::pow(vd2_2 - vg2, 2) / TS2));

		integral += 0.5 * (term1 + term2) * dt;
	}

	const double E_gas_initial = P_INITIAL / (quokka::EOS_Traits<DustDamping>::gamma - 1.0) + 0.5 * 1.0 * std::pow(v_gas_analytic(0), 2);
	return E_gas_initial + integral;
}

auto problem_main() -> int
{
	// problem parameters
	const double CFL_number = 1000000.0; // large CFL number to avoid CFL violation

	// problem initialization
	QuokkaSimulation<DustDamping> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;

	// initialize
	sim.setInitialConditions();

	// store initial values for t=0 plotting
	auto [_, val_ini] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.t_vec_.push_back(0.0);

		const double initial_density = val_ini.at(HydroSystem<DustDamping>::density_index)[0];
		const double initial_momentum_x = val_ini.at(HydroSystem<DustDamping>::x1Momentum_index)[0];
		const double initial_Egas_total = val_ini.at(HydroSystem<DustDamping>::energy_index)[0];
		const double initial_v_gas = initial_momentum_x / initial_density;
		sim.userData_.v_gas_vec_.push_back(initial_v_gas);
		sim.userData_.E_gas_vec_.push_back(initial_Egas_total);

		if constexpr (Physics_Traits<DustDamping>::is_dust_enabled) {
			const double initial_dust1_density = val_ini.at(HydroSystem<DustDamping>::dustDensity_index)[0];
			const double initial_dust1_momentum_x = val_ini.at(HydroSystem<DustDamping>::x1DustMomentum_index)[0];
			const double initial_v_dust1 = initial_dust1_momentum_x / initial_dust1_density;
			sim.userData_.v_dust1_vec_.push_back(initial_v_dust1);

			const double initial_dust2_density = val_ini.at(HydroSystem<DustDamping>::dustDensity_index + numDustVars)[0];
			const double initial_dust2_momentum_x = val_ini.at(HydroSystem<DustDamping>::x1DustMomentum_index + numDustVars)[0];
			const double initial_v_dust2 = initial_dust2_momentum_x / initial_dust2_density;
			sim.userData_.v_dust2_vec_.push_back(initial_v_dust2);
		}
	}

	// evolve
	sim.evolve();

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> &t = sim.userData_.t_vec_;
		std::vector<double> const &v_gas = sim.userData_.v_gas_vec_;
		std::vector<double> const &v_dust1 = sim.userData_.v_dust1_vec_;
		std::vector<double> const &v_dust2 = sim.userData_.v_dust2_vec_;
		std::vector<double> const &E_gas = sim.userData_.E_gas_vec_;

		// calculate dense analytic solution for plotting
		const size_t n_dense_points = 1000;
		std::vector<double> t_dense(n_dense_points);
		std::vector<double> v_gas_exact_dense(n_dense_points);
		std::vector<double> v_dust1_exact_dense(n_dense_points);
		std::vector<double> v_dust2_exact_dense(n_dense_points);
		std::vector<double> E_gas_exact_dense(n_dense_points);

		double const t_max = t.empty() ? 0.0 : t.back();
		for (size_t i = 0; i < n_dense_points; ++i) {
			t_dense[i] = t_max * static_cast<double>(i) / (n_dense_points - 1);
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

		const double rel_err_tol = 0.03;
		if ((rel_err_gas_vx > rel_err_tol) || (rel_err_dust1_vx > rel_err_tol) || (rel_err_dust2_vx > rel_err_tol) || (rel_err_gas_E > rel_err_tol)) {
			status = 1;
		}

#ifdef HAVE_PYTHON
		// plot gas velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_gas, {{"label", "numerical"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, v_gas_exact_dense, {{"label", "analytic"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_g$)");
		matplotlibcpp::title("Gas Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_gas_velocity.pdf");

		// plot dust1 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_dust1, {{"label", "numerical"}, {"color", "b"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, v_dust1_exact_dense, {{"label", "analytic"}, {"color", "b"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,1}$)");
		matplotlibcpp::title("Dust1 Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_dust1_velocity.pdf");

		// plot dust2 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_dust2, {{"label", "numerical"}, {"color", "g"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, v_dust2_exact_dense, {{"label", "analytic"}, {"color", "g"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,2}$)");
		matplotlibcpp::title("Dust2 Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_dust2_velocity.pdf");

		// plot gas energy
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, E_gas, {{"label", "numerical"}, {"color", "m"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_dense, E_gas_exact_dense, {{"label", "analytic"}, {"color", "m"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($E_g$)");
		matplotlibcpp::title("Gas Energy Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_gas_energy.pdf");
#endif
		amrex::Print() << "Finished.\n";
	}
	return status;
}
