/// \file testDustDampingWithExternalForce.cpp
/// \brief Defines a test problem for dust drag with an external force
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <format>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

// analytic solution parameters for Krapp (2024) Section 3.2
constexpr double LAMBDA1 = -1.125;
constexpr double LAMBDA2 = -0.8;

// coefficients for the exact analytic solution (from Krapp 2024 eq 42, 43)
// u(t) = A*exp(lambda1*t) + B*exp(lambda2*t) + D*t + E
constexpr double C_GAS_A = 0.145014245014245;
constexpr double C_GAS_B = 0.0596153846153846;
constexpr double C_GAS_D = 0.833333333333333;
constexpr double C_GAS_E = 1.79537037037037;

constexpr double C_DUST1_A = -0.116011396011396;
constexpr double C_DUST1_B = 0.0298076923076923;
constexpr double C_DUST1_D = 0.0833333333333333;
constexpr double C_DUST1_E = 0.0962037037037037;

constexpr double C_DUST2_A = -0.029002849002849;
constexpr double C_DUST2_B = -0.0894230769230769;
constexpr double C_DUST2_D = 0.0833333333333333;
constexpr double C_DUST2_E = 0.0684259259259259;

constexpr double rho_gas = 1.0;
constexpr double rho_dust1 = 0.1;
constexpr double rho_dust2 = 0.1;

// alpha_d1 = 1.0 -> TS1 = 1.0
// alpha_d2 = 0.75 -> TS2 = 4.0 / 3.0
constexpr double TS1 = 1.0;
constexpr double TS2 = 4.0 / 3.0;
constexpr double OMEGA = 1.0;
constexpr double P_INITIAL = 1.0;

// the constant external force G0 = 1.0
constexpr double EXTERNAL_FORCE = 1.0;

// analytic solution function declarations
auto v_gas_analytic(double t) -> double;
auto v_dust1_analytic(double t) -> double;
auto v_dust2_analytic(double t) -> double;
auto E_gas_analytic(double t) -> double;

struct DustDampingWithExternalForce {
};

template <> struct SimulationData<DustDampingWithExternalForce> {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_vec_;
	std::vector<double> v_dust1_vec_;
	std::vector<double> v_dust2_vec_;
	std::vector<double> E_gas_vec_;
};

template <> struct quokka::EOS_Traits<DustDampingWithExternalForce> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
};

constexpr double v0 = 2.0;
constexpr double Egas0 = P_INITIAL / (quokka::EOS_Traits<DustDampingWithExternalForce>::gamma - 1.0) + 0.5 * rho_gas * v0 * v0;
constexpr double Egas0_internal = P_INITIAL / (quokka::EOS_Traits<DustDampingWithExternalForce>::gamma - 1.0);
constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

template <> struct Physics_Traits<DustDampingWithExternalForce> : DefaultPhysicsTraits {
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
AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingWithExternalForce>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
												    amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
												    amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/,
												    double /*cs*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 2> alpha{};
	alpha[0] = 1.0 / TS1;
	alpha[1] = 1.0 / TS2;
	return alpha;
}

template <> void QuokkaSimulation<DustDampingWithExternalForce>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto vx0 = v0;	    // gas velocity = 2.0
	const auto vx_dust1 = 0.1;  // dust1 velocity
	const auto vx_dust2 = -0.5; // dust2 velocity

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// for gas
		state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::density_index) = rho_gas;
		state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::internalEnergy_index) = Egas0_internal;
		state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x1Momentum_index) = rho_gas * vx0;
		state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x3Momentum_index) = 0.;

		// first-capture for CUDA
		const auto vx_dust1_local = vx_dust1;
		const auto vx_dust2_local = vx_dust2;

		if constexpr (Physics_Traits<DustDampingWithExternalForce>::is_dust_enabled) {
			// for dust1
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::dustDensity_index) = rho_dust1;
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x1DustMomentum_index) = rho_dust1 * vx_dust1_local;
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x3DustMomentum_index) = 0.;
			// for dust2
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::dustDensity_index + numDustVars) = rho_dust2;
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x1DustMomentum_index + numDustVars) = rho_dust2 * vx_dust2_local;
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x2DustMomentum_index + numDustVars) = 0.;
			state_cc(i, j, k, HydroSystem<DustDampingWithExternalForce>::x3DustMomentum_index + numDustVars) = 0.;
		}
	});
}

template <> void QuokkaSimulation<DustDampingWithExternalForce>::computeAfterTimestep()
{
	auto [_, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]); // store current time

		// extract physical quantities
		const double density = values.at(HydroSystem<DustDampingWithExternalForce>::density_index)[0];
		const double momentum_x = values.at(HydroSystem<DustDampingWithExternalForce>::x1Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<DustDampingWithExternalForce>::energy_index)[0];

		// store gas velocity
		const double v_gas = momentum_x / density;
		userData_.v_gas_vec_.push_back(v_gas);

		// store gas total energy
		userData_.E_gas_vec_.push_back(Egas_total);

		if constexpr (Physics_Traits<DustDampingWithExternalForce>::is_dust_enabled) {
			// store dust1 velocity
			const double dust1_density = values.at(HydroSystem<DustDampingWithExternalForce>::dustDensity_index)[0];
			const double dust1_momentum_x = values.at(HydroSystem<DustDampingWithExternalForce>::x1DustMomentum_index)[0];
			const double v_dust1 = dust1_momentum_x / dust1_density;
			userData_.v_dust1_vec_.push_back(v_dust1);

			// store dust2 velocity
			const double dust2_density = values.at(HydroSystem<DustDampingWithExternalForce>::dustDensity_index + numDustVars)[0];
			const double dust2_momentum_x = values.at(HydroSystem<DustDampingWithExternalForce>::x1DustMomentum_index + numDustVars)[0];
			const double v_dust2 = dust2_momentum_x / dust2_density;
			userData_.v_dust2_vec_.push_back(v_dust2);
		}
	}
}

auto v_gas_analytic(double t) -> double { return (C_GAS_A * std::exp(LAMBDA1 * t) + C_GAS_B * std::exp(LAMBDA2 * t) + C_GAS_D * t + C_GAS_E) / rho_gas; }

auto v_dust1_analytic(double t) -> double
{
	return (C_DUST1_A * std::exp(LAMBDA1 * t) + C_DUST1_B * std::exp(LAMBDA2 * t) + C_DUST1_D * t + C_DUST1_E) / rho_dust1;
}

auto v_dust2_analytic(double t) -> double
{
	return (C_DUST2_A * std::exp(LAMBDA1 * t) + C_DUST2_B * std::exp(LAMBDA2 * t) + C_DUST2_D * t + C_DUST2_E) / rho_dust2;
}

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
				      OMEGA * (rho_dust1 * std::pow(vd1_1 - vg1, 2) / TS1 + rho_dust2 * std::pow(vd2_1 - vg1, 2) / TS2)) +
				     EXTERNAL_FORCE * rho_gas * vg1;

		double const term2 = (rho_dust1 * (vd1_2 - vg2) / TS1 * vg2 + rho_dust2 * (vd2_2 - vg2) / TS2 * vg2 +
				      OMEGA * (rho_dust1 * std::pow(vd1_2 - vg2, 2) / TS1 + rho_dust2 * std::pow(vd2_2 - vg2, 2) / TS2)) +
				     EXTERNAL_FORCE * rho_gas * vg2;

		integral += 0.5 * (term1 + term2) * dt;
	}

	const double E_gas_initial =
	    P_INITIAL / (quokka::EOS_Traits<DustDampingWithExternalForce>::gamma - 1.0) + 0.5 * rho_gas * std::pow(v_gas_analytic(0), 2);
	return E_gas_initial + integral;
}

// add Strang Split source term for constant external force
template <>
void QuokkaSimulation<DustDampingWithExternalForce>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) // NOLINT
{
	const amrex::Real dt = dt_lev;

	// define constant external force（dp/dt = G_0）
	const double G_0 = 1.0;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = state(i, j, k, HydroSystem<DustDampingWithExternalForce>::density_index);
			const amrex::Real x1mom = state(i, j, k, HydroSystem<DustDampingWithExternalForce>::x1Momentum_index);
			const amrex::Real x2mom = state(i, j, k, HydroSystem<DustDampingWithExternalForce>::x2Momentum_index);
			const amrex::Real x3mom = state(i, j, k, HydroSystem<DustDampingWithExternalForce>::x3Momentum_index);
			const amrex::Real Egas = state(i, j, k, HydroSystem<DustDampingWithExternalForce>::energy_index);

			static_assert(!Physics_Traits<DustDampingWithExternalForce>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			const amrex::Real Eint = quokka::EOS<DustDampingWithExternalForce>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas, 0.0);

			double const x1mom_new = x1mom + dt * G_0;

			AMREX_ASSERT(!std::isnan(x1mom_new));

			state(i, j, k, HydroSystem<DustDampingWithExternalForce>::x1Momentum_index) = x1mom_new;

			static_assert(!Physics_Traits<DustDampingWithExternalForce>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			const amrex::Real Egas_new = quokka::EOS<DustDampingWithExternalForce>::ComputeEgasFromEint(rho, x1mom_new, x2mom, x3mom, Eint, 0.0);
			AMREX_ASSERT(!std::isnan(Egas_new));

			state(i, j, k, HydroSystem<DustDampingWithExternalForce>::energy_index) = Egas_new;
		});
	}
}

auto problem_main() -> int
{
	// problem parameters
	const double CFL_number = 1000000.0; // large CFL number to avoid CFL violation

	// problem initialization
	QuokkaSimulation<DustDampingWithExternalForce> sim;

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

		const double initial_density = val_ini.at(HydroSystem<DustDampingWithExternalForce>::density_index)[0];
		const double initial_momentum_x = val_ini.at(HydroSystem<DustDampingWithExternalForce>::x1Momentum_index)[0];
		const double initial_Egas_total = val_ini.at(HydroSystem<DustDampingWithExternalForce>::energy_index)[0];
		const double initial_v_gas = initial_momentum_x / initial_density;
		sim.userData_.v_gas_vec_.push_back(initial_v_gas);
		sim.userData_.E_gas_vec_.push_back(initial_Egas_total);

		if constexpr (Physics_Traits<DustDampingWithExternalForce>::is_dust_enabled) {
			const double initial_dust1_density = val_ini.at(HydroSystem<DustDampingWithExternalForce>::dustDensity_index)[0];
			const double initial_dust1_momentum_x = val_ini.at(HydroSystem<DustDampingWithExternalForce>::x1DustMomentum_index)[0];
			const double initial_v_dust1 = initial_dust1_momentum_x / initial_dust1_density;
			sim.userData_.v_dust1_vec_.push_back(initial_v_dust1);

			const double initial_dust2_density = val_ini.at(HydroSystem<DustDampingWithExternalForce>::dustDensity_index + numDustVars)[0];
			const double initial_dust2_momentum_x = val_ini.at(HydroSystem<DustDampingWithExternalForce>::x1DustMomentum_index + numDustVars)[0];
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

		const double rel_err_tol = 0.0001;
		if ((rel_err_gas_vx > rel_err_tol) || (rel_err_dust1_vx > rel_err_tol) || (rel_err_dust2_vx > rel_err_tol) || (rel_err_gas_E > rel_err_tol)) {
			status = 1;
		}

#ifdef HAVE_PYTHON
		std::vector<double> rel_v_d1(t.size());
		std::vector<double> rel_v_d2(t.size());
		for (size_t i = 0; i < t.size(); ++i) {
			rel_v_d1[i] = std::abs(v_gas[i] - v_dust1[i]);
			rel_v_d2[i] = std::abs(v_gas[i] - v_dust2[i]);
		}

		std::vector<double> rel_v_d1_exact_dense(t_dense.size());
		std::vector<double> rel_v_d2_exact_dense(t_dense.size());
		for (size_t i = 0; i < t_dense.size(); ++i) {
			rel_v_d1_exact_dense[i] = std::abs(v_gas_exact_dense[i] - v_dust1_exact_dense[i]);
			rel_v_d2_exact_dense[i] = std::abs(v_gas_exact_dense[i] - v_dust2_exact_dense[i]);
		}

		// plot relative velocity between gas and dust
		matplotlibcpp::clf();
		matplotlibcpp::plot(t_dense, rel_v_d1_exact_dense, {{"label", "analytic"}, {"color", "k"}, {"linestyle", "--"}});
		matplotlibcpp::plot(t_dense, rel_v_d2_exact_dense, {{"color", "k"}, {"linestyle", "--"}});
		matplotlibcpp::plot(t, rel_v_d1,
				    {{"label", R"($|v_g - v_{d,1}|$)"},
				     {"color", "orange"},
				     {"linestyle", "None"},
				     {"marker", "o"},
				     {"markerfacecolor", "none"},
				     {"markersize", "4"}});
		matplotlibcpp::plot(t, rel_v_d2,
				    {{"label", R"($|v_g - v_{d,2}|$)"},
				     {"color", "blue"},
				     {"linestyle", "None"},
				     {"marker", "o"},
				     {"markerfacecolor", "none"},
				     {"markersize", "4"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($|v_g - v_d|$)");
		matplotlibcpp::title("Relative Velocity");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_relative_velocity_external.pdf");
#endif
		amrex::Print() << "Finished.\n";
	}
	return status;
}