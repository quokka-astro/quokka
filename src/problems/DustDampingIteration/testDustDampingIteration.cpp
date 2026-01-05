/// \file testDustDampingIteration.cpp
/// \brief Defines a test problem for dust drag
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <cmath>
#include <fmt/format.h>
#include <fstream>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

constexpr double rho_dust1 = 1.0;
constexpr double rho_dust2 = 1.0;
constexpr double P_INITIAL = 1.0;

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
static constexpr amrex::GpuArray<amrex::Real, 2> dust_grain_radius = {0.02, 0.01};
static constexpr amrex::GpuArray<amrex::Real, 2> dust_grain_density = {1.0, 1.0};
static constexpr bool enable_supersonic_correction = true;

template <> struct Physics_Traits<DustDamping> {
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

template <>
AMREX_GPU_HOST_DEVICE auto DustDrag<DustDamping>::ComputeReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
										amrex::GpuArray<amrex::Real, nDustGroups_ + 1> vel_mag, double cs)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return ComputeReciprocalStoppingTimeHelper(rho_g, rho_d, vel_mag, cs, dust_grain_radius, dust_grain_density, enable_supersonic_correction);
}

template <> void QuokkaSimulation<DustDamping>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto vx0 = v0;		 // gas velocity
	const auto vx_dust1 = 2 * v0;	 // dust1 velocity
	const auto vx_dust2 = 10.0 * v0; // dust2 velocity

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

template <> void QuokkaSimulation<DustDamping>::computeBeforeTimestep()
{
	// extract initial physical quantities at t=0
	if (amrex::ParallelDescriptor::IOProcessor() && userData_.t_vec_.empty()) {
		auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

		userData_.t_vec_.push_back(0.0); // initial time t=0

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

template <> void QuokkaSimulation<DustDamping>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

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

auto run_simulation(double dt, int enableIterDustStoptime) -> SimulationData<DustDamping>
{
	QuokkaSimulation<DustDamping> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 1000000.0; // set large CFL to avoid CFL violation
	sim.constantDt_ = dt;
	sim.enableIterDustStoptime_ = enableIterDustStoptime;

	sim.setInitialConditions();

	sim.evolve();

	return sim.userData_;
}

auto problem_main() -> int
{
	amrex::Print() << "Running dust damping test with supersonic correction...\n";

	const double dt_ref = 0.00001;
	int const enableIterDustStoptime_ref = 0;
	const double dt_test = 0.005;
	int const enableIterDustStoptime_test = 1;

	// step 1: run reference solution (dt = 0.00001)
	auto ref_data = run_simulation(dt_ref, enableIterDustStoptime_ref);

	// step 2: run numerical solution (dt = 0.005)
	auto test_data = run_simulation(dt_test, enableIterDustStoptime_test);

	// step 3: calculate errors
	// calculate sampling step size
	const auto step = static_cast<size_t>(std::round(dt_test / dt_ref));
	auto compute_relative_error = [](const std::vector<double> &sim, const std::vector<double> &ref, size_t step) {
		double err_sum = 0.0;
		double ref_sum = 0.0;
		int count = 0;

		for (size_t i = 0; i < sim.size(); ++i) {
			size_t const ref_idx = i * step;
			if (ref_idx >= ref.size()) {
				break;
			}

			err_sum += std::abs(sim[i] - ref[ref_idx]);
			ref_sum += std::abs(ref[ref_idx]);
			count++;
		}

		if (count == 0 || ref_sum == 0.0) {
			return 1.0; // error value
		}
		return err_sum / ref_sum;
	};

	double const rel_err_gas_vx = compute_relative_error(test_data.v_gas_vec_, ref_data.v_gas_vec_, step);
	double const rel_err_dust1_vx = compute_relative_error(test_data.v_dust1_vec_, ref_data.v_dust1_vec_, step);
	double const rel_err_dust2_vx = compute_relative_error(test_data.v_dust2_vec_, ref_data.v_dust2_vec_, step);
	double const rel_err_gas_E = compute_relative_error(test_data.E_gas_vec_, ref_data.E_gas_vec_, step);

	amrex::Print() << "Comparison with reference solution (supersonic correction enabled):\n";
	amrex::Print() << "Relative L1 norm for gas vx    = " << rel_err_gas_vx << "\n";
	amrex::Print() << "Relative L1 norm for dust1 vx  = " << rel_err_dust1_vx << "\n";
	amrex::Print() << "Relative L1 norm for dust2 vx  = " << rel_err_dust2_vx << "\n";
	amrex::Print() << "Relative L1 norm for gas E     = " << rel_err_gas_E << "\n";

	// determine whether the test has passed
	int status = 0;
	const double rel_err_tol = 0.01;

	if ((rel_err_gas_vx > rel_err_tol) || (rel_err_dust1_vx > rel_err_tol) || (rel_err_dust2_vx > rel_err_tol) || (rel_err_gas_E > rel_err_tol)) {
		status = 1;
		amrex::Print() << "Test FAILED: one or more errors exceed tolerance of " << rel_err_tol << "\n";
	} else {
		amrex::Print() << "Test PASSED: all errors within tolerance of " << rel_err_tol << "\n";
	}

#ifdef HAVE_PYTHON
	if (!ref_data.t_vec_.empty() && !test_data.t_vec_.empty()) {
		// gas velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(test_data.t_vec_, test_data.v_gas_vec_,
				    {{"label", "numerical (iter, dt=0.005)"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_gas_vec_, {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_g$)");
		matplotlibcpp::title("Gas Velocity (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_gas_velocity.pdf");

		// dust1 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(test_data.t_vec_, test_data.v_dust1_vec_,
				    {{"label", "numerical (iter, dt=0.005)"}, {"color", "b"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_dust1_vec_,
				    {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "b"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,1}$)");
		matplotlibcpp::title("Dust1 Velocity (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_dust1_velocity.pdf");

		// dust2 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(test_data.t_vec_, test_data.v_dust2_vec_,
				    {{"label", "numerical (iter, dt=0.005)"}, {"color", "g"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_dust2_vec_,
				    {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "g"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,2}$)");
		matplotlibcpp::title("Dust2 Velocity (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_dust2_velocity.pdf");

		// gas energy
		matplotlibcpp::clf();
		matplotlibcpp::plot(test_data.t_vec_, test_data.E_gas_vec_,
				    {{"label", "numerical (iter, dt=0.005)"}, {"color", "m"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.E_gas_vec_, {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "m"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($E_g$)");
		matplotlibcpp::title("Gas Energy (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_gas_energy.pdf");
	}
#endif

	amrex::Print() << "\nTest complete.\n";
	return status;
}