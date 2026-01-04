/// \file testDustDampingNonIter.cpp
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
constexpr double TS1 = 0.01;
constexpr double TS2 = 0.002;
constexpr double OMEGA = 1.0;
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
AMREX_GPU_HOST_DEVICE auto DustDrag<DustDamping>::ComputeReciprocalStoppingTime(
    amrex::Real rho_g, amrex::GpuArray<amrex::Real, Physics_Traits<DustDamping>::nDustGroups> rho_d,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, Physics_Traits<DustDamping>::nDustGroups + 1> vel, double cs)
    -> amrex::GpuArray<amrex::Real, Physics_Traits<DustDamping>::nDustGroups>
{
	return ComputeReciprocalStoppingTimeHelper(rho_g, rho_d, vel, cs, dust_grain_radius, dust_grain_density, enable_supersonic_correction);
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

// save reference solution to file
void save_reference_solution(const std::vector<double> &t, const std::vector<double> &v_gas, const std::vector<double> &v_dust1,
			     const std::vector<double> &v_dust2, const std::vector<double> &E_gas, const std::string &filename)
{
	std::ofstream outfile(filename);
	if (!outfile) {
		amrex::Print() << "Error: Could not open file " << filename << " for writing.\n";
		return;
	}

	// write file header
	outfile << "# Reference solution for dust damping test (supersonic correction enabled)\n";
	outfile << "# dt = 0.00001, enable_supersonic_correction = true\n";
	outfile << "# Columns: time, v_gas, v_dust1, v_dust2, E_gas\n";
	outfile << "# Total points: " << t.size() << "\n";

	// write data
	outfile << std::scientific << std::setprecision(15);
	for (auto i = 0; i < t.size(); ++i) {
		outfile << t[i] << " " << v_gas[i] << " " << v_dust1[i] << " " << v_dust2[i] << " " << E_gas[i] << "\n";
	}

	outfile.close();
	amrex::Print() << "Reference solution saved to " << filename << "\n";
}

// read reference solution from file
auto load_reference_solution(const std::string &filename, std::vector<double> &t_ref, std::vector<double> &v_gas_ref, std::vector<double> &v_dust1_ref,
			     std::vector<double> &v_dust2_ref, std::vector<double> &E_gas_ref) -> bool
{
	std::ifstream infile(filename);
	if (!infile) {
		amrex::Print() << "Error: Could not open reference file " << filename << "\n";
		return false;
	}

	t_ref.clear();
	v_gas_ref.clear();
	v_dust1_ref.clear();
	v_dust2_ref.clear();
	E_gas_ref.clear();

	std::string line;
	// skip file header
	while (std::getline(infile, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}

		std::istringstream iss(line);
		double time = NAN;
		double vg = NAN;
		double vd1 = NAN;
		double vd2 = NAN;
		double eg = NAN;
		if (iss >> time >> vg >> vd1 >> vd2 >> eg) {
			t_ref.push_back(time);
			v_gas_ref.push_back(vg);
			v_dust1_ref.push_back(vd1);
			v_dust2_ref.push_back(vd2);
			E_gas_ref.push_back(eg);
		}
	}

	infile.close();
	amrex::Print() << "Loaded " << t_ref.size() << " points from reference file " << filename << "\n";
	return !t_ref.empty();
}

auto problem_main() -> int
{
	// problem parameters
	const double CFL_number = 1000000.0; // set large CFL to avoid CFL violation

	// problem initialization
	QuokkaSimulation<DustDamping> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;
	sim.constantDt_ = 0.005; // usually 0.005 for test B, 0.05 for test C
	// sim.constantDt_ = 0.00001;

	// determine whether to generate reference solution or run test
	bool generating_reference = false;
	if (sim.constantDt_ == 0.00001) {
		generating_reference = true;
		amrex::Print() << "Running in reference generation mode (dt = 0.00001)\n";
	} else {
		amrex::Print() << "Running in test mode (dt = " << sim.constantDt_ << ")\n";
	}

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	std::vector<double> const &t = sim.userData_.t_vec_;
	std::vector<double> const &v_gas = sim.userData_.v_gas_vec_;
	std::vector<double> const &v_dust1 = sim.userData_.v_dust1_vec_;
	std::vector<double> const &v_dust2 = sim.userData_.v_dust2_vec_;
	std::vector<double> const &E_gas = sim.userData_.E_gas_vec_;

	if (generating_reference) {
		// generate reference solution mode: save reference solution to file
		save_reference_solution(t, v_gas, v_dust1, v_dust2, E_gas, "dust_damping_reference_supersonic.txt");

		amrex::Print() << "Reference solution generation complete.\n";
		amrex::Print() << "Please copy dust_damping_reference_supersonic.txt to quokka/src/problems/DustDampingIter/\n";
		return 0;
	}
	// test mode: read reference solution and calculate error
	std::string const ref_file_problem = "../src/problems/DustDampingIter/dust_damping_reference_supersonic.txt";

	std::vector<double> t_ref;
	std::vector<double> v_gas_ref;
	std::vector<double> v_dust1_ref;
	std::vector<double> v_dust2_ref;
	std::vector<double> E_gas_ref;

	bool ref_loaded = false;
	if (load_reference_solution(ref_file_problem, t_ref, v_gas_ref, v_dust1_ref, v_dust2_ref, E_gas_ref)) {
		ref_loaded = true;
	}

	if (!ref_loaded) {
		amrex::Print() << "Error: Could not load reference solution.\n";
		amrex::Print() << "Please ensure dust_damping_reference_supersonic.txt is in the current directory or problem directory.\n";
		return 1;
	}

	// check if the time points match (the test time points should be a subset of the reference solution time points)
	if (t_ref.size() < t.size()) {
		amrex::Print() << "Error: Reference solution has fewer points (" << t_ref.size() << ") than test solution (" << t.size() << ")\n";
		return 1;
	}

	// calculate the sampling step size
	auto step = static_cast<int>(t_ref.size() / t.size());
	if (step == 0) {
		step = 1;
	}

	// calculate relative error
	auto compute_relative_error = [](const std::vector<double> &sim, const std::vector<double> &ref, int step) {
		double err_sum = 0.0;
		double ref_sum = 0.0;
		int count = 0;

		for (auto i = 0; i < sim.size(); ++i) {
			int const ref_idx = i * step;
			if (ref_idx >= static_cast<int>(ref.size())) {
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

	double const rel_err_gas_vx = compute_relative_error(v_gas, v_gas_ref, step);
	double const rel_err_dust1_vx = compute_relative_error(v_dust1, v_dust1_ref, step);
	double const rel_err_dust2_vx = compute_relative_error(v_dust2, v_dust2_ref, step);
	double const rel_err_gas_E = compute_relative_error(E_gas, E_gas_ref, step);

	amrex::Print() << "Comparison with reference solution (supersonic correction enabled):\n";
	amrex::Print() << "Relative L1 norm for gas vx    = " << rel_err_gas_vx << "\n";
	amrex::Print() << "Relative L1 norm for dust1 vx  = " << rel_err_dust1_vx << "\n";
	amrex::Print() << "Relative L1 norm for dust2 vx  = " << rel_err_dust2_vx << "\n";
	amrex::Print() << "Relative L1 norm for gas E     = " << rel_err_gas_E << "\n";

	// determine whether the test has passed
	int status = 0;
	const double rel_err_tol = 0.03;

	if ((rel_err_gas_vx > rel_err_tol) || (rel_err_dust1_vx > rel_err_tol) || (rel_err_dust2_vx > rel_err_tol) || (rel_err_gas_E > rel_err_tol)) {
		status = 1;
		amrex::Print() << "Test FAILED: one or more errors exceed tolerance of " << rel_err_tol << "\n";
	} else {
		amrex::Print() << "Test PASSED: all errors within tolerance of " << rel_err_tol << "\n";
	}

#ifdef HAVE_PYTHON
	// plot
	if (!t_ref.empty() && !t.empty()) {
		// gas velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_gas,
				    {{"label", "numerical (non-iter, dt=0.005)"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_ref, v_gas_ref, {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "r"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_g$)");
		matplotlibcpp::title("Gas Velocity (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_supersonic_gas_velocity_noiter.pdf");

		// dust1 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_dust1,
				    {{"label", "numerical (non-iter, dt=0.005)"}, {"color", "b"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_ref, v_dust1_ref, {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "b"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,1}$)");
		matplotlibcpp::title("Dust1 Velocity (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_supersonic_dust1_velocity_noiter.pdf");

		// dust2 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, v_dust2,
				    {{"label", "numerical (non-iter, dt=0.005)"}, {"color", "g"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_ref, v_dust2_ref, {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "g"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,2}$)");
		matplotlibcpp::title("Dust2 Velocity (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_supersonic_dust2_velocity_noiter.pdf");

		// gas energy
		matplotlibcpp::clf();
		matplotlibcpp::plot(t, E_gas,
				    {{"label", "numerical (non-iter, dt=0.005)"}, {"color", "m"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(t_ref, E_gas_ref, {{"label", "reference (non-iter, dt=0.00001)"}, {"color", "m"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($E_g$)");
		matplotlibcpp::title("Gas Energy (with supersonic correction)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_supersonic_gas_energy_noiter.pdf");
	}
#endif

	amrex::Print() << "Test complete.\n";
	return status;
}