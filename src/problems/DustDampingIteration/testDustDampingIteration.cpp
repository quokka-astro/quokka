/// \file testDustDampingIteration.cpp
/// \brief Defines a test problem for dust iterative stopping time
///

#include "QuokkaSimulation.hpp"
#include "dust/DustRuntimeParams.hpp"
#include "util/fextract.hpp"
#include <cmath>
#include <format>
#include <fstream>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

constexpr double rho_dust1 = 1.0;
constexpr double rho_dust2 = 1.0;
constexpr double P_INITIAL = 1.0;

struct DustDampingWithCorrection {
};

struct DustDampingWithoutCorrection {
};

template <> struct SimulationData<DustDampingWithCorrection> {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_vec_;
	std::vector<double> v_dust1_vec_;
	std::vector<double> v_dust2_vec_;
	std::vector<double> E_gas_vec_;
};

template <> struct SimulationData<DustDampingWithoutCorrection> {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_vec_;
	std::vector<double> v_dust1_vec_;
	std::vector<double> v_dust2_vec_;
	std::vector<double> E_gas_vec_;
};

template <> struct quokka::EOS_Traits<DustDampingWithCorrection> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
	// static constexpr double cs_isothermal = 1.0; // only used when gamma = 1
};

template <> struct quokka::EOS_Traits<DustDampingWithoutCorrection> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
	// static constexpr double cs_isothermal = 1.0; // only used when gamma = 1
};

constexpr double rho = 1.0;
constexpr double v0 = 1.0;

constexpr double Egas0_with_corr = P_INITIAL / (quokka::EOS_Traits<DustDampingWithCorrection>::gamma - 1.0) + 0.5 * rho * v0 * v0;
constexpr double Egas0_internal_with_corr = P_INITIAL / (quokka::EOS_Traits<DustDampingWithCorrection>::gamma - 1.0);

constexpr double Egas0_without_corr = P_INITIAL / (quokka::EOS_Traits<DustDampingWithoutCorrection>::gamma - 1.0) + 0.5 * rho * v0 * v0;
constexpr double Egas0_internal_without_corr = P_INITIAL / (quokka::EOS_Traits<DustDampingWithoutCorrection>::gamma - 1.0);

constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;
namespace
{
// problem-specific grain defaults; input files may override them with dust.grain_radius and dust.grain_density
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 2> g_dust_grain_radius = {0.02, 0.01}; // NOLINT
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 2> g_dust_grain_density = {1.0, 1.0};  // NOLINT
} // namespace
static constexpr bool enable_supersonic_correction_with = true;
static constexpr bool enable_supersonic_correction_without = false;

template <> struct Physics_Traits<DustDampingWithCorrection> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 2; // number of dust groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> struct Physics_Traits<DustDampingWithoutCorrection> : DefaultPhysicsTraits {
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
AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingWithCorrection>::ComputeReciprocalStoppingTime(amrex::Real rho_g,
												 amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
												 amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag,
												 double cs) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return ComputeReciprocalStoppingTimeKwok(rho_g, rho_d, rel_vel_mag, cs, g_dust_grain_radius, g_dust_grain_density, enable_supersonic_correction_with);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingWithoutCorrection>::ComputeReciprocalStoppingTime(amrex::Real rho_g,
												    amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
												    amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag,
												    double cs) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return ComputeReciprocalStoppingTimeKwok(rho_g, rho_d, rel_vel_mag, cs, g_dust_grain_radius, g_dust_grain_density,
						 enable_supersonic_correction_without);
}

template <> void QuokkaSimulation<DustDampingWithCorrection>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto vx0 = v0;		 // gas velocity
	const auto vx_dust1 = 2 * v0;	 // dust1 velocity
	const auto vx_dust2 = 10.0 * v0; // dust2 velocity

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// for gas
		state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::energy_index) = Egas0_with_corr;
		state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::internalEnergy_index) = Egas0_internal_with_corr;
		state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x1Momentum_index) = rho * vx0;
		state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x3Momentum_index) = 0.;

		// first-capture for CUDA
		const auto vx_dust1_local = vx_dust1;
		const auto vx_dust2_local = vx_dust2;

		if constexpr (Physics_Traits<DustDampingWithCorrection>::is_dust_enabled) {
			// for dust1
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::dustDensity_index) = rho_dust1;
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index) = rho_dust1 * vx_dust1_local;
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x3DustMomentum_index) = 0.;
			// for dust2
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::dustDensity_index + numDustVars) = rho_dust2;
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index + numDustVars) = rho_dust2 * vx_dust2_local;
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x2DustMomentum_index + numDustVars) = 0.;
			state_cc(i, j, k, HydroSystem<DustDampingWithCorrection>::x3DustMomentum_index + numDustVars) = 0.;
		}
	});
}

template <> void QuokkaSimulation<DustDampingWithoutCorrection>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto vx0 = v0;		 // gas velocity
	const auto vx_dust1 = 2 * v0;	 // dust1 velocity
	const auto vx_dust2 = 10.0 * v0; // dust2 velocity

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// for gas
		state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::energy_index) = Egas0_without_corr;
		state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::internalEnergy_index) = Egas0_internal_without_corr;
		state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x1Momentum_index) = rho * vx0;
		state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x3Momentum_index) = 0.;

		// first-capture for CUDA
		const auto vx_dust1_local = vx_dust1;
		const auto vx_dust2_local = vx_dust2;

		if constexpr (Physics_Traits<DustDampingWithoutCorrection>::is_dust_enabled) {
			// for dust1
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::dustDensity_index) = rho_dust1;
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x1DustMomentum_index) = rho_dust1 * vx_dust1_local;
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x3DustMomentum_index) = 0.;
			// for dust2
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::dustDensity_index + numDustVars) = rho_dust2;
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x1DustMomentum_index + numDustVars) = rho_dust2 * vx_dust2_local;
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x2DustMomentum_index + numDustVars) = 0.;
			state_cc(i, j, k, HydroSystem<DustDampingWithoutCorrection>::x3DustMomentum_index + numDustVars) = 0.;
		}
	});
}

template <> void QuokkaSimulation<DustDampingWithCorrection>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]); // store current time

		// extract physical quantities
		const double density = values.at(HydroSystem<DustDampingWithCorrection>::density_index)[0];
		const double momentum_x = values.at(HydroSystem<DustDampingWithCorrection>::x1Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<DustDampingWithCorrection>::energy_index)[0];

		// store gas velocity
		const double v_gas = momentum_x / density;
		userData_.v_gas_vec_.push_back(v_gas);

		// store gas total energy
		userData_.E_gas_vec_.push_back(Egas_total);

		if constexpr (Physics_Traits<DustDampingWithCorrection>::is_dust_enabled) {
			// store dust1 velocity
			const double dust1_density = values.at(HydroSystem<DustDampingWithCorrection>::dustDensity_index)[0];
			const double dust1_momentum_x = values.at(HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index)[0];
			const double v_dust1 = dust1_momentum_x / dust1_density;
			userData_.v_dust1_vec_.push_back(v_dust1);

			// store dust2 velocity
			const double dust2_density = values.at(HydroSystem<DustDampingWithCorrection>::dustDensity_index + numDustVars)[0];
			const double dust2_momentum_x = values.at(HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index + numDustVars)[0];
			const double v_dust2 = dust2_momentum_x / dust2_density;
			userData_.v_dust2_vec_.push_back(v_dust2);
		}
	}
}

template <> void QuokkaSimulation<DustDampingWithoutCorrection>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]); // store current time

		// extract physical quantities
		const double density = values.at(HydroSystem<DustDampingWithoutCorrection>::density_index)[0];
		const double momentum_x = values.at(HydroSystem<DustDampingWithoutCorrection>::x1Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<DustDampingWithoutCorrection>::energy_index)[0];

		// store gas velocity
		const double v_gas = momentum_x / density;
		userData_.v_gas_vec_.push_back(v_gas);

		// store gas total energy
		userData_.E_gas_vec_.push_back(Egas_total);

		if constexpr (Physics_Traits<DustDampingWithoutCorrection>::is_dust_enabled) {
			// store dust1 velocity
			const double dust1_density = values.at(HydroSystem<DustDampingWithoutCorrection>::dustDensity_index)[0];
			const double dust1_momentum_x = values.at(HydroSystem<DustDampingWithoutCorrection>::x1DustMomentum_index)[0];
			const double v_dust1 = dust1_momentum_x / dust1_density;
			userData_.v_dust1_vec_.push_back(v_dust1);

			// store dust2 velocity
			const double dust2_density = values.at(HydroSystem<DustDampingWithoutCorrection>::dustDensity_index + numDustVars)[0];
			const double dust2_momentum_x = values.at(HydroSystem<DustDampingWithoutCorrection>::x1DustMomentum_index + numDustVars)[0];
			const double v_dust2 = dust2_momentum_x / dust2_density;
			userData_.v_dust2_vec_.push_back(v_dust2);
		}
	}
}

auto run_reference_simulation() -> SimulationData<DustDampingWithCorrection>
{
	QuokkaSimulation<DustDampingWithCorrection> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 1000000.0; // large CFL number to avoid CFL violation
	sim.constantDt_ = 0.00005;  // fixed small timestep for reference solution
	sim.enableIterDustStoptime_ = 0;
	sim.print_dust_counter_ = false;

	sim.setInitialConditions();
	// store initial values for t=0 plotting
	auto [_, val_ini] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.t_vec_.push_back(0.0);

		const double initial_density = val_ini.at(HydroSystem<DustDampingWithCorrection>::density_index)[0];
		const double initial_momentum_x = val_ini.at(HydroSystem<DustDampingWithCorrection>::x1Momentum_index)[0];
		const double initial_Egas_total = val_ini.at(HydroSystem<DustDampingWithCorrection>::energy_index)[0];
		const double initial_v_gas = initial_momentum_x / initial_density;
		sim.userData_.v_gas_vec_.push_back(initial_v_gas);
		sim.userData_.E_gas_vec_.push_back(initial_Egas_total);

		if constexpr (Physics_Traits<DustDampingWithCorrection>::is_dust_enabled) {
			const double initial_dust1_density = val_ini.at(HydroSystem<DustDampingWithCorrection>::dustDensity_index)[0];
			const double initial_dust1_momentum_x = val_ini.at(HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index)[0];
			const double initial_v_dust1 = initial_dust1_momentum_x / initial_dust1_density;
			sim.userData_.v_dust1_vec_.push_back(initial_v_dust1);

			const double initial_dust2_density = val_ini.at(HydroSystem<DustDampingWithCorrection>::dustDensity_index + numDustVars)[0];
			const double initial_dust2_momentum_x = val_ini.at(HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index + numDustVars)[0];
			const double initial_v_dust2 = initial_dust2_momentum_x / initial_dust2_density;
			sim.userData_.v_dust2_vec_.push_back(initial_v_dust2);
		}
	}

	sim.evolve();

	return sim.userData_;
}

auto run_iterative_with_correction() -> SimulationData<DustDampingWithCorrection>
{
	QuokkaSimulation<DustDampingWithCorrection> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 0.3;
	sim.constantDt_ = -1.0;
	sim.enableIterDustStoptime_ = 1;
	sim.print_dust_counter_ = true;

	sim.setInitialConditions();
	// store initial values for t=0 plotting
	auto [_, val_ini] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.t_vec_.push_back(0.0);

		const double initial_density = val_ini.at(HydroSystem<DustDampingWithCorrection>::density_index)[0];
		const double initial_momentum_x = val_ini.at(HydroSystem<DustDampingWithCorrection>::x1Momentum_index)[0];
		const double initial_Egas_total = val_ini.at(HydroSystem<DustDampingWithCorrection>::energy_index)[0];
		const double initial_v_gas = initial_momentum_x / initial_density;
		sim.userData_.v_gas_vec_.push_back(initial_v_gas);
		sim.userData_.E_gas_vec_.push_back(initial_Egas_total);

		if constexpr (Physics_Traits<DustDampingWithCorrection>::is_dust_enabled) {
			const double initial_dust1_density = val_ini.at(HydroSystem<DustDampingWithCorrection>::dustDensity_index)[0];
			const double initial_dust1_momentum_x = val_ini.at(HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index)[0];
			const double initial_v_dust1 = initial_dust1_momentum_x / initial_dust1_density;
			sim.userData_.v_dust1_vec_.push_back(initial_v_dust1);

			const double initial_dust2_density = val_ini.at(HydroSystem<DustDampingWithCorrection>::dustDensity_index + numDustVars)[0];
			const double initial_dust2_momentum_x = val_ini.at(HydroSystem<DustDampingWithCorrection>::x1DustMomentum_index + numDustVars)[0];
			const double initial_v_dust2 = initial_dust2_momentum_x / initial_dust2_density;
			sim.userData_.v_dust2_vec_.push_back(initial_v_dust2);
		}
	}
	sim.evolve();

	return sim.userData_;
}

auto run_iterative_without_correction() -> SimulationData<DustDampingWithoutCorrection>
{
	QuokkaSimulation<DustDampingWithoutCorrection> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 0.3;
	sim.constantDt_ = -1.0;
	sim.enableIterDustStoptime_ = 1;
	sim.print_dust_counter_ = true;

	sim.setInitialConditions();
	// store initial values for t=0 plotting
	auto [_, val_ini] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.t_vec_.push_back(0.0);

		const double initial_density = val_ini.at(HydroSystem<DustDampingWithoutCorrection>::density_index)[0];
		const double initial_momentum_x = val_ini.at(HydroSystem<DustDampingWithoutCorrection>::x1Momentum_index)[0];
		const double initial_Egas_total = val_ini.at(HydroSystem<DustDampingWithoutCorrection>::energy_index)[0];
		const double initial_v_gas = initial_momentum_x / initial_density;
		sim.userData_.v_gas_vec_.push_back(initial_v_gas);
		sim.userData_.E_gas_vec_.push_back(initial_Egas_total);

		if constexpr (Physics_Traits<DustDampingWithoutCorrection>::is_dust_enabled) {
			const double initial_dust1_density = val_ini.at(HydroSystem<DustDampingWithoutCorrection>::dustDensity_index)[0];
			const double initial_dust1_momentum_x = val_ini.at(HydroSystem<DustDampingWithoutCorrection>::x1DustMomentum_index)[0];
			const double initial_v_dust1 = initial_dust1_momentum_x / initial_dust1_density;
			sim.userData_.v_dust1_vec_.push_back(initial_v_dust1);

			const double initial_dust2_density = val_ini.at(HydroSystem<DustDampingWithoutCorrection>::dustDensity_index + numDustVars)[0];
			const double initial_dust2_momentum_x = val_ini.at(HydroSystem<DustDampingWithoutCorrection>::x1DustMomentum_index + numDustVars)[0];
			const double initial_v_dust2 = initial_dust2_momentum_x / initial_dust2_density;
			sim.userData_.v_dust2_vec_.push_back(initial_v_dust2);
		}
	}
	sim.evolve();

	return sim.userData_;
}

auto compute_relative_error(const std::vector<double> &t_test, const std::vector<double> &v_test, const std::vector<double> &t_ref,
			    const std::vector<double> &v_ref) -> double
{
	if (t_ref.empty() || t_test.empty()) {
		return 1.0;
	}

	double err_sum = 0.0;
	double ref_sum = 0.0;
	int count = 0;

	for (size_t i = 0; i < t_test.size(); ++i) {
		double const t = t_test[i];

		// find the closest reference time point
		auto it = std::lower_bound(t_ref.begin(), t_ref.end(), t);

		size_t ref_idx = 0;
		if (it == t_ref.end()) {
			ref_idx = t_ref.size() - 1;
		} else if (it == t_ref.begin()) {
			ref_idx = 0;
		} else {
			size_t const idx = it - t_ref.begin();
			size_t const prev_idx = idx - 1;

			double const diff1 = std::abs(t - t_ref[prev_idx]);
			double const diff2 = std::abs(t - t_ref[idx]);

			ref_idx = (diff1 <= diff2) ? prev_idx : idx;
		}

		err_sum += std::abs(v_test[i] - v_ref[ref_idx]);
		ref_sum += std::abs(v_ref[ref_idx]);
		count++;
	}

	if (count == 0 || ref_sum == 0.0) {
		return 1.0;
	}
	return err_sum / ref_sum;
}

auto problem_main() -> int
{
	quokka::dust::readDustGrainParams(g_dust_grain_radius, g_dust_grain_density);

	// step 1: run the reference solution (non-iterative, fixed small time step, with correction enabled)
	auto ref_data = run_reference_simulation();

	// step 2: run the iterative solution (with correction enabled, CFL=0.3)
	auto iter_with_corr_data = run_iterative_with_correction();

	// step 3: run the iterative solution (without correction, CFL=0.3)
	auto iter_without_corr_data = run_iterative_without_correction();

	double const rel_err_with_corr_gas_vx =
	    compute_relative_error(iter_with_corr_data.t_vec_, iter_with_corr_data.v_gas_vec_, ref_data.t_vec_, ref_data.v_gas_vec_);
	double const rel_err_with_corr_dust1_vx =
	    compute_relative_error(iter_with_corr_data.t_vec_, iter_with_corr_data.v_dust1_vec_, ref_data.t_vec_, ref_data.v_dust1_vec_);
	double const rel_err_with_corr_dust2_vx =
	    compute_relative_error(iter_with_corr_data.t_vec_, iter_with_corr_data.v_dust2_vec_, ref_data.t_vec_, ref_data.v_dust2_vec_);
	double const rel_err_with_corr_gas_E =
	    compute_relative_error(iter_with_corr_data.t_vec_, iter_with_corr_data.E_gas_vec_, ref_data.t_vec_, ref_data.E_gas_vec_);

	double const rel_err_without_corr_gas_vx =
	    compute_relative_error(iter_without_corr_data.t_vec_, iter_without_corr_data.v_gas_vec_, ref_data.t_vec_, ref_data.v_gas_vec_);
	double const rel_err_without_corr_dust1_vx =
	    compute_relative_error(iter_without_corr_data.t_vec_, iter_without_corr_data.v_dust1_vec_, ref_data.t_vec_, ref_data.v_dust1_vec_);
	double const rel_err_without_corr_dust2_vx =
	    compute_relative_error(iter_without_corr_data.t_vec_, iter_without_corr_data.v_dust2_vec_, ref_data.t_vec_, ref_data.v_dust2_vec_);
	double const rel_err_without_corr_gas_E =
	    compute_relative_error(iter_without_corr_data.t_vec_, iter_without_corr_data.E_gas_vec_, ref_data.t_vec_, ref_data.E_gas_vec_);

	amrex::Print() << "\nComparison with reference solution:\n";
	amrex::Print() << "Iterative WITH correction:\n";
	amrex::Print() << "  Relative L1 norm for gas vx    = " << rel_err_with_corr_gas_vx << "\n";
	amrex::Print() << "  Relative L1 norm for dust1 vx  = " << rel_err_with_corr_dust1_vx << "\n";
	amrex::Print() << "  Relative L1 norm for dust2 vx  = " << rel_err_with_corr_dust2_vx << "\n";
	amrex::Print() << "  Relative L1 norm for gas E     = " << rel_err_with_corr_gas_E << "\n";

	amrex::Print() << "\nIterative WITHOUT correction:\n";
	amrex::Print() << "  Relative L1 norm for gas vx    = " << rel_err_without_corr_gas_vx << "\n";
	amrex::Print() << "  Relative L1 norm for dust1 vx  = " << rel_err_without_corr_dust1_vx << "\n";
	amrex::Print() << "  Relative L1 norm for dust2 vx  = " << rel_err_without_corr_dust2_vx << "\n";
	amrex::Print() << "  Relative L1 norm for gas E     = " << rel_err_without_corr_gas_E << "\n";

	// determine whether the test has passed
	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const double rel_err_tol = 0.01;

		bool const with_corr_passed = (rel_err_with_corr_gas_vx <= rel_err_tol) && (rel_err_with_corr_dust1_vx <= rel_err_tol) &&
					      (rel_err_with_corr_dust2_vx <= rel_err_tol) && (rel_err_with_corr_gas_E <= rel_err_tol);

		bool const without_corr_passed = (rel_err_without_corr_gas_vx <= rel_err_tol) && (rel_err_without_corr_dust1_vx <= rel_err_tol) &&
						 (rel_err_without_corr_dust2_vx <= rel_err_tol) && (rel_err_without_corr_gas_E <= rel_err_tol);

		if (!with_corr_passed || !without_corr_passed) {
			status = 1;
			amrex::Print() << "\nTest FAILED: one or more errors exceed tolerance of " << rel_err_tol << "\n";
			if (!with_corr_passed) {
				amrex::Print() << "  - Iterative with correction failed\n";
			}
			if (!without_corr_passed) {
				amrex::Print() << "  - Iterative without correction failed\n";
			}
		} else {
			amrex::Print() << "\nTest PASSED: all errors within tolerance of " << rel_err_tol << "\n";
		}

#ifdef HAVE_PYTHON
		// gas velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_gas_vec_,
				    {{"label", "reference (non-iter, dt=0.00005)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(iter_with_corr_data.t_vec_, iter_with_corr_data.v_gas_vec_,
				    {{"label", "iterative with correction"}, {"color", "r"}, {"linestyle", "--"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(iter_without_corr_data.t_vec_, iter_without_corr_data.v_gas_vec_,
				    {{"label", "iterative without correction"}, {"color", "b"}, {"linestyle", ":"}, {"marker", "s"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_g$)");
		matplotlibcpp::title("Gas Velocity");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_gas_velocity.pdf");

		// dust1 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_dust1_vec_,
				    {{"label", "reference (non-iter, dt=0.00005)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(iter_with_corr_data.t_vec_, iter_with_corr_data.v_dust1_vec_,
				    {{"label", "iterative with correction"}, {"color", "r"}, {"linestyle", "--"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(iter_without_corr_data.t_vec_, iter_without_corr_data.v_dust1_vec_,
				    {{"label", "iterative without correction"}, {"color", "b"}, {"linestyle", ":"}, {"marker", "s"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,1}$)");
		matplotlibcpp::title("Dust1 Velocity");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_dust1_velocity.pdf");

		// dust2 velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_dust2_vec_,
				    {{"label", "reference (non-iter, dt=0.00005)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(iter_with_corr_data.t_vec_, iter_with_corr_data.v_dust2_vec_,
				    {{"label", "iterative with correction"}, {"color", "r"}, {"linestyle", "--"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(iter_without_corr_data.t_vec_, iter_without_corr_data.v_dust2_vec_,
				    {{"label", "iterative without correction"}, {"color", "b"}, {"linestyle", ":"}, {"marker", "s"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,2}$)");
		matplotlibcpp::title("Dust2 Velocity");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_dust2_velocity.pdf");

		// gas energy
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.E_gas_vec_,
				    {{"label", "reference (non-iter, dt=0.00005)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(iter_with_corr_data.t_vec_, iter_with_corr_data.E_gas_vec_,
				    {{"label", "iterative with correction"}, {"color", "r"}, {"linestyle", "--"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::plot(iter_without_corr_data.t_vec_, iter_without_corr_data.E_gas_vec_,
				    {{"label", "iterative without correction"}, {"color", "b"}, {"linestyle", ":"}, {"marker", "s"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($E_g$)");
		matplotlibcpp::title("Gas Energy");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_gas_energy.pdf");
#endif
	}

	amrex::Print() << "\nTest complete.\n";
	return status;
}