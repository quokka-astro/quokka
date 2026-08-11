/// \file testDustDampingIterationMHDZeroCharge.cpp
/// \brief Dust damping test for the MHD dust drag and Lorentz integrator with zero charge-to-mass ratio.
///

#include "QuokkaSimulation.hpp"
#include "dust/DustRuntimeParams.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

namespace
{
constexpr double rho = 1.0;
constexpr double rho_dust1 = 1.0;
constexpr double rho_dust2 = 1.0;
constexpr double v0 = 1.0;
constexpr double P_INITIAL = 1.0;
constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

constexpr double B_X = 0.3;
constexpr double B_Y = 0.4;
constexpr double B_Z = 0.5;
constexpr double MAGNETIC_ENERGY = 0.5 * (B_X * B_X + B_Y * B_Y + B_Z * B_Z);

AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 2> g_dust_grain_radius = {0.02, 0.01}; // NOLINT
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 2> g_dust_grain_density = {1.0, 1.0};  // NOLINT
constexpr bool enable_supersonic_correction = true;
} // namespace

struct DustDampingDragReference {
};

struct DustDampingMHDZeroCharge {
};

struct DustDampingHistory {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_x_vec_;
	std::vector<double> v_gas_y_vec_;
	std::vector<double> v_gas_z_vec_;
	std::vector<double> v_dust1_x_vec_;
	std::vector<double> v_dust1_y_vec_;
	std::vector<double> v_dust1_z_vec_;
	std::vector<double> v_dust2_x_vec_;
	std::vector<double> v_dust2_y_vec_;
	std::vector<double> v_dust2_z_vec_;
	std::vector<double> E_gas_vec_;
};

template <> struct SimulationData<DustDampingDragReference> : DustDampingHistory {
};

template <> struct SimulationData<DustDampingMHDZeroCharge> : DustDampingHistory {
};

template <> struct quokka::EOS_Traits<DustDampingDragReference> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
};

template <> struct quokka::EOS_Traits<DustDampingMHDZeroCharge> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.4;
};

constexpr double Egas0 = P_INITIAL / (quokka::EOS_Traits<DustDampingDragReference>::gamma - 1.0) + 0.5 * rho * v0 * v0;
constexpr double Egas0_internal = P_INITIAL / (quokka::EOS_Traits<DustDampingDragReference>::gamma - 1.0);

template <> struct Physics_Traits<DustDampingDragReference> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 2;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> struct Physics_Traits<DustDampingMHDZeroCharge> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 2;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingDragReference>::ComputeReciprocalStoppingTime(amrex::Real rho_g,
												amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
												amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag,
												double cs) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return ComputeReciprocalStoppingTimeKwok(rho_g, rho_d, rel_vel_mag, cs, g_dust_grain_radius, g_dust_grain_density, enable_supersonic_correction);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingMHDZeroCharge>::ComputeReciprocalStoppingTime(amrex::Real rho_g,
												amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
												amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag,
												double cs) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return ComputeReciprocalStoppingTimeKwok(rho_g, rho_d, rel_vel_mag, cs, g_dust_grain_radius, g_dust_grain_density, enable_supersonic_correction);
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingMHDZeroCharge>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> charge_to_mass_ratio{};
	charge_to_mass_ratio.fill(0.0);
	return charge_to_mass_ratio;
}

template <typename problem_t> void setDustDampingInitialConditions(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto vx0 = v0;
	const auto vx_dust1 = 2.0 * v0;
	const auto vx_dust2 = 10.0 * v0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<problem_t>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<problem_t>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = Egas0_internal;
		state_cc(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = rho * vx0;
		state_cc(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<problem_t>::dustDensity_index) = rho_dust1;
		state_cc(i, j, k, HydroSystem<problem_t>::x1DustMomentum_index) = rho_dust1 * vx_dust1;
		state_cc(i, j, k, HydroSystem<problem_t>::x2DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<problem_t>::x3DustMomentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<problem_t>::dustDensity_index + numDustVars) = rho_dust2;
		state_cc(i, j, k, HydroSystem<problem_t>::x1DustMomentum_index + numDustVars) = rho_dust2 * vx_dust2;
		state_cc(i, j, k, HydroSystem<problem_t>::x2DustMomentum_index + numDustVars) = 0.0;
		state_cc(i, j, k, HydroSystem<problem_t>::x3DustMomentum_index + numDustVars) = 0.0;
	});
}

template <> void QuokkaSimulation<DustDampingDragReference>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setDustDampingInitialConditions<DustDampingDragReference>(grid_elem);
}

template <> void QuokkaSimulation<DustDampingMHDZeroCharge>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setDustDampingInitialConditions<DustDampingMHDZeroCharge>(grid_elem);
}

template <> void QuokkaSimulation<DustDampingMHDZeroCharge>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_fc = Physics_Indices<DustDampingMHDZeroCharge>::nvarPerDim_fc;

	double bfield = B_Z;
	if (grid_elem.dir_ == quokka::direction::x) {
		bfield = B_X;
	} else if (grid_elem.dir_ == quokka::direction::y) {
		bfield = B_Y;
	}

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}
		state_fc(i, j, k, Physics_Indices<DustDampingMHDZeroCharge>::mhdFirstIndex) = bfield;
	});
}

template <typename problem_t> void appendStateHistory(QuokkaSimulation<problem_t> &sim)
{
	auto [_, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto &data = sim.userData_;
		data.t_vec_.push_back(sim.tNew_[0]);

		const double density = values.at(HydroSystem<problem_t>::density_index)[0];
		const double gas_momentum_x = values.at(HydroSystem<problem_t>::x1Momentum_index)[0];
		const double gas_momentum_y = values.at(HydroSystem<problem_t>::x2Momentum_index)[0];
		const double gas_momentum_z = values.at(HydroSystem<problem_t>::x3Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<problem_t>::energy_index)[0];
		const double magnetic_energy = Physics_Traits<problem_t>::is_mhd_enabled ? MAGNETIC_ENERGY : 0.0;

		data.v_gas_x_vec_.push_back(gas_momentum_x / density);
		data.v_gas_y_vec_.push_back(gas_momentum_y / density);
		data.v_gas_z_vec_.push_back(gas_momentum_z / density);
		data.E_gas_vec_.push_back(Egas_total - magnetic_energy);

		const double dust1_density = values.at(HydroSystem<problem_t>::dustDensity_index)[0];
		const double dust1_momentum_x = values.at(HydroSystem<problem_t>::x1DustMomentum_index)[0];
		const double dust1_momentum_y = values.at(HydroSystem<problem_t>::x2DustMomentum_index)[0];
		const double dust1_momentum_z = values.at(HydroSystem<problem_t>::x3DustMomentum_index)[0];
		data.v_dust1_x_vec_.push_back(dust1_momentum_x / dust1_density);
		data.v_dust1_y_vec_.push_back(dust1_momentum_y / dust1_density);
		data.v_dust1_z_vec_.push_back(dust1_momentum_z / dust1_density);

		const double dust2_density = values.at(HydroSystem<problem_t>::dustDensity_index + numDustVars)[0];
		const double dust2_momentum_x = values.at(HydroSystem<problem_t>::x1DustMomentum_index + numDustVars)[0];
		const double dust2_momentum_y = values.at(HydroSystem<problem_t>::x2DustMomentum_index + numDustVars)[0];
		const double dust2_momentum_z = values.at(HydroSystem<problem_t>::x3DustMomentum_index + numDustVars)[0];
		data.v_dust2_x_vec_.push_back(dust2_momentum_x / dust2_density);
		data.v_dust2_y_vec_.push_back(dust2_momentum_y / dust2_density);
		data.v_dust2_z_vec_.push_back(dust2_momentum_z / dust2_density);
	}
}

template <> void QuokkaSimulation<DustDampingDragReference>::computeAfterTimestep() { appendStateHistory(*this); }

template <> void QuokkaSimulation<DustDampingMHDZeroCharge>::computeAfterTimestep() { appendStateHistory(*this); }

auto run_reference_simulation() -> SimulationData<DustDampingDragReference>
{
	QuokkaSimulation<DustDampingDragReference> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 1000000.0; // large CFL number to avoid CFL violation
	sim.constantDt_ = 0.00005;  // fixed small timestep for reference solution
	sim.enableIterDustStoptime_ = 0;
	sim.print_dust_counter_ = false;

	sim.setInitialConditions();
	appendStateHistory(sim);
	sim.evolve();

	return sim.userData_;
}

auto makePeriodicFaceBCs() -> amrex::Vector<amrex::BCRec>
{
	const int nvars_fc = Physics_Indices<DustDampingMHDZeroCharge>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}
	return BCs_fc;
}

auto run_mhd_zero_charge_simulation() -> SimulationData<DustDampingMHDZeroCharge>
{
	auto BCs_cc = quokka::BC<DustDampingMHDZeroCharge>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	auto BCs_fc = makePeriodicFaceBCs();
	QuokkaSimulation<DustDampingMHDZeroCharge> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 0.3;
	sim.constantDt_ = -1.0;
	sim.enableIterDustStoptime_ = 1;
	sim.dust_omega_res_ = 1.0; // make the energy calculation method in computeDustDragAndLorentz() the same as the reference solution
	sim.print_dust_counter_ = true;

	sim.setInitialConditions();
	appendStateHistory(sim);
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
		auto it = std::lower_bound(t_ref.begin(), t_ref.end(), t);

		size_t ref_idx = 0;
		if (it == t_ref.end()) {
			ref_idx = t_ref.size() - 1;
		} else if (it == t_ref.begin()) {
			ref_idx = 0;
		} else {
			size_t const idx = static_cast<size_t>(it - t_ref.begin());
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

auto max_abs_component(const std::vector<double> &values) -> double
{
	double max_value = 0.0;
	for (double const value : values) {
		max_value = std::max(max_value, std::abs(value));
	}
	return max_value;
}

auto problem_main() -> int
{
	quokka::dust::readDustGrainParams(g_dust_grain_radius, g_dust_grain_density);

	auto ref_data = run_reference_simulation();
	auto mhd_data = run_mhd_zero_charge_simulation();

	double const rel_err_gas_vx = compute_relative_error(mhd_data.t_vec_, mhd_data.v_gas_x_vec_, ref_data.t_vec_, ref_data.v_gas_x_vec_);
	double const rel_err_dust1_vx = compute_relative_error(mhd_data.t_vec_, mhd_data.v_dust1_x_vec_, ref_data.t_vec_, ref_data.v_dust1_x_vec_);
	double const rel_err_dust2_vx = compute_relative_error(mhd_data.t_vec_, mhd_data.v_dust2_x_vec_, ref_data.t_vec_, ref_data.v_dust2_x_vec_);
	double const rel_err_gas_E = compute_relative_error(mhd_data.t_vec_, mhd_data.E_gas_vec_, ref_data.t_vec_, ref_data.E_gas_vec_);

	double const max_transverse_velocity =
	    std::max({max_abs_component(mhd_data.v_gas_y_vec_), max_abs_component(mhd_data.v_gas_z_vec_), max_abs_component(mhd_data.v_dust1_y_vec_),
		      max_abs_component(mhd_data.v_dust1_z_vec_), max_abs_component(mhd_data.v_dust2_y_vec_), max_abs_component(mhd_data.v_dust2_z_vec_)});

	amrex::Print() << "\nMHD zero-charge dust damping comparison:\n";
	amrex::Print() << "  B = (" << B_X << ", " << B_Y << ", " << B_Z << ")\n";
	amrex::Print() << "  Relative L1 norm for gas vx    = " << rel_err_gas_vx << "\n";
	amrex::Print() << "  Relative L1 norm for dust1 vx  = " << rel_err_dust1_vx << "\n";
	amrex::Print() << "  Relative L1 norm for dust2 vx  = " << rel_err_dust2_vx << "\n";
	amrex::Print() << "  Relative L1 norm for gas E     = " << rel_err_gas_E << "\n";
	amrex::Print() << "  Max transverse velocity        = " << max_transverse_velocity << "\n";

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const double rel_err_tol = 0.01;
		const double transverse_velocity_tol = 1.0e-12;
		bool const passed = (rel_err_gas_vx <= rel_err_tol) && (rel_err_dust1_vx <= rel_err_tol) && (rel_err_dust2_vx <= rel_err_tol) &&
				    (rel_err_gas_E <= rel_err_tol) && (max_transverse_velocity <= transverse_velocity_tol);

		if (!passed) {
			status = 1;
			amrex::Print() << "\nTest FAILED: MHD zero-charge solution did not match pure drag reference.\n";
		} else {
			amrex::Print() << "\nTest PASSED: MHD zero-charge solution matches pure drag reference.\n";
		}

#ifdef HAVE_PYTHON
		// gas x-velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_gas_x_vec_,
				    {{"label", "reference (pure drag)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(mhd_data.t_vec_, mhd_data.v_gas_x_vec_,
				    {{"label", "MHD zero-charge"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{g,x}$)");
		matplotlibcpp::title("Gas X-Velocity");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_mhd_gas_velocity_x.pdf");

		// dust1 x-velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_dust1_x_vec_,
				    {{"label", "reference (pure drag)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(mhd_data.t_vec_, mhd_data.v_dust1_x_vec_,
				    {{"label", "MHD zero-charge"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d1,x}$)");
		matplotlibcpp::title("Dust1 X-Velocity");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_mhd_dust1_velocity_x.pdf");

		// dust2 x-velocity
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.v_dust2_x_vec_,
				    {{"label", "reference (pure drag)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(mhd_data.t_vec_, mhd_data.v_dust2_x_vec_,
				    {{"label", "MHD zero-charge"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d2,x}$)");
		matplotlibcpp::title("Dust2 X-Velocity");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_mhd_dust2_velocity_x.pdf");

		// gas energy
		matplotlibcpp::clf();
		matplotlibcpp::plot(ref_data.t_vec_, ref_data.E_gas_vec_,
				    {{"label", "reference (pure drag)"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "0.7"}});
		matplotlibcpp::plot(mhd_data.t_vec_, mhd_data.E_gas_vec_,
				    {{"label", "MHD zero-charge"}, {"color", "r"}, {"linestyle", "-"}, {"marker", "o"}, {"markersize", "3"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($E_g$)");
		matplotlibcpp::title("Gas Energy");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_iteration_mhd_gas_energy.pdf");
#endif
	}

	return status;
}
