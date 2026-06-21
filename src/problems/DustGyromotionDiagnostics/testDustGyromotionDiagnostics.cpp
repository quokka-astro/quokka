/// \file testDustGyromotionDiagnostics.cpp
/// \brief One-step pure gyromotion magnitude and phase diagnostics for charged dust.
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

struct DustGyromotionDiagnostics {
};

using DustPureGyromotion = DustGyromotionDiagnostics;

namespace
{
constexpr double rho_gas = 1.0;
constexpr double epsilon = 1.0;
constexpr double rho_dust = epsilon * rho_gas;
constexpr double sound_speed = 1.0;
constexpr double charge_to_mass_ratio = 1.0;
constexpr double magnetic_field_z = 1.0;
constexpr double initial_relative_drift = 1.0;

constexpr double omega_L = charge_to_mass_ratio * magnetic_field_z;
constexpr double omega_rel = (1.0 + epsilon) * omega_L;
constexpr double gas_velocity_x0 = -epsilon * initial_relative_drift / (1.0 + epsilon);
constexpr double dust_velocity_x0 = initial_relative_drift / (1.0 + epsilon);
using ResolvedRkScheme = quokka::dust::ResolvedRkScheme;

constexpr std::array<double, 11> requested_dt_values = {1.0e-2, 3.0e-2, 1.0e-1, 2.0e-1, 3.0e-1, 5.0e-1, 1.0e0, 2.0e0, 4.0e0, 8.0e0, 1.6e1};
constexpr std::array<ResolvedRkScheme, 3> resolved_rk_schemes = {ResolvedRkScheme::TP2025, ResolvedRkScheme::GL4, ResolvedRkScheme::Midpoint};
constexpr double plot_floor = 1.0e-16;

struct DriftState {
	double wx;
	double wy;
};

struct GyroSample {
	double requested_dt;
	double effective_dt;
	double end_time;
	double theta;
	double amplitude_ratio;
	double delta_log_amplitude;
	double delta_phase;
	double abs_delta_log_amplitude;
	double abs_delta_phase;
	double theory_delta_log_amplitude;
	double theory_delta_phase;
	double conservation_error;
	bool used_resolved_branch;
};

struct SchemeSweepResult {
	ResolvedRkScheme scheme;
	std::vector<GyroSample> samples;
};

auto resolvedRkSchemeSlug(ResolvedRkScheme scheme) -> std::string_view
{
	switch (scheme) {
		case ResolvedRkScheme::TP2025:
			return "tp2025";
		case ResolvedRkScheme::GL4:
			return "gl4";
		case ResolvedRkScheme::Midpoint:
			return "midpoint";
	}
	return "unknown";
}
} // namespace

template <> struct SimulationData<DustPureGyromotion> {
	std::vector<double> t_vec_;
	std::vector<double> wx_vec_;
	std::vector<double> wy_vec_;
	std::vector<double> wz_vec_;
	std::vector<double> center_momentum_x_vec_;
	std::vector<double> center_momentum_y_vec_;
	std::vector<double> center_momentum_z_vec_;
};

template <> struct quokka::EOS_Traits<DustPureGyromotion> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = sound_speed;
};

template <> struct Physics_Traits<DustPureGyromotion> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustPureGyromotion>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
											  amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
											  amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/,
											  double /*cs*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 1> alpha{};
	alpha[0] = 0.0;
	return alpha;
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustPureGyromotion>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 1> q_over_m{};
	q_over_m[0] = charge_to_mass_ratio;
	return q_over_m;
}

template <> void QuokkaSimulation<DustPureGyromotion>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<DustPureGyromotion>::nvarTotal_cc;
	const double magnetic_energy = 0.5 * magnetic_field_z * magnetic_field_z;
	const double gas_kinetic_energy = 0.5 * rho_gas * gas_velocity_x0 * gas_velocity_x0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::density_index) = rho_gas;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::energy_index) = magnetic_energy + gas_kinetic_energy;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::x1Momentum_index) = rho_gas * gas_velocity_x0;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::x3Momentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::dustDensity_index) = rho_dust;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::x1DustMomentum_index) = rho_dust * dust_velocity_x0;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::x2DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustPureGyromotion>::x3DustMomentum_index) = 0.0;
	});
}

template <> void QuokkaSimulation<DustPureGyromotion>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_fc = Physics_Indices<DustPureGyromotion>::nvarPerDim_fc;
	const double bfield = (grid_elem.dir_ == quokka::direction::z) ? magnetic_field_z : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}
		state_fc(i, j, k, Physics_Indices<DustPureGyromotion>::mhdFirstIndex) = bfield;
	});
}

void recordHistory(QuokkaSimulation<DustPureGyromotion> &sim)
{
	auto [_, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);

	if (!amrex::ParallelDescriptor::IOProcessor()) {
		return;
	}

	auto &data = sim.userData_;
	data.t_vec_.push_back(sim.tNew_[0]);

	const double gas_density = values.at(HydroSystem<DustPureGyromotion>::density_index)[0];
	const double gas_momentum_x = values.at(HydroSystem<DustPureGyromotion>::x1Momentum_index)[0];
	const double gas_momentum_y = values.at(HydroSystem<DustPureGyromotion>::x2Momentum_index)[0];
	const double gas_momentum_z = values.at(HydroSystem<DustPureGyromotion>::x3Momentum_index)[0];
	const double gas_vx = gas_momentum_x / gas_density;
	const double gas_vy = gas_momentum_y / gas_density;
	const double gas_vz = gas_momentum_z / gas_density;

	const double dust_density = values.at(HydroSystem<DustPureGyromotion>::dustDensity_index)[0];
	const double dust_momentum_x = values.at(HydroSystem<DustPureGyromotion>::x1DustMomentum_index)[0];
	const double dust_momentum_y = values.at(HydroSystem<DustPureGyromotion>::x2DustMomentum_index)[0];
	const double dust_momentum_z = values.at(HydroSystem<DustPureGyromotion>::x3DustMomentum_index)[0];
	const double dust_vx = dust_momentum_x / dust_density;
	const double dust_vy = dust_momentum_y / dust_density;
	const double dust_vz = dust_momentum_z / dust_density;

	data.wx_vec_.push_back(dust_vx - gas_vx);
	data.wy_vec_.push_back(dust_vy - gas_vy);
	data.wz_vec_.push_back(dust_vz - gas_vz);
	data.center_momentum_x_vec_.push_back(gas_momentum_x + dust_momentum_x);
	data.center_momentum_y_vec_.push_back(gas_momentum_y + dust_momentum_y);
	data.center_momentum_z_vec_.push_back(gas_momentum_z + dust_momentum_z);
}

template <> void QuokkaSimulation<DustPureGyromotion>::computeAfterTimestep() { recordHistory(*this); }

auto resolvedBranchThresholdDt() -> double { return 1.0 / std::abs(omega_L); }

auto usesResolvedBranch(double effective_dt) -> bool { return effective_dt < (1.0 / std::abs(omega_L)); }

auto runGyroSimulation(ResolvedRkScheme scheme, double constant_dt) -> SimulationData<DustPureGyromotion>
{
	amrex::ParmParse pp;
	pp.add("constant_dt", constant_dt);
	pp.add("stop_time", constant_dt);
	pp.add("max_timesteps", 4);
	pp.add("suppress_output", 1);
	pp.add("show_performance_hints", 0);

	using quokka::BCType::mathematicalBndryTypes;
	auto BCs_cc = quokka::BC<DustPureGyromotion>(quokka::BCType::int_dir);
	auto BCs_fc = quokka::BC_fc<DustPureGyromotion>(mathematicalBndryTypes::periodic, mathematicalBndryTypes::periodic, mathematicalBndryTypes::periodic);
	QuokkaSimulation<DustPureGyromotion> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 1000000.0;
	sim.constantDt_ = constant_dt;
	sim.stopTime_ = constant_dt;
	sim.maxTimesteps_ = 4;
	sim.dustResolvedRkScheme_ = scheme;
	sim.print_dust_counter_ = false;

	sim.setInitialConditions();
	recordHistory(sim);
	sim.evolve();

	return sim.userData_;
}

auto phaseFromRelativeState(double wx, double wy) -> double { return std::atan2(-wy, wx); }

auto maxConservationError(SimulationData<DustPureGyromotion> const &data) -> double
{
	double max_err = 0.0;
	for (size_t i = 0; i < data.t_vec_.size(); ++i) {
		max_err = std::max(max_err, std::abs(data.center_momentum_x_vec_[i]));
		max_err = std::max(max_err, std::abs(data.center_momentum_y_vec_[i]));
		max_err = std::max(max_err, std::abs(data.center_momentum_z_vec_[i]));
		max_err = std::max(max_err, std::abs(data.wz_vec_[i]));
	}
	return max_err;
}

auto theoryResolvedDeltaLogAmplitude(ResolvedRkScheme scheme, double theta) -> double
{
	switch (scheme) {
		case ResolvedRkScheme::GL4:
		case ResolvedRkScheme::Midpoint:
			return 0.0;
		case ResolvedRkScheme::TP2025:
			return -std::pow(theta, 4) / 192.0;
	}
	return 0.0;
}

auto theoryResolvedDeltaPhase(ResolvedRkScheme scheme, double theta) -> double
{
	switch (scheme) {
		case ResolvedRkScheme::GL4:
			return -std::pow(theta, 5) / 11520.0;
		case ResolvedRkScheme::Midpoint:
			return -std::pow(theta, 3) / 48.0;
		case ResolvedRkScheme::TP2025:
			return -std::pow(theta, 5) / 720.0;
	}
	return 0.0;
}

auto theoryStiffDeltaPhase(double theta) -> double
{
	const double arg_g = 2.0 * (std::atan2(theta, 1.0 - theta * theta / 2.0) - std::atan(theta / 2.0));
	return std::remainder(arg_g - theta, 2.0 * std::numbers::pi);
}

auto theoryStiffDeltaLogAmplitude(double theta) -> double { return std::log((1.0 + theta * theta / 4.0) / (1.0 + theta * theta * theta * theta / 4.0)); }

auto computeGyroSample(ResolvedRkScheme scheme, double requested_dt) -> GyroSample
{
	SimulationData<DustPureGyromotion> const data = runGyroSimulation(scheme, requested_dt);
	double effective_dt = 0.0;
	if (data.t_vec_.size() >= 2) {
		effective_dt = data.t_vec_[1] - data.t_vec_[0];
	}
	const double end_time = data.t_vec_.empty() ? 0.0 : data.t_vec_.back();
	const size_t last = data.t_vec_.size() - 1;
	const double wx = data.wx_vec_[last];
	const double wy = data.wy_vec_[last];
	const double amplitude_ratio = std::sqrt(wx * wx + wy * wy) / initial_relative_drift;
	const double delta_log_amplitude = std::log(amplitude_ratio);
	const double theta = omega_rel * end_time;
	const double numerical_phase = phaseFromRelativeState(wx, wy);
	const double delta_phase = std::remainder(numerical_phase - theta, 2.0 * std::numbers::pi);
	const bool used_resolved_branch = usesResolvedBranch(effective_dt);
	const double theory_delta_log_amplitude = used_resolved_branch ? theoryResolvedDeltaLogAmplitude(scheme, theta) : theoryStiffDeltaLogAmplitude(theta);
	const double theory_delta_phase = used_resolved_branch ? theoryResolvedDeltaPhase(scheme, theta) : theoryStiffDeltaPhase(theta);

	return GyroSample{
	    .requested_dt = requested_dt,
	    .effective_dt = effective_dt,
	    .end_time = end_time,
	    .theta = theta,
	    .amplitude_ratio = amplitude_ratio,
	    .delta_log_amplitude = delta_log_amplitude,
	    .delta_phase = delta_phase,
	    .abs_delta_log_amplitude = std::max(std::abs(delta_log_amplitude), plot_floor),
	    .abs_delta_phase = std::max(std::abs(delta_phase), plot_floor),
	    .theory_delta_log_amplitude = theory_delta_log_amplitude,
	    .theory_delta_phase = theory_delta_phase,
	    .conservation_error = maxConservationError(data),
	    .used_resolved_branch = used_resolved_branch,
	};
}

auto runSchemeSweep(ResolvedRkScheme scheme) -> SchemeSweepResult
{
	SchemeSweepResult result{.scheme = scheme, .samples = {}};
	result.samples.reserve(requested_dt_values.size());
	for (double const requested_dt : requested_dt_values) {
		result.samples.push_back(computeGyroSample(scheme, requested_dt));
	}
	return result;
}

void writeSweepCsv(std::vector<SchemeSweepResult> const &runs)
{
	std::ofstream file("dust_gyromotion_diagnostics.csv");
	file << std::setprecision(17);
	file << "scheme,requested_dt,effective_dt,end_time,theta,amplitude_ratio,delta_log_amplitude,delta_phase,abs_delta_log_amplitude,abs_delta_phase,"
		"theory_delta_log_amplitude,theory_delta_phase,conservation_error,used_resolved_branch,resolved_stiff_boundary_dt,plot_floor\n";
	for (auto const &run : runs) {
		for (auto const &sample : run.samples) {
			file << resolvedRkSchemeSlug(run.scheme) << "," << sample.requested_dt << "," << sample.effective_dt << "," << sample.end_time << ","
			     << sample.theta << "," << sample.amplitude_ratio << "," << sample.delta_log_amplitude << "," << sample.delta_phase << ","
			     << sample.abs_delta_log_amplitude << "," << sample.abs_delta_phase << "," << sample.theory_delta_log_amplitude << ","
			     << sample.theory_delta_phase << "," << sample.conservation_error << "," << (sample.used_resolved_branch ? 1 : 0) << ","
			     << resolvedBranchThresholdDt() << "," << plot_floor << "\n";
		}
	}
}

auto problem_main() -> int
{
	bool write_csv = true;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);

	std::vector<SchemeSweepResult> runs;
	runs.reserve(resolved_rk_schemes.size());
	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		runs.push_back(runSchemeSweep(scheme));
	}

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const double conservation_tol = 1.0e-12;
		bool passed = true;

		amrex::Print() << "\nPure gyromotion diagnostic parameters:\n";
		amrex::Print() << "  omega_L = " << omega_L << "\n";
		amrex::Print() << "  omega_rel = " << omega_rel << "\n";
		amrex::Print() << "  initial relative drift = " << initial_relative_drift << "\n";
		amrex::Print() << "  resolved/stiff threshold dt = " << resolvedBranchThresholdDt() << "\n";
		amrex::Print() << "\nPure gyromotion one-step diagnostics:\n";

		for (auto const &run : runs) {
			for (auto const &sample : run.samples) {
				amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] requested dt = " << sample.requested_dt
					       << ", effective dt = " << sample.effective_dt << ", theta = " << sample.theta
					       << ", branch = " << (sample.used_resolved_branch ? "resolved" : "stiff")
					       << ", amplitude ratio = " << sample.amplitude_ratio << ", delta log amplitude = " << sample.delta_log_amplitude
					       << ", delta phase = " << sample.delta_phase;
				if (std::isfinite(sample.theory_delta_log_amplitude)) {
					amrex::Print() << ", theory delta log amplitude = " << sample.theory_delta_log_amplitude;
				}
				if (std::isfinite(sample.theory_delta_phase)) {
					amrex::Print() << ", theory delta phase = " << sample.theory_delta_phase;
				}
				amrex::Print() << ", conservation error = " << sample.conservation_error << "\n";

				if (!std::isfinite(sample.delta_log_amplitude) || !std::isfinite(sample.delta_phase) ||
				    !std::isfinite(sample.conservation_error) || (sample.conservation_error > conservation_tol)) {
					passed = false;
				}
			}
		}

		if (!passed) {
			status = 1;
			amrex::Print() << "\nTest FAILED: pure gyromotion diagnostics produced invalid values.\n";
		} else {
			amrex::Print() << "\nTest PASSED: pure gyromotion diagnostics completed with finite values.\n";
		}
		if (write_csv) {
			writeSweepCsv(runs);
		}
	}

	return status;
}
