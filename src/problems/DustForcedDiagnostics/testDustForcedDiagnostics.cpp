/// \file testDustForcedDiagnostics.cpp
/// \brief Forced Hall/Pedersen diagnostics separating transient and terminal-drift errors.
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

struct DustForcedDiagnostics {
};

using DustHallPedersenForcedDiagnostics = DustForcedDiagnostics;

namespace
{
constexpr double rho_gas = 1.0;
constexpr double epsilon = 1.0;
constexpr double rho_dust = epsilon * rho_gas;
constexpr double sound_speed = 1.0;
constexpr double alpha_d = 1.0;
constexpr double dimensionless_charge_to_mass_ratio = 1.0;
constexpr double magnetic_field_z = 1.0;
constexpr double external_force = 1.0;

constexpr double sweep_stop_time = 20.0;
constexpr int minimum_timesteps = 2;
constexpr int theory_sample_count = 401;

constexpr double omega_L = dimensionless_charge_to_mass_ratio * magnetic_field_z;
constexpr double alpha_rel = (1.0 + epsilon) * alpha_d;
constexpr double omega_rel = (1.0 + epsilon) * omega_L;
constexpr double g_rel_x = -external_force / rho_gas;
using ResolvedRkScheme = quokka::dust::ResolvedRkScheme;
using Complex = std::complex<double>;

// Sweep from resolved to stiff timesteps for comparison with the analytic
// forced-response diagnostics.
constexpr double half_decade_factor = 3.1622776601683795;
constexpr std::array<double, 13> requested_dt_values = {
    1.0e-3, half_decade_factor * 1.0e-3, 1.0e-2, half_decade_factor * 1.0e-2, 1.0e-1, half_decade_factor * 1.0e-1, 1.0e0, half_decade_factor * 1.0e0,
    1.0e1,  half_decade_factor * 1.0e1,	 1.0e2,	 half_decade_factor * 1.0e2,  1.0e3};
constexpr std::array<ResolvedRkScheme, 3> resolved_rk_schemes = {ResolvedRkScheme::TP2025, ResolvedRkScheme::GL4, ResolvedRkScheme::Midpoint};
constexpr double plot_floor = 1.0e-16;

struct DriftState {
	double ux;
	double uy;
};

struct SweepSample {
	double requested_dt;
	double effective_dt;
	double end_time;
	double transient_l2_error;
	double transient_final_error;
	double terminal_error;
	double final_data_error;
	double final_to_fixed_point_error;
	double predicted_final_to_fixed_point_error;
	double predicted_final_data_error;
	double final_state_map_consistency_error;
	double momentum_residual;
	bool used_resolved_branch;
};

struct SchemeSweepResult {
	ResolvedRkScheme scheme;
	std::vector<SweepSample> samples;
};

constexpr Complex lambda_rel{-alpha_rel, omega_rel};
constexpr Complex forcing_rel{g_rel_x, 0.0};
constexpr Complex initial_rel{0.0, 0.0};

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

template <> struct SimulationData<DustHallPedersenForcedDiagnostics> {
	std::vector<double> t_vec_;
	std::vector<double> ux_vec_;
	std::vector<double> uy_vec_;
	std::vector<double> uz_vec_;
	std::vector<double> center_momentum_x_vec_;
	std::vector<double> center_momentum_y_vec_;
	std::vector<double> center_momentum_z_vec_;
};

template <> struct quokka::EOS_Traits<DustHallPedersenForcedDiagnostics> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = sound_speed;
};

template <> struct Physics_Traits<DustHallPedersenForcedDiagnostics> {
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
	static constexpr ResistivityModel resistivity_model = ResistivityModel::none;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustHallPedersenForcedDiagnostics>::ComputeReciprocalStoppingTime(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 1> alpha{};
	alpha[0] = alpha_d;
	return alpha;
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustHallPedersenForcedDiagnostics>::ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 1> dimensionless_charge_to_mass_ratio_array{};
	dimensionless_charge_to_mass_ratio_array[0] = dimensionless_charge_to_mass_ratio;
	return dimensionless_charge_to_mass_ratio_array;
}

template <> void QuokkaSimulation<DustHallPedersenForcedDiagnostics>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<DustHallPedersenForcedDiagnostics>::nvarTotal_cc;
	const double magnetic_energy = 0.5 * magnetic_field_z * magnetic_field_z;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::density_index) = rho_gas;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::energy_index) = magnetic_energy;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x3Momentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::dustDensity_index) = rho_dust;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x1DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x2DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x3DustMomentum_index) = 0.0;
	});
}

template <> void QuokkaSimulation<DustHallPedersenForcedDiagnostics>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_fc = Physics_Indices<DustHallPedersenForcedDiagnostics>::nvarPerDim_fc;
	const double bfield = (grid_elem.dir_ == quokka::direction::z) ? magnetic_field_z : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}
		state_fc(i, j, k, Physics_Indices<DustHallPedersenForcedDiagnostics>::mhdFirstIndex) = bfield;
	});
}

void recordHistory(QuokkaSimulation<DustHallPedersenForcedDiagnostics> &sim)
{
	auto [_, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);

	if (!amrex::ParallelDescriptor::IOProcessor()) {
		return;
	}

	auto &data = sim.userData_;
	data.t_vec_.push_back(sim.tNew_[0]);

	const double gas_density = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::density_index)[0];
	const double gas_momentum_x = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::x1Momentum_index)[0];
	const double gas_momentum_y = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::x2Momentum_index)[0];
	const double gas_momentum_z = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::x3Momentum_index)[0];
	const double gas_vx = gas_momentum_x / gas_density;
	const double gas_vy = gas_momentum_y / gas_density;
	const double gas_vz = gas_momentum_z / gas_density;

	const double dust_density = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::dustDensity_index)[0];
	const double dust_momentum_x = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::x1DustMomentum_index)[0];
	const double dust_momentum_y = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::x2DustMomentum_index)[0];
	const double dust_momentum_z = values.at(HydroSystem<DustHallPedersenForcedDiagnostics>::x3DustMomentum_index)[0];
	const double dust_vx = dust_momentum_x / dust_density;
	const double dust_vy = dust_momentum_y / dust_density;
	const double dust_vz = dust_momentum_z / dust_density;

	data.ux_vec_.push_back(dust_vx - gas_vx);
	data.uy_vec_.push_back(dust_vy - gas_vy);
	data.uz_vec_.push_back(dust_vz - gas_vz);
	data.center_momentum_x_vec_.push_back(gas_momentum_x + dust_momentum_x);
	data.center_momentum_y_vec_.push_back(gas_momentum_y + dust_momentum_y);
	data.center_momentum_z_vec_.push_back(gas_momentum_z + dust_momentum_z);
}

template <> void QuokkaSimulation<DustHallPedersenForcedDiagnostics>::computeAfterTimestep() { recordHistory(*this); }

template <>
void QuokkaSimulation<DustHallPedersenForcedDiagnostics>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time,
										amrex::Real dt_lev) // NOLINT
{
	amrex::ignore_unused(lev);
	amrex::ignore_unused(time);

	const double magnetic_energy = 0.5 * magnetic_field_z * magnetic_field_z;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = state(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::density_index);
			const amrex::Real x1mom = state(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x1Momentum_index);
			const amrex::Real x2mom = state(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x2Momentum_index);
			const amrex::Real x3mom = state(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x3Momentum_index);

			const amrex::Real x1mom_new = x1mom + dt_lev * external_force;
			const amrex::Real gas_kinetic_energy_new = (x1mom_new * x1mom_new + x2mom * x2mom + x3mom * x3mom) / (2.0 * rho);
			const amrex::Real Egas_new = magnetic_energy + gas_kinetic_energy_new;

			state(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::energy_index) = Egas_new;
			state(i, j, k, HydroSystem<DustHallPedersenForcedDiagnostics>::internalEnergy_index) = 0.0;
		});
	}
}

auto resolvedBranchThresholdDt() -> double { return 1.0 / std::sqrt(alpha_d * alpha_d + omega_L * omega_L); }

auto usesResolvedBranch(double effective_dt) -> bool { return effective_dt < resolvedBranchThresholdDt(); }

auto runForcedSimulation(ResolvedRkScheme scheme, double constant_dt) -> SimulationData<DustHallPedersenForcedDiagnostics>
{
	const int step_count = std::max(static_cast<int>(std::ceil(sweep_stop_time / constant_dt)), minimum_timesteps);
	const double run_stop_time = static_cast<double>(step_count) * constant_dt;
	amrex::ParmParse pp;
	pp.add("constant_dt", constant_dt);
	pp.add("stop_time", run_stop_time);
	pp.add("max_timesteps", step_count);
	pp.add("suppress_output", 1);
	pp.add("show_performance_hints", 0);

	using quokka::BCType::mathematicalBndryTypes;
	auto BCs_cc = quokka::BC<DustHallPedersenForcedDiagnostics>(quokka::BCType::int_dir);
	auto BCs_fc = quokka::BC_fc<DustHallPedersenForcedDiagnostics>(mathematicalBndryTypes::periodic, mathematicalBndryTypes::periodic,
								       mathematicalBndryTypes::periodic);
	QuokkaSimulation<DustHallPedersenForcedDiagnostics> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	// This is a spatially uniform source-term diagnostic. Very large requested
	// timesteps can generate large uniform center-of-mass velocities from the
	// imposed forcing, which would otherwise trip the hydro CFL retry logic
	// even though transport is physically inactive in this test.
	sim.cflNumber_ = 1.0e12;
	sim.constantDt_ = constant_dt;
	sim.stopTime_ = run_stop_time;
	sim.maxTimesteps_ = step_count;
	sim.dustResolvedRkScheme_ = scheme;
	sim.print_dust_counter_ = false;

	sim.setInitialConditions();
	recordHistory(sim);
	sim.evolve();

	return sim.userData_;
}

auto exactSteadyRelativeDrift() -> Complex { return -forcing_rel / lambda_rel; }

auto exactRelativeDrift(double t) -> Complex { return exactSteadyRelativeDrift() + std::exp(lambda_rel * t) * (initial_rel - exactSteadyRelativeDrift()); }

auto stabilityFunction(ResolvedRkScheme scheme, Complex z, bool use_resolved_branch) -> Complex
{
	if (!use_resolved_branch) {
		return (1.0 - z) / (1.0 - 2.0 * z + 2.0 * z * z);
	}

	switch (scheme) {
		case ResolvedRkScheme::TP2025:
			return (1.0 - z * z / 6.0) / (1.0 - z + z * z / 3.0);
		case ResolvedRkScheme::GL4:
			return (1.0 + z / 2.0 + z * z / 12.0) / (1.0 - z / 2.0 + z * z / 12.0);
		case ResolvedRkScheme::Midpoint:
			return (1.0 + z / 2.0) / (1.0 - z / 2.0);
	}

	return 0.0;
}

auto numericalFixedPoint(ResolvedRkScheme scheme, double effective_dt) -> Complex
{
	const bool use_resolved_branch = usesResolvedBranch(effective_dt);
	const Complex z = 0.5 * effective_dt * lambda_rel;
	const Complex r = stabilityFunction(scheme, z, use_resolved_branch);
	const Complex g = r * r;
	return effective_dt * r * forcing_rel / (1.0 - g);
}

auto advanceDiscreteMapOneStep(ResolvedRkScheme scheme, double dt_step, Complex state_n) -> Complex
{
	const bool use_resolved_branch = usesResolvedBranch(dt_step);
	const Complex z = 0.5 * dt_step * lambda_rel;
	const Complex r = stabilityFunction(scheme, z, use_resolved_branch);
	const Complex g = r * r;
	return g * state_n + dt_step * r * forcing_rel;
}

auto finalNumericalState(SimulationData<DustHallPedersenForcedDiagnostics> const &data) -> Complex
{
	const size_t i = data.t_vec_.size() - 1;
	return {data.ux_vec_[i], -data.uy_vec_[i]};
}

auto predictedDiscreteFinalState(ResolvedRkScheme scheme, double dt, int step_count) -> Complex
{
	Complex state = initial_rel;
	for (int i = 0; i < step_count; ++i) {
		state = advanceDiscreteMapOneStep(scheme, dt, state);
	}
	return state;
}

auto transientRelativeL2Error(SimulationData<DustHallPedersenForcedDiagnostics> const &data, Complex numerical_fixed_point) -> double
{
	double err_sq = 0.0;
	double ref_sq = 0.0;
	for (size_t i = 1; i < data.t_vec_.size(); ++i) {
		const Complex numerical_state{data.ux_vec_[i], -data.uy_vec_[i]};
		const Complex exact_transient = exactRelativeDrift(data.t_vec_[i]) - exactSteadyRelativeDrift();
		const Complex diff = (numerical_state - numerical_fixed_point) - exact_transient;
		err_sq += std::norm(diff);
		ref_sq += std::norm(exact_transient);
	}
	return (ref_sq > 0.0) ? std::sqrt(err_sq / ref_sq) : 1.0;
}

auto finalTransientError(SimulationData<DustHallPedersenForcedDiagnostics> const &data, Complex numerical_fixed_point) -> double
{
	const size_t i = data.t_vec_.size() - 1;
	const Complex numerical_state{data.ux_vec_[i], -data.uy_vec_[i]};
	const Complex exact_transient = exactRelativeDrift(data.t_vec_[i]) - exactSteadyRelativeDrift();
	return std::abs((numerical_state - numerical_fixed_point) - exact_transient);
}

auto terminalDriftError(Complex numerical_fixed_point) -> double { return std::abs(numerical_fixed_point - exactSteadyRelativeDrift()); }

auto finalDataError(SimulationData<DustHallPedersenForcedDiagnostics> const &data) -> double
{
	return std::abs(finalNumericalState(data) - exactSteadyRelativeDrift());
}

auto maxMomentumResidual(SimulationData<DustHallPedersenForcedDiagnostics> const &data) -> double
{
	double max_residual = 0.0;
	for (size_t i = 0; i < data.t_vec_.size(); ++i) {
		const double px_exact = external_force * data.t_vec_[i];
		max_residual = std::max(max_residual, std::abs(data.center_momentum_x_vec_[i] - px_exact));
		max_residual = std::max(max_residual, std::abs(data.center_momentum_y_vec_[i]));
		max_residual = std::max(max_residual, std::abs(data.center_momentum_z_vec_[i]));
		max_residual = std::max(max_residual, std::abs(data.uz_vec_[i]));
	}
	return max_residual;
}

auto computeSweepSample(ResolvedRkScheme scheme, double requested_dt) -> SweepSample
{
	SimulationData<DustHallPedersenForcedDiagnostics> const data = runForcedSimulation(scheme, requested_dt);
	const double effective_dt = data.t_vec_[1] - data.t_vec_[0];
	const double end_time = data.t_vec_.back();
	const Complex numerical_fixed_point = numericalFixedPoint(scheme, effective_dt);
	const Complex predicted_final_state = predictedDiscreteFinalState(scheme, effective_dt, static_cast<int>(data.t_vec_.size()) - 1);
	const Complex final_state = finalNumericalState(data);

	return SweepSample{
	    .requested_dt = requested_dt,
	    .effective_dt = effective_dt,
	    .end_time = end_time,
	    .transient_l2_error = transientRelativeL2Error(data, numerical_fixed_point),
	    .transient_final_error = finalTransientError(data, numerical_fixed_point),
	    .terminal_error = terminalDriftError(numerical_fixed_point),
	    .final_data_error = finalDataError(data),
	    .final_to_fixed_point_error = std::abs(final_state - numerical_fixed_point),
	    .predicted_final_to_fixed_point_error = std::abs(predicted_final_state - numerical_fixed_point),
	    .predicted_final_data_error = std::abs(predicted_final_state - exactSteadyRelativeDrift()),
	    .final_state_map_consistency_error = std::abs(final_state - predicted_final_state),
	    .momentum_residual = maxMomentumResidual(data),
	    .used_resolved_branch = usesResolvedBranch(effective_dt),
	};
}

auto runSchemeSweep(ResolvedRkScheme scheme) -> SchemeSweepResult
{
	SchemeSweepResult result{.scheme = scheme, .samples = {}};
	result.samples.reserve(requested_dt_values.size());
	for (double const requested_dt : requested_dt_values) {
		result.samples.push_back(computeSweepSample(scheme, requested_dt));
	}
	return result;
}

auto safeLogSlope(double x1, double y1, double x2, double y2) -> double
{
	if ((x1 <= 0.0) || (x2 <= 0.0) || (y1 <= 0.0) || (y2 <= 0.0) || !std::isfinite(y1) || !std::isfinite(y2)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	return std::log(y2 / y1) / std::log(x2 / x1);
}

void writeSweepCsv(std::vector<SchemeSweepResult> const &runs)
{
	std::ofstream file("dust_forced_diagnostics.csv");
	file << std::setprecision(17);
	file << "scheme,requested_dt,effective_dt,end_time,transient_l2_error,transient_final_error,terminal_error,final_data_error,"
		"final_to_fixed_point_error,predicted_final_to_fixed_point_error,predicted_final_data_error,"
		"final_state_map_consistency_error,momentum_residual,"
		"used_resolved_branch,resolved_stiff_boundary_dt,plot_floor\n";
	for (auto const &run : runs) {
		for (auto const &sample : run.samples) {
			file << resolvedRkSchemeSlug(run.scheme) << "," << sample.requested_dt << "," << sample.effective_dt << "," << sample.end_time << ","
			     << sample.transient_l2_error << "," << sample.transient_final_error << "," << sample.terminal_error << ","
			     << sample.final_data_error << "," << sample.final_to_fixed_point_error << "," << sample.predicted_final_to_fixed_point_error << ","
			     << sample.predicted_final_data_error << "," << sample.final_state_map_consistency_error << "," << sample.momentum_residual << ","
			     << (sample.used_resolved_branch ? 1 : 0) << "," << resolvedBranchThresholdDt() << "," << plot_floor << "\n";
		}
	}
}

void writeTheoryCsv()
{
	std::ofstream file("dust_forced_diagnostics_theory.csv");
	file << std::setprecision(17);
	file << "scheme,requested_dt,terminal_error,used_resolved_branch\n";
	const double log_dt_min = std::log(requested_dt_values.front());
	const double log_dt_max = std::log(requested_dt_values.back());
	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		for (int i = 0; i < theory_sample_count; ++i) {
			const double fraction = static_cast<double>(i) / static_cast<double>(theory_sample_count - 1);
			const double requested_dt = std::exp(log_dt_min + fraction * (log_dt_max - log_dt_min));
			const Complex numerical_fixed_point = numericalFixedPoint(scheme, requested_dt);
			file << resolvedRkSchemeSlug(scheme) << "," << requested_dt << "," << terminalDriftError(numerical_fixed_point) << ","
			     << (usesResolvedBranch(requested_dt) ? 1 : 0) << "\n";
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
		const DriftState steady{
		    .ux = exactSteadyRelativeDrift().real(),
		    .uy = -exactSteadyRelativeDrift().imag(),
		};
		const double momentum_tol = 1.0e-12;
		const double map_consistency_tol = 1.0e-10;
		const double small_transient_tol = 2.0e-3;
		bool passed = true;

		amrex::Print() << "\nForced Hall/Pedersen analytic steady state:\n";
		amrex::Print() << "  w_x* = " << steady.ux << "\n";
		amrex::Print() << "  w_y* = " << steady.uy << "\n";
		amrex::Print() << "  resolved/stiff threshold dt = " << resolvedBranchThresholdDt() << "\n";
		amrex::Print() << "\nForced Hall/Pedersen diagnostics:\n";

		for (auto const &run : runs) {
			for (size_t i = 0; i < run.samples.size(); ++i) {
				auto const &sample = run.samples[i];
				amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] requested dt = " << sample.requested_dt
					       << ", effective dt = " << sample.effective_dt << ", t_end = " << sample.end_time
					       << ", branch = " << (sample.used_resolved_branch ? "resolved" : "stiff")
					       << ", transient L2 error = " << sample.transient_l2_error
					       << ", transient final error = " << sample.transient_final_error
					       << ", terminal drift error = " << sample.terminal_error << ", final data error = " << sample.final_data_error
					       << ", final-to-fixed-point error = " << sample.final_to_fixed_point_error
					       << ", predicted final-to-fixed-point error = " << sample.predicted_final_to_fixed_point_error
					       << ", predicted final-data error = " << sample.predicted_final_data_error
					       << ", final-state map consistency error = " << sample.final_state_map_consistency_error
					       << ", max momentum residual = " << sample.momentum_residual;
				if (i > 0) {
					auto const &prev = run.samples[i - 1];
					if (prev.used_resolved_branch == sample.used_resolved_branch) {
						amrex::Print()
						    << ", p(transient final) = "
						    << safeLogSlope(prev.requested_dt, prev.transient_final_error, sample.requested_dt,
								    sample.transient_final_error)
						    << ", p(terminal) = "
						    << safeLogSlope(prev.requested_dt, prev.terminal_error, sample.requested_dt, sample.terminal_error)
						    << ", p(final data) = "
						    << safeLogSlope(prev.requested_dt, prev.final_data_error, sample.requested_dt, sample.final_data_error);
					}
				}
				amrex::Print() << "\n";

				if (!std::isfinite(sample.transient_l2_error) || !std::isfinite(sample.transient_final_error) ||
				    !std::isfinite(sample.terminal_error) || !std::isfinite(sample.final_data_error) ||
				    !std::isfinite(sample.final_to_fixed_point_error) || !std::isfinite(sample.predicted_final_to_fixed_point_error) ||
				    !std::isfinite(sample.predicted_final_data_error) || !std::isfinite(sample.final_state_map_consistency_error) ||
				    !std::isfinite(sample.momentum_residual) || (sample.momentum_residual > momentum_tol) ||
				    (sample.final_state_map_consistency_error > map_consistency_tol) ||
				    (sample.predicted_final_to_fixed_point_error > small_transient_tol) ||
				    (sample.final_to_fixed_point_error > small_transient_tol)) {
					passed = false;
				}
			}
		}

		if (!passed) {
			status = 1;
			amrex::Print() << "\nTest FAILED: forced Hall/Pedersen diagnostics produced invalid values.\n";
		} else {
			amrex::Print() << "\nTest PASSED: forced Hall/Pedersen diagnostics completed with finite values.\n";
		}
		if (write_csv) {
			writeSweepCsv(runs);
			writeTheoryCsv();
		}
	}

	return status;
}
