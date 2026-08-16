/// \file testDustDampedGyromotion.cpp
/// \brief Damped dust-gas gyromotion test from Moseley et al. (2023).
///

#include "QuokkaSimulation.hpp"
#include "dust/DustRuntimeParams.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
constexpr double rho_gas = 1.0;
constexpr double epsilon = 1.0;
constexpr double rho_dust = epsilon * rho_gas;
constexpr double sound_speed = 1.0;
constexpr double initial_drift = 10.0 * sound_speed;
constexpr double gamma_iso = 1.0;
constexpr double eta = 9.0 * std::numbers::pi * gamma_iso / 128.0;
constexpr double default_grain_density = 1.0;
constexpr double default_grain_radius = 1.5957691216057308; // sqrt(8 / pi) gives alpha0 = 1 for gamma = rho_g = c_s = rho_gr = 1.
constexpr double dimensionless_charge_to_mass_ratio = 1.0;
constexpr double dynamic_charge_offset = 0.95;
constexpr double default_coefficient_tolerance = 1.0e-6;
constexpr double convergence_coefficient_tolerance = 1.0e-12;

AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 1> g_dust_grain_radius = {default_grain_radius};	  // NOLINT
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 1> g_dust_grain_density = {default_grain_density}; // NOLINT

constexpr double gas_velocity_x0 = -epsilon * initial_drift / (1.0 + epsilon);
constexpr double dust_velocity_x0 = initial_drift / (1.0 + epsilon);

struct DriftState {
	double wx;
	double wy;
};

struct DustGyroHistory {
	std::vector<double> t_vec_;
	std::vector<double> wx_vec_;
	std::vector<double> wy_vec_;
	std::vector<double> wz_vec_;
	std::vector<double> center_momentum_x_vec_;
	std::vector<double> center_momentum_y_vec_;
	std::vector<double> center_momentum_z_vec_;
};

auto computeInitialReciprocalStoppingTime() -> double
{
	return (2.0 * std::numbers::sqrt2 * rho_gas * sound_speed) /
	       (std::sqrt(std::numbers::pi * gamma_iso) * g_dust_grain_radius[0] * g_dust_grain_density[0]);
}

AMREX_GPU_HOST_DEVICE auto dynamicChargeFromDrift(amrex::Real drift) -> amrex::Real
{
	amrex::Real const normalized_drift = drift / initial_drift;
	return normalized_drift - dynamic_charge_offset;
}
} // namespace

struct DustGyroEpsteinNoB {
};

struct DustGyroNoDrag {
};

struct DustGyroEpsteinWithB {
};

struct DustGyroDynamicCharge {
};

namespace
{
template <typename problem_t> struct GyroCaseParams;

template <> struct GyroCaseParams<DustGyroEpsteinNoB> {
	static constexpr bool enable_epstein_drag = true;
	static constexpr double magnetic_field_z = 0.0;
	static constexpr double omega_L = 0.0;
	static constexpr double stop_time = 2.0;
	static constexpr double constant_dt = 0.1;
};

template <> struct GyroCaseParams<DustGyroNoDrag> {
	static constexpr bool enable_epstein_drag = false;
	static constexpr double magnetic_field_z = 5.0;
	static constexpr double omega_L = dimensionless_charge_to_mass_ratio * magnetic_field_z;
	static constexpr double stop_time = 2.0;
	static constexpr double constant_dt = 0.1;
};

template <> struct GyroCaseParams<DustGyroEpsteinWithB> {
	static constexpr bool enable_epstein_drag = true;
	static constexpr double magnetic_field_z = 5.0;
	static constexpr double omega_L = dimensionless_charge_to_mass_ratio * magnetic_field_z;
	static constexpr double stop_time = 2.0;
	static constexpr double constant_dt = 0.1;
};

template <> struct GyroCaseParams<DustGyroDynamicCharge> {
	static constexpr bool enable_epstein_drag = true;
	static constexpr double magnetic_field_z = 5.0;
	static constexpr double stop_time = 2.0;
	static constexpr double constant_dt = 0.1;
};

struct DustGyroEOSTraits {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = gamma_iso;
	static constexpr double cs_isothermal = sound_speed;
};

struct DustGyroPhysicsTraits {
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

using ResolvedRkScheme = quokka::dust::ResolvedRkScheme;

struct SchemeRunResult {
	ResolvedRkScheme scheme;
	DustGyroHistory data;
	double drift_l2_error;
	double amplitude_error;
	double conservation_error;
};

struct CoefficientTreatmentConvergencePoint {
	ResolvedRkScheme scheme{};
	int steps = 0;
	double dt = 0.0;
	double frozen_error = 0.0;
	double frozen_order = 0.0;
	double stage_error = 0.0;
	double stage_order = 0.0;
	double endpoint_error = 0.0;
	double endpoint_order = 0.0;
};

struct IterationStatistics {
	int total = 0;
	int solves = 0;
	int maximum = 0;

	[[nodiscard]] auto average() const -> double { return static_cast<double>(total) / static_cast<double>(solves); }
};

struct EndpointPicardHistory {
	std::vector<double> t;
	std::vector<double> wx;
	std::vector<double> wy;
	IterationStatistics iteration_statistics;
	bool converged = true;
};

struct SchemeEndpointPicardHistory {
	ResolvedRkScheme scheme{};
	EndpointPicardHistory data;
};

struct EndpointPicardResult {
	std::complex<double> endpoint;
	int iterations = 0;
	bool converged = false;
};

struct TwoStageTableau {
	double a11 = 0.0;
	double a12 = 0.0;
	double a21 = 0.0;
	double a22 = 0.0;
	double b1 = 0.0;
	double b2 = 0.0;
};

constexpr std::array<ResolvedRkScheme, 3> resolved_rk_schemes = {ResolvedRkScheme::TP2025, ResolvedRkScheme::GL4, ResolvedRkScheme::Midpoint};

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

auto resolvedRkTableau(ResolvedRkScheme scheme) -> TwoStageTableau
{
	switch (scheme) {
		case ResolvedRkScheme::TP2025:
			return {.a11 = 1.0, .a12 = -0.5, .a21 = 2.0 / 3.0, .a22 = 0.0, .b1 = 1.0, .b2 = 0.0};
		case ResolvedRkScheme::GL4:
			return {
			    .a11 = 0.25, .a12 = 0.25 - std::numbers::sqrt3 / 6.0, .a21 = 0.25 + std::numbers::sqrt3 / 6.0, .a22 = 0.25, .b1 = 0.5, .b2 = 0.5};
		case ResolvedRkScheme::Midpoint:
			return {.a11 = 0.25, .a12 = 0.25, .a21 = 0.25, .a22 = 0.25, .b1 = 0.5, .b2 = 0.5};
	}
	return {};
}
} // namespace

template <> struct SimulationData<DustGyroEpsteinNoB> : DustGyroHistory {
};

template <> struct SimulationData<DustGyroNoDrag> : DustGyroHistory {
};

template <> struct SimulationData<DustGyroEpsteinWithB> : DustGyroHistory {
};

template <> struct SimulationData<DustGyroDynamicCharge> : DustGyroHistory {
};

template <> struct quokka::EOS_Traits<DustGyroEpsteinNoB> : DustGyroEOSTraits {
};

template <> struct quokka::EOS_Traits<DustGyroNoDrag> : DustGyroEOSTraits {
};

template <> struct quokka::EOS_Traits<DustGyroEpsteinWithB> : DustGyroEOSTraits {
};

template <> struct quokka::EOS_Traits<DustGyroDynamicCharge> : DustGyroEOSTraits {
};

template <> struct Physics_Traits<DustGyroEpsteinNoB> : DustGyroPhysicsTraits {
};

template <> struct Physics_Traits<DustGyroNoDrag> : DustGyroPhysicsTraits {
};

template <> struct Physics_Traits<DustGyroEpsteinWithB> : DustGyroPhysicsTraits {
};

template <> struct Physics_Traits<DustGyroDynamicCharge> : DustGyroPhysicsTraits {
};

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto computeDustGyroReciprocalStoppingTime(typename DustSources<problem_t>::DustCoefficientState const &state)
    -> amrex::GpuArray<amrex::Real, 1>
{
	if constexpr (GyroCaseParams<problem_t>::enable_epstein_drag) {
		return DustSources<problem_t>::ComputeReciprocalStoppingTimeKwok(state.rhoGas, state.rhoDust, state.relativeVelocityMagnitude, state.soundSpeed,
										 g_dust_grain_radius, g_dust_grain_density, true);
	} else {
		amrex::GpuArray<amrex::Real, 1> alpha{};
		alpha.fill(0.0);
		return alpha;
	}
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinNoB>::ComputeReciprocalStoppingTime(DustCoefficientState const &state)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroReciprocalStoppingTime<DustGyroEpsteinNoB>(state);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroNoDrag>::ComputeReciprocalStoppingTime(DustCoefficientState const &state)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroReciprocalStoppingTime<DustGyroNoDrag>(state);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinWithB>::ComputeReciprocalStoppingTime(DustCoefficientState const &state)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroReciprocalStoppingTime<DustGyroEpsteinWithB>(state);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroDynamicCharge>::ComputeReciprocalStoppingTime(DustCoefficientState const &state)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroReciprocalStoppingTime<DustGyroDynamicCharge>(state);
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto computeDustGyroDimensionlessChargeToMassRatio() -> amrex::GpuArray<amrex::Real, 1>
{
	amrex::GpuArray<amrex::Real, 1> dimensionless_charge_to_mass_ratio_array{};
	dimensionless_charge_to_mass_ratio_array[0] = dimensionless_charge_to_mass_ratio;
	return dimensionless_charge_to_mass_ratio_array;
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinNoB>::ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroDimensionlessChargeToMassRatio<DustGyroEpsteinNoB>();
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroNoDrag>::ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroDimensionlessChargeToMassRatio<DustGyroNoDrag>();
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinWithB>::ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroDimensionlessChargeToMassRatio<DustGyroEpsteinWithB>();
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroDynamicCharge>::ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const &state)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> charge_to_mass_ratio{};
	charge_to_mass_ratio[0] = dynamicChargeFromDrift(state.relativeVelocityMagnitude[0]);
	return charge_to_mass_ratio;
}

template <typename problem_t> void setDustGyroInitialConditions(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
	const double magnetic_energy = 0.5 * GyroCaseParams<problem_t>::magnetic_field_z * GyroCaseParams<problem_t>::magnetic_field_z;
	const double gas_energy = 0.5 * rho_gas * gas_velocity_x0 * gas_velocity_x0 + magnetic_energy;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		state_cc(i, j, k, HydroSystem<problem_t>::density_index) = rho_gas;
		state_cc(i, j, k, HydroSystem<problem_t>::energy_index) = gas_energy;
		state_cc(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = 0.0; // isothermal setup
		state_cc(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = rho_gas * gas_velocity_x0;
		state_cc(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<problem_t>::dustDensity_index) = rho_dust;
		state_cc(i, j, k, HydroSystem<problem_t>::x1DustMomentum_index) = rho_dust * dust_velocity_x0;
		state_cc(i, j, k, HydroSystem<problem_t>::x2DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<problem_t>::x3DustMomentum_index) = 0.0;
	});
}

template <typename problem_t> void setDustGyroFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
	double bfield = 0.0;
	if (grid_elem.dir_ == quokka::direction::z) {
		bfield = GyroCaseParams<problem_t>::magnetic_field_z;
	}

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}
		state_fc(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) = bfield;
	});
}

template <> void QuokkaSimulation<DustGyroEpsteinNoB>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setDustGyroInitialConditions<DustGyroEpsteinNoB>(grid_elem);
}

template <> void QuokkaSimulation<DustGyroNoDrag>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setDustGyroInitialConditions<DustGyroNoDrag>(grid_elem);
}

template <> void QuokkaSimulation<DustGyroEpsteinWithB>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setDustGyroInitialConditions<DustGyroEpsteinWithB>(grid_elem);
}

template <> void QuokkaSimulation<DustGyroDynamicCharge>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setDustGyroInitialConditions<DustGyroDynamicCharge>(grid_elem);
}

template <> void QuokkaSimulation<DustGyroEpsteinNoB>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setDustGyroFaceVars<DustGyroEpsteinNoB>(grid_elem);
}

template <> void QuokkaSimulation<DustGyroNoDrag>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setDustGyroFaceVars<DustGyroNoDrag>(grid_elem);
}

template <> void QuokkaSimulation<DustGyroEpsteinWithB>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setDustGyroFaceVars<DustGyroEpsteinWithB>(grid_elem);
}

template <> void QuokkaSimulation<DustGyroDynamicCharge>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setDustGyroFaceVars<DustGyroDynamicCharge>(grid_elem);
}

template <typename problem_t> void appendDustGyroHistory(QuokkaSimulation<problem_t> &sim)
{
	auto [_, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto &data = sim.userData_;
		data.t_vec_.push_back(sim.tNew_[0]);

		const double density = values.at(HydroSystem<problem_t>::density_index)[0];
		const double gas_momentum_x = values.at(HydroSystem<problem_t>::x1Momentum_index)[0];
		const double gas_momentum_y = values.at(HydroSystem<problem_t>::x2Momentum_index)[0];
		const double gas_momentum_z = values.at(HydroSystem<problem_t>::x3Momentum_index)[0];
		const double gas_vx = gas_momentum_x / density;
		const double gas_vy = gas_momentum_y / density;
		const double gas_vz = gas_momentum_z / density;
		const double dust_density = values.at(HydroSystem<problem_t>::dustDensity_index)[0];
		const double dust_momentum_x = values.at(HydroSystem<problem_t>::x1DustMomentum_index)[0];
		const double dust_momentum_y = values.at(HydroSystem<problem_t>::x2DustMomentum_index)[0];
		const double dust_momentum_z = values.at(HydroSystem<problem_t>::x3DustMomentum_index)[0];
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
}

template <> void QuokkaSimulation<DustGyroEpsteinNoB>::computeAfterTimestep() { appendDustGyroHistory(*this); }

template <> void QuokkaSimulation<DustGyroNoDrag>::computeAfterTimestep() { appendDustGyroHistory(*this); }

template <> void QuokkaSimulation<DustGyroEpsteinWithB>::computeAfterTimestep() { appendDustGyroHistory(*this); }

template <> void QuokkaSimulation<DustGyroDynamicCharge>::computeAfterTimestep() { appendDustGyroHistory(*this); }

template <typename problem_t> auto makePeriodicFaceBCs() -> amrex::Vector<amrex::BCRec>
{
	const int nvars_fc = Physics_Indices<problem_t>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}
	return BCs_fc;
}

template <typename problem_t>
auto runDustGyroSimulation(ResolvedRkScheme scheme, double constant_dt, double stop_time, bool enable_coefficient_iteration, double coefficient_tolerance)
    -> SimulationData<problem_t>
{
	auto BCs_cc = quokka::BC<problem_t>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	auto BCs_fc = makePeriodicFaceBCs<problem_t>();
	QuokkaSimulation<problem_t> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 1000000.0; // large CFL number to avoid CFL violation
	sim.constantDt_ = constant_dt;
	sim.stopTime_ = stop_time;
	sim.maxTimesteps_ = 10000000;
	sim.dustCoefficientIteration_.enabled = GyroCaseParams<problem_t>::enable_epstein_drag && enable_coefficient_iteration;
	sim.dustCoefficientIteration_.alphaRelativeTolerance = coefficient_tolerance;
	sim.dustCoefficientIteration_.chargeRelativeTolerance = coefficient_tolerance;
	sim.dustResolvedRkScheme_ = scheme;
	sim.print_dust_counter_ = false;

	sim.setInitialConditions();
	appendDustGyroHistory(sim);
	sim.evolve();

	return sim.userData_;
}

auto analyticEpsteinDriftAmplitude(double t) -> double
{
	const double drift_factor = std::sqrt(1.0 + eta * initial_drift * initial_drift / (sound_speed * sound_speed));
	const double alpha0 = computeInitialReciprocalStoppingTime();
	const double tau = (1.0 + epsilon) * alpha0 * t;
	const double numerator = std::sinh(tau) + drift_factor * std::cosh(tau);
	const double denominator = std::cosh(tau) + drift_factor * std::sinh(tau);
	const double ratio = numerator / denominator;
	return sound_speed * std::sqrt(std::max(ratio * ratio - 1.0, 0.0)) / std::sqrt(eta);
}

auto analyticEpsteinDrift(double t, double omega_L) -> DriftState
{
	const double amplitude = analyticEpsteinDriftAmplitude(t);
	const double phase = (1.0 + epsilon) * omega_L * t;
	return {.wx = amplitude * std::cos(phase), .wy = -amplitude * std::sin(phase)};
}

auto analyticDynamicChargeDrift(double t) -> DriftState
{
	const double amplitude = analyticEpsteinDriftAmplitude(t);
	const double alpha0 = computeInitialReciprocalStoppingTime();
	const double inverse_speed_scale = std::sqrt(eta) / sound_speed;
	const double magnetic_field = GyroCaseParams<DustGyroDynamicCharge>::magnetic_field_z;
	const double phase = magnetic_field / (alpha0 * initial_drift * inverse_speed_scale) *
				 (std::asinh(inverse_speed_scale * initial_drift) - std::asinh(inverse_speed_scale * amplitude)) -
			     (1.0 + epsilon) * magnetic_field * dynamic_charge_offset * t;
	return {.wx = amplitude * std::cos(phase), .wy = -amplitude * std::sin(phase)};
}

auto dynamicChargeRate(std::complex<double> drift) -> std::complex<double>
{
	double const speed = std::abs(drift);
	double const alpha0 = computeInitialReciprocalStoppingTime();
	double const alpha = alpha0 * std::sqrt(1.0 + eta * speed * speed / (sound_speed * sound_speed));
	double const charge = dynamicChargeFromDrift(speed);
	double const response_factor = 1.0 + epsilon;
	return response_factor * std::complex<double>{-alpha, -GyroCaseParams<DustGyroDynamicCharge>::magnetic_field_z * charge};
}

auto epsteinRate(std::complex<double> drift, double omega_L) -> std::complex<double>
{
	double const speed = std::abs(drift);
	double const alpha0 = computeInitialReciprocalStoppingTime();
	double const alpha = alpha0 * std::sqrt(1.0 + eta * speed * speed / (sound_speed * sound_speed));
	return (1.0 + epsilon) * std::complex<double>{-alpha, -omega_L};
}

auto linearTwoStageEndpoint(std::complex<double> initial, double dt, std::complex<double> rate, TwoStageTableau const &tableau) -> std::complex<double>
{
	std::complex<double> const block11 = 1.0 - dt * tableau.a11 * rate;
	std::complex<double> const block12 = -dt * tableau.a12 * rate;
	std::complex<double> const block21 = -dt * tableau.a21 * rate;
	std::complex<double> const block22 = 1.0 - dt * tableau.a22 * rate;
	std::complex<double> const determinant = block11 * block22 - block12 * block21;
	std::complex<double> const stage1 = initial * (block22 - block12) / determinant;
	std::complex<double> const stage2 = initial * (block11 - block21) / determinant;
	return initial + dt * rate * (tableau.b1 * stage1 + tableau.b2 * stage2);
}

auto epsteinCoefficientConverged(std::complex<double> used, std::complex<double> updated, double tolerance) -> bool
{
	double const alpha0 = computeInitialReciprocalStoppingTime();
	double const alpha_used = alpha0 * std::sqrt(1.0 + eta * std::norm(used) / (sound_speed * sound_speed));
	double const alpha_updated = alpha0 * std::sqrt(1.0 + eta * std::norm(updated) / (sound_speed * sound_speed));
	return std::abs(alpha_updated - alpha_used) <= tolerance * alpha_used;
}

auto dynamicChargeCoefficientsConverged(std::complex<double> used, std::complex<double> updated, double tolerance) -> bool
{
	double const used_speed = std::abs(used);
	double const updated_speed = std::abs(updated);
	double const alpha0 = computeInitialReciprocalStoppingTime();
	double const alpha_used = alpha0 * std::sqrt(1.0 + eta * used_speed * used_speed / (sound_speed * sound_speed));
	double const alpha_updated = alpha0 * std::sqrt(1.0 + eta * updated_speed * updated_speed / (sound_speed * sound_speed));
	double const charge_used = dynamicChargeFromDrift(used_speed);
	double const charge_updated = dynamicChargeFromDrift(updated_speed);
	bool const charge_sign_changed =
	    (charge_used < 0.0 && charge_updated >= 0.0) || (charge_used > 0.0 && charge_updated <= 0.0) || (charge_used == 0.0 && charge_updated != 0.0);
	bool const alpha_converged = std::abs(alpha_updated - alpha_used) <= tolerance * alpha_used;
	bool const charge_converged = !charge_sign_changed && std::abs(charge_updated - charge_used) <= tolerance * std::abs(charge_used);
	return alpha_converged && charge_converged;
}

template <typename RateFn, typename ConvergenceFn>
auto advanceEndpointPicard(std::complex<double> initial, double dt, ResolvedRkScheme scheme, double tolerance, RateFn rate, ConvergenceFn coefficientsConverged)
    -> EndpointPicardResult
{
	constexpr int max_iterations = 100;
	TwoStageTableau const tableau = resolvedRkTableau(scheme);
	std::complex<double> endpoint = initial;
	for (int iteration = 0; iteration < max_iterations; ++iteration) {
		std::complex<double> const updated = linearTwoStageEndpoint(initial, dt, rate(endpoint), tableau);
		if (coefficientsConverged(endpoint, updated, tolerance)) {
			return {.endpoint = updated, .iterations = iteration + 1, .converged = true};
		}
		endpoint = updated;
	}
	return {.endpoint = endpoint, .iterations = max_iterations, .converged = false};
}

template <typename RateFn, typename ConvergenceFn>
auto integrateEndpointPicardHistory(int full_steps, double full_dt, ResolvedRkScheme scheme, double tolerance, RateFn rate, ConvergenceFn coefficientsConverged)
    -> EndpointPicardHistory
{
	EndpointPicardHistory history{.t = {0.0}, .wx = {initial_drift}, .wy = {0.0}};
	std::complex<double> drift{initial_drift, 0.0};
	for (int step = 0; step < full_steps; ++step) {
		EndpointPicardResult const first_half_step = advanceEndpointPicard(drift, 0.5 * full_dt, scheme, tolerance, rate, coefficientsConverged);
		EndpointPicardResult const second_half_step =
		    advanceEndpointPicard(first_half_step.endpoint, 0.5 * full_dt, scheme, tolerance, rate, coefficientsConverged);
		drift = second_half_step.endpoint;
		history.t.push_back(static_cast<double>(step + 1) * full_dt);
		history.wx.push_back(drift.real());
		history.wy.push_back(drift.imag());
		history.iteration_statistics.total += first_half_step.iterations + second_half_step.iterations;
		history.iteration_statistics.solves += 2;
		history.iteration_statistics.maximum =
		    std::max({history.iteration_statistics.maximum, first_half_step.iterations, second_half_step.iterations});
		history.converged = history.converged && first_half_step.converged && second_half_step.converged;
	}
	return history;
}

auto analyticGyroDrift(double t, double omega_L) -> DriftState
{
	const double phase = (1.0 + epsilon) * omega_L * t;
	return {.wx = initial_drift * std::cos(phase), .wy = -initial_drift * std::sin(phase)};
}

template <typename AnalyticFn> auto relativeDriftL2Error(const DustGyroHistory &data, AnalyticFn analytic) -> double
{
	double err_sq = 0.0;
	double ref_sq = 0.0;
	for (size_t i = 0; i < data.t_vec_.size(); ++i) {
		DriftState const exact = analytic(data.t_vec_[i]);
		const double dx = data.wx_vec_[i] - exact.wx;
		const double dy = data.wy_vec_[i] - exact.wy;
		err_sq += dx * dx + dy * dy;
		ref_sq += exact.wx * exact.wx + exact.wy * exact.wy;
	}
	return (ref_sq > 0.0) ? std::sqrt(err_sq / ref_sq) : 1.0;
}

template <typename AnalyticFn>
auto timeAveragedRelativeDriftError(const std::vector<double> &time, const std::vector<double> &wx, const std::vector<double> &wy, AnalyticFn analytic)
    -> double
{
	double error_integral = 0.0;
	for (size_t i = 1; i < time.size(); ++i) {
		DriftState const exact_left = analytic(time[i - 1]);
		DriftState const exact_right = analytic(time[i]);
		double const error_left = std::hypot(wx[i - 1] - exact_left.wx, wy[i - 1] - exact_left.wy);
		double const error_right = std::hypot(wx[i] - exact_right.wx, wy[i] - exact_right.wy);
		error_integral += 0.5 * (error_left + error_right) * (time[i] - time[i - 1]);
	}
	return error_integral / ((time.back() - time.front()) * initial_drift);
}

template <typename AnalyticFn> auto timeAveragedRelativeDriftError(const DustGyroHistory &data, AnalyticFn analytic) -> double
{
	return timeAveragedRelativeDriftError(data.t_vec_, data.wx_vec_, data.wy_vec_, analytic);
}

template <typename AnalyticFn> auto timeAveragedRelativeDriftError(const EndpointPicardHistory &data, AnalyticFn analytic) -> double
{
	return timeAveragedRelativeDriftError(data.t, data.wx, data.wy, analytic);
}

auto observedOrder(double coarse_error, double fine_error) -> double { return std::log2(coarse_error / fine_error); }

void printEndpointPicardStatistics(std::string_view label, EndpointPicardHistory const &history)
{
	amrex::Print() << "[" << label << "] Endpoint Picard iterations per source half-step: average = " << history.iteration_statistics.average()
		       << ", maximum = " << history.iteration_statistics.maximum << "\n";
}

auto maxRelativeAmplitudeError(const DustGyroHistory &data) -> double
{
	double max_err = 0.0;
	for (size_t i = 0; i < data.t_vec_.size(); ++i) {
		const double amplitude = std::sqrt(data.wx_vec_[i] * data.wx_vec_[i] + data.wy_vec_[i] * data.wy_vec_[i]);
		max_err = std::max(max_err, std::abs(amplitude - initial_drift) / initial_drift);
	}
	return max_err;
}

auto maxAbsVectorComponent(const std::vector<double> &values) -> double
{
	double max_value = 0.0;
	for (double const value : values) {
		max_value = std::max(max_value, std::abs(value));
	}
	return max_value;
}

auto maxConservationError(const DustGyroHistory &data) -> double
{
	return std::max({maxAbsVectorComponent(data.center_momentum_x_vec_), maxAbsVectorComponent(data.center_momentum_y_vec_),
			 maxAbsVectorComponent(data.center_momentum_z_vec_), maxAbsVectorComponent(data.wz_vec_)});
}

auto maxRelativeHistoryDifference(const DustGyroHistory &first, const DustGyroHistory &second) -> double
{
	double max_difference = 0.0;
	for (size_t i = 0; i < first.t_vec_.size(); ++i) {
		max_difference = std::max(max_difference, std::abs(first.wx_vec_[i] - second.wx_vec_[i]) / initial_drift);
		max_difference = std::max(max_difference, std::abs(first.wy_vec_[i] - second.wy_vec_[i]) / initial_drift);
	}
	return max_difference;
}

template <typename problem_t, typename AnalyticFn>
auto computeRunResult(ResolvedRkScheme scheme, SimulationData<problem_t> data, AnalyticFn analytic) -> SchemeRunResult
{
	SchemeRunResult result{
	    .scheme = scheme,
	    .data = std::move(data),
	    .drift_l2_error = 0.0,
	    .amplitude_error = 0.0,
	    .conservation_error = 0.0,
	};
	result.drift_l2_error = relativeDriftL2Error(result.data, analytic);
	result.amplitude_error = maxRelativeAmplitudeError(result.data);
	result.conservation_error = maxConservationError(result.data);
	return result;
}

template <typename AnalyticFn>
void fillDenseDriftXData(const DustGyroHistory &data, AnalyticFn analytic, double x_scale, std::vector<double> &t_dense, std::vector<double> &x_dense,
			 std::vector<double> &wx_dense)
{
	const size_t n_dense = 1000;
	t_dense.resize(n_dense);
	x_dense.resize(n_dense);
	wx_dense.resize(n_dense);
	const double t_max = data.t_vec_.back();
	for (size_t i = 0; i < n_dense; ++i) {
		const double t = t_max * static_cast<double>(i) / static_cast<double>(n_dense - 1);
		DriftState const exact = analytic(t);
		t_dense[i] = t;
		x_dense[i] = x_scale * t;
		wx_dense[i] = exact.wx / initial_drift;
	}
}

template <typename AnalyticFn> void writeDenseExactCsv(const DustGyroHistory &data, AnalyticFn analytic, std::string_view filename, double x_scale)
{
	std::vector<double> t_dense;
	std::vector<double> x_dense;
	std::vector<double> wx_dense;
	fillDenseDriftXData(data, analytic, x_scale, t_dense, x_dense, wx_dense);

	std::ofstream file{std::string(filename)};
	file << std::setprecision(17);
	file << "t,x_plot,wx_exact_norm\n";
	for (size_t i = 0; i < t_dense.size(); ++i) {
		file << t_dense[i] << "," << x_dense[i] << "," << wx_dense[i] << "\n";
	}
}

template <typename AnalyticFn> void writeHistoryCsv(const std::vector<SchemeRunResult> &runs, AnalyticFn analytic, std::string_view filename, double x_scale)
{
	size_t n_samples = runs.front().data.t_vec_.size();
	for (auto const &run : runs) {
		n_samples = std::min(n_samples, run.data.t_vec_.size());
	}

	std::ofstream file{std::string(filename)};
	file << std::setprecision(17);
	file << "t,x_plot";
	for (auto const &run : runs) {
		file << ",wx_" << resolvedRkSchemeSlug(run.scheme) << "_norm";
	}
	file << ",wx_exact_norm\n";

	for (size_t i = 0; i < n_samples; ++i) {
		double const t = runs.front().data.t_vec_[i];
		file << t << "," << x_scale * t;
		for (auto const &run : runs) {
			file << "," << run.data.wx_vec_[i] / initial_drift;
		}
		file << "," << analytic(t).wx / initial_drift << "\n";
	}
}

template <typename AnalyticFn> void writeCaseOutputs(const std::vector<SchemeRunResult> &runs, AnalyticFn analytic, std::string_view case_tag, double x_scale)
{
	std::string const history_filename = "dust_damped_gyromotion_" + std::string(case_tag) + "_history.csv";
	std::string const exact_filename = "dust_damped_gyromotion_" + std::string(case_tag) + "_exact.csv";
	writeHistoryCsv(runs, analytic, history_filename, x_scale);
	writeDenseExactCsv(runs.front().data, analytic, exact_filename, x_scale);
}

void writeSummaryCsv(const std::string_view case_tag, const std::vector<SchemeRunResult> &runs, std::ofstream &file)
{
	for (auto const &run : runs) {
		file << case_tag << "," << resolvedRkSchemeSlug(run.scheme) << "," << run.drift_l2_error << "," << run.amplitude_error << ","
		     << run.conservation_error << "\n";
	}
}

void writeDynamicChargeHistoryCsv(const DustGyroHistory &stage_run, const DustGyroHistory &frozen_run, const EndpointPicardHistory &endpoint_run)
{
	size_t const n_samples = stage_run.t_vec_.size();
	std::ofstream file("dust_dynamic_charge_iteration_history.csv");
	file << std::setprecision(17);
	file << "t,wx_stage_norm,wy_stage_norm,xi_stage,wx_frozen_norm,wy_frozen_norm,xi_frozen,wx_endpoint_norm,wy_endpoint_norm,xi_endpoint\n";
	for (size_t i = 0; i < n_samples; ++i) {
		double const stage_drift = std::hypot(stage_run.wx_vec_[i], stage_run.wy_vec_[i]);
		double const frozen_drift = std::hypot(frozen_run.wx_vec_[i], frozen_run.wy_vec_[i]);
		double const endpoint_drift = std::hypot(endpoint_run.wx[i], endpoint_run.wy[i]);
		file << stage_run.t_vec_[i] << "," << stage_run.wx_vec_[i] / initial_drift << "," << stage_run.wy_vec_[i] / initial_drift << ","
		     << dynamicChargeFromDrift(stage_drift) << "," << frozen_run.wx_vec_[i] / initial_drift << "," << frozen_run.wy_vec_[i] / initial_drift
		     << "," << dynamicChargeFromDrift(frozen_drift) << "," << endpoint_run.wx[i] / initial_drift << "," << endpoint_run.wy[i] / initial_drift
		     << "," << dynamicChargeFromDrift(endpoint_drift) << "\n";
	}
}

void writeDynamicChargeExactCsv(const DustGyroHistory &data)
{
	constexpr size_t n_samples = 1000;
	double const t_max = data.t_vec_.back();
	std::ofstream file("dust_dynamic_charge_iteration_exact.csv");
	file << std::setprecision(17);
	file << "t,wx_exact_norm,wy_exact_norm,xi_exact\n";
	for (size_t i = 0; i < n_samples; ++i) {
		double const t = t_max * static_cast<double>(i) / static_cast<double>(n_samples - 1);
		DriftState const exact = analyticDynamicChargeDrift(t);
		double const drift = std::hypot(exact.wx, exact.wy);
		file << t << "," << exact.wx / initial_drift << "," << exact.wy / initial_drift << "," << dynamicChargeFromDrift(drift) << "\n";
	}
}

void writeDynamicChargeConvergenceCsv(const std::vector<CoefficientTreatmentConvergencePoint> &points)
{
	std::ofstream file("dust_dynamic_charge_convergence.csv");
	file << std::setprecision(17);
	file << "steps,dt,frozen_error,frozen_order,stage_error,stage_order,endpoint_error,endpoint_order\n";
	for (auto const &point : points) {
		file << point.steps << "," << point.dt << "," << point.frozen_error << "," << point.frozen_order << "," << point.stage_error << ","
		     << point.stage_order << "," << point.endpoint_error << "," << point.endpoint_order << "\n";
	}
}

template <typename AnalyticFn>
void writeDynamicEpsteinHistoryCsv(const std::vector<SchemeRunResult> &stage_runs, const std::vector<SchemeRunResult> &frozen_runs,
				   const std::vector<SchemeEndpointPicardHistory> &endpoint_picard_runs, AnalyticFn analytic)
{
	size_t const n_samples = stage_runs.front().data.t_vec_.size();

	std::ofstream file("dust_dynamic_epstein_iteration_history.csv");
	file << std::setprecision(17);
	file << "t,wx_exact_norm,wy_exact_norm";
	for (auto const &run : stage_runs) {
		std::string_view const slug = resolvedRkSchemeSlug(run.scheme);
		file << ",wx_" << slug << "_frozen_norm,wx_" << slug << "_stage_norm,wx_" << slug << "_endpoint_norm,wy_" << slug << "_frozen_norm,wy_" << slug
		     << "_stage_norm,wy_" << slug << "_endpoint_norm";
	}
	file << "\n";

	for (size_t i = 0; i < n_samples; ++i) {
		double const t = stage_runs.front().data.t_vec_[i];
		DriftState const exact = analytic(t);
		file << t << "," << exact.wx / initial_drift << "," << exact.wy / initial_drift;
		for (size_t j = 0; j < stage_runs.size(); ++j) {
			file << "," << frozen_runs[j].data.wx_vec_[i] / initial_drift << "," << stage_runs[j].data.wx_vec_[i] / initial_drift << ","
			     << endpoint_picard_runs[j].data.wx[i] / initial_drift << "," << frozen_runs[j].data.wy_vec_[i] / initial_drift << ","
			     << stage_runs[j].data.wy_vec_[i] / initial_drift << "," << endpoint_picard_runs[j].data.wy[i] / initial_drift;
		}
		file << "\n";
	}
}

void writeDynamicEpsteinConvergenceCsv(const std::vector<CoefficientTreatmentConvergencePoint> &points)
{
	std::ofstream file("dust_dynamic_epstein_iteration_convergence.csv");
	file << std::setprecision(17);
	file << "scheme,steps,dt,frozen_error,frozen_order,stage_error,stage_order,endpoint_error,endpoint_order\n";
	for (auto const &point : points) {
		file << resolvedRkSchemeSlug(point.scheme) << "," << point.steps << "," << point.dt << "," << point.frozen_error << "," << point.frozen_order
		     << "," << point.stage_error << "," << point.stage_order << "," << point.endpoint_error << "," << point.endpoint_order << "\n";
	}
}

auto problem_main() -> int
{
	bool write_csv = true;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);

	quokka::dust::readDustGrainParams(g_dust_grain_radius, g_dust_grain_density);

	auto epstein_no_b_exact = [](double t) { return analyticEpsteinDrift(t, GyroCaseParams<DustGyroEpsteinNoB>::omega_L); };
	auto gyro_no_drag_exact = [](double t) { return analyticGyroDrift(t, GyroCaseParams<DustGyroNoDrag>::omega_L); };
	auto epstein_with_b_exact = [](double t) { return analyticEpsteinDrift(t, GyroCaseParams<DustGyroEpsteinWithB>::omega_L); };
	auto dynamic_charge_exact = [](double t) { return analyticDynamicChargeDrift(t); };
	auto epstein_rate = [](std::complex<double> drift) { return epsteinRate(drift, GyroCaseParams<DustGyroEpsteinWithB>::omega_L); };

	std::vector<SchemeRunResult> epstein_no_b_runs;
	std::vector<SchemeRunResult> gyro_no_drag_runs;
	std::vector<SchemeRunResult> epstein_with_b_stage_runs;
	std::vector<SchemeRunResult> epstein_with_b_frozen_runs;
	std::vector<SchemeEndpointPicardHistory> epstein_with_b_endpoint_picard_runs;
	epstein_no_b_runs.reserve(resolved_rk_schemes.size());
	gyro_no_drag_runs.reserve(resolved_rk_schemes.size());
	epstein_with_b_stage_runs.reserve(resolved_rk_schemes.size());
	epstein_with_b_frozen_runs.reserve(resolved_rk_schemes.size());
	epstein_with_b_endpoint_picard_runs.reserve(resolved_rk_schemes.size());

	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		epstein_no_b_runs.push_back(computeRunResult(scheme,
							     runDustGyroSimulation<DustGyroEpsteinNoB>(scheme, GyroCaseParams<DustGyroEpsteinNoB>::constant_dt,
												       GyroCaseParams<DustGyroEpsteinNoB>::stop_time, true,
												       default_coefficient_tolerance),
							     epstein_no_b_exact));
		gyro_no_drag_runs.push_back(
		    computeRunResult(scheme,
				     runDustGyroSimulation<DustGyroNoDrag>(scheme, GyroCaseParams<DustGyroNoDrag>::constant_dt,
									   GyroCaseParams<DustGyroNoDrag>::stop_time, true, default_coefficient_tolerance),
				     gyro_no_drag_exact));
		epstein_with_b_stage_runs.push_back(computeRunResult(
		    scheme,
		    runDustGyroSimulation<DustGyroEpsteinWithB>(scheme, GyroCaseParams<DustGyroEpsteinWithB>::constant_dt,
								GyroCaseParams<DustGyroEpsteinWithB>::stop_time, true, default_coefficient_tolerance),
		    epstein_with_b_exact));
		epstein_with_b_frozen_runs.push_back(computeRunResult(
		    scheme,
		    runDustGyroSimulation<DustGyroEpsteinWithB>(scheme, GyroCaseParams<DustGyroEpsteinWithB>::constant_dt,
								GyroCaseParams<DustGyroEpsteinWithB>::stop_time, false, default_coefficient_tolerance),
		    epstein_with_b_exact));
		int const epstein_full_steps =
		    static_cast<int>(std::lround(GyroCaseParams<DustGyroEpsteinWithB>::stop_time / GyroCaseParams<DustGyroEpsteinWithB>::constant_dt));
		epstein_with_b_endpoint_picard_runs.push_back(
		    {.scheme = scheme,
		     .data = integrateEndpointPicardHistory(epstein_full_steps, GyroCaseParams<DustGyroEpsteinWithB>::constant_dt, scheme,
							    default_coefficient_tolerance, epstein_rate, epsteinCoefficientConverged)});
	}
	auto dynamic_charge_stage_run =
	    runDustGyroSimulation<DustGyroDynamicCharge>(ResolvedRkScheme::GL4, GyroCaseParams<DustGyroDynamicCharge>::constant_dt,
							 GyroCaseParams<DustGyroDynamicCharge>::stop_time, true, default_coefficient_tolerance);
	auto dynamic_charge_frozen_run =
	    runDustGyroSimulation<DustGyroDynamicCharge>(ResolvedRkScheme::GL4, GyroCaseParams<DustGyroDynamicCharge>::constant_dt,
							 GyroCaseParams<DustGyroDynamicCharge>::stop_time, false, default_coefficient_tolerance);
	int const dynamic_charge_full_steps =
	    static_cast<int>(std::lround(GyroCaseParams<DustGyroDynamicCharge>::stop_time / GyroCaseParams<DustGyroDynamicCharge>::constant_dt));
	auto dynamic_charge_endpoint_picard_run =
	    integrateEndpointPicardHistory(dynamic_charge_full_steps, GyroCaseParams<DustGyroDynamicCharge>::constant_dt, ResolvedRkScheme::GL4,
					   default_coefficient_tolerance, dynamicChargeRate, dynamicChargeCoefficientsConverged);

	constexpr std::array<int, 5> convergence_step_counts = {20, 40, 80, 160, 320};
	constexpr double convergence_stop_time = 2.0;
	std::vector<CoefficientTreatmentConvergencePoint> dynamic_charge_convergence;
	dynamic_charge_convergence.reserve(convergence_step_counts.size());
	bool endpoint_picard_converged = true;
	for (int const steps : convergence_step_counts) {
		double const dt = convergence_stop_time / static_cast<double>(steps);
		auto const stage_run =
		    runDustGyroSimulation<DustGyroDynamicCharge>(ResolvedRkScheme::GL4, dt, convergence_stop_time, true, convergence_coefficient_tolerance);
		auto const frozen_run =
		    runDustGyroSimulation<DustGyroDynamicCharge>(ResolvedRkScheme::GL4, dt, convergence_stop_time, false, convergence_coefficient_tolerance);
		EndpointPicardHistory const endpoint_run = integrateEndpointPicardHistory(steps, dt, ResolvedRkScheme::GL4, convergence_coefficient_tolerance,
											  dynamicChargeRate, dynamicChargeCoefficientsConverged);
		endpoint_picard_converged = endpoint_picard_converged && endpoint_run.converged;
		CoefficientTreatmentConvergencePoint point{.scheme = ResolvedRkScheme::GL4, .steps = steps, .dt = dt};
		if (amrex::ParallelDescriptor::IOProcessor()) {
			point.frozen_error = timeAveragedRelativeDriftError(frozen_run, dynamic_charge_exact);
			point.stage_error = timeAveragedRelativeDriftError(stage_run, dynamic_charge_exact);
			point.endpoint_error = timeAveragedRelativeDriftError(endpoint_run, dynamic_charge_exact);
			if (!dynamic_charge_convergence.empty()) {
				auto const &previous = dynamic_charge_convergence.back();
				point.frozen_order = observedOrder(previous.frozen_error, point.frozen_error);
				point.stage_order = observedOrder(previous.stage_error, point.stage_error);
				point.endpoint_order = observedOrder(previous.endpoint_error, point.endpoint_error);
			}
		}
		dynamic_charge_convergence.push_back(point);
	}

	std::vector<CoefficientTreatmentConvergencePoint> dynamic_epstein_convergence;
	dynamic_epstein_convergence.reserve(resolved_rk_schemes.size() * convergence_step_counts.size());
	bool epstein_endpoint_picard_converged = true;
	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		CoefficientTreatmentConvergencePoint previous{.scheme = scheme};
		for (int const steps : convergence_step_counts) {
			double const dt = convergence_stop_time / static_cast<double>(steps);
			auto const stage_run =
			    runDustGyroSimulation<DustGyroEpsteinWithB>(scheme, dt, convergence_stop_time, true, convergence_coefficient_tolerance);
			auto const frozen_run =
			    runDustGyroSimulation<DustGyroEpsteinWithB>(scheme, dt, convergence_stop_time, false, convergence_coefficient_tolerance);
			EndpointPicardHistory const endpoint_run =
			    integrateEndpointPicardHistory(steps, dt, scheme, convergence_coefficient_tolerance, epstein_rate, epsteinCoefficientConverged);
			epstein_endpoint_picard_converged = epstein_endpoint_picard_converged && endpoint_run.converged;
			CoefficientTreatmentConvergencePoint point{.scheme = scheme, .steps = steps, .dt = dt};
			if (amrex::ParallelDescriptor::IOProcessor()) {
				point.frozen_error = timeAveragedRelativeDriftError(frozen_run, epstein_with_b_exact);
				point.stage_error = timeAveragedRelativeDriftError(stage_run, epstein_with_b_exact);
				point.endpoint_error = timeAveragedRelativeDriftError(endpoint_run, epstein_with_b_exact);
				if (previous.steps > 0) {
					point.frozen_order = observedOrder(previous.frozen_error, point.frozen_error);
					point.stage_order = observedOrder(previous.stage_error, point.stage_error);
					point.endpoint_order = observedOrder(previous.endpoint_error, point.endpoint_error);
				}
			}
			dynamic_epstein_convergence.push_back(point);
			previous = point;
		}
	}

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const double epstein_no_b_tol = 5.0e-2;
		const double gyro_no_drag_tol = 8.0e-2;
		const double gyro_no_drag_midpoint_tol = 2.5e-1;
		const double gyro_amplitude_tol = 0.1;
		const double epstein_with_b_tol = 8.0e-2;
		const double conservation_tol = 1.0e-10;
		const double dynamic_charge_drift_tol = 1.0e-3;
		const double dynamic_charge_minimum_change = 1.0e-2;
		const double dynamic_charge_minimum_order = 3.8;

		bool passed = true;
		for (auto const &run : epstein_no_b_runs) {
			amrex::Print() << "[Pure Damping][" << quokka::dust::resolvedRkSchemeName(run.scheme)
				       << "] Relative L2 drift error = " << run.drift_l2_error << "\n";
			amrex::Print() << "[Pure Damping][" << quokka::dust::resolvedRkSchemeName(run.scheme)
				       << "] Conservation error     = " << run.conservation_error << "\n";
			if ((run.drift_l2_error > epstein_no_b_tol) || (run.conservation_error > conservation_tol)) {
				passed = false;
			}
		}

		for (auto const &run : gyro_no_drag_runs) {
			const double drift_tol = (run.scheme == ResolvedRkScheme::Midpoint) ? gyro_no_drag_midpoint_tol : gyro_no_drag_tol;
			amrex::Print() << "[Undamped Gyromotion][" << quokka::dust::resolvedRkSchemeName(run.scheme)
				       << "] Relative L2 drift error = " << run.drift_l2_error << "\n";
			amrex::Print() << "[Undamped Gyromotion][" << quokka::dust::resolvedRkSchemeName(run.scheme)
				       << "] Relative amplitude error = " << run.amplitude_error << "\n";
			amrex::Print() << "[Undamped Gyromotion][" << quokka::dust::resolvedRkSchemeName(run.scheme)
				       << "] Conservation error       = " << run.conservation_error << "\n";
			if ((run.drift_l2_error > drift_tol) || (run.amplitude_error > gyro_amplitude_tol) || (run.conservation_error > conservation_tol)) {
				passed = false;
			}
		}

		for (auto const &run : epstein_with_b_stage_runs) {
			amrex::Print() << "[Damped Gyromotion][" << quokka::dust::resolvedRkSchemeName(run.scheme)
				       << "] Relative L2 drift error = " << run.drift_l2_error << "\n";
			amrex::Print() << "[Damped Gyromotion][" << quokka::dust::resolvedRkSchemeName(run.scheme)
				       << "] Conservation error     = " << run.conservation_error << "\n";
			if ((run.drift_l2_error > epstein_with_b_tol) || (run.conservation_error > conservation_tol)) {
				passed = false;
			}
		}
		for (auto const &run : epstein_with_b_frozen_runs) {
			if (run.conservation_error > conservation_tol) {
				passed = false;
			}
		}
		for (auto const &run : epstein_with_b_endpoint_picard_runs) {
			if (!run.data.converged) {
				passed = false;
			}
			if (run.scheme == ResolvedRkScheme::GL4) {
				printEndpointPicardStatistics("Dynamic Epstein", run.data);
			}
		}
		for (auto const &point : dynamic_epstein_convergence) {
			amrex::Print() << "[Dynamic Epstein Convergence][" << quokka::dust::resolvedRkSchemeName(point.scheme) << "] steps = " << point.steps
				       << ", frozen error/order = " << point.frozen_error << "/" << point.frozen_order
				       << ", stage error/order = " << point.stage_error << "/" << point.stage_order
				       << ", endpoint error/order = " << point.endpoint_error << "/" << point.endpoint_order << "\n";
			if (point.steps == convergence_step_counts.back()) {
				double const minimum_stage_order = (point.scheme == ResolvedRkScheme::GL4) ? 3.8 : 1.8;
				if ((point.stage_order < minimum_stage_order) || (point.stage_error >= point.endpoint_error) ||
				    (point.stage_error >= point.frozen_error)) {
					passed = false;
				}
			}
		}
		if (!epstein_endpoint_picard_converged) {
			passed = false;
		}

		double const dynamic_charge_drift_error = relativeDriftL2Error(dynamic_charge_stage_run, dynamic_charge_exact);
		double const dynamic_charge_frozen_drift_error = relativeDriftL2Error(dynamic_charge_frozen_run, dynamic_charge_exact);
		double const dynamic_charge_difference = maxRelativeHistoryDifference(dynamic_charge_stage_run, dynamic_charge_frozen_run);
		double const final_drift = std::hypot(dynamic_charge_stage_run.wx_vec_.back(), dynamic_charge_stage_run.wy_vec_.back());
		double const initial_charge = dynamicChargeFromDrift(initial_drift);
		double const final_charge = dynamicChargeFromDrift(final_drift);
		double const dynamic_charge_change = std::abs(final_charge - initial_charge);
		bool const dynamic_charge_sign_changed = initial_charge * final_charge < 0.0;
		double const dynamic_charge_conservation_error = maxConservationError(dynamic_charge_stage_run);
		double const dynamic_charge_frozen_conservation_error = maxConservationError(dynamic_charge_frozen_run);
		printEndpointPicardStatistics("Dynamic Charge", dynamic_charge_endpoint_picard_run);
		amrex::Print() << "[Dynamic Charge] Stage-consistent analytic drift error = " << dynamic_charge_drift_error << "\n";
		amrex::Print() << "[Dynamic Charge] Frozen-coefficient drift error    = " << dynamic_charge_frozen_drift_error << "\n";
		amrex::Print() << "[Dynamic Charge] Stage/frozen solution difference  = " << dynamic_charge_difference << "\n";
		amrex::Print() << "[Dynamic Charge] Initial/final charge-to-mass ratio = " << initial_charge << ", " << final_charge << "\n";
		amrex::Print() << "[Dynamic Charge] Charge-to-mass ratio change      = " << dynamic_charge_change << "\n";
		amrex::Print() << "[Dynamic Charge] Conservation error               = " << dynamic_charge_conservation_error << "\n";
		amrex::Print() << "[Dynamic Charge] Frozen-coefficient conservation error = " << dynamic_charge_frozen_conservation_error << "\n";
		for (auto const &point : dynamic_charge_convergence) {
			amrex::Print() << "[Dynamic Charge Convergence] steps = " << point.steps << ", frozen error/order = " << point.frozen_error << "/"
				       << point.frozen_order << ", stage error/order = " << point.stage_error << "/" << point.stage_order
				       << ", endpoint error/order = " << point.endpoint_error << "/" << point.endpoint_order << "\n";
		}
		if (!dynamic_charge_endpoint_picard_run.converged || !dynamic_charge_sign_changed || (dynamic_charge_drift_error > dynamic_charge_drift_tol) ||
		    (dynamic_charge_change < dynamic_charge_minimum_change) || (dynamic_charge_conservation_error > conservation_tol) ||
		    !endpoint_picard_converged || (dynamic_charge_convergence.back().stage_order < dynamic_charge_minimum_order) ||
		    (dynamic_charge_convergence.back().stage_error >= dynamic_charge_convergence.back().endpoint_error) ||
		    (dynamic_charge_convergence.back().stage_error >= dynamic_charge_convergence.back().frozen_error)) {
			passed = false;
		}

		if (!passed) {
			status = 1;
			amrex::Print() << "\nTest FAILED: dust-gas gyromotion errors exceeded tolerance.\n";
		} else {
			amrex::Print() << "\nTest PASSED: dust-gas gyromotion matches analytic solutions.\n";
		}
		if (write_csv) {
			const double alpha0 = computeInitialReciprocalStoppingTime();
			writeCaseOutputs(epstein_no_b_runs, epstein_no_b_exact, "pure_damping", alpha0);
			writeCaseOutputs(gyro_no_drag_runs, gyro_no_drag_exact, "undamped_gyromotion", GyroCaseParams<DustGyroNoDrag>::omega_L);
			writeCaseOutputs(epstein_with_b_stage_runs, epstein_with_b_exact, "damped_gyromotion", alpha0);
			writeDynamicChargeHistoryCsv(dynamic_charge_stage_run, dynamic_charge_frozen_run, dynamic_charge_endpoint_picard_run);
			writeDynamicChargeExactCsv(dynamic_charge_stage_run);
			writeDynamicChargeConvergenceCsv(dynamic_charge_convergence);
			writeDynamicEpsteinHistoryCsv(epstein_with_b_stage_runs, epstein_with_b_frozen_runs, epstein_with_b_endpoint_picard_runs,
						      epstein_with_b_exact);
			writeDynamicEpsteinConvergenceCsv(dynamic_epstein_convergence);
			std::ofstream summary_file("dust_damped_gyromotion_summary.csv");
			summary_file << std::setprecision(17);
			summary_file << "case,scheme,drift_l2_error,amplitude_error,conservation_error\n";
			writeSummaryCsv("pure_damping", epstein_no_b_runs, summary_file);
			writeSummaryCsv("undamped_gyromotion", gyro_no_drag_runs, summary_file);
			writeSummaryCsv("damped_gyromotion", epstein_with_b_stage_runs, summary_file);
		}
	}

	return status;
}
