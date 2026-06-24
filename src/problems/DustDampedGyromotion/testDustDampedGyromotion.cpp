/// \file testDustDampedGyromotion.cpp
/// \brief Damped dust-gas gyromotion test from Moseley et al. (2022).
///

#include "QuokkaSimulation.hpp"
#include "dust/DustRuntimeParams.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

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
constexpr double charge_to_mass_ratio = 1.0;

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
} // namespace

struct DustGyroEpsteinNoB {
};

struct DustGyroNoDrag {
};

struct DustGyroEpsteinWithB {
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
	static constexpr double omega_L = charge_to_mass_ratio * magnetic_field_z;
	static constexpr double stop_time = 2.0;
	static constexpr double constant_dt = 0.1;
};

template <> struct GyroCaseParams<DustGyroEpsteinWithB> {
	static constexpr bool enable_epstein_drag = true;
	static constexpr double magnetic_field_z = 5.0;
	static constexpr double omega_L = charge_to_mass_ratio * magnetic_field_z;
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
} // namespace

template <> struct SimulationData<DustGyroEpsteinNoB> : DustGyroHistory {
};

template <> struct SimulationData<DustGyroNoDrag> : DustGyroHistory {
};

template <> struct SimulationData<DustGyroEpsteinWithB> : DustGyroHistory {
};

template <> struct quokka::EOS_Traits<DustGyroEpsteinNoB> : DustGyroEOSTraits {
};

template <> struct quokka::EOS_Traits<DustGyroNoDrag> : DustGyroEOSTraits {
};

template <> struct quokka::EOS_Traits<DustGyroEpsteinWithB> : DustGyroEOSTraits {
};

template <> struct Physics_Traits<DustGyroEpsteinNoB> : DustGyroPhysicsTraits {
};

template <> struct Physics_Traits<DustGyroNoDrag> : DustGyroPhysicsTraits {
};

template <> struct Physics_Traits<DustGyroEpsteinWithB> : DustGyroPhysicsTraits {
};

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto computeDustGyroReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, 1> rho_d,
								 amrex::GpuArray<amrex::Real, 1> rel_vel_mag, double cs) -> amrex::GpuArray<amrex::Real, 1>
{
	if constexpr (GyroCaseParams<problem_t>::enable_epstein_drag) {
		return DustSources<problem_t>::ComputeReciprocalStoppingTimeKwok(rho_g, rho_d, rel_vel_mag, cs, g_dust_grain_radius, g_dust_grain_density,
										 true);
	} else {
		amrex::GpuArray<amrex::Real, 1> alpha{};
		alpha.fill(0.0);
		return alpha;
	}
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinNoB>::ComputeReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
											  amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroReciprocalStoppingTime<DustGyroEpsteinNoB>(rho_g, rho_d, rel_vel_mag, cs);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroNoDrag>::ComputeReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
										      amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroReciprocalStoppingTime<DustGyroNoDrag>(rho_g, rho_d, rel_vel_mag, cs);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinWithB>::ComputeReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
											    amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroReciprocalStoppingTime<DustGyroEpsteinWithB>(rho_g, rho_d, rel_vel_mag, cs);
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto computeDustGyroChargeToMassRatio() -> amrex::GpuArray<amrex::Real, 1>
{
	amrex::GpuArray<amrex::Real, 1> q_over_m{};
	q_over_m[0] = charge_to_mass_ratio;
	return q_over_m;
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinNoB>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroChargeToMassRatio<DustGyroEpsteinNoB>();
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroNoDrag>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroChargeToMassRatio<DustGyroNoDrag>();
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustGyroEpsteinWithB>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return computeDustGyroChargeToMassRatio<DustGyroEpsteinWithB>();
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

template <typename problem_t> auto runDustGyroSimulation() -> SimulationData<problem_t>
{
	auto BCs_cc = quokka::BC<problem_t>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	auto BCs_fc = makePeriodicFaceBCs<problem_t>();
	QuokkaSimulation<problem_t> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 1000000.0; // large CFL number to avoid CFL violation
	sim.constantDt_ = GyroCaseParams<problem_t>::constant_dt;
	sim.stopTime_ = GyroCaseParams<problem_t>::stop_time;
	sim.maxTimesteps_ = 10000000;
	sim.enableIterDustStoptime_ = GyroCaseParams<problem_t>::enable_epstein_drag ? 1 : 0;
	sim.print_dust_counter_ = false;

	sim.setInitialConditions();
	appendDustGyroHistory(sim);
	sim.evolve();

	return sim.userData_;
}

auto analyticEpsteinDrift(double t, double omega_L) -> DriftState
{
	const double drift_factor = std::sqrt(1.0 + eta * initial_drift * initial_drift / (sound_speed * sound_speed));
	const double alpha0 = computeInitialReciprocalStoppingTime();
	const double tau = (1.0 + epsilon) * alpha0 * t;
	const double numerator = std::sinh(tau) + drift_factor * std::cosh(tau);
	const double denominator = std::cosh(tau) + drift_factor * std::sinh(tau);
	const double ratio = numerator / denominator;
	const double amplitude = sound_speed * std::sqrt(std::max(ratio * ratio - 1.0, 0.0)) / std::sqrt(eta);
	const double phase = (1.0 + epsilon) * omega_L * t;
	return {.wx = amplitude * std::cos(phase), .wy = -amplitude * std::sin(phase)};
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

template <typename AnalyticFn>
void fillDenseDriftXData(const DustGyroHistory &data, AnalyticFn analytic, double x_scale, std::vector<double> &x_dense, std::vector<double> &wx_dense)
{
	const size_t n_dense = 1000;
	x_dense.resize(n_dense);
	wx_dense.resize(n_dense);
	const double t_max = data.t_vec_.empty() ? 0.0 : data.t_vec_.back();
	for (size_t i = 0; i < n_dense; ++i) {
		const double t = t_max * static_cast<double>(i) / static_cast<double>(n_dense - 1);
		DriftState const exact = analytic(t);
		x_dense[i] = x_scale * t;
		wx_dense[i] = exact.wx / initial_drift;
	}
}

#ifdef HAVE_PYTHON
template <typename AnalyticFn>
void plotDriftX(const DustGyroHistory &data, AnalyticFn analytic, const std::string &filename, const std::string &title, double x_scale,
		const std::string &xlabel)
{
	std::vector<double> x_dense;
	std::vector<double> wx_dense;
	fillDenseDriftXData(data, analytic, x_scale, x_dense, wx_dense);

	std::vector<double> x_num(data.t_vec_.size());
	std::vector<double> wx_num(data.wx_vec_.size());
	for (size_t i = 0; i < data.wx_vec_.size(); ++i) {
		x_num[i] = x_scale * data.t_vec_[i];
		wx_num[i] = data.wx_vec_[i] / initial_drift;
	}

	matplotlibcpp::clf();
	matplotlibcpp::plot(x_dense, wx_dense, {{"label", "analytic"}, {"color", "k"}, {"linestyle", "--"}, {"linewidth", "1.0"}});
	matplotlibcpp::plot(x_num, wx_num, {{"label", "numerical"}, {"color", "C0"}, {"linestyle", "None"}, {"marker", "o"}, {"markersize", "3"}});
	matplotlibcpp::legend();
	matplotlibcpp::xlabel(xlabel);
	matplotlibcpp::ylabel(R"($w_x/w_0$)");
	matplotlibcpp::title(title);
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(filename);
}
#endif

auto problem_main() -> int
{
	quokka::dust::readDustGrainParams(g_dust_grain_radius, g_dust_grain_density);

	auto epstein_no_b = runDustGyroSimulation<DustGyroEpsteinNoB>();
	auto gyro_no_drag = runDustGyroSimulation<DustGyroNoDrag>();
	auto epstein_with_b = runDustGyroSimulation<DustGyroEpsteinWithB>();

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto epstein_no_b_exact = [](double t) { return analyticEpsteinDrift(t, GyroCaseParams<DustGyroEpsteinNoB>::omega_L); };
		auto gyro_no_drag_exact = [](double t) { return analyticGyroDrift(t, GyroCaseParams<DustGyroNoDrag>::omega_L); };
		auto epstein_with_b_exact = [](double t) { return analyticEpsteinDrift(t, GyroCaseParams<DustGyroEpsteinWithB>::omega_L); };

		const double epstein_no_b_error = relativeDriftL2Error(epstein_no_b, epstein_no_b_exact);
		const double gyro_no_drag_error = relativeDriftL2Error(gyro_no_drag, gyro_no_drag_exact);
		const double epstein_with_b_error = relativeDriftL2Error(epstein_with_b, epstein_with_b_exact);
		const double gyro_amplitude_error = maxRelativeAmplitudeError(gyro_no_drag);
		const double epstein_no_b_conservation_error = maxConservationError(epstein_no_b);
		const double gyro_no_drag_conservation_error = maxConservationError(gyro_no_drag);
		const double epstein_with_b_conservation_error = maxConservationError(epstein_with_b);

		const double epstein_no_b_tol = 5.0e-2;
		const double gyro_no_drag_tol = 8.0e-2;
		const double gyro_amplitude_tol = 0.1;
		const double epstein_with_b_tol = 8.0e-2;
		const double conservation_tol = 1.0e-10;

		const bool passed = (epstein_no_b_error <= epstein_no_b_tol) && (gyro_no_drag_error <= gyro_no_drag_tol) &&
				    (gyro_amplitude_error <= gyro_amplitude_tol) && (epstein_with_b_error <= epstein_with_b_tol) &&
				    (epstein_no_b_conservation_error <= conservation_tol) && (gyro_no_drag_conservation_error <= conservation_tol) &&
				    (epstein_with_b_conservation_error <= conservation_tol);

		if (!passed) {
			status = 1;
			amrex::Print() << "\nTest FAILED: dust-gas gyromotion errors exceeded tolerance.\n";
		} else {
			amrex::Print() << "\nTest PASSED: dust-gas gyromotion matches analytic solutions.\n";
		}

#ifdef HAVE_PYTHON
		const double alpha0 = computeInitialReciprocalStoppingTime();
		plotDriftX(epstein_no_b, epstein_no_b_exact, "./dust_gyromotion_PureDamping.pdf", "Pure Damping", alpha0, R"($t/t_{s,0}$)");
		plotDriftX(gyro_no_drag, gyro_no_drag_exact, "./dust_gyromotion_UndampedGyromotion.pdf", "Undamped Gyromotion",
			   GyroCaseParams<DustGyroNoDrag>::omega_L, R"($\omega_L t$)");
		plotDriftX(epstein_with_b, epstein_with_b_exact, "./dust_gyromotion_DampedGyromotion.pdf", "Damped Gyromotion", alpha0, R"($t/t_{s,0}$)");
#endif
	}

	return status;
}
