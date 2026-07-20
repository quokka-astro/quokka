/// \file testDustyAlfvenWave.cpp
/// \brief Circularly polarized dusty Alfven wave test inspired by Moseley et al. (2022).

#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

struct DustyAlfvenWave {
};

namespace
{
constexpr double rho_gas = 1.0;
constexpr double sound_speed = 1.0;
constexpr double u0 = 0.1;
constexpr double domain_length = 1.0;
constexpr double wave_number = 2.0 * std::numbers::pi / domain_length;
constexpr double bz0 = 1.0;
constexpr double alfven_frequency = 2.0 * std::numbers::pi;
constexpr double stop_time_default = 10.0 / alfven_frequency;
constexpr double final_time = 5.0;
constexpr double dust_density_floor = 1.0e-14;
constexpr double initial_b_magnitude = 1.004987562112089; // sqrt(1 + u0^2)
constexpr double sample_z_target = 0.5;
constexpr double time_tolerance = 1.0e-12;
constexpr double advance_tolerance = 1.0e-14;
constexpr int history_stride = 20;
constexpr size_t tracer_particle_count = 128;
constexpr int numerical_tracer_substeps = 8;
constexpr int reference_dense_history_points = 1001;

AMREX_GPU_MANAGED double g_mu = 0.01;			       // NOLINT
AMREX_GPU_MANAGED double g_stopping_time = 0.1;		       // NOLINT
AMREX_GPU_MANAGED double g_omega_l_target = -alfven_frequency; // NOLINT

// input parameter table for a test case
struct CaseConfig {
	std::string sweep_;
	std::string tag_;
	std::string label_;
	double mu_ = 0.0;
	double omega_l_target_ = -alfven_frequency;
	double stopping_time_ = stop_time_default;
};

// historical data
struct CaseHistory {
	double sample_z_ = sample_z_target;
	std::vector<double> t_;
	std::vector<double> dust_vx_;
	std::vector<double> ref_dust_vx_;
	std::vector<double> gas_amp_re_;
	std::vector<double> gas_amp_im_;
	std::vector<double> b_amp_re_;
	std::vector<double> b_amp_im_;
	std::vector<double> gas_vz_mean_;
	std::vector<double> bz_mean_;
	std::vector<double> tracer_dust_vx_;
	std::vector<double> ref_tracer_dust_vx_;
	std::vector<double> ref_tracer_t_dense_;
	std::vector<double> ref_tracer_dust_vx_dense_;
};

} // namespace

template <> struct SimulationData<DustyAlfvenWave> : CaseHistory {
};

template <> struct quokka::EOS_Traits<DustyAlfvenWave> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = sound_speed;
};

template <> struct Physics_Traits<DustyAlfvenWave> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = 0;
	static constexpr bool is_dust_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

namespace
{
// final profile data
struct CaseProfile {
	std::vector<double> z_;
	std::vector<double> gas_vx_;
	std::vector<double> gas_vy_;
	std::vector<double> gas_vz_;
	std::vector<double> dust_vx_;
	std::vector<double> dust_vy_;
	std::vector<double> bx_;
	std::vector<double> by_;
	std::vector<double> bz_;
	std::vector<double> ref_gas_vx_;
	std::vector<double> ref_gas_vy_;
	std::vector<double> ref_dust_vx_;
	std::vector<double> ref_dust_vy_;
	std::vector<double> ref_bx_;
	std::vector<double> ref_by_;
};

// final profile data of tracer particles
struct ParticleProfile {
	std::vector<double> z_num_;
	std::vector<double> gas_vx_num_;
	std::vector<double> dust_vx_num_;
	std::vector<double> z_ref_;
	std::vector<double> gas_vx_ref_;
	std::vector<double> dust_vx_ref_;
};

// 	the complete result package of a case
struct CaseResult {
	CaseConfig config_;
	CaseHistory history_;
	CaseProfile profile_;
	ParticleProfile tracer_profile_;
	double dust_profile_error_ = 0.0;
	double gas_helical_error_ = 0.0;
	double b_helical_error_ = 0.0;
	double max_dust_speed_ = 0.0;
	bool finite_ = true;
};

// state variables of reference solution ODE
struct ReferenceState {
	std::complex<double> gas_perp_;
	std::complex<double> b_perp_;
	std::complex<double> dust_perp_;
	double gas_z_ = 1.0;
	double dust_z_ = 1.0;
};

// the value of the reference solution at a specific z position
// ReferenceState = mode amplitude, ReferencePoint = physical-space value at one z
struct ReferencePoint {
	double gas_vx = 0.0;
	double gas_vy = 0.0;
	double dust_vx = 0.0;
	double dust_vy = 0.0;
	double bx = 0.0;
	double by = 0.0;
};

// helical gas/B field samples at certain time
struct HelicalFieldSample {
	double t_ = 0.0;
	std::complex<double> gas_perp_ = 0.0;
	std::complex<double> b_perp_ = 0.0;
	double gas_z_ = 1.0;
	double b_z_ = 1.0;
};

// the state of a post-processing tracer particle, satisfying dz/dt = v_z, dv/dt = drag + Lorentz
struct TracerState {
	double z_ = 0.0;
	double vx_ = 0.0;
	double vy_ = 0.0;
	double vz_ = 1.0;
};

template <typename T> auto square(T value) -> T { return value * value; }

auto dustDensityFromMu(double mu) -> double { return std::max(mu * rho_gas, dust_density_floor); }

AMREX_GPU_HOST_DEVICE auto chargeToMassRatio(double omega_l_target) -> double { return omega_l_target / initial_b_magnitude; }

auto projectHelicalAmplitude(const std::vector<double> &z, const std::vector<double> &vx, const std::vector<double> &vy) -> std::complex<double>
{
	std::complex<double> amplitude = 0.0;
	for (size_t i = 0; i < z.size(); ++i) {
		const std::complex<double> value(vx[i], vy[i]);
		amplitude += value * std::exp(std::complex<double>(0.0, -wave_number * z[i]));
	}
	if (!z.empty()) {
		amplitude /= static_cast<double>(z.size());
	}
	return amplitude;
}

auto meanValue(const std::vector<double> &values) -> double
{
	if (values.empty()) {
		return 0.0;
	}
	double sum = 0.0;
	for (double const value : values) {
		sum += value;
	}
	return sum / static_cast<double>(values.size());
}

auto referenceRhs(const ReferenceState &state, const CaseConfig &config) -> ReferenceState
{
	const std::complex<double> imaginary(0.0, 1.0);
	const double alpha = 1.0 / config.stopping_time_;
	const double lorentz_qom = chargeToMassRatio(config.omega_l_target_);

	const std::complex<double> w_perp = state.dust_perp_ - state.gas_perp_;
	const double w_z = state.dust_z_ - state.gas_z_;
	const std::complex<double> cross_perp = imaginary * (w_z * state.b_perp_ - bz0 * w_perp);
	const double cross_z = -std::imag(w_perp * std::conj(state.b_perp_));

	const std::complex<double> dust_source_perp = -alpha * w_perp + lorentz_qom * cross_perp;
	const double dust_source_z = -alpha * w_z + lorentz_qom * cross_z;
	const std::complex<double> gas_source_perp = -config.mu_ * dust_source_perp;
	const double gas_source_z = -config.mu_ * dust_source_z;

	ReferenceState rhs;
	rhs.gas_perp_ = imaginary * wave_number * (bz0 * state.b_perp_ - state.gas_z_ * state.gas_perp_) + gas_source_perp;
	rhs.b_perp_ = imaginary * wave_number * (bz0 * state.gas_perp_ - state.gas_z_ * state.b_perp_);
	rhs.dust_perp_ = -imaginary * wave_number * state.dust_z_ * state.dust_perp_ + dust_source_perp;
	rhs.gas_z_ = gas_source_z;
	rhs.dust_z_ = dust_source_z;
	return rhs;
}

auto operator+(const ReferenceState &a, const ReferenceState &b) -> ReferenceState
{
	return {.gas_perp_ = a.gas_perp_ + b.gas_perp_,
		.b_perp_ = a.b_perp_ + b.b_perp_,
		.dust_perp_ = a.dust_perp_ + b.dust_perp_,
		.gas_z_ = a.gas_z_ + b.gas_z_,
		.dust_z_ = a.dust_z_ + b.dust_z_};
}

auto operator*(double scale, const ReferenceState &state) -> ReferenceState
{
	return {.gas_perp_ = scale * state.gas_perp_,
		.b_perp_ = scale * state.b_perp_,
		.dust_perp_ = scale * state.dust_perp_,
		.gas_z_ = scale * state.gas_z_,
		.dust_z_ = scale * state.dust_z_};
}

auto rk4Step(const ReferenceState &state, const CaseConfig &config, double dt) -> ReferenceState
{
	const ReferenceState k1 = referenceRhs(state, config);
	const ReferenceState k2 = referenceRhs(state + (0.5 * dt) * k1, config);
	const ReferenceState k3 = referenceRhs(state + (0.5 * dt) * k2, config);
	const ReferenceState k4 = referenceRhs(state + dt * k3, config);
	return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

auto evaluateReference(const ReferenceState &state, double z) -> ReferencePoint
{
	const std::complex<double> phase = std::exp(std::complex<double>(0.0, wave_number * z));
	const std::complex<double> gas = state.gas_perp_ * phase;
	const std::complex<double> dust = state.dust_perp_ * phase;
	const std::complex<double> bfield = state.b_perp_ * phase;
	return {.gas_vx = std::real(gas),
		.gas_vy = std::imag(gas),
		.dust_vx = std::real(dust),
		.dust_vy = std::imag(dust),
		.bx = std::real(bfield),
		.by = std::imag(bfield)};
}

auto integrateReferenceToTimes(const CaseConfig &config, const std::vector<double> &times, double sample_z, int reference_steps)
    -> std::pair<ReferenceState, std::vector<double>>
{
	ReferenceState state{.gas_perp_ = u0, .b_perp_ = u0, .dust_perp_ = 0.0, .gas_z_ = 1.0, .dust_z_ = 1.0};
	std::vector<double> ref_dust_vx;
	ref_dust_vx.reserve(times.size());

	double t = 0.0;
	const double dt_ref = final_time / static_cast<double>(std::max(reference_steps, 1));
	for (double const target_time : times) {
		while (t + advance_tolerance < target_time) {
			const double dt = std::min(dt_ref, target_time - t);
			state = rk4Step(state, config, dt);
			t += dt;
		}
		const ReferencePoint point = evaluateReference(state, sample_z);
		ref_dust_vx.push_back(point.dust_vx);
	}

	return {state, ref_dust_vx};
}

auto operator+(const TracerState &a, const TracerState &b) -> TracerState
{
	return {.z_ = a.z_ + b.z_, .vx_ = a.vx_ + b.vx_, .vy_ = a.vy_ + b.vy_, .vz_ = a.vz_ + b.vz_};
}

auto operator*(double scale, const TracerState &state) -> TracerState
{
	return {.z_ = scale * state.z_, .vx_ = scale * state.vx_, .vy_ = scale * state.vy_, .vz_ = scale * state.vz_};
}

auto makeFieldSample(double t, const std::complex<double> &gas_perp, const std::complex<double> &b_perp, double gas_z, double b_z) -> HelicalFieldSample
{
	return {.t_ = t, .gas_perp_ = gas_perp, .b_perp_ = b_perp, .gas_z_ = gas_z, .b_z_ = b_z};
}

auto makeNumericalFieldHistory(const CaseHistory &history) -> std::vector<HelicalFieldSample>
{
	std::vector<HelicalFieldSample> samples;
	samples.reserve(history.t_.size());
	for (size_t i = 0; i < history.t_.size(); ++i) {
		samples.push_back(makeFieldSample(history.t_[i], std::complex<double>(history.gas_amp_re_[i], history.gas_amp_im_[i]),
						  std::complex<double>(history.b_amp_re_[i], history.b_amp_im_[i]), history.gas_vz_mean_[i],
						  history.bz_mean_[i]));
	}
	return samples;
}

auto makeInitialTracers() -> std::vector<TracerState>
{
	std::vector<TracerState> particles(tracer_particle_count);
	for (size_t i = 0; i < particles.size(); ++i) {
		particles[i].z_ = static_cast<double>(i) / static_cast<double>(particles.size());
	}
	return particles;
}

auto interpolateFieldHistory(const std::vector<HelicalFieldSample> &samples, double t) -> HelicalFieldSample
{
	AMREX_ALWAYS_ASSERT(!samples.empty());
	if (t <= samples.front().t_) {
		return samples.front();
	}
	if (t >= samples.back().t_) {
		return samples.back();
	}

	size_t upper = 1;
	while (upper < samples.size() && samples[upper].t_ < t) {
		++upper;
	}
	const size_t lower = upper - 1;
	const double dt = samples[upper].t_ - samples[lower].t_;
	const double weight = (dt > 0.0) ? (t - samples[lower].t_) / dt : 0.0;

	return makeFieldSample(t, (1.0 - weight) * samples[lower].gas_perp_ + weight * samples[upper].gas_perp_,
			       (1.0 - weight) * samples[lower].b_perp_ + weight * samples[upper].b_perp_,
			       (1.0 - weight) * samples[lower].gas_z_ + weight * samples[upper].gas_z_,
			       (1.0 - weight) * samples[lower].b_z_ + weight * samples[upper].b_z_);
}

// HelicalFieldSample = mode amplitude at a time point, EvaluatedField = local gas/B at this time point and position
struct EvaluatedField {
	double ux_ = 0.0;
	double uy_ = 0.0;
	double uz_ = 1.0;
	double bx_ = 0.0;
	double by_ = 0.0;
	double bz_ = 1.0;
};

auto evaluateFieldAtPosition(const HelicalFieldSample &sample, double z) -> EvaluatedField
{
	const std::complex<double> phase = std::exp(std::complex<double>(0.0, wave_number * z));
	const std::complex<double> gas = sample.gas_perp_ * phase;
	const std::complex<double> bfield = sample.b_perp_ * phase;
	return {.ux_ = std::real(gas), .uy_ = std::imag(gas), .uz_ = sample.gas_z_, .bx_ = std::real(bfield), .by_ = std::imag(bfield), .bz_ = sample.b_z_};
}

void fillTracerProfileColumns(const std::vector<TracerState> &particles, const HelicalFieldSample &field, std::vector<double> &z_out,
			      std::vector<double> &gas_vx_out, std::vector<double> &dust_vx_out)
{
	std::vector<std::tuple<double, double, double>> rows;
	rows.reserve(particles.size());
	for (TracerState const &particle : particles) {
		const double z_mod = particle.z_ - std::floor(particle.z_);
		const EvaluatedField local = evaluateFieldAtPosition(field, z_mod);
		rows.emplace_back(z_mod, local.ux_, particle.vx_);
	}
	std::sort(rows.begin(), rows.end(), [](auto const &a, auto const &b) { return std::get<0>(a) < std::get<0>(b); });

	z_out.clear();
	gas_vx_out.clear();
	dust_vx_out.clear();
	z_out.reserve(rows.size());
	gas_vx_out.reserve(rows.size());
	dust_vx_out.reserve(rows.size());
	for (auto const &[z, gas_vx, dust_vx] : rows) {
		z_out.push_back(z);
		gas_vx_out.push_back(gas_vx);
		dust_vx_out.push_back(dust_vx);
	}
}

auto tracerRhs(const TracerState &state, const HelicalFieldSample &field, const CaseConfig &config) -> TracerState
{
	const EvaluatedField local = evaluateFieldAtPosition(field, state.z_);
	const double alpha = 1.0 / config.stopping_time_;
	const double qom = chargeToMassRatio(config.omega_l_target_);
	const double wx = state.vx_ - local.ux_;
	const double wy = state.vy_ - local.uy_;
	const double wz = state.vz_ - local.uz_;

	const double cross_x = wy * local.bz_ - wz * local.by_;
	const double cross_y = wz * local.bx_ - wx * local.bz_;
	const double cross_z = wx * local.by_ - wy * local.bx_;

	return {.z_ = state.vz_, .vx_ = -alpha * wx + qom * cross_x, .vy_ = -alpha * wy + qom * cross_y, .vz_ = -alpha * wz + qom * cross_z};
}

void rk4StepTracersWithSampledFields(std::vector<TracerState> &particles, double t, double dt, const std::vector<HelicalFieldSample> &field_history,
				     const CaseConfig &config)
{
	const HelicalFieldSample field1 = interpolateFieldHistory(field_history, t);
	const HelicalFieldSample field2 = interpolateFieldHistory(field_history, t + 0.5 * dt);
	const HelicalFieldSample field3 = field2;
	const HelicalFieldSample field4 = interpolateFieldHistory(field_history, t + dt);

	for (TracerState &particle : particles) {
		const TracerState k1 = tracerRhs(particle, field1, config);
		const TracerState k2 = tracerRhs(particle + (0.5 * dt) * k1, field2, config);
		const TracerState k3 = tracerRhs(particle + (0.5 * dt) * k2, field3, config);
		const TracerState k4 = tracerRhs(particle + dt * k3, field4, config);
		particle = particle + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
	}
}

auto integrateTracerThroughSampledFields(const CaseConfig &config, const std::vector<HelicalFieldSample> &field_history,
					 const std::vector<double> &sample_times) -> std::pair<std::vector<double>, ParticleProfile>
{
	std::vector<TracerState> particles = makeInitialTracers();
	std::vector<double> tracer_dust_vx;
	ParticleProfile profile;
	if (sample_times.empty()) {
		return {tracer_dust_vx, profile};
	}

	double t = sample_times.front();
	tracer_dust_vx.push_back(particles.front().vx_);

	for (size_t i = 0; i + 1 < sample_times.size(); ++i) {
		const double target_time = sample_times[i + 1];
		const int substeps = std::max(numerical_tracer_substeps, 1);
		for (int step = 0; step < substeps; ++step) {
			const double dt = (target_time - t) / static_cast<double>(substeps - step);
			rk4StepTracersWithSampledFields(particles, t, dt, field_history, config);
			t += dt;
		}
		tracer_dust_vx.push_back(particles.front().vx_);
	}

	const HelicalFieldSample field_final = interpolateFieldHistory(field_history, sample_times.back());
	fillTracerProfileColumns(particles, field_final, profile.z_num_, profile.gas_vx_num_, profile.dust_vx_num_);

	return {tracer_dust_vx, profile};
}

void rk4StepReferenceFieldsAndTracers(ReferenceState &field_state, std::vector<TracerState> &particles, const CaseConfig &config, double dt)
{
	const ReferenceState field_k1 = referenceRhs(field_state, config);
	std::vector<TracerState> particle_k1(particles.size());
	for (size_t i = 0; i < particles.size(); ++i) {
		particle_k1[i] = tracerRhs(particles[i], makeFieldSample(0.0, field_state.gas_perp_, field_state.b_perp_, field_state.gas_z_, bz0), config);
	}

	const ReferenceState field_2 = field_state + (0.5 * dt) * field_k1;
	const ReferenceState field_k2 = referenceRhs(field_2, config);
	std::vector<TracerState> particle_2(particles.size());
	std::vector<TracerState> particle_k2(particles.size());
	for (size_t i = 0; i < particles.size(); ++i) {
		particle_2[i] = particles[i] + (0.5 * dt) * particle_k1[i];
		particle_k2[i] = tracerRhs(particle_2[i], makeFieldSample(0.0, field_2.gas_perp_, field_2.b_perp_, field_2.gas_z_, bz0), config);
	}

	const ReferenceState field_3 = field_state + (0.5 * dt) * field_k2;
	const ReferenceState field_k3 = referenceRhs(field_3, config);
	std::vector<TracerState> particle_3(particles.size());
	std::vector<TracerState> particle_k3(particles.size());
	for (size_t i = 0; i < particles.size(); ++i) {
		particle_3[i] = particles[i] + (0.5 * dt) * particle_k2[i];
		particle_k3[i] = tracerRhs(particle_3[i], makeFieldSample(0.0, field_3.gas_perp_, field_3.b_perp_, field_3.gas_z_, bz0), config);
	}

	const ReferenceState field_4 = field_state + dt * field_k3;
	const ReferenceState field_k4 = referenceRhs(field_4, config);
	std::vector<TracerState> particle_k4(particles.size());
	for (size_t i = 0; i < particles.size(); ++i) {
		const TracerState particle_4 = particles[i] + dt * particle_k3[i];
		particle_k4[i] = tracerRhs(particle_4, makeFieldSample(0.0, field_4.gas_perp_, field_4.b_perp_, field_4.gas_z_, bz0), config);
	}

	field_state = field_state + (dt / 6.0) * (field_k1 + 2.0 * field_k2 + 2.0 * field_k3 + field_k4);
	for (size_t i = 0; i < particles.size(); ++i) {
		particles[i] = particles[i] + (dt / 6.0) * (particle_k1[i] + 2.0 * particle_k2[i] + 2.0 * particle_k3[i] + particle_k4[i]);
	}
}

auto integrateReferenceTracerEnsemble(const CaseConfig &config, const std::vector<double> &sample_times, int reference_steps)
    -> std::tuple<std::vector<double>, ParticleProfile, std::vector<double>, std::vector<double>>
{
	ReferenceState field_state{.gas_perp_ = u0, .b_perp_ = u0, .dust_perp_ = 0.0, .gas_z_ = 1.0, .dust_z_ = 1.0};
	std::vector<TracerState> particles = makeInitialTracers();

	std::vector<double> tracer_dust_vx;
	ParticleProfile profile;
	std::vector<double> t_dense;
	std::vector<double> dust_vx_dense;

	double t = 0.0;
	const double dt_ref = final_time / static_cast<double>(std::max(reference_steps, 1));
	size_t sample_index = 0;
	const double dense_dt = final_time / static_cast<double>(std::max(reference_dense_history_points - 1, 1));
	double next_dense_time = 0.0;

	auto recordDensePoint = [&](double time) {
		t_dense.push_back(time);
		dust_vx_dense.push_back(particles.front().vx_);
	};
	recordDensePoint(0.0);

	while (t + advance_tolerance < final_time) {
		double target_time = std::min(t + dt_ref, final_time);
		if (sample_index < sample_times.size()) {
			target_time = std::min(target_time, sample_times[sample_index]);
		}
		target_time = std::min(target_time, next_dense_time + dense_dt);

		const double dt = target_time - t;
		if (dt > 0.0) {
			rk4StepReferenceFieldsAndTracers(field_state, particles, config, dt);
			t = target_time;
		}

		if (sample_index < sample_times.size() && std::abs(t - sample_times[sample_index]) < time_tolerance) {
			tracer_dust_vx.push_back(particles.front().vx_);
			++sample_index;
		}
		if ((t_dense.empty() || std::abs(t - t_dense.back()) > time_tolerance) && std::abs(t - (next_dense_time + dense_dt)) < time_tolerance) {
			recordDensePoint(t);
			next_dense_time = t;
		}
	}

	const HelicalFieldSample field_final = makeFieldSample(final_time, field_state.gas_perp_, field_state.b_perp_, field_state.gas_z_, bz0);
	fillTracerProfileColumns(particles, field_final, profile.z_ref_, profile.gas_vx_ref_, profile.dust_vx_ref_);

	return {tracer_dust_vx, profile, t_dense, dust_vx_dense};
}

auto extractProfile(QuokkaSimulation<DustyAlfvenWave> &sim) -> CaseProfile;

template <typename problem_t> void appendHistory(QuokkaSimulation<problem_t> &sim, bool force = false)
{
	if (!force && (sim.istep[0] % history_stride != 0)) {
		return;
	}

	CaseProfile const profile = extractProfile(sim);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!sim.userData_.t_.empty() && std::abs(sim.userData_.t_.back() - sim.tNew_[0]) < time_tolerance) {
			return;
		}
		size_t sample_index = 0;
		double min_distance = std::numeric_limits<double>::max();
		for (size_t i = 0; i < profile.z_.size(); ++i) {
			const double distance = std::abs(profile.z_[i] - sample_z_target);
			if (distance < min_distance) {
				min_distance = distance;
				sample_index = i;
			}
		}
		const std::complex<double> gas_amp = projectHelicalAmplitude(profile.z_, profile.gas_vx_, profile.gas_vy_);
		const std::complex<double> b_amp = projectHelicalAmplitude(profile.z_, profile.bx_, profile.by_);

		sim.userData_.sample_z_ = profile.z_[sample_index];
		sim.userData_.t_.push_back(sim.tNew_[0]);
		sim.userData_.dust_vx_.push_back(profile.dust_vx_[sample_index]);
		sim.userData_.gas_amp_re_.push_back(std::real(gas_amp));
		sim.userData_.gas_amp_im_.push_back(std::imag(gas_amp));
		sim.userData_.b_amp_re_.push_back(std::real(b_amp));
		sim.userData_.b_amp_im_.push_back(std::imag(b_amp));
		sim.userData_.gas_vz_mean_.push_back(meanValue(profile.gas_vz_));
		sim.userData_.bz_mean_.push_back(meanValue(profile.bz_));
	}
}

auto extractMagneticProfile(QuokkaSimulation<DustyAlfvenWave> &sim) -> amrex::Vector<amrex::Gpu::HostVector<amrex::Real>>
{
	amrex::MultiFab b_cc(sim.state_new_cc_[0].boxArray(), sim.state_new_cc_[0].DistributionMap(), 3, 0);
	auto const &b_cc_arrays = b_cc.arrays();
	auto const &fcx = sim.state_new_fc_[0][0].const_arrays();
	auto const &fcy = sim.state_new_fc_[0][1].const_arrays();
	auto const &fcz = sim.state_new_fc_[0][2].const_arrays();
	amrex::ParallelFor(b_cc, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		b_cc_arrays[bx](i, j, k, 0) =
		    0.5 * (fcx[bx](i, j, k, MHDSystem<DustyAlfvenWave>::bfield_index) + fcx[bx](i + 1, j, k, MHDSystem<DustyAlfvenWave>::bfield_index));
		b_cc_arrays[bx](i, j, k, 1) =
		    0.5 * (fcy[bx](i, j, k, MHDSystem<DustyAlfvenWave>::bfield_index) + fcy[bx](i, j + 1, k, MHDSystem<DustyAlfvenWave>::bfield_index));
		b_cc_arrays[bx](i, j, k, 2) =
		    0.5 * (fcz[bx](i, j, k, MHDSystem<DustyAlfvenWave>::bfield_index) + fcz[bx](i, j, k + 1, MHDSystem<DustyAlfvenWave>::bfield_index));
	});
	auto extracted = fextract(b_cc, sim.Geom(0), 2, 0.5, true);
	return std::move(std::get<1>(extracted));
}

auto extractProfile(QuokkaSimulation<DustyAlfvenWave> &sim) -> CaseProfile
{
	auto [z, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 2, 0.5, true);
	auto const b_values = extractMagneticProfile(sim);

	CaseProfile profile;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &rho_g = values.at(HydroSystem<DustyAlfvenWave>::density_index);
		auto const &mom_x = values.at(HydroSystem<DustyAlfvenWave>::x1Momentum_index);
		auto const &mom_y = values.at(HydroSystem<DustyAlfvenWave>::x2Momentum_index);
		auto const &mom_z = values.at(HydroSystem<DustyAlfvenWave>::x3Momentum_index);
		auto const &rho_d = values.at(HydroSystem<DustyAlfvenWave>::dustDensity_index);
		auto const &dust_mom_x = values.at(HydroSystem<DustyAlfvenWave>::x1DustMomentum_index);
		auto const &dust_mom_y = values.at(HydroSystem<DustyAlfvenWave>::x2DustMomentum_index);
		auto const &bx = b_values.at(0);
		auto const &by = b_values.at(1);
		auto const &bz = b_values.at(2);
		const size_t npts = z.size();

		profile.z_.assign(z.begin(), z.end());
		profile.gas_vx_.resize(npts);
		profile.gas_vy_.resize(npts);
		profile.gas_vz_.resize(npts);
		profile.dust_vx_.resize(npts);
		profile.dust_vy_.resize(npts);
		profile.bx_.resize(npts);
		profile.by_.resize(npts);
		profile.bz_.resize(npts);

		for (size_t i = 0; i < npts; ++i) {
			profile.gas_vx_[i] = mom_x[i] / rho_g[i];
			profile.gas_vy_[i] = mom_y[i] / rho_g[i];
			profile.gas_vz_[i] = mom_z[i] / rho_g[i];
			profile.dust_vx_[i] = dust_mom_x[i] / rho_d[i];
			profile.dust_vy_[i] = dust_mom_y[i] / rho_d[i];
			profile.bx_[i] = bx[i];
			profile.by_[i] = by[i];
			profile.bz_[i] = bz[i];
		}
	}

	return profile;
}

void fillReferenceProfile(CaseProfile &profile, const ReferenceState &reference_state)
{
	profile.ref_gas_vx_.resize(profile.z_.size());
	profile.ref_gas_vy_.resize(profile.z_.size());
	profile.ref_dust_vx_.resize(profile.z_.size());
	profile.ref_dust_vy_.resize(profile.z_.size());
	profile.ref_bx_.resize(profile.z_.size());
	profile.ref_by_.resize(profile.z_.size());

	for (size_t i = 0; i < profile.z_.size(); ++i) {
		const ReferencePoint point = evaluateReference(reference_state, profile.z_[i]);
		profile.ref_gas_vx_[i] = point.gas_vx;
		profile.ref_gas_vy_[i] = point.gas_vy;
		profile.ref_dust_vx_[i] = point.dust_vx;
		profile.ref_dust_vy_[i] = point.dust_vy;
		profile.ref_bx_[i] = point.bx;
		profile.ref_by_[i] = point.by;
	}
}

auto helicalResidual(const std::vector<double> &z, const std::vector<double> &vx, const std::vector<double> &vy) -> double
{
	if (z.empty()) {
		return 0.0;
	}
	const std::complex<double> amplitude = projectHelicalAmplitude(z, vx, vy);

	double residual_sq = 0.0;
	for (size_t i = 0; i < z.size(); ++i) {
		const std::complex<double> model = amplitude * std::exp(std::complex<double>(0.0, wave_number * z[i]));
		const std::complex<double> value(vx[i], vy[i]);
		residual_sq += std::norm(value - model);
	}
	residual_sq /= static_cast<double>(z.size());
	const double norm = std::abs(amplitude);
	return (norm > 0.0) ? std::sqrt(residual_sq) / norm : std::sqrt(residual_sq);
}

auto relativeTransverseError(const std::vector<double> &vx, const std::vector<double> &vy, const std::vector<double> &ref_vx, const std::vector<double> &ref_vy)
    -> double
{
	double err_sq = 0.0;
	double ref_sq = 0.0;
	for (size_t i = 0; i < vx.size(); ++i) {
		err_sq += square(vx[i] - ref_vx[i]) + square(vy[i] - ref_vy[i]);
		ref_sq += square(ref_vx[i]) + square(ref_vy[i]);
	}
	return (ref_sq > 0.0) ? std::sqrt(err_sq / ref_sq) : std::sqrt(err_sq);
}

auto maxDustSpeed(const CaseProfile &profile) -> double
{
	double result = 0.0;
	for (size_t i = 0; i < profile.dust_vx_.size(); ++i) {
		result = std::max(result, std::hypot(profile.dust_vx_[i], profile.dust_vy_[i]));
	}
	return result;
}

auto profileIsFinite(const CaseProfile &profile) -> bool
{
	auto check = [](const std::vector<double> &values) {
		return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
	};
	return check(profile.gas_vx_) && check(profile.gas_vy_) && check(profile.dust_vx_) && check(profile.dust_vy_) && check(profile.bx_) &&
	       check(profile.by_);
}

void writeProfileCsv(const CaseResult &result)
{
	std::ofstream file(std::format("dusty_alfven_{}_{}_profile.csv", result.config_.sweep_, result.config_.tag_));
	file << "z,gas_vx,gas_vy,dust_vx,dust_vy,bx,by,ref_gas_vx,ref_gas_vy,ref_dust_vx,ref_dust_vy,ref_bx,ref_by\n";
	for (size_t i = 0; i < result.profile_.z_.size(); ++i) {
		file << result.profile_.z_[i] << "," << result.profile_.gas_vx_[i] << "," << result.profile_.gas_vy_[i] << "," << result.profile_.dust_vx_[i]
		     << "," << result.profile_.dust_vy_[i] << "," << result.profile_.bx_[i] << "," << result.profile_.by_[i] << ","
		     << result.profile_.ref_gas_vx_[i] << "," << result.profile_.ref_gas_vy_[i] << "," << result.profile_.ref_dust_vx_[i] << ","
		     << result.profile_.ref_dust_vy_[i] << "," << result.profile_.ref_bx_[i] << "," << result.profile_.ref_by_[i] << "\n";
	}
}

void writeHistoryCsv(const CaseResult &result)
{
	std::ofstream file(std::format("dusty_alfven_{}_{}_history.csv", result.config_.sweep_, result.config_.tag_));
	file << "t,dust_vx,ref_dust_vx\n";
	for (size_t i = 0; i < result.history_.t_.size(); ++i) {
		file << result.history_.t_[i] << "," << result.history_.dust_vx_[i] << "," << result.history_.ref_dust_vx_[i] << "\n";
	}
}

void writeTracerProfileCsv(const CaseResult &result)
{
	std::ofstream file(std::format("dusty_alfven_{}_{}_particle_profile.csv", result.config_.sweep_, result.config_.tag_));
	file << "z_num,gas_vx_num,dust_vx_num,z_ref,gas_vx_ref,dust_vx_ref\n";
	const size_t nrows = std::max(result.tracer_profile_.z_num_.size(), result.tracer_profile_.z_ref_.size());
	for (size_t i = 0; i < nrows; ++i) {
		if (i < result.tracer_profile_.z_num_.size()) {
			file << result.tracer_profile_.z_num_[i] << "," << result.tracer_profile_.gas_vx_num_[i] << ","
			     << result.tracer_profile_.dust_vx_num_[i];
		} else {
			file << ",,";
		}
		file << ",";
		if (i < result.tracer_profile_.z_ref_.size()) {
			file << result.tracer_profile_.z_ref_[i] << "," << result.tracer_profile_.gas_vx_ref_[i] << ","
			     << result.tracer_profile_.dust_vx_ref_[i];
		} else {
			file << ",,";
		}
		file << "\n";
	}
}

void writeTracerHistoryCsv(const CaseResult &result)
{
	std::ofstream file(std::format("dusty_alfven_{}_{}_particle_history.csv", result.config_.sweep_, result.config_.tag_));
	file << "t,dust_vx,ref_dust_vx\n";
	for (size_t i = 0; i < result.history_.t_.size(); ++i) {
		file << result.history_.t_[i] << "," << result.history_.tracer_dust_vx_[i] << "," << result.history_.ref_tracer_dust_vx_[i] << "\n";
	}
}

void writeTracerHistoryDenseCsv(const CaseResult &result)
{
	std::ofstream file(std::format("dusty_alfven_{}_{}_particle_history_dense.csv", result.config_.sweep_, result.config_.tag_));
	file << "t,ref_dust_vx\n";
	for (size_t i = 0; i < result.history_.ref_tracer_t_dense_.size(); ++i) {
		file << result.history_.ref_tracer_t_dense_[i] << "," << result.history_.ref_tracer_dust_vx_dense_[i] << "\n";
	}
}

auto runCase(const CaseConfig &config, int reference_steps) -> CaseResult
{
	g_mu = config.mu_;
	g_omega_l_target = config.omega_l_target_;
	g_stopping_time = config.stopping_time_;

	amrex::Print() << std::format("Running DustyAlfvenWave case: {} ({})\n", config.label_, config.sweep_);

	auto BCs_cc = quokka::BC<DustyAlfvenWave>(quokka::BCType::int_dir);
	auto BCs_fc = quokka::BC_fc<DustyAlfvenWave>(quokka::BCType::mathematicalBndryTypes::periodic, quokka::BCType::mathematicalBndryTypes::periodic,
						     quokka::BCType::mathematicalBndryTypes::periodic);
	QuokkaSimulation<DustyAlfvenWave> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 2;
	sim.radiationReconstructionOrder_ = 2;
	sim.plotfileInterval_ = -1;
	sim.stopTime_ = final_time;

	sim.setInitialConditions();
	appendHistory(sim, true);
	sim.evolve();
	appendHistory(sim, true);

	CaseResult result;
	result.config_ = config;
	result.history_ = sim.userData_;
	result.profile_ = extractProfile(sim);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto [reference_state, ref_dust_vx] = integrateReferenceToTimes(config, result.history_.t_, result.history_.sample_z_, reference_steps);
		result.history_.ref_dust_vx_ = std::move(ref_dust_vx);
		fillReferenceProfile(result.profile_, reference_state);

		auto const numerical_field_history = makeNumericalFieldHistory(result.history_);
		auto [tracer_dust_vx, tracer_profile] = integrateTracerThroughSampledFields(config, numerical_field_history, result.history_.t_);
		auto [ref_tracer_dust_vx, reference_tracer_profile, reference_tracer_t_dense, reference_tracer_vx_dense] =
		    integrateReferenceTracerEnsemble(config, result.history_.t_, reference_steps);
		result.history_.tracer_dust_vx_ = std::move(tracer_dust_vx);
		result.history_.ref_tracer_dust_vx_ = std::move(ref_tracer_dust_vx);
		result.history_.ref_tracer_t_dense_ = std::move(reference_tracer_t_dense);
		result.history_.ref_tracer_dust_vx_dense_ = std::move(reference_tracer_vx_dense);
		result.tracer_profile_ = std::move(tracer_profile);
		result.tracer_profile_.z_ref_ = std::move(reference_tracer_profile.z_ref_);
		result.tracer_profile_.gas_vx_ref_ = std::move(reference_tracer_profile.gas_vx_ref_);
		result.tracer_profile_.dust_vx_ref_ = std::move(reference_tracer_profile.dust_vx_ref_);

		result.dust_profile_error_ =
		    relativeTransverseError(result.profile_.dust_vx_, result.profile_.dust_vy_, result.profile_.ref_dust_vx_, result.profile_.ref_dust_vy_);
		result.gas_helical_error_ = helicalResidual(result.profile_.z_, result.profile_.gas_vx_, result.profile_.gas_vy_);
		result.b_helical_error_ = helicalResidual(result.profile_.z_, result.profile_.bx_, result.profile_.by_);
		result.max_dust_speed_ = maxDustSpeed(result.profile_);
		result.finite_ = profileIsFinite(result.profile_);

		amrex::Print() << std::format("  dust profile rel L2 error = {:.4e}\n", result.dust_profile_error_);
		amrex::Print() << std::format("  gas helical residual      = {:.4e}\n", result.gas_helical_error_);
		amrex::Print() << std::format("  B helical residual        = {:.4e}\n", result.b_helical_error_);
		amrex::Print() << std::format("  max transverse dust speed = {:.4e}\n", result.max_dust_speed_);
	}

	return result;
}

auto makeMuCases() -> std::vector<CaseConfig>
{
	return {{"mu", "mu0", "mu = 0", 0.0, -alfven_frequency, stop_time_default},
		{"mu", "mu0p01", "mu = 0.01", 0.01, -alfven_frequency, stop_time_default},
		{"mu", "mu0p1", "mu = 0.1", 0.1, -alfven_frequency, stop_time_default},
		{"mu", "mu1", "mu = 1", 1.0, -alfven_frequency, stop_time_default}};
}

auto makeOmegaCases() -> std::vector<CaseConfig>
{
	return {{"omega", "omega_high", "-omega_L/Omega_AW = 10", 0.01, -10.0 * alfven_frequency, stop_time_default},
		{"omega", "omega_resonant", "-omega_L/Omega_AW = 1", 0.01, -alfven_frequency, stop_time_default},
		{"omega", "omega_low", "-omega_L/Omega_AW = 0.1", 0.01, -0.1 * alfven_frequency, stop_time_default}};
}
} // namespace

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustyAlfvenWave>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
										       amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
										       amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/,
										       double /*cs*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha{};
	alpha[0] = 1.0 / g_stopping_time;
	return alpha;
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustyAlfvenWave>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> q_over_m{};
	q_over_m[0] = chargeToMassRatio(g_omega_l_target);
	return q_over_m;
}

template <> void QuokkaSimulation<DustyAlfvenWave>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<DustyAlfvenWave>::nvarTotal_cc;
	const double rho_dust = dustDensityFromMu(g_mu);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		const double z = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
		const double phase = wave_number * z;
		const double ux = u0 * std::cos(phase);
		const double uy = u0 * std::sin(phase);
		const double uz = 1.0;
		const double kinetic = 0.5 * rho_gas * (ux * ux + uy * uy + uz * uz);
		const double magnetic = 0.5 * (u0 * u0 + bz0 * bz0);

		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::density_index) = rho_gas;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::x1Momentum_index) = rho_gas * ux;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::x2Momentum_index) = rho_gas * uy;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::x3Momentum_index) = rho_gas * uz;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::energy_index) = kinetic + magnetic;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::internalEnergy_index) = 0.0;

		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::dustDensity_index) = rho_dust;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::x1DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::x2DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustyAlfvenWave>::x3DustMomentum_index) = rho_dust;
	});
}

template <> void QuokkaSimulation<DustyAlfvenWave>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const quokka::direction dir = grid_elem.dir_;
	const int ncomp_fc = Physics_Indices<DustyAlfvenWave>::nvarPerDim_fc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}

		const double z_center = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
		const double phase = wave_number * z_center;
		double bfield = 0.0;
		if (dir == quokka::direction::x) {
			bfield = u0 * std::cos(phase);
		} else if (dir == quokka::direction::y) {
			bfield = u0 * std::sin(phase);
		} else if (dir == quokka::direction::z) {
			bfield = bz0;
		}
		state_fc(i, j, k, MHDSystem<DustyAlfvenWave>::bfield_index) = bfield;
	});
}

template <> void QuokkaSimulation<DustyAlfvenWave>::computeAfterTimestep() { appendHistory(*this); }

auto problem_main() -> int
{
	bool write_csv = true;
	int reference_steps = 50000;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);
	pp.query("reference_steps", reference_steps);

	std::vector<CaseResult> results;
	for (CaseConfig const &config : makeMuCases()) {
		results.push_back(runCase(config, reference_steps));
	}
	for (CaseConfig const &config : makeOmegaCases()) {
		results.push_back(runCase(config, reference_steps));
	}

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		for (CaseResult const &result : results) {
			if (write_csv) {
				writeProfileCsv(result);
				writeHistoryCsv(result);
				writeTracerProfileCsv(result);
				writeTracerHistoryCsv(result);
				writeTracerHistoryDenseCsv(result);
			}
			const bool case_passed =
			    result.finite_ && result.gas_helical_error_ < 0.3 && result.b_helical_error_ < 0.3 && result.dust_profile_error_ < 0.3;
			if (!case_passed) {
				status = 1;
			}
		}

		if (status == 0) {
			amrex::Print() << "DustyAlfvenWave PASSED.\n";
		} else {
			amrex::Print() << "DustyAlfvenWave FAILED.\n";
		}
	}

	amrex::ParallelDescriptor::ReduceIntMax(status);
	return status;
}
