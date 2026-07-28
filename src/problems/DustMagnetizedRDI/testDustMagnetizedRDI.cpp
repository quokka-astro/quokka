/// \file testDustMagnetizedRDI.cpp
/// \brief Magnetized RDI test inspired by Moseley et al. (2023), Section 3.5.

#include "AMReX_Gpu.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Reduce.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "dust/DustRuntimeParams.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

struct DustMagnetizedRDI {
};

namespace
{
using Vec3 = std::array<double, 3>;

constexpr double pi = std::numbers::pi;
constexpr double rho_gas0 = 1.0;
constexpr double sound_speed = 1.0;
constexpr double gamma_iso = 1.0;
constexpr double tiny_number = 1.0e-14;
constexpr double supersonic_eta = 9.0 * pi * gamma_iso / 128.0;
constexpr double time_tolerance = 1.0e-10;
constexpr double bar_a = 5.0;
constexpr double grain_radius_default = 5.0;
constexpr double grain_density_default = 1.0;
constexpr double dimensionless_charge_to_mass_ratio = -10.0;
constexpr double dust_to_gas_mass_ratio = 0.01;
constexpr double beta_param = 2.0;
constexpr double theta_Ba_deg = 87.0;
constexpr double noise_amplitude_default = 1.0e-7;
constexpr int noise_seed_default = 20250305;

constexpr std::array<char const *, 3> stage_labels = {"linear", "nonlinear", "saturation"};
constexpr std::array<char const *, 3> slice_tags = {"xmax_slice", "ymin_slice", "zmax_slice"};
constexpr std::array<double, 3> stage_times_over_ts0_default = {5.8, 7.5, 17.0};
constexpr double history_dt_over_ts0_default = 0.1;

double g_history_dt_over_ts0 = history_dt_over_ts0_default;		     // NOLINT
double g_history_dt_code = history_dt_over_ts0_default;			     // NOLINT
bool g_write_csv = true;						     // NOLINT
std::array<double, 3> g_stage_times_over_ts0 = stage_times_over_ts0_default; // NOLINT
std::array<double, 3> g_stage_target_times = {0.0, 0.0, 0.0};		     // NOLINT
double g_equilibrium_ts = 0.0;						     // NOLINT

AMREX_GPU_MANAGED double g_grain_radius = grain_radius_default;	      // NOLINT
AMREX_GPU_MANAGED double g_grain_density = grain_density_default;     // NOLINT
AMREX_GPU_MANAGED double g_noise_amplitude = noise_amplitude_default; // NOLINT
AMREX_GPU_MANAGED int g_noise_seed = noise_seed_default;	      // NOLINT
AMREX_GPU_MANAGED double g_Bx0 = 0.0;				      // NOLINT
AMREX_GPU_MANAGED double g_By0 = 0.0;				      // NOLINT
AMREX_GPU_MANAGED double g_Bz0 = 1.0;				      // NOLINT
AMREX_GPU_MANAGED double g_gas_vx0 = 0.0;			      // NOLINT
AMREX_GPU_MANAGED double g_gas_vy0 = 0.0;			      // NOLINT
AMREX_GPU_MANAGED double g_gas_vz0 = 0.0;			      // NOLINT
AMREX_GPU_MANAGED double g_dust_vx0 = 0.0;			      // NOLINT
AMREX_GPU_MANAGED double g_dust_vy0 = 0.0;			      // NOLINT
AMREX_GPU_MANAGED double g_dust_vz0 = 0.0;			      // NOLINT

struct EquilibriumState {
	Vec3 drift_{};
	Vec3 gas_velocity_{};
	Vec3 dust_velocity_{};
	Vec3 magnetic_field_{};
	double stop_time_ = 0.0;
	double tau_ = 0.0;
	double drift_angle_to_b_deg_ = 0.0;
	double drift_speed_ = 0.0;
};

struct DiagnosticsRecord {
	double time_ = 0.0;
	double sigma_log_rho_g_ = 0.0;
	double sigma_log_rho_d_ = 0.0;
	double sigma_vgx_ = 0.0;
	double sigma_vgy_ = 0.0;
	double sigma_vgz_ = 0.0;
	double sigma_vdx_ = 0.0;
	double sigma_vdy_ = 0.0;
	double sigma_vdz_ = 0.0;
	double sigma_bx_ = 0.0;
	double sigma_by_ = 0.0;
	double sigma_bz_ = 0.0;
	double sigma_bmag_ = 0.0;
	bool finite_ = true;
};

struct OuterSlice {
	std::vector<double> u_;
	std::vector<double> v_;
	std::vector<double> magnetic_perturbation_magnitude_;
	std::vector<double> dust_density_ratio_;
};

struct DustMagnetizedRDIHistory {
	double next_history_time_ = 0.0;
	std::vector<double> t_;
	std::vector<double> sigma_log_rho_g_;
	std::vector<double> sigma_log_rho_d_;
	std::vector<double> sigma_vgx_;
	std::vector<double> sigma_vgy_;
	std::vector<double> sigma_vgz_;
	std::vector<double> sigma_vdx_;
	std::vector<double> sigma_vdy_;
	std::vector<double> sigma_vdz_;
	std::vector<double> sigma_bx_;
	std::vector<double> sigma_by_;
	std::vector<double> sigma_bz_;
	std::vector<double> sigma_bmag_;
	std::array<bool, 3> stage_written_ = {false, false, false};
	std::array<double, 3> stage_times_ = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
					      std::numeric_limits<double>::quiet_NaN()};
	std::array<std::string, 3> stage_plotfiles_;
};

template <typename T> auto square(T value) -> T { return value * value; }

auto dot(Vec3 const &a, Vec3 const &b) -> double { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

auto cross(Vec3 const &a, Vec3 const &b) -> Vec3 { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }

auto norm(Vec3 const &a) -> double { return std::sqrt(dot(a, a)); }

auto operator+(Vec3 const &a, Vec3 const &b) -> Vec3 { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }

auto operator-(Vec3 const &a, Vec3 const &b) -> Vec3 { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

auto operator*(double scalar, Vec3 const &a) -> Vec3 { return {scalar * a[0], scalar * a[1], scalar * a[2]}; }

auto operator/(Vec3 const &a, double scalar) -> Vec3 { return (1.0 / scalar) * a; }

auto angleDegrees(Vec3 const &a, Vec3 const &b) -> double
{
	double const denom = norm(a) * norm(b);
	if (denom <= 0.0) {
		return 0.0;
	}
	double cosine = dot(a, b) / denom;
	cosine = std::clamp(cosine, -1.0, 1.0);
	return std::acos(cosine) * 180.0 / pi;
}

auto computeMagneticFieldMagnitudeFromBeta(double beta) -> double
{
	double const pressure = rho_gas0 * sound_speed * sound_speed;
	return std::sqrt(2.0 * pressure / beta);
}

auto makeBackgroundMagneticField(double beta, double theta_Ba_deg) -> Vec3
{
	double const theta = theta_Ba_deg * pi / 180.0;
	double const Bmag = computeMagneticFieldMagnitudeFromBeta(beta);
	return {Bmag * std::cos(theta), 0.0, Bmag * std::sin(theta)};
}

auto grainSizeParameter() -> double { return g_grain_radius * g_grain_density; }

auto computeSubsonicStoppingTime() -> double { return std::sqrt(pi * gamma_iso) * grainSizeParameter() / (2.0 * std::numbers::sqrt2 * rho_gas0 * sound_speed); }

auto solveDriftEquilibrium() -> EquilibriumState
{
	EquilibriumState result;
	result.magnetic_field_ = makeBackgroundMagneticField(beta_param, theta_Ba_deg);
	double const magnetic_field_norm = norm(result.magnetic_field_);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(magnetic_field_norm > 0.0, "DustMagnetizedRDI requires a non-zero background magnetic field.");

	Vec3 const acceleration = {bar_a, 0.0, 0.0};
	Vec3 const b_hat = result.magnetic_field_ / magnetic_field_norm;
	double const ts_sub = computeSubsonicStoppingTime();
	double const omega_L = dimensionless_charge_to_mass_ratio * magnetic_field_norm;

	Vec3 drift = {(ts_sub / (1.0 + dust_to_gas_mass_ratio)) * bar_a, 0.0, 0.0};
	for (int iter = 0; iter < 64; ++iter) {
		double const drift_speed = norm(drift);
		double const stop_time = ts_sub / std::sqrt(1.0 + supersonic_eta * square(drift_speed / sound_speed));
		double const tau_local = omega_L * stop_time;

		Vec3 const rhs = (stop_time / (1.0 + dust_to_gas_mass_ratio)) * acceleration;
		Vec3 const rhs_parallel = dot(rhs, b_hat) * b_hat;
		Vec3 const rhs_perp = rhs - rhs_parallel;
		Vec3 const hall = cross(rhs, b_hat);
		Vec3 const updated = rhs_parallel + (rhs_perp + tau_local * hall) / (1.0 + square(tau_local));
		if (norm(updated - drift) < 1.0e-13) {
			drift = updated;
			break;
		}
		drift = updated;
	}

	result.drift_ = drift;
	result.stop_time_ = ts_sub / std::sqrt(1.0 + supersonic_eta * square(norm(drift) / sound_speed));
	result.tau_ = std::abs(omega_L) * result.stop_time_;
	result.drift_speed_ = norm(drift);
	result.drift_angle_to_b_deg_ = angleDegrees(drift, result.magnetic_field_);
	result.gas_velocity_ = (-dust_to_gas_mass_ratio / (1.0 + dust_to_gas_mass_ratio)) * drift;
	result.dust_velocity_ = (1.0 / (1.0 + dust_to_gas_mass_ratio)) * drift;
	return result;
}

void loadProblemParameters()
{
	amrex::GpuArray<amrex::Real, 1> grain_radius = {g_grain_radius};
	amrex::GpuArray<amrex::Real, 1> grain_density = {g_grain_density};
	quokka::dust::readDustGrainParams(grain_radius, grain_density);
	g_grain_radius = grain_radius[0];
	g_grain_density = grain_density[0];

	amrex::ParmParse const pp("problem");
	pp.query("write_csv", g_write_csv);
	pp.query("history_dt_over_ts0", g_history_dt_over_ts0);
	pp.query("noise_amplitude", g_noise_amplitude);
	pp.query("noise_seed", g_noise_seed);
	amrex::Vector<double> stage_times_over_ts0;
	if (pp.queryarr("stage_times_over_ts0", stage_times_over_ts0) != 0) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(stage_times_over_ts0.size() == static_cast<amrex::Long>(g_stage_times_over_ts0.size()),
						 "problem.stage_times_over_ts0 must contain exactly 3 values.");
		for (std::size_t i = 0; i < g_stage_times_over_ts0.size(); ++i) {
			g_stage_times_over_ts0[i] = stage_times_over_ts0[i];
		}
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(g_history_dt_over_ts0) && g_history_dt_over_ts0 > 0.0,
					 "problem.history_dt_over_ts0 must be finite and positive.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(g_noise_amplitude) && g_noise_amplitude >= 0.0,
					 "problem.noise_amplitude must be finite and non-negative.");
	for (double const stage_time : g_stage_times_over_ts0) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(stage_time) && stage_time > 0.0,
						 "problem.stage_times_over_ts0 values must all be finite and positive.");
	}
	for (std::size_t i = 1; i < g_stage_times_over_ts0.size(); ++i) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(g_stage_times_over_ts0[i] > g_stage_times_over_ts0[i - 1],
						 "problem.stage_times_over_ts0 values must be strictly increasing.");
	}
}

void applyEquilibriumState(EquilibriumState const &equilibrium)
{
	g_Bx0 = equilibrium.magnetic_field_[0];
	g_By0 = equilibrium.magnetic_field_[1];
	g_Bz0 = equilibrium.magnetic_field_[2];

	g_gas_vx0 = equilibrium.gas_velocity_[0];
	g_gas_vy0 = equilibrium.gas_velocity_[1];
	g_gas_vz0 = equilibrium.gas_velocity_[2];
	g_dust_vx0 = equilibrium.dust_velocity_[0];
	g_dust_vy0 = equilibrium.dust_velocity_[1];
	g_dust_vz0 = equilibrium.dust_velocity_[2];

	g_equilibrium_ts = equilibrium.stop_time_;
	g_history_dt_code = g_history_dt_over_ts0 * equilibrium.stop_time_;
	for (std::size_t i = 0; i < g_stage_target_times.size(); ++i) {
		g_stage_target_times[i] = g_stage_times_over_ts0[i] * equilibrium.stop_time_;
	}
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto mixBits(std::uint64_t key) -> std::uint64_t
{
	key ^= key >> 33U;
	key *= 0xff51afd7ed558ccdULL;
	key ^= key >> 33U;
	key *= 0xc4ceb9fe1a85ec53ULL;
	key ^= key >> 33U;
	return key;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto deterministicNoise(int i, int j, int k, int component) -> double
{
	auto key = static_cast<std::uint64_t>(static_cast<std::uint32_t>(g_noise_seed)) * 0xd6e8feb86659fd93ULL;
	key ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) * 0x9e3779b185ebca87ULL;
	key ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(j)) * 0xc2b2ae3d27d4eb4fULL;
	key ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(k)) * 0x165667b19e3779f9ULL;
	key ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(component)) * 0x85ebca77c2b2ae63ULL;
	std::uint64_t const mixed = mixBits(key);
	double const unit = static_cast<double>(mixed & 0xffffffffULL) / static_cast<double>(0xffffffffULL);
	return 2.0 * unit - 1.0;
}

auto computeStd(double sum, double sum_sq, double count) -> double
{
	if (count <= 0.0) {
		return 0.0;
	}
	double const mean = sum / count;
	double const variance = std::max(0.0, (sum_sq / count) - square(mean));
	return std::sqrt(variance);
}

template <typename problem_t> auto computeDiagnostics(QuokkaSimulation<problem_t> &sim) -> DiagnosticsRecord
{
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
			 amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
			 amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
			 amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
			 amrex::ReduceOpSum>
	    reduce_op;
	using ReduceDataType =
	    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
			      amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
			      amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>;
	ReduceDataType reduce_data(reduce_op);

	auto &state_mf = sim.state_new_cc_[0];
	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto const &state = state_mf.const_array(mfi);
		auto const &bx_fc = sim.state_new_fc_[0][0].const_array(mfi);
		auto const &by_fc = sim.state_new_fc_[0][1].const_array(mfi);
		auto const &bz_fc = sim.state_new_fc_[0][2].const_array(mfi);
		constexpr int mhd_idx = Physics_Indices<problem_t>::mhdFirstIndex;

		reduce_op.eval(
		    box, reduce_data,
		    [=] AMREX_GPU_DEVICE(int i, int j, int k)
			-> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
					   amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
					   amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
			    amrex::Real const rho_g = state(i, j, k, HydroSystem<problem_t>::density_index);
			    amrex::Real const rho_d = state(i, j, k, HydroSystem<problem_t>::dustDensity_index);
			    amrex::Real const vx_g = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / rho_g;
			    amrex::Real const vy_g = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / rho_g;
			    amrex::Real const vz_g = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / rho_g;
			    amrex::Real const vx_d = state(i, j, k, HydroSystem<problem_t>::x1DustMomentum_index) / rho_d;
			    amrex::Real const vy_d = state(i, j, k, HydroSystem<problem_t>::x2DustMomentum_index) / rho_d;
			    amrex::Real const vz_d = state(i, j, k, HydroSystem<problem_t>::x3DustMomentum_index) / rho_d;
			    amrex::Real const bx = 0.5_rt * (bx_fc(i, j, k, mhd_idx) + bx_fc(i + 1, j, k, mhd_idx));
			    amrex::Real const by = 0.5_rt * (by_fc(i, j, k, mhd_idx) + by_fc(i, j + 1, k, mhd_idx));
			    amrex::Real const bz = 0.5_rt * (bz_fc(i, j, k, mhd_idx) + bz_fc(i, j, k + 1, mhd_idx));
			    amrex::Real const bmag = std::sqrt(bx * bx + by * by + bz * bz);
			    amrex::Real const log_rho_g = std::log(amrex::max(rho_g, static_cast<amrex::Real>(tiny_number)));
			    amrex::Real const log_rho_d = std::log(rho_d);

			    return {1.0_rt,
				    log_rho_g,
				    log_rho_g * log_rho_g,
				    log_rho_d,
				    log_rho_d * log_rho_d,
				    vx_g,
				    vx_g * vx_g,
				    vy_g,
				    vy_g * vy_g,
				    vz_g,
				    vz_g * vz_g,
				    vx_d,
				    vx_d * vx_d,
				    vy_d,
				    vy_d * vy_d,
				    vz_d,
				    vz_d * vz_d,
				    bx,
				    bx * bx,
				    by,
				    by * by,
				    bz,
				    bz * bz,
				    bmag,
				    bmag * bmag};
		    });
	}

	auto [count, sum_log_rho_g, sum_log_rho_g2, sum_log_rho_d, sum_log_rho_d2, sum_vgx, sum_vgx2, sum_vgy, sum_vgy2, sum_vgz, sum_vgz2, sum_vdx, sum_vdx2,
	      sum_vdy, sum_vdy2, sum_vdz, sum_vdz2, sum_bx, sum_bx2, sum_by, sum_by2, sum_bz, sum_bz2, sum_bmag, sum_bmag2] = reduce_data.value();

	amrex::GpuArray<amrex::Real, 25> reduced = {
	    count,   sum_log_rho_g, sum_log_rho_g2, sum_log_rho_d, sum_log_rho_d2, sum_vgx, sum_vgx2, sum_vgy, sum_vgy2, sum_vgz, sum_vgz2, sum_vdx,  sum_vdx2,
	    sum_vdy, sum_vdy2,	    sum_vdz,	    sum_vdz2,	   sum_bx,	   sum_bx2, sum_by,   sum_by2, sum_bz,	 sum_bz2, sum_bmag, sum_bmag2};
	amrex::ParallelDescriptor::ReduceRealSum(reduced.data(), 25);

	DiagnosticsRecord record;
	record.time_ = sim.tNew_[0];
	record.sigma_log_rho_g_ = computeStd(reduced[1], reduced[2], reduced[0]);
	record.sigma_log_rho_d_ = computeStd(reduced[3], reduced[4], reduced[0]);
	record.sigma_vgx_ = computeStd(reduced[5], reduced[6], reduced[0]);
	record.sigma_vgy_ = computeStd(reduced[7], reduced[8], reduced[0]);
	record.sigma_vgz_ = computeStd(reduced[9], reduced[10], reduced[0]);
	record.sigma_vdx_ = computeStd(reduced[11], reduced[12], reduced[0]);
	record.sigma_vdy_ = computeStd(reduced[13], reduced[14], reduced[0]);
	record.sigma_vdz_ = computeStd(reduced[15], reduced[16], reduced[0]);
	record.sigma_bx_ = computeStd(reduced[17], reduced[18], reduced[0]);
	record.sigma_by_ = computeStd(reduced[19], reduced[20], reduced[0]);
	record.sigma_bz_ = computeStd(reduced[21], reduced[22], reduced[0]);
	record.sigma_bmag_ = computeStd(reduced[23], reduced[24], reduced[0]);

	record.finite_ = std::isfinite(record.sigma_log_rho_g_) && std::isfinite(record.sigma_log_rho_d_) && std::isfinite(record.sigma_vgx_) &&
			 std::isfinite(record.sigma_vgy_) && std::isfinite(record.sigma_vgz_) && std::isfinite(record.sigma_vdx_) &&
			 std::isfinite(record.sigma_vdy_) && std::isfinite(record.sigma_vdz_) && std::isfinite(record.sigma_bx_) &&
			 std::isfinite(record.sigma_by_) && std::isfinite(record.sigma_bz_) && std::isfinite(record.sigma_bmag_);
	return record;
}

void writeGrowthHistoryCsv(DustMagnetizedRDIHistory const &history)
{
	std::ofstream file("dust_magnetized_rdi_growth.csv");
	file << std::setprecision(17);
	file << "t,sigma_log_rho_g,sigma_log_rho_d,sigma_vgx,sigma_vgy,sigma_vgz,sigma_vdx,sigma_vdy,sigma_vdz,sigma_bx,sigma_by,sigma_bz,sigma_bmag\n";
	for (size_t i = 0; i < history.t_.size(); ++i) {
		file << history.t_[i] << "," << history.sigma_log_rho_g_[i] << "," << history.sigma_log_rho_d_[i] << "," << history.sigma_vgx_[i] << ","
		     << history.sigma_vgy_[i] << "," << history.sigma_vgz_[i] << "," << history.sigma_vdx_[i] << "," << history.sigma_vdy_[i] << ","
		     << history.sigma_vdz_[i] << "," << history.sigma_bx_[i] << "," << history.sigma_by_[i] << "," << history.sigma_bz_[i] << ","
		     << history.sigma_bmag_[i] << "\n";
	}
}

template <typename problem_t>
void writeSummaryCsv(QuokkaSimulation<problem_t> const &sim, EquilibriumState const &equilibrium, DustMagnetizedRDIHistory const &history)
{
	auto const domain = sim.Geom(0).Domain();
	auto const prob_lo = sim.Geom(0).ProbLoArray();
	auto const prob_hi = sim.Geom(0).ProbHiArray();
	double const B0 = norm(equilibrium.magnetic_field_);
	double const rho_d0 = dust_to_gas_mass_ratio * rho_gas0;
	double const gas_accel_x = -dust_to_gas_mass_ratio * bar_a / (1.0 + dust_to_gas_mass_ratio);
	double const dust_accel_x = bar_a / (1.0 + dust_to_gas_mass_ratio);

	std::ofstream file("dust_magnetized_rdi_summary.csv");
	file << std::setprecision(17);
	file << "key,value\n";
	file << "bar_a," << bar_a << "\n";
	file << "grain_size_parameter," << grainSizeParameter() << "\n";
	file << "xi," << dimensionless_charge_to_mass_ratio << "\n";
	file << "bar_phi_d," << grainSizeParameter() * std::abs(dimensionless_charge_to_mass_ratio) << "\n";
	file << "dust_to_gas_mass_ratio," << dust_to_gas_mass_ratio << "\n";
	file << "beta," << beta_param << "\n";
	file << "gamma," << gamma_iso << "\n";
	file << "theta_Ba_deg," << theta_Ba_deg << "\n";
	file << "grain_radius," << g_grain_radius << "\n";
	file << "grain_density," << g_grain_density << "\n";
	file << "drag_law,Epstein-Baines\n";
	file << "frame,zero-center-of-mass\n";
	file << "gas_acceleration_x," << gas_accel_x << "\n";
	file << "dust_acceleration_x," << dust_accel_x << "\n";
	file << "relative_acceleration_x," << dust_accel_x - gas_accel_x << "\n";
	file << "noise_amplitude," << g_noise_amplitude << "\n";
	file << "noise_seed," << g_noise_seed << "\n";
	file << "noise_distribution,\"uniform[-A,A]\"\n";
	file << "equilibrium_stop_time," << equilibrium.stop_time_ << "\n";
	file << "equilibrium_tau," << equilibrium.tau_ << "\n";
	file << "equilibrium_drift_speed," << equilibrium.drift_speed_ << "\n";
	file << "equilibrium_angle_wB_deg," << equilibrium.drift_angle_to_b_deg_ << "\n";
	file << "history_dt_over_ts0," << g_history_dt_over_ts0 << "\n";
	file << "history_dt_code," << g_history_dt_code << "\n";
	file << "gas_vx0," << equilibrium.gas_velocity_[0] << "\n";
	file << "gas_vy0," << equilibrium.gas_velocity_[1] << "\n";
	file << "gas_vz0," << equilibrium.gas_velocity_[2] << "\n";
	file << "dust_vx0," << equilibrium.dust_velocity_[0] << "\n";
	file << "dust_vy0," << equilibrium.dust_velocity_[1] << "\n";
	file << "dust_vz0," << equilibrium.dust_velocity_[2] << "\n";
	file << "Bx0," << equilibrium.magnetic_field_[0] << "\n";
	file << "By0," << equilibrium.magnetic_field_[1] << "\n";
	file << "Bz0," << equilibrium.magnetic_field_[2] << "\n";
	file << "B0," << B0 << "\n";
	file << "rho_g0," << rho_gas0 << "\n";
	file << "rho_d0," << rho_d0 << "\n";
	file << "cs0," << sound_speed << "\n";
	file << "dust_density_floor," << sim.dustDensityFloor_ << "\n";
	file << "grid_nx," << domain.length(0) << "\n";
	file << "grid_ny," << domain.length(1) << "\n";
	file << "grid_nz," << domain.length(2) << "\n";
	file << "actual_resolution," << domain.length(0) << "x" << domain.length(1) << "x" << domain.length(2) << "\n";
	file << "box_xlo," << prob_lo[0] << "\n";
	file << "box_ylo," << prob_lo[1] << "\n";
	file << "box_zlo," << prob_lo[2] << "\n";
	file << "box_xhi," << prob_hi[0] << "\n";
	file << "box_yhi," << prob_hi[1] << "\n";
	file << "box_zhi," << prob_hi[2] << "\n";
	file << "box_length_x," << prob_hi[0] - prob_lo[0] << "\n";
	file << "box_length_y," << prob_hi[1] - prob_lo[1] << "\n";
	file << "box_length_z," << prob_hi[2] - prob_lo[2] << "\n";
	file << "slice_sampling,outermost_cell_centers\n";
	for (int i = 0; i < 3; ++i) {
		file << "stage_" << stage_labels[i] << "_target_time_over_ts0," << g_stage_times_over_ts0[i] << "\n";
		file << "stage_" << stage_labels[i] << "_target_time_code," << g_stage_target_times[i] << "\n";
		file << "stage_" << stage_labels[i] << "_actual_time_code," << history.stage_times_[i] << "\n";
		file << "stage_" << stage_labels[i] << "_actual_time_over_ts0," << history.stage_times_[i] / equilibrium.stop_time_ << "\n";
		file << "stage_" << stage_labels[i] << "_reached," << static_cast<int>(history.stage_written_[i]) << "\n";
		file << "stage_" << stage_labels[i] << "_plotfile," << history.stage_plotfiles_[i] << "\n";
	}
}

void writeOuterSliceCsv(std::string const &stage_label, std::string const &slice_tag, OuterSlice const &slice)
{
	std::ofstream file(std::format("dust_magnetized_rdi_{}_{}.csv", stage_label, slice_tag));
	file << std::setprecision(17);
	file << "u,v,magnetic_perturbation_magnitude,dust_density_ratio\n";
	for (size_t i = 0; i < slice.u_.size(); ++i) {
		file << slice.u_[i] << "," << slice.v_[i] << "," << slice.magnetic_perturbation_magnitude_[i] << "," << slice.dust_density_ratio_[i] << "\n";
	}
}

template <typename problem_t> auto extractOuterSlice(QuokkaSimulation<problem_t> &sim, int normal_dir) -> OuterSlice
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.finest_level == 0, "DustMagnetizedRDI slice extraction only supports single-level runs.");

	auto const &state_mf = sim.state_new_cc_[0];
	const auto domain = sim.Geom(0).Domain();
	const auto lo = amrex::lbound(domain);
	const auto dx = sim.Geom(0).CellSizeArray();
	const auto prob_lo = sim.Geom(0).ProbLoArray();
	const int nx = domain.length(0);
	const int ny = domain.length(1);
	const int nz = domain.length(2);
	const double dust_density0 = dust_to_gas_mass_ratio * rho_gas0;
	const double bx0 = g_Bx0;
	const double by0 = g_By0;
	const double bz0 = g_Bz0;
	constexpr int mhd_idx = Physics_Indices<problem_t>::mhdFirstIndex;

	int nu = 0;
	int nv = 0;
	if (normal_dir == 0) {
		nu = ny;
		nv = nz;
	} else if (normal_dir == 1) {
		nu = nx;
		nv = nz;
	} else {
		nu = nx;
		nv = ny;
	}

	amrex::Box slice_box = domain;
	if (normal_dir == 0) {
		int const hi_x = lo.x + nx - 1;
		slice_box.setSmall(0, hi_x);
		slice_box.setBig(0, hi_x);
	} else if (normal_dir == 1) {
		slice_box.setSmall(1, lo.y);
		slice_box.setBig(1, lo.y);
	} else {
		int const hi_z = lo.z + nz - 1;
		slice_box.setSmall(2, hi_z);
		slice_box.setBig(2, hi_z);
	}

	const int npts = nu * nv;
	amrex::Gpu::DeviceVector<amrex::Real> magnetic_perturbation_magnitude_d(npts, 0.0);
	amrex::Gpu::DeviceVector<amrex::Real> dust_density_ratio_d(npts, 0.0);
	auto *magnetic_perturbation_magnitude_ptr = magnetic_perturbation_magnitude_d.data();
	auto *dust_density_ratio_ptr = dust_density_ratio_d.data();

	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		amrex::Box const box = mfi.validbox() & slice_box;
		if (!box.ok()) {
			continue;
		}

		auto const &state = state_mf.const_array(mfi);
		auto const &bx_fc = sim.state_new_fc_[0][0].const_array(mfi);
		auto const &by_fc = sim.state_new_fc_[0][1].const_array(mfi);
		auto const &bz_fc = sim.state_new_fc_[0][2].const_array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			int idx = 0;
			if (normal_dir == 0) {
				idx = (k - lo.z) * ny + (j - lo.y);
			} else if (normal_dir == 1) {
				idx = (k - lo.z) * nx + (i - lo.x);
			} else {
				idx = (j - lo.y) * nx + (i - lo.x);
			}

			amrex::Real const bx = 0.5_rt * (bx_fc(i, j, k, mhd_idx) + bx_fc(i + 1, j, k, mhd_idx));
			amrex::Real const by = 0.5_rt * (by_fc(i, j, k, mhd_idx) + by_fc(i, j + 1, k, mhd_idx));
			amrex::Real const bz = 0.5_rt * (bz_fc(i, j, k, mhd_idx) + bz_fc(i, j, k + 1, mhd_idx));
			amrex::Real const dbx = bx - bx0;
			amrex::Real const dby = by - by0;
			amrex::Real const dbz = bz - bz0;
			amrex::Real const magnetic_perturbation_magnitude = std::sqrt(dbx * dbx + dby * dby + dbz * dbz);
			amrex::Real const dust_density_ratio = state(i, j, k, HydroSystem<problem_t>::dustDensity_index) / dust_density0;

			magnetic_perturbation_magnitude_ptr[idx] = magnetic_perturbation_magnitude;
			dust_density_ratio_ptr[idx] = dust_density_ratio;
		});
	}
	amrex::Gpu::streamSynchronize();
	amrex::Gpu::HostVector<amrex::Real> magnetic_perturbation_magnitude(npts);
	amrex::Gpu::HostVector<amrex::Real> dust_density_ratio(npts);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, magnetic_perturbation_magnitude_d.begin(), magnetic_perturbation_magnitude_d.end(),
			 magnetic_perturbation_magnitude.begin());
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, dust_density_ratio_d.begin(), dust_density_ratio_d.end(), dust_density_ratio.begin());
	amrex::ParallelDescriptor::ReduceRealSum(magnetic_perturbation_magnitude.data(), npts);
	amrex::ParallelDescriptor::ReduceRealSum(dust_density_ratio.data(), npts);

	OuterSlice slice;
	slice.u_.resize(npts);
	slice.v_.resize(npts);
	slice.magnetic_perturbation_magnitude_.resize(npts);
	slice.dust_density_ratio_.resize(npts);

	for (int iv = 0; iv < nv; ++iv) {
		for (int iu = 0; iu < nu; ++iu) {
			int const idx = iv * nu + iu;
			double u = 0.0;
			double v = 0.0;
			if (normal_dir == 0) {
				u = prob_lo[1] + (iu + 0.5) * dx[1];
				v = prob_lo[2] + (iv + 0.5) * dx[2];
			} else if (normal_dir == 1) {
				u = prob_lo[0] + (iu + 0.5) * dx[0];
				v = prob_lo[2] + (iv + 0.5) * dx[2];
			} else {
				u = prob_lo[0] + (iu + 0.5) * dx[0];
				v = prob_lo[1] + (iv + 0.5) * dx[1];
			}
			slice.u_[idx] = u;
			slice.v_[idx] = v;
			slice.magnetic_perturbation_magnitude_[idx] = magnetic_perturbation_magnitude[idx];
			slice.dust_density_ratio_[idx] = dust_density_ratio[idx];
		}
	}

	return slice;
}

template <typename problem_t> void captureStage(QuokkaSimulation<problem_t> &sim, int stage_index, double time, std::string const &plotfile)
{
	if (g_write_csv) {
		for (int normal_dir = 0; normal_dir < 3; ++normal_dir) {
			OuterSlice const slice = extractOuterSlice(sim, normal_dir);
			if (amrex::ParallelDescriptor::IOProcessor()) {
				writeOuterSliceCsv(stage_labels[stage_index], slice_tags[normal_dir], slice);
			}
		}
	}

	sim.userData_.stage_written_[stage_index] = true;
	sim.userData_.stage_times_[stage_index] = time;
	sim.userData_.stage_plotfiles_[stage_index] = plotfile;
	amrex::Print() << std::format("Captured DustMagnetizedRDI stage '{}' at t = {:.6f} = {:.3f} t_s^0\n", stage_labels[stage_index], time,
				      time / std::max(g_equilibrium_ts, tiny_number));
}

template <typename problem_t> void recordHistory(QuokkaSimulation<problem_t> &sim, DiagnosticsRecord const &diagnostics, bool force = false)
{
	bool const should_record = force || (diagnostics.time_ + time_tolerance >= sim.userData_.next_history_time_);
	if (!should_record) {
		return;
	}

	if (!sim.userData_.t_.empty() && std::abs(sim.userData_.t_.back() - diagnostics.time_) < time_tolerance) {
		while (sim.userData_.next_history_time_ <= diagnostics.time_ + time_tolerance) {
			sim.userData_.next_history_time_ += g_history_dt_code;
		}
		return;
	}

	sim.userData_.t_.push_back(diagnostics.time_);
	sim.userData_.sigma_log_rho_g_.push_back(diagnostics.sigma_log_rho_g_);
	sim.userData_.sigma_log_rho_d_.push_back(diagnostics.sigma_log_rho_d_);
	sim.userData_.sigma_vgx_.push_back(diagnostics.sigma_vgx_);
	sim.userData_.sigma_vgy_.push_back(diagnostics.sigma_vgy_);
	sim.userData_.sigma_vgz_.push_back(diagnostics.sigma_vgz_);
	sim.userData_.sigma_vdx_.push_back(diagnostics.sigma_vdx_);
	sim.userData_.sigma_vdy_.push_back(diagnostics.sigma_vdy_);
	sim.userData_.sigma_vdz_.push_back(diagnostics.sigma_vdz_);
	sim.userData_.sigma_bx_.push_back(diagnostics.sigma_bx_);
	sim.userData_.sigma_by_.push_back(diagnostics.sigma_by_);
	sim.userData_.sigma_bz_.push_back(diagnostics.sigma_bz_);
	sim.userData_.sigma_bmag_.push_back(diagnostics.sigma_bmag_);

	while (sim.userData_.next_history_time_ <= diagnostics.time_ + time_tolerance) {
		sim.userData_.next_history_time_ += g_history_dt_code;
	}
}
} // namespace

template <> struct SimulationData<DustMagnetizedRDI> : DustMagnetizedRDIHistory {
};

template <> struct quokka::EOS_Traits<DustMagnetizedRDI> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = gamma_iso;
	static constexpr double cs_isothermal = sound_speed;
};

template <> struct Physics_Traits<DustMagnetizedRDI> : DefaultPhysicsTraits {
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

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustMagnetizedRDI>::ComputeReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
											 amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> const grain_radius = {g_grain_radius};
	amrex::GpuArray<amrex::Real, nDustGroups_> const grain_density = {g_grain_density};
	return DustSources<DustMagnetizedRDI>::ComputeReciprocalStoppingTimeKwok(rho_g, rho_d, rel_vel_mag, cs, grain_radius, grain_density, true);
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustMagnetizedRDI>::ComputeDustDimensionlessChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> dimensionless_charge_to_mass_ratio_array{};
	dimensionless_charge_to_mass_ratio_array[0] = dimensionless_charge_to_mass_ratio;
	return dimensionless_charge_to_mass_ratio_array;
}

template <> void QuokkaSimulation<DustMagnetizedRDI>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<DustMagnetizedRDI>::nvarTotal_cc;
	const double dust_density0 = dust_to_gas_mass_ratio * rho_gas0;
	const double magnetic_energy = 0.5 * (g_Bx0 * g_Bx0 + g_By0 * g_By0 + g_Bz0 * g_Bz0);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		double const dv_gx = g_noise_amplitude * deterministicNoise(i, j, k, 0);
		double const dv_gy = g_noise_amplitude * deterministicNoise(i, j, k, 1);
		double const dv_gz = g_noise_amplitude * deterministicNoise(i, j, k, 2);
		double const dv_dx = g_noise_amplitude * deterministicNoise(i, j, k, 3);
		double const dv_dy = g_noise_amplitude * deterministicNoise(i, j, k, 4);
		double const dv_dz = g_noise_amplitude * deterministicNoise(i, j, k, 5);

		double const vx_g = g_gas_vx0 + dv_gx;
		double const vy_g = g_gas_vy0 + dv_gy;
		double const vz_g = g_gas_vz0 + dv_gz;
		double const vx_d = g_dust_vx0 + dv_dx;
		double const vy_d = g_dust_vy0 + dv_dy;
		double const vz_d = g_dust_vz0 + dv_dz;

		double const gas_kinetic = 0.5 * rho_gas0 * (vx_g * vx_g + vy_g * vy_g + vz_g * vz_g);

		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::density_index) = rho_gas0;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::energy_index) = gas_kinetic + magnetic_energy;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::x1Momentum_index) = rho_gas0 * vx_g;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::x2Momentum_index) = rho_gas0 * vy_g;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::x3Momentum_index) = rho_gas0 * vz_g;

		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::dustDensity_index) = dust_density0;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::x1DustMomentum_index) = dust_density0 * vx_d;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::x2DustMomentum_index) = dust_density0 * vy_d;
		state_cc(i, j, k, HydroSystem<DustMagnetizedRDI>::x3DustMomentum_index) = dust_density0 * vz_d;
	});
}

template <> void QuokkaSimulation<DustMagnetizedRDI>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const int ncomp_fc = Physics_Indices<DustMagnetizedRDI>::nvarPerDim_fc;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}

		double bfield = 0.0;
		if (dir == quokka::direction::x) {
			bfield = g_Bx0;
		} else if (dir == quokka::direction::y) {
			bfield = g_By0;
		} else if (dir == quokka::direction::z) {
			bfield = g_Bz0;
		}
		state_fc(i, j, k, Physics_Indices<DustMagnetizedRDI>::mhdFirstIndex) = bfield;
	});
}

template <> void QuokkaSimulation<DustMagnetizedRDI>::computeBeforeTimestep()
{
	for (int i = 0; i < 3; ++i) {
		if (!userData_.stage_written_[i]) {
			double const time_to_stage = g_stage_target_times[i] - tNew_[0];
			if (time_to_stage > time_tolerance) {
				dt_[0] = std::min(dt_[0], time_to_stage);
			}
			break;
		}
	}
}

template <> void QuokkaSimulation<DustMagnetizedRDI>::computeAfterTimestep()
{
	double const time = tNew_[0];
	for (int i = 0; i < 3; ++i) {
		if (!userData_.stage_written_[i] && time + time_tolerance >= g_stage_target_times[i]) {
			std::string const original_prefix = plot_file;
			plot_file = std::format("dust_magnetized_rdi_{}_plt", stage_labels[i]);
			std::string const plotfile = PlotFileName(istep[0]);
			WritePlotFile();
			plot_file = original_prefix;
			captureStage(*this, i, time, plotfile);
		}
	}

	if (time + time_tolerance >= userData_.next_history_time_) {
		DiagnosticsRecord const diagnostics = computeDiagnostics(*this);
		recordHistory(*this, diagnostics);
	}
}

template <> void QuokkaSimulation<DustMagnetizedRDI>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) // NOLINT
{
	amrex::ignore_unused(lev);
	amrex::ignore_unused(time);

	double const gas_accel_x = -dust_to_gas_mass_ratio * bar_a / (1.0 + dust_to_gas_mass_ratio);
	double const dust_accel_x = bar_a / (1.0 + dust_to_gas_mass_ratio);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const rho_g = state(i, j, k, HydroSystem<DustMagnetizedRDI>::density_index);
			amrex::Real px_g = state(i, j, k, HydroSystem<DustMagnetizedRDI>::x1Momentum_index);
			amrex::Real const py_g = state(i, j, k, HydroSystem<DustMagnetizedRDI>::x2Momentum_index);
			amrex::Real const pz_g = state(i, j, k, HydroSystem<DustMagnetizedRDI>::x3Momentum_index);
			amrex::Real const kinetic_old = (px_g * px_g + py_g * py_g + pz_g * pz_g) / (2.0 * rho_g);

			px_g += dt_lev * rho_g * gas_accel_x;
			amrex::Real const kinetic_new = (px_g * px_g + py_g * py_g + pz_g * pz_g) / (2.0 * rho_g);

			state(i, j, k, HydroSystem<DustMagnetizedRDI>::x1Momentum_index) = px_g;
			state(i, j, k, HydroSystem<DustMagnetizedRDI>::energy_index) += kinetic_new - kinetic_old;
			state(i, j, k, HydroSystem<DustMagnetizedRDI>::internalEnergy_index) = 0.0;

			amrex::Real const rho_d = state(i, j, k, HydroSystem<DustMagnetizedRDI>::dustDensity_index);
			state(i, j, k, HydroSystem<DustMagnetizedRDI>::x1DustMomentum_index) += dt_lev * rho_d * dust_accel_x;
		});
	}
}

auto problem_main() -> int
{
	loadProblemParameters();
	EquilibriumState const equilibrium = solveDriftEquilibrium();
	applyEquilibriumState(equilibrium);

	amrex::Print() << "DustMagnetizedRDI setup:\n";
	amrex::Print() << std::format("  bar_a              = {:.6f}\n", bar_a);
	amrex::Print() << std::format("  grain radius       = {:.6f}\n", g_grain_radius);
	amrex::Print() << std::format("  grain density      = {:.6f}\n", g_grain_density);
	amrex::Print() << std::format("  grain size param.  = {:.6f}\n", grainSizeParameter());
	amrex::Print() << std::format("  xi                 = {:.6f}\n", dimensionless_charge_to_mass_ratio);
	amrex::Print() << std::format("  dust-to-gas ratio  = {:.6f}\n", dust_to_gas_mass_ratio);
	amrex::Print() << std::format("  beta               = {:.6f}\n", beta_param);
	amrex::Print() << std::format("  theta_Ba [deg]     = {:.6f}\n", theta_Ba_deg);
	amrex::Print() << std::format("  equilibrium t_s    = {:.6f}\n", equilibrium.stop_time_);
	amrex::Print() << std::format("  equilibrium tau    = {:.6f}\n", equilibrium.tau_);
	amrex::Print() << std::format("  |w_s| / c_s        = {:.6f}\n", equilibrium.drift_speed_);
	amrex::Print() << std::format("  angle(w_s, B) [deg]= {:.6f}\n", equilibrium.drift_angle_to_b_deg_);
	amrex::Print() << std::format("  history dt / t_s^0 = {:.6f}\n", g_history_dt_over_ts0);
	amrex::Print() << std::format("  noise amplitude    = {:.6e}\n", g_noise_amplitude);
	amrex::Print() << std::format("  noise seed         = {}\n", g_noise_seed);

	auto BCs_cc = quokka::BC<DustMagnetizedRDI>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	auto BCs_fc = quokka::BC_fc<DustMagnetizedRDI>(quokka::BCType::mathematicalBndryTypes::periodic, quokka::BCType::mathematicalBndryTypes::periodic,
						       quokka::BCType::mathematicalBndryTypes::periodic);
	QuokkaSimulation<DustMagnetizedRDI> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 2;

	sim.setInitialConditions();
	DiagnosticsRecord const initial_diagnostics = computeDiagnostics(sim);
	recordHistory(sim, initial_diagnostics, true);
	sim.evolve();

	DiagnosticsRecord const final_diagnostics = computeDiagnostics(sim);
	recordHistory(sim, final_diagnostics, true);
	for (int i = 0; i < 3; ++i) {
		if (!sim.userData_.stage_written_[i]) {
			amrex::Print() << std::format("Warning: DustMagnetizedRDI stage '{}' was not reached: target t = {:.6f} ({:.3f} t_s^0), final t = "
						      "{:.6f} ({:.3f} t_s^0). No stage output was written.\n",
						      stage_labels[i], g_stage_target_times[i], g_stage_times_over_ts0[i], final_diagnostics.time_,
						      final_diagnostics.time_ / equilibrium.stop_time_);
		}
	}

	if (amrex::ParallelDescriptor::IOProcessor() && g_write_csv) {
		writeGrowthHistoryCsv(sim.userData_);
		writeSummaryCsv(sim, equilibrium, sim.userData_);
	}

	double max_sigma_b = 0.0;
	for (double const value : sim.userData_.sigma_bmag_) {
		max_sigma_b = std::max(max_sigma_b, value);
	}
	double max_sigma_log_rho_d = 0.0;
	for (double const value : sim.userData_.sigma_log_rho_d_) {
		max_sigma_log_rho_d = std::max(max_sigma_log_rho_d, value);
	}

	bool const finite = final_diagnostics.finite_;
	bool const magnetic_growth_visible = max_sigma_b > 1.0e-4;
	bool const dust_growth_visible = max_sigma_log_rho_d > 1.0e-4;
	bool const all_stages_reached = std::ranges::all_of(sim.userData_.stage_written_, [](bool reached) { return reached; });

	amrex::Print() << std::format("  max sigma(|B|)         = {:.6e}\n", max_sigma_b);
	amrex::Print() << std::format("  max sigma(log rho_d)   = {:.6e}\n", max_sigma_log_rho_d);
	amrex::Print() << std::format("  all target stages      = {}\n", all_stages_reached ? "yes" : "no");
	if (!(magnetic_growth_visible && dust_growth_visible)) {
		amrex::Print() << "Warning: physical RDI growth is not visible at this resolution or run time.\n";
	}

	if (!(finite && all_stages_reached)) {
		amrex::Print() << "DustMagnetizedRDI FAILED.\n";
		return 1;
	}

	amrex::Print() << "DustMagnetizedRDI PASSED.\n";
	return 0;
}
