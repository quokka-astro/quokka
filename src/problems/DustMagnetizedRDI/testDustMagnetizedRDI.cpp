/// \file testDustMagnetizedRDI.cpp
/// \brief Magnetized RDI analogue inspired by Moseley et al. (2022), Section 3.5.

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
constexpr double dust_density_floor = 1.0e-12;
constexpr double supersonic_eta = 9.0 * pi * gamma_iso / 128.0;
constexpr double time_tolerance = 1.0e-10;
constexpr double bar_a = 5.0;
constexpr double grain_radius_density_param = 5.0;
constexpr double xi_param = 10.0;
constexpr double mu_param = 0.01;
constexpr double beta_param = 2.0;
constexpr double theta_Ba_deg = 87.0;
constexpr double grain_density0 = 1.0;
constexpr double noise_amplitude_param = 1.0e-7;

constexpr std::array<char const *, 3> snapshot_tags = {"t6p2ts0", "t8p3ts0", "t17p0ts0"};
constexpr std::array<char const *, 3> face_tags = {"xface", "yface", "zface"};
constexpr std::array<double, 3> snapshot_times_over_ts0_default = {6.2, 8.3, 17.0};
constexpr double history_dt_over_ts0_default = 0.1;

double g_history_dt_over_ts0 = history_dt_over_ts0_default;			   // NOLINT
double g_history_dt_code = history_dt_over_ts0_default;				   // NOLINT
int g_slice_thickness_cells = 1;						   // NOLINT
bool g_write_csv = true;							   // NOLINT
std::array<double, 3> g_snapshot_times_over_ts0 = snapshot_times_over_ts0_default; // NOLINT
std::array<double, 3> g_snapshot_target_times = {0.0, 0.0, 0.0};		   // NOLINT
double g_equilibrium_ts = 0.0;							   // NOLINT

AMREX_GPU_MANAGED double g_grain_radius = grain_radius_density_param / grain_density0; // NOLINT
AMREX_GPU_MANAGED double g_grain_density = grain_density0;			       // NOLINT
AMREX_GPU_MANAGED double g_charge_to_mass = xi_param;				       // NOLINT
AMREX_GPU_MANAGED double g_noise_amplitude = noise_amplitude_param;		       // NOLINT
AMREX_GPU_MANAGED double g_Bx0 = 0.0;						       // NOLINT
AMREX_GPU_MANAGED double g_By0 = 0.0;						       // NOLINT
AMREX_GPU_MANAGED double g_Bz0 = 1.0;						       // NOLINT
AMREX_GPU_MANAGED double g_gas_vx0 = 0.0;					       // NOLINT
AMREX_GPU_MANAGED double g_gas_vy0 = 0.0;					       // NOLINT
AMREX_GPU_MANAGED double g_gas_vz0 = 0.0;					       // NOLINT
AMREX_GPU_MANAGED double g_dust_vx0 = 0.0;					       // NOLINT
AMREX_GPU_MANAGED double g_dust_vy0 = 0.0;					       // NOLINT
AMREX_GPU_MANAGED double g_dust_vz0 = 0.0;					       // NOLINT

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

struct FaceProjection {
	std::vector<double> u_;
	std::vector<double> v_;
	std::vector<double> bvec_minus_b0_norm_;
	std::vector<double> dust_overdensity_;
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
	std::array<bool, 3> snapshot_written_ = {false, false, false};
	std::array<double, 3> snapshot_times_ = {-1.0, -1.0, -1.0};
	std::array<double, 3> snapshot_sigmas_ = {0.0, 0.0, 0.0};
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

auto grainRadiusDensityProduct() -> double { return g_grain_radius * g_grain_density; }

auto computeSubsonicStoppingTime() -> double
{
	return std::sqrt(pi * gamma_iso) * grainRadiusDensityProduct() / (2.0 * std::numbers::sqrt2 * rho_gas0 * sound_speed);
}

auto solveDriftEquilibrium() -> EquilibriumState
{
	EquilibriumState result;
	result.magnetic_field_ = makeBackgroundMagneticField(beta_param, theta_Ba_deg);
	double const magnetic_field_norm = norm(result.magnetic_field_);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(magnetic_field_norm > 0.0, "DustMagnetizedRDI requires a non-zero background magnetic field.");

	Vec3 const acceleration = {bar_a, 0.0, 0.0};
	Vec3 const b_hat = result.magnetic_field_ / magnetic_field_norm;
	double const ts_sub = computeSubsonicStoppingTime();
	double const omega_L = xi_param * magnetic_field_norm;

	Vec3 drift = {(ts_sub / (1.0 + mu_param)) * bar_a, 0.0, 0.0};
	for (int iter = 0; iter < 64; ++iter) {
		double const drift_speed = norm(drift);
		double const stop_time = ts_sub / std::sqrt(1.0 + supersonic_eta * square(drift_speed / sound_speed));
		double const tau_local = omega_L * stop_time;

		Vec3 const rhs = (stop_time / (1.0 + mu_param)) * acceleration;
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
	result.tau_ = omega_L * result.stop_time_;
	result.drift_speed_ = norm(drift);
	result.drift_angle_to_b_deg_ = angleDegrees(drift, result.magnetic_field_);
	result.gas_velocity_ = (-mu_param / (1.0 + mu_param)) * drift;
	result.dust_velocity_ = (1.0 / (1.0 + mu_param)) * drift;
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
	pp.query("slice_thickness_cells", g_slice_thickness_cells);
	amrex::Vector<double> snapshot_times_over_ts0_vec;
	if (pp.queryarr("snapshot_times_over_ts0", snapshot_times_over_ts0_vec) != 0) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<amrex::Long>(snapshot_times_over_ts0_vec.size()) ==
						     static_cast<amrex::Long>(g_snapshot_times_over_ts0.size()),
						 "problem.snapshot_times_over_ts0 must contain exactly 3 values.");
		for (std::size_t i = 0; i < g_snapshot_times_over_ts0.size(); ++i) {
			g_snapshot_times_over_ts0[i] = snapshot_times_over_ts0_vec[i];
		}
	}

	if (g_history_dt_over_ts0 <= 0.0) {
		g_history_dt_over_ts0 = history_dt_over_ts0_default;
	}
	if (g_slice_thickness_cells <= 0) {
		g_slice_thickness_cells = 1;
	}
	for (double const snapshot_time_over_ts0 : g_snapshot_times_over_ts0) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(snapshot_time_over_ts0 > 0.0, "problem.snapshot_times_over_ts0 values must all be positive.");
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
	for (std::size_t i = 0; i < g_snapshot_target_times.size(); ++i) {
		g_snapshot_target_times[i] = g_snapshot_times_over_ts0[i] * equilibrium.stop_time_;
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
	auto key = static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) + 0x9e3779b9ULL;
	key = (key << 21U) ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(j)) + 0x7f4a7c15ULL);
	key = (key << 17U) ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(k)) + 0x94d049bbULL);
	key = (key << 13U) ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(component)) + 0x27d4eb2dULL);
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
			    amrex::Real const log_rho_d = std::log(amrex::max(rho_d, static_cast<amrex::Real>(dust_density_floor)));

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
	file << "t,sigma_log_rho_g,sigma_log_rho_d,sigma_vgx,sigma_vgy,sigma_vgz,sigma_vdx,sigma_vdy,sigma_vdz,sigma_bx,sigma_by,sigma_bz,sigma_bmag\n";
	for (size_t i = 0; i < history.t_.size(); ++i) {
		file << history.t_[i] << "," << history.sigma_log_rho_g_[i] << "," << history.sigma_log_rho_d_[i] << "," << history.sigma_vgx_[i] << ","
		     << history.sigma_vgy_[i] << "," << history.sigma_vgz_[i] << "," << history.sigma_vdx_[i] << "," << history.sigma_vdy_[i] << ","
		     << history.sigma_vdz_[i] << "," << history.sigma_bx_[i] << "," << history.sigma_by_[i] << "," << history.sigma_bz_[i] << ","
		     << history.sigma_bmag_[i] << "\n";
	}
}

void writeSummaryCsv(EquilibriumState const &equilibrium, DustMagnetizedRDIHistory const &history)
{
	std::ofstream file("dust_magnetized_rdi_summary.csv");
	file << "key,value\n";
	file << "bar_a," << bar_a << "\n";
	file << "epsilon," << grainRadiusDensityProduct() << "\n";
	file << "xi," << xi_param << "\n";
	file << "mu," << mu_param << "\n";
	file << "beta," << beta_param << "\n";
	file << "theta_Ba_deg," << theta_Ba_deg << "\n";
	file << "grain_radius," << g_grain_radius << "\n";
	file << "grain_density," << g_grain_density << "\n";
	file << "noise_amplitude," << noise_amplitude_param << "\n";
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
	file << "rho_g0," << rho_gas0 << "\n";
	file << "cs0," << sound_speed << "\n";
	for (int i = 0; i < 3; ++i) {
		file << std::format("snapshot_{}_target_time_ts0,{}\n", snapshot_tags[i], g_snapshot_times_over_ts0[i]);
		file << std::format("snapshot_{}_target_time_code,{}\n", snapshot_tags[i], g_snapshot_target_times[i]);
		file << std::format("snapshot_{}_time,{}\n", snapshot_tags[i], history.snapshot_times_[i]);
		file << std::format("snapshot_{}_time_ts0,{}\n", snapshot_tags[i], history.snapshot_times_[i] / std::max(equilibrium.stop_time_, tiny_number));
		file << std::format("snapshot_{}_sigma_bmag,{}\n", snapshot_tags[i], history.snapshot_sigmas_[i]);
	}
}

void writeFaceProjectionCsv(std::string const &snapshot_tag, std::string const &face_tag, FaceProjection const &projection)
{
	std::ofstream file(std::format("dust_magnetized_rdi_{}_{}.csv", snapshot_tag, face_tag));
	file << "u,v,bvec_minus_b0_norm,dust_overdensity\n";
	for (size_t i = 0; i < projection.u_.size(); ++i) {
		file << projection.u_[i] << "," << projection.v_[i] << "," << projection.bvec_minus_b0_norm_[i] << "," << projection.dust_overdensity_[i]
		     << "\n";
	}
}

template <typename problem_t> auto extractFaceProjection(QuokkaSimulation<problem_t> &sim, int normal_dir) -> FaceProjection
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.finest_level == 0, "DustMagnetizedRDI face extraction only supports single-level runs.");

	auto const &state_mf = sim.state_new_cc_[0];
	const auto domain = sim.Geom(0).Domain();
	const auto lo = amrex::lbound(domain);
	const auto dx = sim.Geom(0).CellSizeArray();
	const auto prob_lo = sim.Geom(0).ProbLoArray();
	const int nx = domain.length(0);
	const int ny = domain.length(1);
	const int nz = domain.length(2);
	const int slab_cells = std::clamp(g_slice_thickness_cells, 1, domain.length(normal_dir));
	const double mean_dust_density = std::max(mu_param * rho_gas0, dust_density_floor);
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

	amrex::Box slab = domain;
	if (normal_dir == 0) {
		int const hi_x = lo.x + nx - 1;
		slab.setSmall(0, hi_x - slab_cells + 1);
		slab.setBig(0, hi_x);
	} else if (normal_dir == 1) {
		slab.setSmall(1, lo.y);
		slab.setBig(1, lo.y + slab_cells - 1);
	} else {
		int const hi_z = lo.z + nz - 1;
		slab.setSmall(2, hi_z - slab_cells + 1);
		slab.setBig(2, hi_z);
	}

	const int npts = nu * nv;
	amrex::Gpu::DeviceVector<amrex::Real> bvec_minus_b0_norm_sum_d(npts, 0.0);
	amrex::Gpu::DeviceVector<amrex::Real> dust_sum_d(npts, 0.0);
	amrex::Gpu::DeviceVector<amrex::Real> count_sum_d(npts, 0.0);
	auto *bvec_minus_b0_norm_ptr = bvec_minus_b0_norm_sum_d.data();
	auto *dust_ptr = dust_sum_d.data();
	auto *count_ptr = count_sum_d.data();

	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		amrex::Box const box = mfi.validbox() & slab;
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
			amrex::Real const bvec_minus_b0_norm = std::sqrt(dbx * dbx + dby * dby + dbz * dbz);
			amrex::Real const dust_overdensity = state(i, j, k, HydroSystem<problem_t>::dustDensity_index) / mean_dust_density;

			amrex::Gpu::Atomic::Add(&bvec_minus_b0_norm_ptr[idx], bvec_minus_b0_norm);
			amrex::Gpu::Atomic::Add(&dust_ptr[idx], dust_overdensity);
			amrex::Gpu::Atomic::Add(&count_ptr[idx], 1.0_rt);
		});
	}
	amrex::Gpu::streamSynchronize();
	amrex::Gpu::HostVector<amrex::Real> bvec_minus_b0_norm_sum(npts);
	amrex::Gpu::HostVector<amrex::Real> dust_sum(npts);
	amrex::Gpu::HostVector<amrex::Real> count_sum(npts);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, bvec_minus_b0_norm_sum_d.begin(), bvec_minus_b0_norm_sum_d.end(), bvec_minus_b0_norm_sum.begin());
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, dust_sum_d.begin(), dust_sum_d.end(), dust_sum.begin());
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, count_sum_d.begin(), count_sum_d.end(), count_sum.begin());
	amrex::ParallelDescriptor::ReduceRealSum(bvec_minus_b0_norm_sum.data(), npts);
	amrex::ParallelDescriptor::ReduceRealSum(dust_sum.data(), npts);
	amrex::ParallelDescriptor::ReduceRealSum(count_sum.data(), npts);

	FaceProjection projection;
	projection.u_.resize(npts);
	projection.v_.resize(npts);
	projection.bvec_minus_b0_norm_.resize(npts);
	projection.dust_overdensity_.resize(npts);

	for (int iv = 0; iv < nv; ++iv) {
		for (int iu = 0; iu < nu; ++iu) {
			int const idx = iv * nu + iu;
			double const count = std::max(static_cast<double>(count_sum[idx]), 1.0);
			double u = 0.0;
			double v = 0.0;
			if (normal_dir == 0) {
				u = prob_lo[1] + (lo.y + iu + 0.5) * dx[1];
				v = prob_lo[2] + (lo.z + iv + 0.5) * dx[2];
			} else if (normal_dir == 1) {
				u = prob_lo[0] + (lo.x + iu + 0.5) * dx[0];
				v = prob_lo[2] + (lo.z + iv + 0.5) * dx[2];
			} else {
				u = prob_lo[0] + (lo.x + iu + 0.5) * dx[0];
				v = prob_lo[1] + (lo.y + iv + 0.5) * dx[1];
			}
			projection.u_[idx] = u;
			projection.v_[idx] = v;
			projection.bvec_minus_b0_norm_[idx] = bvec_minus_b0_norm_sum[idx] / count;
			projection.dust_overdensity_[idx] = dust_sum[idx] / count;
		}
	}

	return projection;
}

template <typename problem_t> void captureSnapshot(QuokkaSimulation<problem_t> &sim, int snapshot_index, DiagnosticsRecord const &diagnostics)
{
	sim.userData_.snapshot_written_[snapshot_index] = true;
	sim.userData_.snapshot_times_[snapshot_index] = diagnostics.time_;
	sim.userData_.snapshot_sigmas_[snapshot_index] = diagnostics.sigma_bmag_;

	if (g_write_csv) {
		for (int face = 0; face < 3; ++face) {
			FaceProjection const projection = extractFaceProjection(sim, face);
			if (amrex::ParallelDescriptor::IOProcessor()) {
				writeFaceProjectionCsv(snapshot_tags[snapshot_index], face_tags[face], projection);
			}
		}
	}
	amrex::Print() << std::format("Captured DustMagnetizedRDI snapshot '{}' at t = {:.6f} = {:.3f} t_s^0\n", snapshot_tags[snapshot_index],
				      diagnostics.time_, diagnostics.time_ / std::max(g_equilibrium_ts, tiny_number));
}

template <typename problem_t> void maybeCaptureSnapshots(QuokkaSimulation<problem_t> &sim, DiagnosticsRecord const &diagnostics)
{
	for (int i = 0; i < 3; ++i) {
		if (!sim.userData_.snapshot_written_[i] && diagnostics.time_ + time_tolerance >= g_snapshot_target_times[i]) {
			captureSnapshot(sim, i, diagnostics);
		}
	}
}

template <typename problem_t> void recordHistory(QuokkaSimulation<problem_t> &sim, bool force = false)
{
	DiagnosticsRecord const diagnostics = computeDiagnostics(sim);
	maybeCaptureSnapshots(sim, diagnostics);

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

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustMagnetizedRDI>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> q_over_m{};
	q_over_m[0] = g_charge_to_mass;
	return q_over_m;
}

template <> void QuokkaSimulation<DustMagnetizedRDI>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<DustMagnetizedRDI>::nvarTotal_cc;
	const double dust_density0 = std::max(mu_param * rho_gas0, dust_density_floor);
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

template <> void QuokkaSimulation<DustMagnetizedRDI>::computeAfterTimestep() { recordHistory(*this); }

template <> void QuokkaSimulation<DustMagnetizedRDI>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) // NOLINT
{
	amrex::ignore_unused(lev);
	amrex::ignore_unused(time);

	double const gas_accel_x = -mu_param * bar_a / (1.0 + mu_param);
	double const dust_accel_x = bar_a / (1.0 + mu_param);

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
	amrex::Print() << std::format("  a * rho_gr         = {:.6f}\n", grainRadiusDensityProduct());
	amrex::Print() << std::format("  xi                 = {:.6f}\n", xi_param);
	amrex::Print() << std::format("  mu                 = {:.6f}\n", mu_param);
	amrex::Print() << std::format("  beta               = {:.6f}\n", beta_param);
	amrex::Print() << std::format("  theta_Ba [deg]     = {:.6f}\n", theta_Ba_deg);
	amrex::Print() << std::format("  equilibrium t_s    = {:.6f}\n", equilibrium.stop_time_);
	amrex::Print() << std::format("  equilibrium tau    = {:.6f}\n", equilibrium.tau_);
	amrex::Print() << std::format("  |w_s| / c_s        = {:.6f}\n", equilibrium.drift_speed_);
	amrex::Print() << std::format("  angle(w_s, B) [deg]= {:.6f}\n", equilibrium.drift_angle_to_b_deg_);
	amrex::Print() << std::format("  history dt / t_s^0 = {:.6f}\n", g_history_dt_over_ts0);

	auto BCs_cc = quokka::BC<DustMagnetizedRDI>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	auto BCs_fc = quokka::BC_fc<DustMagnetizedRDI>(quokka::BCType::mathematicalBndryTypes::periodic, quokka::BCType::mathematicalBndryTypes::periodic,
						       quokka::BCType::mathematicalBndryTypes::periodic);
	QuokkaSimulation<DustMagnetizedRDI> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 2;
	sim.plotfileInterval_ = -1;
	sim.enableIterDustStoptime_ = 1;
	sim.print_dust_counter_ = false;
	sim.dust_omega_res_ = 1.0;

	sim.setInitialConditions();
	recordHistory(sim, true);
	sim.evolve();
	recordHistory(sim, true);

	DiagnosticsRecord const final_diagnostics = computeDiagnostics(sim);
	for (int i = 0; i < 3; ++i) {
		if (!sim.userData_.snapshot_written_[i] && final_diagnostics.finite_) {
			amrex::Print() << std::format("Warning: snapshot '{}' was not reached by the end of the run; writing the final state instead.\n",
						      snapshot_tags[i]);
			captureSnapshot(sim, i, final_diagnostics);
		}
	}

	if (amrex::ParallelDescriptor::IOProcessor() && g_write_csv) {
		writeGrowthHistoryCsv(sim.userData_);
		writeSummaryCsv(equilibrium, sim.userData_);
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
	bool const all_snapshot_targets_reached = [&]() {
		for (std::size_t i = 0; i < sim.userData_.snapshot_times_.size(); ++i) {
			if (sim.userData_.snapshot_times_[i] + time_tolerance < g_snapshot_target_times[i]) {
				return false;
			}
		}
		return true;
	}();

	amrex::Print() << std::format("  max sigma(|B|)         = {:.6e}\n", max_sigma_b);
	amrex::Print() << std::format("  max sigma(log rho_d)   = {:.6e}\n", max_sigma_log_rho_d);
	amrex::Print() << std::format("  all target snapshots   = {}\n", all_snapshot_targets_reached ? "yes" : "no");

	if (!(finite && magnetic_growth_visible && dust_growth_visible && all_snapshot_targets_reached)) {
		amrex::Print() << "DustMagnetizedRDI FAILED.\n";
		return 1;
	}

	amrex::Print() << "DustMagnetizedRDI PASSED.\n";
	return 0;
}
