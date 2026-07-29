/// \file testDustyOrszagTangPaper.cpp
/// \brief Paper-only Dusty Orszag-Tang figure generator inspired by Moseley et al. (2023), Section 3.4.

#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "QuokkaSimulation.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <gcem.hpp>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct DustyOrszagTangPaper {
};

namespace
{
constexpr double pi = std::numbers::pi;
constexpr double gamma_gas = 5.0 / 3.0;
constexpr double inv_sqrt_4pi = 1.0 / gcem::sqrt(4.0 * pi);
constexpr double rho_gas0 = 25.0 / (36.0 * pi);
constexpr double pressure0 = 5.0 / (12.0 * pi);
constexpr double stopping_time0 = 0.1;
constexpr double dimensionless_charge_to_mass_ratio0 = 100.0;
constexpr double tiny_number = 1.0e-14;
constexpr double first_snapshot_time = 0.25;
constexpr double second_snapshot_time = 0.5;
constexpr double shock_window_ymax = 0.3;

AMREX_GPU_MANAGED double g_initial_dust_density = 1.0e-1;    // NOLINT
AMREX_GPU_MANAGED double g_stopping_time = stopping_time0;   // NOLINT
AMREX_GPU_MANAGED double g_dimensionless_charge_to_mass_ratio = dimensionless_charge_to_mass_ratio0; // NOLINT
std::string g_active_case_tag;				     // NOLINT
std::string g_active_case_label;			     // NOLINT
std::string g_output_prefix = "dusty_orszag_tang_paper";     // NOLINT
std::string g_resolution_tag = "64";			     // NOLINT
bool g_capture_slice_csv = true;			     // NOLINT
bool g_capture_profile_csv = true;			     // NOLINT

struct ProblemRuntimeConfig {
	bool write_csv_ = true;
	bool write_slice_csv_ = true;
	bool write_profile_csv_ = true;
	std::string output_prefix_ = "dusty_orszag_tang_paper";
	amrex::Vector<std::string> case_tags_;
};

// input parameters for one dusty Orszag-Tang run
struct CaseConfig {
	std::string tag_;
	std::string label_;
	double dust_density0_ = 0.0;
	double epsilon0_ = 0.0;
};

// 2D mid-plane slice exported for the Fig. 6 analogue
struct SliceData {
	std::string case_tag_;
	std::string snapshot_tag_;
	double time_ = 0.0;
	std::vector<double> x_;
	std::vector<double> y_;
	std::vector<double> rho_g_;
	std::vector<double> rho_d_scaled_;
	std::vector<double> epsilon_local_;
};

// 1D x = 0.5 profile exported for the Fig. 7 analogue
struct ProfileData {
	std::string case_tag_;
	std::string snapshot_tag_;
	double time_ = 0.0;
	std::vector<double> y_;
	std::vector<double> rho_g_;
	std::vector<double> rho_d_scaled_;
	std::vector<double> v_gx_;
	std::vector<double> v_gy_;
	std::vector<double> v_dx_;
	std::vector<double> v_dy_;
};

// diagnostics collected at one output time
struct SnapshotData {
	std::optional<SliceData> slice_;
	std::optional<ProfileData> profile_;
	double shock_position_ = 0.0;
	double max_drift_ = 0.0;
	bool finite_ = true;
};

// complete output bundle for one dust-loading case
struct CaseResult {
	CaseConfig config_;
	SnapshotData snap_025_;
	SnapshotData snap_050_;
};

// reconstruct the active case metadata for mid-run snapshot capture
auto activeCaseConfig() -> CaseConfig
{
	return {
	    .tag_ = g_active_case_tag, .label_ = g_active_case_label, .dust_density0_ = g_initial_dust_density, .epsilon0_ = g_initial_dust_density / rho_gas0};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto vectorPotentialAz(double x, double y) -> double
{
	return inv_sqrt_4pi * (std::cos(4.0 * pi * x) + 2.0 * std::cos(2.0 * pi * y)) / (4.0 * pi);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto BxFace(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return (vectorPotentialAz(xL, yL + dx[1]) - vectorPotentialAz(xL, yL)) / dx[1];
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto ByFace(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return -(vectorPotentialAz(xL + dx[0], yL) - vectorPotentialAz(xL, yL)) / dx[0];
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeTotalEnergy(double rho_g, double vx_g, double vy_g, double bx, double by) -> double
{
	const double kinetic = 0.5 * rho_g * (vx_g * vx_g + vy_g * vy_g);
	const double internal = pressure0 / (gamma_gas - 1.0);
	const double magnetic = 0.5 * (bx * bx + by * by);
	return internal + kinetic + magnetic;
}

auto makeCaseConfigs() -> std::vector<CaseConfig>
{
	return {{"high_epsilon", "high epsilon", 1.0e-1, 1.0e-1 / rho_gas0}, {"low_epsilon", "low epsilon", 1.0e-6, 1.0e-6 / rho_gas0}};
}

auto selectCases(amrex::Vector<std::string> const &requested_tags) -> std::vector<CaseConfig>
{
	std::vector<CaseConfig> all_cases = makeCaseConfigs();
	if (requested_tags.empty()) {
		return all_cases;
	}

	std::vector<CaseConfig> selected_cases;
	selected_cases.reserve(requested_tags.size());
	for (auto const &tag : requested_tags) {
		auto const it = std::find_if(all_cases.begin(), all_cases.end(), [&](CaseConfig const &config) { return config.tag_ == tag; });
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(it != all_cases.end(), "DustyOrszagTangPaper received an unknown problem.case_tags entry.");
		selected_cases.push_back(*it);
	}
	return selected_cases;
}

auto readProblemRuntimeConfig() -> ProblemRuntimeConfig
{
	ProblemRuntimeConfig config;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", config.write_csv_);
	pp.query("write_slice_csv", config.write_slice_csv_);
	pp.query("write_profile_csv", config.write_profile_csv_);
	pp.query("output_prefix", config.output_prefix_);
	pp.queryarr("case_tags", config.case_tags_);

	if (!config.write_csv_) {
		config.write_slice_csv_ = false;
		config.write_profile_csv_ = false;
	}
	if (config.write_csv_) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(config.write_slice_csv_ || config.write_profile_csv_,
						 "DustyOrszagTangPaper requires at least one of problem.write_slice_csv or problem.write_profile_csv.");
	}
	return config;
}

auto snapshotTag(double time) -> std::string
{
	if (std::abs(time - first_snapshot_time) < 1.0e-12) {
		return "t0p25";
	}
	return "t0p50";
}

auto csvFilename(std::string const &case_tag, std::string const &snapshot_tag, std::string const &kind) -> std::string
{
	return std::format("{}_{}_{}_{}_{}.csv", g_output_prefix, g_resolution_tag, case_tag, snapshot_tag, kind);
}

void writeSliceCsv(SliceData const &slice)
{
	std::ofstream file(csvFilename(slice.case_tag_, slice.snapshot_tag_, "slice"));
	file << "x,y,rho_g,rho_d_scaled,epsilon_local\n";
	for (size_t idx = 0; idx < slice.x_.size(); ++idx) {
		file << slice.x_[idx] << "," << slice.y_[idx] << "," << slice.rho_g_[idx] << "," << slice.rho_d_scaled_[idx] << "," << slice.epsilon_local_[idx]
		     << "\n";
	}
}

void writeProfileCsv(ProfileData const &profile)
{
	std::ofstream file(csvFilename(profile.case_tag_, profile.snapshot_tag_, "profile"));
	file << "y,rho_g,rho_d_scaled,v_gx,v_gy,v_dx,v_dy\n";
	for (size_t idx = 0; idx < profile.y_.size(); ++idx) {
		file << profile.y_[idx] << "," << profile.rho_g_[idx] << "," << profile.rho_d_scaled_[idx] << "," << profile.v_gx_[idx] << ","
		     << profile.v_gy_[idx] << "," << profile.v_dx_[idx] << "," << profile.v_dy_[idx] << "\n";
	}
}

auto profileIsFinite(ProfileData const &profile) -> bool
{
	auto const check = [](std::vector<double> const &values) {
		return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
	};
	return check(profile.rho_g_) && check(profile.rho_d_scaled_) && check(profile.v_gx_) && check(profile.v_gy_) && check(profile.v_dx_) &&
	       check(profile.v_dy_);
}

auto sliceIsFinite(SliceData const &slice) -> bool
{
	auto const check = [](std::vector<double> const &values) {
		return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
	};
	return check(slice.rho_g_) && check(slice.rho_d_scaled_) && check(slice.epsilon_local_);
}

// locate the outer shock-like front in the y profile
auto detectShockPosition(ProfileData const &profile) -> double
{
	double max_jump = -1.0;
	for (size_t i = 0; i + 1 < profile.y_.size(); ++i) {
		if (profile.y_[i] > shock_window_ymax) {
			break;
		}
		double const jump = std::abs(profile.rho_g_[i + 1] - profile.rho_g_[i]);
		max_jump = std::max(jump, max_jump);
	}

	double shock_y = 0.0;
	double const threshold = 0.6 * max_jump;
	for (size_t i = 0; i + 1 < profile.y_.size(); ++i) {
		if (profile.y_[i] > shock_window_ymax) {
			break;
		}
		double const jump = std::abs(profile.rho_g_[i + 1] - profile.rho_g_[i]);
		if (jump >= threshold) {
			shock_y = 0.5 * (profile.y_[i] + profile.y_[i + 1]);
		}
	}
	return shock_y;
}

// measure the largest dust-gas velocity offset in the plotted window
auto maxDustDrift(ProfileData const &profile) -> double
{
	double max_drift = 0.0;
	for (size_t i = 0; i < profile.y_.size(); ++i) {
		if (profile.y_[i] > shock_window_ymax) {
			break;
		}
		double const dvx = profile.v_dx_[i] - profile.v_gx_[i];
		double const dvy = profile.v_dy_[i] - profile.v_gy_[i];
		max_drift = std::max(max_drift, std::hypot(dvx, dvy));
	}
	return max_drift;
}

template <typename problem_t> auto makePeriodicBCsCC() -> amrex::Vector<amrex::BCRec>
{
	return quokka::BC<problem_t>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
}

template <typename problem_t> auto makePeriodicBCsFC() -> amrex::Vector<amrex::BCRec>
{
	return quokka::BC_fc<problem_t>(quokka::BCType::mathematicalBndryTypes::periodic, quokka::BCType::mathematicalBndryTypes::periodic,
					quokka::BCType::mathematicalBndryTypes::periodic);
}

template <typename problem_t> auto extractSlice(QuokkaSimulation<problem_t> &sim, CaseConfig const &config, double time) -> SliceData
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.finest_level == 0, "DustyOrszagTangPaper slice extraction only supports single-level runs.");

	auto const domain = sim.Geom(0).Domain();
	auto const lo = amrex::lbound(domain);
	int const nx = domain.length(0);
	int const ny = domain.length(1);
	int const kslice = lo.z + domain.length(2) / 2;
	int const npts = nx * ny;
	auto const prob_lo = sim.Geom(0).ProbLoArray();
	auto const dx = sim.Geom(0).CellSizeArray();

	amrex::Gpu::HostVector<amrex::Real> rho_g(npts, 0.0);
	amrex::Gpu::HostVector<amrex::Real> rho_d_scaled(npts, 0.0);
	amrex::Gpu::HostVector<amrex::Real> epsilon_local(npts, 0.0);
	auto *rho_g_ptr = rho_g.data();
	auto *rho_d_scaled_ptr = rho_d_scaled.data();
	auto *epsilon_local_ptr = epsilon_local.data();
	double const epsilon0 = config.epsilon0_;
	auto const &state_mf = sim.state_new_cc_[0];
	amrex::Real const tiny_number_local = tiny_number;

	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		amrex::Box slice_box = mfi.validbox();
		slice_box.setSmall(2, kslice);
		slice_box.setBig(2, kslice);
		auto const state = state_mf.const_array(mfi);

		if (!slice_box.ok()) {
			continue;
		}

		amrex::ParallelFor(slice_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			int const idx = (j - lo.y) * nx + (i - lo.x);
			double const rho_g_cell = state(i, j, k, HydroSystem<problem_t>::density_index);
			double const rho_d_cell = state(i, j, k, HydroSystem<problem_t>::dustDensity_index);
			rho_g_ptr[idx] = rho_g_cell;
			rho_d_scaled_ptr[idx] = rho_d_cell / amrex::max(epsilon0, tiny_number_local);
			epsilon_local_ptr[idx] = (rho_g_cell > 0.0) ? rho_d_cell / rho_g_cell : 0.0;
		});
	}
	amrex::Gpu::streamSynchronize();

	amrex::ParallelDescriptor::ReduceRealSum(rho_g.data(), npts);
	amrex::ParallelDescriptor::ReduceRealSum(rho_d_scaled.data(), npts);
	amrex::ParallelDescriptor::ReduceRealSum(epsilon_local.data(), npts);

	SliceData slice;
	slice.case_tag_ = config.tag_;
	slice.snapshot_tag_ = snapshotTag(time);
	slice.time_ = time;
	slice.x_.resize(npts);
	slice.y_.resize(npts);
	slice.rho_g_.assign(rho_g.begin(), rho_g.end());
	slice.rho_d_scaled_.assign(rho_d_scaled.begin(), rho_d_scaled.end());
	slice.epsilon_local_.assign(epsilon_local.begin(), epsilon_local.end());

	for (int j = 0; j < ny; ++j) {
		for (int i = 0; i < nx; ++i) {
			int const idx = j * nx + i;
			slice.x_[idx] = prob_lo[0] + (lo.x + i + 0.5) * dx[0];
			slice.y_[idx] = prob_lo[1] + (lo.y + j + 0.5) * dx[1];
		}
	}

	return slice;
}

// average the two cells adjacent to x = 0.5 and then average over z
template <typename problem_t> auto extractProfile(QuokkaSimulation<problem_t> &sim, CaseConfig const &config, double time) -> ProfileData
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.finest_level == 0, "DustyOrszagTangPaper profile extraction only supports single-level runs.");

	auto const domain = sim.Geom(0).Domain();
	auto const lo = amrex::lbound(domain);
	auto const dx = sim.Geom(0).CellSizeArray();
	auto const prob_lo = sim.Geom(0).ProbLoArray();
	int const nx = domain.length(0);
	int const ny = domain.length(1);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE((nx % 2) == 0, "DustyOrszagTangPaper profile extraction expects an even number of x cells.");
	int const i_left = lo.x + nx / 2 - 1;
	int const i_right = lo.x + nx / 2;
	double const rescale = static_cast<double>(nx) / 2.0;
	double const epsilon0 = config.epsilon0_;
	amrex::Real const tiny_number_local = tiny_number;

	auto rho_g_avg = sim.computeAxisAlignedProfile(1, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) {
		return ((i == i_left) || (i == i_right)) ? state(i, j, k, HydroSystem<problem_t>::density_index) : 0.0;
	});
	auto rho_d_avg = sim.computeAxisAlignedProfile(1, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) {
		return ((i == i_left) || (i == i_right)) ? state(i, j, k, HydroSystem<problem_t>::dustDensity_index) : 0.0;
	});
	auto mom_gx_avg = sim.computeAxisAlignedProfile(1, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) {
		return ((i == i_left) || (i == i_right)) ? state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) : 0.0;
	});
	auto mom_gy_avg = sim.computeAxisAlignedProfile(1, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) {
		return ((i == i_left) || (i == i_right)) ? state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) : 0.0;
	});
	auto mom_dx_avg = sim.computeAxisAlignedProfile(1, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) {
		return ((i == i_left) || (i == i_right)) ? state(i, j, k, HydroSystem<problem_t>::x1DustMomentum_index) : 0.0;
	});
	auto mom_dy_avg = sim.computeAxisAlignedProfile(1, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) {
		return ((i == i_left) || (i == i_right)) ? state(i, j, k, HydroSystem<problem_t>::x2DustMomentum_index) : 0.0;
	});

	ProfileData profile;
	profile.case_tag_ = config.tag_;
	profile.snapshot_tag_ = snapshotTag(time);
	profile.time_ = time;
	profile.y_.resize(ny);
	profile.rho_g_.resize(ny);
	profile.rho_d_scaled_.resize(ny);
	profile.v_gx_.resize(ny);
	profile.v_gy_.resize(ny);
	profile.v_dx_.resize(ny);
	profile.v_dy_.resize(ny);

	for (int j = 0; j < ny; ++j) {
		double const rho_g = rescale * rho_g_avg[j];
		double const rho_d = rescale * rho_d_avg[j];
		double const mom_gx = rescale * mom_gx_avg[j];
		double const mom_gy = rescale * mom_gy_avg[j];
		double const mom_dx = rescale * mom_dx_avg[j];
		double const mom_dy = rescale * mom_dy_avg[j];

		profile.y_[j] = prob_lo[1] + (lo.y + j + 0.5) * dx[1];
		profile.rho_g_[j] = rho_g;
		profile.rho_d_scaled_[j] = rho_d / amrex::max(epsilon0, tiny_number_local);
		profile.v_gx_[j] = (rho_g > 0.0) ? mom_gx / rho_g : 0.0;
		profile.v_gy_[j] = (rho_g > 0.0) ? mom_gy / rho_g : 0.0;
		profile.v_dx_[j] = (rho_d > 0.0) ? mom_dx / rho_d : 0.0;
		profile.v_dy_[j] = (rho_d > 0.0) ? mom_dy / rho_d : 0.0;
	}

	return profile;
}

template <typename problem_t>
auto extractSnapshot(QuokkaSimulation<problem_t> &sim, CaseConfig const &config, double time, bool capture_slice, bool capture_profile) -> SnapshotData
{
	SnapshotData snapshot;
	if (capture_slice) {
		snapshot.slice_ = extractSlice(sim, config, time);
		snapshot.finite_ = snapshot.finite_ && sliceIsFinite(*snapshot.slice_);
	}
	if (capture_profile) {
		snapshot.profile_ = extractProfile(sim, config, time);
		snapshot.shock_position_ = detectShockPosition(*snapshot.profile_);
		snapshot.max_drift_ = maxDustDrift(*snapshot.profile_);
		snapshot.finite_ = snapshot.finite_ && profileIsFinite(*snapshot.profile_);
	}
	return snapshot;
}

// cached intermediate output recorded when the run first reaches t = 0.25
struct DustyOrszagTangPaperHistory {
	bool has_snap_025_ = false;
	SnapshotData snap_025_;
};

} // namespace

template <> struct SimulationData<DustyOrszagTangPaper> : DustyOrszagTangPaperHistory {
};

namespace
{

// run one dust-loading case and collect the requested diagnostics at t = 0.25 and t = 0.5
template <typename problem_t> auto runCase(CaseConfig const &config) -> CaseResult
{
	g_initial_dust_density = config.dust_density0_;
	g_stopping_time = stopping_time0;
	g_dimensionless_charge_to_mass_ratio = dimensionless_charge_to_mass_ratio0;
	g_active_case_tag = config.tag_;
	g_active_case_label = config.label_;

	auto BCs_cc = makePeriodicBCsCC<problem_t>();
	auto BCs_fc = makePeriodicBCsFC<problem_t>();
	QuokkaSimulation<problem_t> sim(BCs_cc, BCs_fc);

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.Geom(0).Domain().length(0) == sim.Geom(0).Domain().length(1),
					 "DustyOrszagTangPaper expects equal x and y resolution.");
	g_resolution_tag = std::to_string(sim.Geom(0).Domain().length(0));

	amrex::Print() << std::format("Running DustyOrszagTangPaper case: {} (epsilon = {:.6e}, resolution = {}x{})\n", config.tag_, config.epsilon0_,
				      sim.Geom(0).Domain().length(0), sim.Geom(0).Domain().length(1));

	sim.reconstructionOrder_ = 2;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();
	SnapshotData snap_025 = sim.userData_.has_snap_025_ ? sim.userData_.snap_025_
							    : extractSnapshot(sim, config, first_snapshot_time, g_capture_slice_csv, g_capture_profile_csv);
	SnapshotData snap_050 = extractSnapshot(sim, config, second_snapshot_time, g_capture_slice_csv, g_capture_profile_csv);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (g_capture_slice_csv) {
			if (snap_025.slice_.has_value() && snap_050.slice_.has_value()) {
				writeSliceCsv(*snap_025.slice_);
				writeSliceCsv(*snap_050.slice_);
			} else {
				amrex::Abort("DustyOrszagTangPaper requested slice CSV output, but slice data are missing.");
			}
		}
		if (g_capture_profile_csv) {
			if (snap_025.profile_.has_value() && snap_050.profile_.has_value()) {
				writeProfileCsv(*snap_025.profile_);
				writeProfileCsv(*snap_050.profile_);
			} else {
				amrex::Abort("DustyOrszagTangPaper requested profile CSV output, but profile data are missing.");
			}
		}
	}

	CaseResult result;
	result.config_ = config;
	result.snap_025_ = std::move(snap_025);
	result.snap_050_ = std::move(snap_050);
	return result;
}

} // namespace

template <> struct quokka::EOS_Traits<DustyOrszagTangPaper> {
	static constexpr double gamma = gamma_gas;
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = 1.0;
};

template <> struct Physics_Traits<DustyOrszagTangPaper> : DefaultPhysicsTraits {
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

template <> void QuokkaSimulation<DustyOrszagTangPaper>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::Array4<double> const &state_cc = grid_elem.array_;
	amrex::Box const &indexRange = grid_elem.indexRange_;
	int const ncomp_cc = Physics_Indices<DustyOrszagTangPaper>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		double const x = prob_lo[0] + ((i + 0.5) * dx[0]);
		double const y = prob_lo[1] + ((j + 0.5) * dx[1]);

		double const vx_g = -std::sin(2.0 * pi * y);
		double const vy_g = std::sin(2.0 * pi * x);
		double const bx_cc = 0.5 * (BxFace(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + BxFace(x + 0.5 * dx[0], y - 0.5 * dx[1], dx));
		double const by_cc = 0.5 * (ByFace(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + ByFace(x - 0.5 * dx[0], y + 0.5 * dx[1], dx));

		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::density_index) = rho_gas0;
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::x1Momentum_index) = rho_gas0 * vx_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::x2Momentum_index) = rho_gas0 * vy_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::internalEnergy_index) = pressure0 / (gamma_gas - 1.0);
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::energy_index) = computeTotalEnergy(rho_gas0, vx_g, vy_g, bx_cc, by_cc);

		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::dustDensity_index) = g_initial_dust_density;
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::x1DustMomentum_index) = g_initial_dust_density * vx_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::x2DustMomentum_index) = g_initial_dust_density * vy_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTangPaper>::x3DustMomentum_index) = 0.0;
	});
}

template <> void QuokkaSimulation<DustyOrszagTangPaper>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::Array4<double> const &state_fc = grid_elem.array_;
	amrex::Box const &indexRange = grid_elem.indexRange_;
	quokka::direction const dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double const xL = prob_lo[0] + i * dx[0];
		double const yL = prob_lo[1] + j * dx[1];

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<DustyOrszagTangPaper>::mhdFirstIndex) = BxFace(xL, yL, dx);
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<DustyOrszagTangPaper>::mhdFirstIndex) = ByFace(xL, yL, dx);
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<DustyOrszagTangPaper>::mhdFirstIndex) = 0.0;
		}
	});
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustyOrszagTangPaper>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
											    amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
											    amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/,
											    double /*cs*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha{};
	alpha[0] = 1.0 / g_stopping_time;
	return alpha;
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustyOrszagTangPaper>::ComputeDustDimensionlessChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> dimensionless_charge_to_mass_ratio{};
	dimensionless_charge_to_mass_ratio[0] = g_dimensionless_charge_to_mass_ratio;
	return dimensionless_charge_to_mass_ratio;
}

template <> void QuokkaSimulation<DustyOrszagTangPaper>::computeAfterTimestep()
{
	if (!userData_.has_snap_025_ && (tNew_[0] + 1.0e-12 >= first_snapshot_time)) {
		userData_.snap_025_ = extractSnapshot(*this, activeCaseConfig(), first_snapshot_time, g_capture_slice_csv, g_capture_profile_csv);
		userData_.has_snap_025_ = true;
	}
}

auto problem_main() -> int
{
	ProblemRuntimeConfig const runtime_config = readProblemRuntimeConfig();
	g_output_prefix = runtime_config.output_prefix_;
	g_capture_slice_csv = runtime_config.write_slice_csv_;
	g_capture_profile_csv = runtime_config.write_profile_csv_;

	std::vector<CaseConfig> const cases = selectCases(runtime_config.case_tags_);
	std::vector<CaseResult> results;
	results.reserve(cases.size());
	for (auto const &config : cases) {
		results.push_back(runCase<DustyOrszagTangPaper>(config));
	}

	auto const all_finite =
	    std::all_of(results.begin(), results.end(), [](CaseResult const &result) { return result.snap_025_.finite_ && result.snap_050_.finite_; });

	auto const find_result = [&](std::string const &tag) -> CaseResult const * {
		auto const it = std::find_if(results.begin(), results.end(), [&](CaseResult const &result) { return result.config_.tag_ == tag; });
		return (it != results.end()) ? &(*it) : nullptr;
	};

	bool valid = all_finite;
	CaseResult const *high_epsilon = find_result("high_epsilon");
	CaseResult const *low_epsilon = find_result("low_epsilon");

	if (high_epsilon != nullptr && high_epsilon->snap_025_.profile_.has_value()) {
		amrex::Print() << std::format("  max drift high_epsilon t=0.25 = {:.6e}\n", high_epsilon->snap_025_.max_drift_);
		valid = valid && (high_epsilon->snap_025_.max_drift_ > 5.0e-2);
	}

	if ((high_epsilon != nullptr) && (low_epsilon != nullptr) && high_epsilon->snap_025_.profile_.has_value() &&
	    high_epsilon->snap_050_.profile_.has_value() && low_epsilon->snap_025_.profile_.has_value() && low_epsilon->snap_050_.profile_.has_value()) {
		double const shock_sep_025 = low_epsilon->snap_025_.shock_position_ - high_epsilon->snap_025_.shock_position_;
		double const shock_sep_050 = low_epsilon->snap_050_.shock_position_ - high_epsilon->snap_050_.shock_position_;

		amrex::Print() << std::format("  shock(high_epsilon, t=0.25) = {:.6e}\n", high_epsilon->snap_025_.shock_position_);
		amrex::Print() << std::format("  shock(low_epsilon,  t=0.25) = {:.6e}\n", low_epsilon->snap_025_.shock_position_);
		amrex::Print() << std::format("  shock(high_epsilon, t=0.50) = {:.6e}\n", high_epsilon->snap_050_.shock_position_);
		amrex::Print() << std::format("  shock(low_epsilon,  t=0.50) = {:.6e}\n", low_epsilon->snap_050_.shock_position_);
		amrex::Print() << std::format("  shock separation t=0.25 = {:.6e}\n", shock_sep_025);
		amrex::Print() << std::format("  shock separation t=0.50 = {:.6e}\n", shock_sep_050);

		valid = valid && (shock_sep_025 > 5.0e-3) && (shock_sep_050 > 5.0e-3);
	}

	if (!valid) {
		amrex::Print() << "DustyOrszagTangPaper FAILED.\n";
		return 1;
	}

	amrex::Print() << "DustyOrszagTangPaper PASSED.\n";
	return 0;
}
