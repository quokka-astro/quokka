/// \file testDustyOrszagTang.cpp
/// \brief Dusty Orszag-Tang vortex analogue inspired by Moseley et al. (2022), Section 3.4.

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
#include <string>
#include <vector>

struct DustyOrszagTang {
};

namespace
{
constexpr double pi = std::numbers::pi;
constexpr double gamma_gas = 5.0 / 3.0;
constexpr double inv_sqrt_4pi = 1.0 / gcem::sqrt(4.0 * pi);
constexpr double rho_gas0 = 25.0 / (36.0 * pi);
constexpr double pressure0 = 5.0 / (12.0 * pi);
constexpr double stopping_time0 = 0.1;
constexpr double charge_to_mass0 = 100.0;
constexpr double tiny_number = 1.0e-14;
constexpr double first_snapshot_time = 0.25;
constexpr double second_snapshot_time = 0.5;
constexpr double shock_window_ymax = 0.3;

AMREX_GPU_MANAGED double g_initial_dust_density = 1.0e-1;    // NOLINT
AMREX_GPU_MANAGED double g_stopping_time = stopping_time0;   // NOLINT
AMREX_GPU_MANAGED double g_charge_to_mass = charge_to_mass0; // NOLINT
std::string g_active_case_tag;				     // NOLINT
std::string g_active_case_label;			     // NOLINT

// input parameters for one dusty Orszag-Tang run
struct CaseConfig {
	std::string tag_;
	std::string label_;
	double dust_density0_ = 0.0;
	double mu0_ = 0.0;
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
	std::vector<double> mu_local_;
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
	SliceData slice_;
	ProfileData profile_;
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
	return {.tag_ = g_active_case_tag, .label_ = g_active_case_label, .dust_density0_ = g_initial_dust_density, .mu0_ = g_initial_dust_density / rho_gas0};
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
	return {{"high_mu", "high mu", 1.0e-1, 1.0e-1 / rho_gas0}, {"low_mu", "low mu", 1.0e-6, 1.0e-6 / rho_gas0}};
}

auto snapshotTag(double time) -> std::string
{
	if (std::abs(time - first_snapshot_time) < 1.0e-12) {
		return "t0p25";
	}
	return "t0p50";
}

void writeSliceCsv(const SliceData &slice)
{
	std::ofstream file(std::format("dusty_orszag_tang_{}_{}_slice.csv", slice.case_tag_, slice.snapshot_tag_));
	file << "x,y,rho_g,rho_d_scaled,mu_local\n";
	for (size_t idx = 0; idx < slice.x_.size(); ++idx) {
		file << slice.x_[idx] << "," << slice.y_[idx] << "," << slice.rho_g_[idx] << "," << slice.rho_d_scaled_[idx] << "," << slice.mu_local_[idx]
		     << "\n";
	}
}

void writeProfileCsv(const ProfileData &profile)
{
	std::ofstream file(std::format("dusty_orszag_tang_{}_{}_profile.csv", profile.case_tag_, profile.snapshot_tag_));
	file << "y,rho_g,rho_d_scaled,v_gx,v_gy,v_dx,v_dy\n";
	for (size_t idx = 0; idx < profile.y_.size(); ++idx) {
		file << profile.y_[idx] << "," << profile.rho_g_[idx] << "," << profile.rho_d_scaled_[idx] << "," << profile.v_gx_[idx] << ","
		     << profile.v_gy_[idx] << "," << profile.v_dx_[idx] << "," << profile.v_dy_[idx] << "\n";
	}
}

auto profileIsFinite(const ProfileData &profile) -> bool
{
	auto const check = [](const std::vector<double> &values) {
		return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
	};
	return check(profile.rho_g_) && check(profile.rho_d_scaled_) && check(profile.v_gx_) && check(profile.v_gy_) && check(profile.v_dx_) &&
	       check(profile.v_dy_);
}

auto sliceIsFinite(const SliceData &slice) -> bool
{
	auto const check = [](const std::vector<double> &values) {
		return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
	};
	return check(slice.rho_g_) && check(slice.rho_d_scaled_) && check(slice.mu_local_);
}

// locate the outer shock-like front in the y profile
auto detectShockPosition(const ProfileData &profile) -> double
{
	double max_jump = -1.0;
	for (size_t i = 0; i + 1 < profile.y_.size(); ++i) {
		if (profile.y_[i] > shock_window_ymax) {
			break;
		}
		const double jump = std::abs(profile.rho_g_[i + 1] - profile.rho_g_[i]);
		max_jump = std::max(jump, max_jump);
	}

	double shock_y = 0.0;
	double const threshold = 0.6 * max_jump;
	for (size_t i = 0; i + 1 < profile.y_.size(); ++i) {
		if (profile.y_[i] > shock_window_ymax) {
			break;
		}
		const double jump = std::abs(profile.rho_g_[i + 1] - profile.rho_g_[i]);
		if (jump >= threshold) {
			shock_y = 0.5 * (profile.y_[i] + profile.y_[i + 1]);
		}
	}
	return shock_y;
}

// measure the largest dust-gas velocity offset in the plotted window
auto maxDustDrift(const ProfileData &profile) -> double
{
	double max_drift = 0.0;
	for (size_t i = 0; i < profile.y_.size(); ++i) {
		if (profile.y_[i] > shock_window_ymax) {
			break;
		}
		const double dvx = profile.v_dx_[i] - profile.v_gx_[i];
		const double dvy = profile.v_dy_[i] - profile.v_gy_[i];
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

template <typename problem_t> auto extractSlice(QuokkaSimulation<problem_t> &sim, const CaseConfig &config, double time) -> SliceData
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.finest_level == 0, "DustyOrszagTang slice extraction only supports single-level runs.");

	const auto domain = sim.Geom(0).Domain();
	const auto lo = amrex::lbound(domain);
	const int nx = domain.length(0);
	const int ny = domain.length(1);
	const int kslice = lo.z + domain.length(2) / 2;
	const int npts = nx * ny;
	const auto prob_lo = sim.Geom(0).ProbLoArray();
	const auto dx = sim.Geom(0).CellSizeArray();

	amrex::Gpu::HostVector<amrex::Real> rho_g(npts, 0.0);
	amrex::Gpu::HostVector<amrex::Real> rho_d_scaled(npts, 0.0);
	amrex::Gpu::HostVector<amrex::Real> mu_local(npts, 0.0);
	auto *rho_g_ptr = rho_g.data();
	auto *rho_d_scaled_ptr = rho_d_scaled.data();
	auto *mu_local_ptr = mu_local.data();
	const double mu0 = config.mu0_;
	const auto &state_mf = sim.state_new_cc_[0];
	const amrex::Real tiny_number_local = tiny_number;

	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		amrex::Box slice_box = mfi.validbox();
		slice_box.setSmall(2, kslice);
		slice_box.setBig(2, kslice);
		const auto state = state_mf.const_array(mfi);

		if (!slice_box.ok()) {
			continue;
		}

		amrex::ParallelFor(slice_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const int idx = (j - lo.y) * nx + (i - lo.x);
			const double rho_g_cell = state(i, j, k, HydroSystem<problem_t>::density_index);
			const double rho_d_cell = state(i, j, k, HydroSystem<problem_t>::dustDensity_index);
			rho_g_ptr[idx] = rho_g_cell;
			rho_d_scaled_ptr[idx] = rho_d_cell / amrex::max(mu0, tiny_number_local);
			mu_local_ptr[idx] = (rho_g_cell > 0.0) ? rho_d_cell / rho_g_cell : 0.0;
		});
	}
	amrex::Gpu::streamSynchronize();

	amrex::ParallelDescriptor::ReduceRealSum(rho_g.data(), npts);
	amrex::ParallelDescriptor::ReduceRealSum(rho_d_scaled.data(), npts);
	amrex::ParallelDescriptor::ReduceRealSum(mu_local.data(), npts);

	SliceData slice;
	slice.case_tag_ = config.tag_;
	slice.snapshot_tag_ = snapshotTag(time);
	slice.time_ = time;
	slice.x_.resize(npts);
	slice.y_.resize(npts);
	slice.rho_g_.assign(rho_g.begin(), rho_g.end());
	slice.rho_d_scaled_.assign(rho_d_scaled.begin(), rho_d_scaled.end());
	slice.mu_local_.assign(mu_local.begin(), mu_local.end());

	for (int j = 0; j < ny; ++j) {
		for (int i = 0; i < nx; ++i) {
			const int idx = j * nx + i;
			slice.x_[idx] = prob_lo[0] + (lo.x + i + 0.5) * dx[0];
			slice.y_[idx] = prob_lo[1] + (lo.y + j + 0.5) * dx[1];
		}
	}

	return slice;
}

// average the two cells adjacent to x = 0.5 and then average over z
template <typename problem_t> auto extractProfile(QuokkaSimulation<problem_t> &sim, const CaseConfig &config, double time) -> ProfileData
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.finest_level == 0, "DustyOrszagTang profile extraction only supports single-level runs.");

	const auto domain = sim.Geom(0).Domain();
	const auto lo = amrex::lbound(domain);
	const auto dx = sim.Geom(0).CellSizeArray();
	const auto prob_lo = sim.Geom(0).ProbLoArray();
	const int nx = domain.length(0);
	const int ny = domain.length(1);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE((nx % 2) == 0, "DustyOrszagTang profile extraction expects an even number of x cells.");
	const int i_left = lo.x + nx / 2 - 1;
	const int i_right = lo.x + nx / 2;
	const double rescale = static_cast<double>(nx) / 2.0;
	const double mu0 = config.mu0_;
	const amrex::Real tiny_number_local = tiny_number;

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
		const double rho_g = rescale * rho_g_avg[j];
		const double rho_d = rescale * rho_d_avg[j];
		const double mom_gx = rescale * mom_gx_avg[j];
		const double mom_gy = rescale * mom_gy_avg[j];
		const double mom_dx = rescale * mom_dx_avg[j];
		const double mom_dy = rescale * mom_dy_avg[j];

		profile.y_[j] = prob_lo[1] + (lo.y + j + 0.5) * dx[1];
		profile.rho_g_[j] = rho_g;
		profile.rho_d_scaled_[j] = rho_d / amrex::max(mu0, tiny_number_local);
		profile.v_gx_[j] = (rho_g > 0.0) ? mom_gx / rho_g : 0.0;
		profile.v_gy_[j] = (rho_g > 0.0) ? mom_gy / rho_g : 0.0;
		profile.v_dx_[j] = (rho_d > 0.0) ? mom_dx / rho_d : 0.0;
		profile.v_dy_[j] = (rho_d > 0.0) ? mom_dy / rho_d : 0.0;
	}

	return profile;
}

template <typename problem_t> auto extractSnapshot(QuokkaSimulation<problem_t> &sim, const CaseConfig &config, double time) -> SnapshotData
{
	SnapshotData snapshot;
	snapshot.slice_ = extractSlice(sim, config, time);
	snapshot.profile_ = extractProfile(sim, config, time);
	snapshot.shock_position_ = detectShockPosition(snapshot.profile_);
	snapshot.max_drift_ = maxDustDrift(snapshot.profile_);
	snapshot.finite_ = profileIsFinite(snapshot.profile_) && sliceIsFinite(snapshot.slice_);
	return snapshot;
}

// cached intermediate output recorded when the run first reaches t = 0.25
struct DustyOrszagTangHistory {
	bool has_snap_025_ = false;
	SnapshotData snap_025_;
};

} // namespace

template <> struct SimulationData<DustyOrszagTang> : DustyOrszagTangHistory {
};

namespace
{

// run one dust-loading case and collect the t = 0.25 and t = 0.5 diagnostics
template <typename problem_t> auto runCase(const CaseConfig &config, bool write_csv) -> CaseResult
{
	g_initial_dust_density = config.dust_density0_;
	g_stopping_time = stopping_time0;
	g_charge_to_mass = charge_to_mass0;
	g_active_case_tag = config.tag_;
	g_active_case_label = config.label_;

	amrex::Print() << std::format("Running DustyOrszagTang case: {} (mu = {:.6e})\n", config.tag_, config.mu0_);

	auto BCs_cc = makePeriodicBCsCC<problem_t>();
	auto BCs_fc = makePeriodicBCsFC<problem_t>();
	QuokkaSimulation<problem_t> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 2;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();
	SnapshotData snap_025 = sim.userData_.has_snap_025_ ? sim.userData_.snap_025_ : extractSnapshot(sim, config, first_snapshot_time);
	SnapshotData snap_050 = extractSnapshot(sim, config, second_snapshot_time);

	if (write_csv && amrex::ParallelDescriptor::IOProcessor()) {
		writeSliceCsv(snap_025.slice_);
		writeSliceCsv(snap_050.slice_);
		writeProfileCsv(snap_025.profile_);
		writeProfileCsv(snap_050.profile_);
	}

	CaseResult result;
	result.config_ = config;
	result.snap_025_ = std::move(snap_025);
	result.snap_050_ = std::move(snap_050);
	return result;
}
} // namespace

template <> struct quokka::EOS_Traits<DustyOrszagTang> {
	static constexpr double gamma = gamma_gas;
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = 1.0;
};

template <> struct Physics_Traits<DustyOrszagTang> : DefaultPhysicsTraits {
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

template <> void QuokkaSimulation<DustyOrszagTang>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_cc = Physics_Indices<DustyOrszagTang>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		const double vx_g = -std::sin(2.0 * pi * y);
		const double vy_g = std::sin(2.0 * pi * x);
		const double bx_cc = 0.5 * (BxFace(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + BxFace(x + 0.5 * dx[0], y - 0.5 * dx[1], dx));
		const double by_cc = 0.5 * (ByFace(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + ByFace(x - 0.5 * dx[0], y + 0.5 * dx[1], dx));

		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::density_index) = rho_gas0;
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::x1Momentum_index) = rho_gas0 * vx_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::x2Momentum_index) = rho_gas0 * vy_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::internalEnergy_index) = pressure0 / (gamma_gas - 1.0);
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::energy_index) = computeTotalEnergy(rho_gas0, vx_g, vy_g, bx_cc, by_cc);

		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::dustDensity_index) = g_initial_dust_density;
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::x1DustMomentum_index) = g_initial_dust_density * vx_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::x2DustMomentum_index) = g_initial_dust_density * vy_g;
		state_cc(i, j, k, HydroSystem<DustyOrszagTang>::x3DustMomentum_index) = 0.0;
	});
}

template <> void QuokkaSimulation<DustyOrszagTang>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + i * dx[0];
		const double yL = prob_lo[1] + j * dx[1];

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<DustyOrszagTang>::mhdFirstIndex) = BxFace(xL, yL, dx);
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<DustyOrszagTang>::mhdFirstIndex) = ByFace(xL, yL, dx);
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<DustyOrszagTang>::mhdFirstIndex) = 0.0;
		}
	});
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustyOrszagTang>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
										       amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
										       amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/,
										       double /*cs*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha{};
	alpha[0] = 1.0 / g_stopping_time;
	return alpha;
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustyOrszagTang>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> charge_to_mass{};
	charge_to_mass[0] = g_charge_to_mass;
	return charge_to_mass;
}

template <> void QuokkaSimulation<DustyOrszagTang>::computeAfterTimestep()
{
	if (!userData_.has_snap_025_ && (tNew_[0] + 1.0e-12 >= first_snapshot_time)) {
		userData_.snap_025_ = extractSnapshot(*this, activeCaseConfig(), first_snapshot_time);
		userData_.has_snap_025_ = true;
	}
}

auto problem_main() -> int
{
	bool write_csv = true;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);

	std::vector<CaseConfig> const cases = makeCaseConfigs();
	CaseResult const high_mu = runCase<DustyOrszagTang>(cases[0], write_csv);
	CaseResult const low_mu = runCase<DustyOrszagTang>(cases[1], write_csv);

	const double shock_sep_025 = low_mu.snap_025_.shock_position_ - high_mu.snap_025_.shock_position_;
	const double shock_sep_050 = low_mu.snap_050_.shock_position_ - high_mu.snap_050_.shock_position_;

	amrex::Print() << std::format("  shock(high_mu, t=0.25) = {:.6e}\n", high_mu.snap_025_.shock_position_);
	amrex::Print() << std::format("  shock(low_mu,  t=0.25) = {:.6e}\n", low_mu.snap_025_.shock_position_);
	amrex::Print() << std::format("  shock(high_mu, t=0.50) = {:.6e}\n", high_mu.snap_050_.shock_position_);
	amrex::Print() << std::format("  shock(low_mu,  t=0.50) = {:.6e}\n", low_mu.snap_050_.shock_position_);
	amrex::Print() << std::format("  shock separation t=0.25 = {:.6e}\n", shock_sep_025);
	amrex::Print() << std::format("  shock separation t=0.50 = {:.6e}\n", shock_sep_050);
	amrex::Print() << std::format("  max drift high_mu t=0.25 = {:.6e}\n", high_mu.snap_025_.max_drift_);
	amrex::Print() << std::format("  max drift low_mu  t=0.25 = {:.6e}\n", low_mu.snap_025_.max_drift_);

	const bool finite = high_mu.snap_025_.finite_ && high_mu.snap_050_.finite_ && low_mu.snap_025_.finite_ && low_mu.snap_050_.finite_;
	const bool low_mu_shock_ahead_025 = shock_sep_025 > 5.0e-3;
	const bool low_mu_shock_ahead_050 = shock_sep_050 > 5.0e-3;
	const bool dust_drift_visible = high_mu.snap_025_.max_drift_ > 5.0e-2;

	if (!(finite && low_mu_shock_ahead_025 && low_mu_shock_ahead_050 && dust_drift_visible)) {
		amrex::Print() << "DustyOrszagTang FAILED.\n";
		return 1;
	}

	amrex::Print() << "DustyOrszagTang PASSED.\n";
	return 0;
}
