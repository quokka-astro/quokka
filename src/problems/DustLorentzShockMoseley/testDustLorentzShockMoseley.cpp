/// \file testDustLorentzShockMoseley.cpp
/// \brief Fluid-dust Lorentz shock paper-figure analogue using the Figure 2 parameter triplets of Moseley et al. (2023).

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <string>
#include <vector>

namespace
{
constexpr double rho_ambient = 1.0;
constexpr double u_ambient = 0.0;
constexpr double bz_ambient = 1.0;
constexpr double dust_density_floor = 1.0e-12;
constexpr bool use_local_guiding_center_gyrofrequency = true;

struct ShockProfile {
	std::string output_tag_;
	double epsilon_ = 0.0;
	double target_magnetization_ = 0.0;
	double stopping_time_ = 0.0;
	double dimensionless_charge_to_mass_ratio_ = 0.0;
	std::vector<double> x_;
	std::vector<double> rho_g_;
	std::vector<double> v_gx_;
	std::vector<double> v_gy_;
	std::vector<double> bz_;
	std::vector<double> rho_d_;
	std::vector<double> v_dx_;
	std::vector<double> v_dy_;
};

template <typename problem_t> struct ShockCaseParams;

struct DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004 {
};

struct DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004 {
};

struct DustLorentzShockMoseleyEps1em4OmegaTs12Ts010 {
};

template <> struct ShockCaseParams<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004> {
	static constexpr double sound_speed = 1.0;
	static constexpr double rho_inflow = 3.0;
	static constexpr double u_inflow = 2.0;
	static constexpr double bz_inflow = 3.0;
	static constexpr double dust_to_gas_ratio = 1.0e-4;
	static constexpr double stopping_time = 0.04;
	static constexpr double target_magnetization = 1.8;
	static constexpr double dimensionless_charge_to_mass_ratio = target_magnetization / (stopping_time * bz_ambient);
	static constexpr char const *label = "epsilon = 1e-4, Omega_L t_s = 1.8, t_s = 0.04";
	static constexpr char const *output_tag = "moseley_eps1em4_omega1p8_ts0p04";
};

template <> struct ShockCaseParams<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004> {
	static constexpr double sound_speed = 1.0;
	static constexpr double rho_inflow = 3.0;
	static constexpr double u_inflow = 2.0;
	static constexpr double bz_inflow = 3.0;
	static constexpr double dust_to_gas_ratio = 1.0e-1;
	static constexpr double stopping_time = 0.04;
	static constexpr double target_magnetization = 3.0;
	static constexpr double dimensionless_charge_to_mass_ratio = target_magnetization / (stopping_time * bz_ambient);
	static constexpr char const *label = "epsilon = 1e-1, Omega_L t_s = 3.0, t_s = 0.04";
	static constexpr char const *output_tag = "moseley_eps1em1_omega3p0_ts0p04";
};

template <> struct ShockCaseParams<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010> {
	static constexpr double sound_speed = 1.0;
	static constexpr double rho_inflow = 3.0;
	static constexpr double u_inflow = 2.0;
	static constexpr double bz_inflow = 3.0;
	static constexpr double dust_to_gas_ratio = 1.0e-4;
	static constexpr double stopping_time = 0.10;
	static constexpr double target_magnetization = 12.0;
	static constexpr double dimensionless_charge_to_mass_ratio = target_magnetization / (stopping_time * bz_ambient);
	static constexpr char const *label = "epsilon = 1e-4, Omega_L t_s = 12, t_s = 0.10";
	static constexpr char const *output_tag = "moseley_eps1em4_omega12_ts0p10";
};

template <typename problem_t> struct ShockEOSTraits {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = ShockCaseParams<problem_t>::sound_speed;
};

struct ShockPhysicsTraits : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

AMREX_GPU_DEVICE auto computeGasEnergy(double rho, double vx, double bz) -> double
{
	const double kinetic = 0.5 * rho * vx * vx;
	const double magnetic = 0.5 * bz * bz;
	return kinetic + magnetic;
}

template <typename problem_t>
AMREX_GPU_DEVICE void fillCellState(const amrex::Array4<double> &state_cc, int i, int j, int k, double rho_g, double vx_g, double rho_d, double vx_d, double bz)
{
	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
	for (int n = 0; n < ncomp_cc; ++n) {
		state_cc(i, j, k, n) = 0.0;
	}

	state_cc(i, j, k, HydroSystem<problem_t>::density_index) = rho_g;
	state_cc(i, j, k, HydroSystem<problem_t>::energy_index) = computeGasEnergy(rho_g, vx_g, bz);
	state_cc(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = 0.0;
	state_cc(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = rho_g * vx_g;
	state_cc(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = 0.0;
	state_cc(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = 0.0;

	state_cc(i, j, k, HydroSystem<problem_t>::dustDensity_index) = rho_d;
	state_cc(i, j, k, HydroSystem<problem_t>::x1DustMomentum_index) = rho_d * vx_d;
	state_cc(i, j, k, HydroSystem<problem_t>::x2DustMomentum_index) = 0.0;
	state_cc(i, j, k, HydroSystem<problem_t>::x3DustMomentum_index) = 0.0;
}

template <typename problem_t> void setShockInitialConditions(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho_d = ShockCaseParams<problem_t>::dust_to_gas_ratio * rho_ambient;
	const double bz = bz_ambient;

	amrex::ParallelFor(indexRange,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k) { fillCellState<problem_t>(state_cc, i, j, k, rho_ambient, u_ambient, rho_d, 0.0, bz); });
}

template <typename problem_t> void setShockFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
	double bfield = 0.0;
	if (grid_elem.dir_ == quokka::direction::z) {
		bfield = bz_ambient;
	}

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}
		state_fc(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) = bfield;
	});
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto constantStoppingTime() -> amrex::GpuArray<amrex::Real, 1>
{
	amrex::GpuArray<amrex::Real, 1> alpha{};
	alpha[0] = 1.0 / ShockCaseParams<problem_t>::stopping_time;
	return alpha;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto constantDimensionlessChargeToMassRatio() -> amrex::GpuArray<amrex::Real, 1>
{
	amrex::GpuArray<amrex::Real, 1> dimensionless_charge_to_mass_ratio{};
	dimensionless_charge_to_mass_ratio[0] = ShockCaseParams<problem_t>::dimensionless_charge_to_mass_ratio;
	return dimensionless_charge_to_mass_ratio;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto makeShockInflowCellState()
{
	constexpr int nvar = Physics_Indices<problem_t>::nvarTotal_cc;
	amrex::GpuArray<amrex::Real, nvar> inflow_state{};
	inflow_state[HydroSystem<problem_t>::density_index] = ShockCaseParams<problem_t>::rho_inflow;
	inflow_state[HydroSystem<problem_t>::energy_index] =
	    computeGasEnergy(ShockCaseParams<problem_t>::rho_inflow, ShockCaseParams<problem_t>::u_inflow, ShockCaseParams<problem_t>::bz_inflow);
	inflow_state[HydroSystem<problem_t>::internalEnergy_index] = 0.0;
	inflow_state[HydroSystem<problem_t>::x1Momentum_index] = ShockCaseParams<problem_t>::rho_inflow * ShockCaseParams<problem_t>::u_inflow;
	inflow_state[HydroSystem<problem_t>::x2Momentum_index] = 0.0;
	inflow_state[HydroSystem<problem_t>::x3Momentum_index] = 0.0;
	inflow_state[HydroSystem<problem_t>::dustDensity_index] = dust_density_floor;
	inflow_state[HydroSystem<problem_t>::x1DustMomentum_index] = 0.0;
	inflow_state[HydroSystem<problem_t>::x2DustMomentum_index] = 0.0;
	inflow_state[HydroSystem<problem_t>::x3DustMomentum_index] = 0.0;
	return inflow_state;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto makeShockInflowFaceState() -> amrex::GpuArray<amrex::Real, 3>
{
	return {0.0, 0.0, ShockCaseParams<problem_t>::bz_inflow};
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setShockBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
								    amrex::GeometryData const &geom)
{
	const auto low_bdr_cells = makeShockInflowCellState<problem_t>();
	AMRSimulation<problem_t>::template setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
}

template <typename problem_t, quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setShockFaceBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar_fc,
									amrex::GeometryData const &geom)
{
	const auto low_bdr_values = makeShockInflowFaceState<problem_t>();
	AMRSimulation<problem_t>::template setConstantDirichletBCFaceVarLo<0, dir, 3>(iv, consVar_fc, geom, low_bdr_values);
}

void setShockHiOutflow(amrex::Vector<amrex::BCRec> &bcs)
{
	for (auto &bc : bcs) {
		bc.setHi(0, amrex::BCType::foextrap);
	}
}

template <typename problem_t> auto makeShockBCsCC() -> amrex::Vector<amrex::BCRec>
{
	auto BCs_cc = quokka::BC<problem_t>(quokka::BCType::ext_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	setShockHiOutflow(BCs_cc);
	return BCs_cc;
}

template <typename problem_t> auto makeShockBCsFC() -> amrex::Vector<amrex::BCRec>
{
	auto BCs_fc = quokka::BC_fc<problem_t>(quokka::BCType::mathematicalBndryTypes::ext_dir, quokka::BCType::mathematicalBndryTypes::periodic,
					       quokka::BCType::mathematicalBndryTypes::periodic);
	setShockHiOutflow(BCs_fc);
	return BCs_fc;
}

template <typename problem_t> auto extractMagneticProfile(QuokkaSimulation<problem_t> &sim) -> amrex::Vector<amrex::Gpu::HostVector<amrex::Real>>
{
	amrex::MultiFab b_cc(sim.state_new_cc_[0].boxArray(), sim.state_new_cc_[0].DistributionMap(), 3, 0);
	auto const &b_cc_arrays = b_cc.arrays();
	auto const &fcx = sim.state_new_fc_[0][0].const_arrays();
	auto const &fcy = sim.state_new_fc_[0][1].const_arrays();
	auto const &fcz = sim.state_new_fc_[0][2].const_arrays();
	amrex::ParallelFor(b_cc, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		b_cc_arrays[bx](i, j, k, 0) =
		    0.5 * (fcx[bx](i, j, k, MHDSystem<problem_t>::bfield_index) + fcx[bx](i + 1, j, k, MHDSystem<problem_t>::bfield_index));
		b_cc_arrays[bx](i, j, k, 1) =
		    0.5 * (fcy[bx](i, j, k, MHDSystem<problem_t>::bfield_index) + fcy[bx](i, j + 1, k, MHDSystem<problem_t>::bfield_index));
		b_cc_arrays[bx](i, j, k, 2) =
		    0.5 * (fcz[bx](i, j, k, MHDSystem<problem_t>::bfield_index) + fcz[bx](i, j, k + 1, MHDSystem<problem_t>::bfield_index));
	});
	auto extracted = fextract(b_cc, sim.Geom(0), 0, 0.5);
	return std::move(std::get<1>(extracted));
}

template <typename problem_t> auto extractShockProfile(QuokkaSimulation<problem_t> &sim) -> ShockProfile
{
	auto [x, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);
	auto const b_values = extractMagneticProfile(sim);

	ShockProfile profile;
	profile.output_tag_ = ShockCaseParams<problem_t>::output_tag;
	profile.epsilon_ = ShockCaseParams<problem_t>::dust_to_gas_ratio;
	profile.target_magnetization_ = ShockCaseParams<problem_t>::target_magnetization;
	profile.stopping_time_ = ShockCaseParams<problem_t>::stopping_time;
	profile.dimensionless_charge_to_mass_ratio_ = ShockCaseParams<problem_t>::dimensionless_charge_to_mass_ratio;
	profile.x_ = x;
	profile.rho_g_.resize(x.size());
	profile.v_gx_.resize(x.size());
	profile.v_gy_.resize(x.size());
	profile.bz_.resize(x.size());
	profile.rho_d_.resize(x.size());
	profile.v_dx_.resize(x.size());
	profile.v_dy_.resize(x.size());

	auto const &bz = b_values[2];
	for (size_t i = 0; i < static_cast<size_t>(x.size()); ++i) {
		const double rho_g = values.at(HydroSystem<problem_t>::density_index)[i];
		const double mom_gx = values.at(HydroSystem<problem_t>::x1Momentum_index)[i];
		const double mom_gy = values.at(HydroSystem<problem_t>::x2Momentum_index)[i];
		const double rho_d = values.at(HydroSystem<problem_t>::dustDensity_index)[i];
		const double mom_dx = values.at(HydroSystem<problem_t>::x1DustMomentum_index)[i];
		const double mom_dy = values.at(HydroSystem<problem_t>::x2DustMomentum_index)[i];

		profile.rho_g_[i] = rho_g;
		profile.v_gx_[i] = mom_gx / rho_g;
		profile.v_gy_[i] = mom_gy / rho_g;
		profile.bz_[i] = bz[i];
		profile.rho_d_[i] = rho_d;
		profile.v_dx_[i] = mom_dx / rho_d;
		profile.v_dy_[i] = mom_dy / rho_d;
	}

	return profile;
}

template <typename problem_t> auto runShockCase() -> ShockProfile
{
	amrex::Print() << std::format("Running DustLorentzShockMoseley case: {}\n", ShockCaseParams<problem_t>::label);

	auto BCs_cc = makeShockBCsCC<problem_t>();
	auto BCs_fc = makeShockBCsFC<problem_t>();
	QuokkaSimulation<problem_t> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 2;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	return extractShockProfile(sim);
}

auto maxAbsValue(const std::vector<double> &values) -> double
{
	double max_value = 0.0;
	for (double const value : values) {
		max_value = std::max(max_value, std::abs(value));
	}
	return max_value;
}

auto computeGuidingCenterVx(const ShockProfile &profile) -> std::vector<double>
{
	std::vector<double> guiding_vx(profile.x_.size());
	const auto omega_ts_at = [&profile](size_t i) {
		if constexpr (use_local_guiding_center_gyrofrequency) {
			return profile.dimensionless_charge_to_mass_ratio_ * profile.bz_[i] * profile.stopping_time_;
		}
		return profile.target_magnetization_;
	};
	if (std::abs(profile.target_magnetization_) <= 0.0) {
		return profile.v_dx_;
	}
	for (size_t i = 0; i < profile.x_.size(); ++i) {
		const double w_y = profile.v_dy_[i] - profile.v_gy_[i];
		const double omega_ts = omega_ts_at(i);
		if (std::abs(omega_ts) <= 0.0) {
			guiding_vx[i] = profile.v_dx_[i];
			continue;
		}
		guiding_vx[i] = profile.v_gx_[i] - w_y / omega_ts;
	}
	return guiding_vx;
}

auto profileIsFinite(const ShockProfile &profile) -> bool
{
	auto const all_finite = [](const std::vector<double> &values) {
		return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
	};
	return all_finite(profile.x_) && all_finite(profile.rho_g_) && all_finite(profile.v_gx_) && all_finite(profile.v_gy_) && all_finite(profile.rho_d_) &&
	       all_finite(profile.v_dx_) && all_finite(profile.v_dy_) && all_finite(profile.bz_);
}

void writeShockProfileCsv(const ShockProfile &profile, const std::vector<double> &guiding_vx)
{
	std::ofstream file(std::format("dust_lorentz_shock_{}.csv", profile.output_tag_));
	file << "x,rho_g,v_gx,v_gy,bz,omega_ts_guiding,rho_d_scaled,v_dx,v_dy,w_y,v_guiding_x\n";

	for (size_t i = 0; i < profile.x_.size(); ++i) {
		const double w_y = profile.v_dy_[i] - profile.v_gy_[i];
		const double omega_ts_guiding = use_local_guiding_center_gyrofrequency
						   ? profile.dimensionless_charge_to_mass_ratio_ * profile.bz_[i] * profile.stopping_time_
						   : profile.target_magnetization_;
		file << profile.x_[i] << "," << profile.rho_g_[i] << "," << profile.v_gx_[i] << "," << profile.v_gy_[i] << "," << profile.bz_[i] << ","
		     << omega_ts_guiding << "," << profile.rho_d_[i] / profile.epsilon_ << "," << profile.v_dx_[i] << ","
		     << profile.v_dy_[i] << "," << w_y << "," << guiding_vx[i] << "\n";
	}
}

auto runShockAnalogue(bool write_csv) -> int
{
	ShockProfile const shock_eps1em4_omega1p8_ts0p04 = runShockCase<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>();
	ShockProfile const shock_eps1em1_omega3p0_ts0p04 = runShockCase<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>();
	ShockProfile const shock_eps1em4_omega12_ts0p10 = runShockCase<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>();

	if (write_csv) {
		const std::vector<double> guiding_vx_eps1em4_omega1p8_ts0p04 = computeGuidingCenterVx(shock_eps1em4_omega1p8_ts0p04);
		const std::vector<double> guiding_vx_eps1em1_omega3p0_ts0p04 = computeGuidingCenterVx(shock_eps1em1_omega3p0_ts0p04);
		const std::vector<double> guiding_vx_eps1em4_omega12_ts0p10 = computeGuidingCenterVx(shock_eps1em4_omega12_ts0p10);

		writeShockProfileCsv(shock_eps1em4_omega1p8_ts0p04, guiding_vx_eps1em4_omega1p8_ts0p04);
		writeShockProfileCsv(shock_eps1em1_omega3p0_ts0p04, guiding_vx_eps1em1_omega3p0_ts0p04);
		writeShockProfileCsv(shock_eps1em4_omega12_ts0p10, guiding_vx_eps1em4_omega12_ts0p10);
	}

	const double vy_max_eps1em4_omega1p8_ts0p04 = maxAbsValue(shock_eps1em4_omega1p8_ts0p04.v_dy_);
	const double vy_max_eps1em1_omega3p0_ts0p04 = maxAbsValue(shock_eps1em1_omega3p0_ts0p04.v_dy_);
	const double vy_max_eps1em4_omega12_ts0p10 = maxAbsValue(shock_eps1em4_omega12_ts0p10.v_dy_);

	constexpr double vy_min_case1 = 1.0e-3;
	constexpr double vy_min_case2 = 1.0e-2;
	constexpr double vy_min_case3 = 5.0e-2;

	amrex::Print() << std::format("  vy_max_eps1em4_omega1p8_ts0p04 = {:.6e} (pass if > {:.6e})\n", vy_max_eps1em4_omega1p8_ts0p04, vy_min_case1);
	amrex::Print() << std::format("  vy_max_eps1em1_omega3p0_ts0p04 = {:.6e} (pass if > {:.6e})\n", vy_max_eps1em1_omega3p0_ts0p04, vy_min_case2);
	amrex::Print() << std::format("  vy_max_eps1em4_omega12_ts0p10 = {:.6e} (pass if > {:.6e})\n", vy_max_eps1em4_omega12_ts0p10, vy_min_case3);

	const bool finite = profileIsFinite(shock_eps1em4_omega1p8_ts0p04) && profileIsFinite(shock_eps1em1_omega3p0_ts0p04) && profileIsFinite(shock_eps1em4_omega12_ts0p10);
	const bool charged_rotates = vy_max_eps1em4_omega1p8_ts0p04 > vy_min_case1 && vy_max_eps1em1_omega3p0_ts0p04 > vy_min_case2 &&
				     vy_max_eps1em4_omega12_ts0p10 > vy_min_case3;

	const bool passed = finite && charged_rotates;
	if (!passed) {
		amrex::Print() << "DustLorentzShockMoseley FAILED.\n";
		return 1;
	}

	amrex::Print() << "DustLorentzShockMoseley PASSED.\n";
	return 0;
}
} // namespace

template <> struct quokka::EOS_Traits<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004> : ShockEOSTraits<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004> {
};
template <> struct quokka::EOS_Traits<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004> : ShockEOSTraits<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004> {
};
template <> struct quokka::EOS_Traits<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010> : ShockEOSTraits<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010> {
};

template <> struct Physics_Traits<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004> : ShockPhysicsTraits {
};
template <> struct Physics_Traits<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004> : ShockPhysicsTraits {
};
template <> struct Physics_Traits<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010> : ShockPhysicsTraits {
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>::ComputeReciprocalStoppingTime(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantStoppingTime<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>();
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>::ComputeDustDimensionlessChargeToMassRatio(
    DustCoefficientState const & /*state*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantDimensionlessChargeToMassRatio<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>();
}

template <> void QuokkaSimulation<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setShockInitialConditions<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>(grid_elem);
}

template <> void QuokkaSimulation<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setShockFaceVars<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>(grid_elem);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
											   int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
											   const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											   int /*orig_comp*/)
{
	setShockBoundaryConditions<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>(iv, consVar, geom);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004>::setCustomBoundaryConditionsFaceVar(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockFaceBoundaryConditions<DustLorentzShockMoseleyEps1em4OmegaTs1p8Ts004, dir>(iv, dest, geom);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>::ComputeReciprocalStoppingTime(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantStoppingTime<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>();
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>::ComputeDustDimensionlessChargeToMassRatio(
    DustCoefficientState const & /*state*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantDimensionlessChargeToMassRatio<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>();
}

template <> void QuokkaSimulation<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setShockInitialConditions<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>(grid_elem);
}

template <> void QuokkaSimulation<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setShockFaceVars<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>(grid_elem);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
											   int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
											   const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											   int /*orig_comp*/)
{
	setShockBoundaryConditions<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>(iv, consVar, geom);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004>::setCustomBoundaryConditionsFaceVar(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockFaceBoundaryConditions<DustLorentzShockMoseleyEps1em1OmegaTs3p0Ts004, dir>(iv, dest, geom);
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>::ComputeReciprocalStoppingTime(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantStoppingTime<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>();
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>::ComputeDustDimensionlessChargeToMassRatio(
    DustCoefficientState const & /*state*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantDimensionlessChargeToMassRatio<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>();
}

template <> void QuokkaSimulation<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setShockInitialConditions<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>(grid_elem);
}

template <> void QuokkaSimulation<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setShockFaceVars<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>(grid_elem);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
											  int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
											  const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											  int /*orig_comp*/)
{
	setShockBoundaryConditions<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>(iv, consVar, geom);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010>::setCustomBoundaryConditionsFaceVar(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockFaceBoundaryConditions<DustLorentzShockMoseleyEps1em4OmegaTs12Ts010, dir>(iv, dest, geom);
}

auto problem_main() -> int
{
	bool write_csv = true;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);

	return runShockAnalogue(write_csv);
}
