/// \file testDustLorentzShock.cpp
/// \brief Fluid-dust Lorentz shock regression test inspired by Moseley et al. (2022).

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

struct ShockProfile {
	std::string output_tag_;
	double mu_ = 0.0;
	double target_magnetization_ = 0.0;
	std::vector<double> x_;
	std::vector<double> rho_g_;
	std::vector<double> v_gx_;
	std::vector<double> rho_d_;
	std::vector<double> v_dx_;
	std::vector<double> v_dy_;
};

template <typename problem_t> struct ShockCaseParams;

struct DustLorentzShockRefNeutral {
};

struct DustLorentzShockChargedDilute {
};

struct DustLorentzShockChargedBackreacting {
};

template <> struct ShockCaseParams<DustLorentzShockRefNeutral> {
	static constexpr double sound_speed = 1.0;
	static constexpr double rho_inflow = 3.0;
	static constexpr double u_inflow = 2.0;
	static constexpr double bz_inflow = 3.0;
	static constexpr double dust_to_gas_ratio = 0.01;
	static constexpr double stopping_time = 0.10;
	static constexpr double target_magnetization = 0.0;
	static constexpr double charge_to_mass_ratio = 0.0;
	static constexpr char const *label = "Neutral, t_s = 0.10";
	static constexpr char const *output_tag = "ref_neutral";
};

template <> struct ShockCaseParams<DustLorentzShockChargedDilute> {
	static constexpr double sound_speed = 1.0;
	static constexpr double rho_inflow = 3.0;
	static constexpr double u_inflow = 2.0;
	static constexpr double bz_inflow = 3.0;
	static constexpr double dust_to_gas_ratio = 0.01;
	static constexpr double stopping_time = 0.10;
	static constexpr double target_magnetization = 20.0;
	static constexpr double charge_to_mass_ratio = target_magnetization / (stopping_time * bz_ambient);
	static constexpr char const *label = "Charged, mu = 0.01";
	static constexpr char const *output_tag = "charged_dilute";
};

template <> struct ShockCaseParams<DustLorentzShockChargedBackreacting> {
	static constexpr double sound_speed = 1.0;
	static constexpr double rho_inflow = 3.0;
	static constexpr double u_inflow = 2.0;
	static constexpr double bz_inflow = 3.0;
	static constexpr double dust_to_gas_ratio = 0.10;
	static constexpr double stopping_time = 0.10;
	static constexpr double target_magnetization = 20.0;
	static constexpr double charge_to_mass_ratio = target_magnetization / (stopping_time * bz_ambient);
	static constexpr char const *label = "Charged, mu = 0.10";
	static constexpr char const *output_tag = "charged_backreacting";
};

template <typename problem_t> struct ShockEOSTraits {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = ShockCaseParams<problem_t>::sound_speed;
};

struct ShockPhysicsTraits {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
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

template <typename problem_t> AMREX_GPU_DEVICE auto computeGasEnergy(double rho, double vx, double bz) -> double
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
	state_cc(i, j, k, HydroSystem<problem_t>::energy_index) = computeGasEnergy<problem_t>(rho_g, vx_g, bz);
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

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto constantChargeToMassRatio() -> amrex::GpuArray<amrex::Real, 1>
{
	amrex::GpuArray<amrex::Real, 1> charge_to_mass{};
	charge_to_mass[0] = ShockCaseParams<problem_t>::charge_to_mass_ratio;
	return charge_to_mass;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto makeShockInflowCellState()
{
	constexpr int nvar = Physics_Indices<problem_t>::nvarTotal_cc;
	amrex::GpuArray<amrex::Real, nvar> inflow_state{};
	inflow_state[HydroSystem<problem_t>::density_index] = ShockCaseParams<problem_t>::rho_inflow;
	inflow_state[HydroSystem<problem_t>::energy_index] =
	    computeGasEnergy<problem_t>(ShockCaseParams<problem_t>::rho_inflow, ShockCaseParams<problem_t>::u_inflow, ShockCaseParams<problem_t>::bz_inflow);
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

template <typename problem_t> auto extractShockProfile(QuokkaSimulation<problem_t> &sim) -> ShockProfile
{
	auto [x, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);

	ShockProfile profile;
	profile.output_tag_ = ShockCaseParams<problem_t>::output_tag;
	profile.mu_ = ShockCaseParams<problem_t>::dust_to_gas_ratio;
	profile.target_magnetization_ = ShockCaseParams<problem_t>::target_magnetization;
	profile.x_ = x;
	profile.rho_g_.resize(x.size());
	profile.v_gx_.resize(x.size());
	profile.rho_d_.resize(x.size());
	profile.v_dx_.resize(x.size());
	profile.v_dy_.resize(x.size());

	for (size_t i = 0; i < static_cast<size_t>(x.size()); ++i) {
		const double rho_g = values.at(HydroSystem<problem_t>::density_index)[i];
		const double mom_gx = values.at(HydroSystem<problem_t>::x1Momentum_index)[i];
		const double rho_d = values.at(HydroSystem<problem_t>::dustDensity_index)[i];
		const double mom_dx = values.at(HydroSystem<problem_t>::x1DustMomentum_index)[i];
		const double mom_dy = values.at(HydroSystem<problem_t>::x2DustMomentum_index)[i];

		profile.rho_g_[i] = rho_g;
		profile.v_gx_[i] = (rho_g > 0.0) ? mom_gx / rho_g : 0.0;
		profile.rho_d_[i] = rho_d;
		profile.v_dx_[i] = (rho_d > 0.0) ? mom_dx / rho_d : 0.0;
		profile.v_dy_[i] = (rho_d > 0.0) ? mom_dy / rho_d : 0.0;
	}

	return profile;
}

template <typename problem_t> auto runShockCase() -> ShockProfile
{
	amrex::Print() << std::format("Running DustLorentzShock case: {}\n", ShockCaseParams<problem_t>::label);

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

auto detectShockPosition(const ShockProfile &profile) -> double
{
	if (profile.rho_g_.size() < 2) {
		return profile.x_.empty() ? 0.0 : profile.x_.front();
	}

	double max_jump = -1.0;
	size_t shock_index = 0;
	for (size_t i = 0; i + 1 < profile.rho_g_.size(); ++i) {
		const double jump = std::abs(profile.rho_g_[i + 1] - profile.rho_g_[i]);
		if (jump > max_jump) {
			max_jump = jump;
			shock_index = i;
		}
	}
	return profile.x_[shock_index];
}

auto meanRelativeWindowDrift(const ShockProfile &profile, double shock_position, double rel_lo, double rel_hi) -> double
{
	double sum = 0.0;
	int count = 0;
	for (size_t i = 0; i < profile.x_.size(); ++i) {
		const double x_rel = profile.x_[i] - shock_position;
		if (x_rel >= rel_lo && x_rel <= rel_hi) {
			sum += std::abs(profile.v_dx_[i] - profile.v_gx_[i]);
			count++;
		}
	}
	return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

auto computeGuidingCenterVx(const ShockProfile &profile) -> std::vector<double>
{
	std::vector<double> guiding_vx(profile.x_.size());
	const double omega_ts = profile.target_magnetization_;
	if (std::abs(omega_ts) <= 0.0) {
		return profile.v_dx_;
	}
	for (size_t i = 0; i < profile.x_.size(); ++i) {
		const double w_y = profile.v_dy_[i];
		guiding_vx[i] = profile.v_gx_[i] - w_y / omega_ts;
	}
	return guiding_vx;
}

void writeShockProfileCsv(const ShockProfile &profile, const std::vector<double> *guiding_vx = nullptr)
{
	std::ofstream file(std::format("dust_lorentz_shock_{}.csv", profile.output_tag_));
	file << "x,rho_g,v_gx,rho_d_scaled,v_dx,v_dy,v_guiding_x\n";

	for (size_t i = 0; i < profile.x_.size(); ++i) {
		file << profile.x_[i] << "," << profile.rho_g_[i] << "," << profile.v_gx_[i] << "," << profile.rho_d_[i] / std::max(profile.mu_, 1.0e-12) << ","
		     << profile.v_dx_[i] << "," << profile.v_dy_[i] << ",";
		if (guiding_vx != nullptr) {
			file << (*guiding_vx)[i];
		}
		file << "\n";
	}
}

auto runLowMachRegression(bool write_csv) -> int
{
	ShockProfile const ref_neutral = runShockCase<DustLorentzShockRefNeutral>();
	ShockProfile const charged_dilute = runShockCase<DustLorentzShockChargedDilute>();
	ShockProfile const charged_backreacting = runShockCase<DustLorentzShockChargedBackreacting>();

	const double shock_charged = detectShockPosition(charged_dilute);
	const double shock_backreact = detectShockPosition(charged_backreacting);

	const double neutral_vy_max = maxAbsValue(ref_neutral.v_dy_);
	const double charged_vy_max = maxAbsValue(charged_dilute.v_dy_);
	const double mean_drift_charged = meanRelativeWindowDrift(charged_dilute, shock_charged, 0.02, 0.18);
	const std::vector<double> guiding_vx = computeGuidingCenterVx(charged_dilute);
	double mean_guiding_drift = 0.0;
	int guiding_count = 0;
	for (size_t i = 0; i < charged_dilute.x_.size(); ++i) {
		const double x_rel = charged_dilute.x_[i] - shock_charged;
		if (x_rel >= 0.02 && x_rel <= 0.18) {
			mean_guiding_drift += std::abs(guiding_vx[i] - charged_dilute.v_gx_[i]);
			guiding_count++;
		}
	}
	if (guiding_count > 0) {
		mean_guiding_drift /= static_cast<double>(guiding_count);
	}

	if (write_csv) {
		writeShockProfileCsv(ref_neutral);
		writeShockProfileCsv(charged_backreacting);
		writeShockProfileCsv(charged_dilute, &guiding_vx);
	}

	constexpr double neutral_vy_tol = 1.0e-8;
	constexpr double charged_vy_min = 5.0e-2;
	constexpr double guiding_center_factor = 0.10;
	constexpr double shock_backreaction_margin = 5.0e-3;

	amrex::Print() << std::format("  neutral_vy_max             = {:.6e} (pass if < {:.6e})\n", neutral_vy_max, neutral_vy_tol);
	amrex::Print() << std::format("  charged_vy_max             = {:.6e} (pass if > {:.6e})\n", charged_vy_max, charged_vy_min);
	amrex::Print() << std::format("  mean_drift_charged         = {:.6e}\n", mean_drift_charged);
	amrex::Print() << std::format("  mean_guiding_drift         = {:.6e} (pass if < {:.6e})\n", mean_guiding_drift,
				      guiding_center_factor * mean_drift_charged);
	amrex::Print() << std::format("  shock_charged              = {:.6e}\n", shock_charged);
	amrex::Print() << std::format("  shock_backreact            = {:.6e} (pass if < {:.6e})\n", shock_backreact, shock_charged - shock_backreaction_margin);

	const bool neutral_uncharged = neutral_vy_max < neutral_vy_tol;
	const bool charged_rotates = charged_vy_max > charged_vy_min;
	const bool guiding_center_improves_coupling = mean_guiding_drift < guiding_center_factor * mean_drift_charged;
	const bool backreaction_slows_shock = shock_backreact < (shock_charged - shock_backreaction_margin);

	const bool passed = neutral_uncharged && charged_rotates && guiding_center_improves_coupling && backreaction_slows_shock;

	if (!passed) {
		amrex::Print() << "DustLorentzShock FAILED.\n";
		return 1;
	}

	amrex::Print() << "DustLorentzShock PASSED.\n";
	return 0;
}
} // namespace

template <> struct quokka::EOS_Traits<DustLorentzShockRefNeutral> : ShockEOSTraits<DustLorentzShockRefNeutral> {
};
template <> struct quokka::EOS_Traits<DustLorentzShockChargedDilute> : ShockEOSTraits<DustLorentzShockChargedDilute> {
};
template <> struct quokka::EOS_Traits<DustLorentzShockChargedBackreacting> : ShockEOSTraits<DustLorentzShockChargedBackreacting> {
};

template <> struct Physics_Traits<DustLorentzShockRefNeutral> : ShockPhysicsTraits {
};
template <> struct Physics_Traits<DustLorentzShockChargedDilute> : ShockPhysicsTraits {
};
template <> struct Physics_Traits<DustLorentzShockChargedBackreacting> : ShockPhysicsTraits {
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockRefNeutral>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
												  amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
												  amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/,
												  double /*cs*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantStoppingTime<DustLorentzShockRefNeutral>();
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockRefNeutral>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantChargeToMassRatio<DustLorentzShockRefNeutral>();
}

template <> void QuokkaSimulation<DustLorentzShockRefNeutral>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setShockInitialConditions<DustLorentzShockRefNeutral>(grid_elem);
}

template <> void QuokkaSimulation<DustLorentzShockRefNeutral>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setShockFaceVars<DustLorentzShockRefNeutral>(grid_elem);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustLorentzShockRefNeutral>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
								       int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								       const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockBoundaryConditions<DustLorentzShockRefNeutral>(iv, consVar, geom);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustLorentzShockRefNeutral>::setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int /*dcomp*/,
									      int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
									      const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockFaceBoundaryConditions<DustLorentzShockRefNeutral, dir>(iv, dest, geom);
}

template <>
AMREX_GPU_HOST_DEVICE auto
DustSources<DustLorentzShockChargedDilute>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
									  amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantStoppingTime<DustLorentzShockChargedDilute>();
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockChargedDilute>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantChargeToMassRatio<DustLorentzShockChargedDilute>();
}

template <> void QuokkaSimulation<DustLorentzShockChargedDilute>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setShockInitialConditions<DustLorentzShockChargedDilute>(grid_elem);
}

template <> void QuokkaSimulation<DustLorentzShockChargedDilute>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setShockFaceVars<DustLorentzShockChargedDilute>(grid_elem);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustLorentzShockChargedDilute>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
									  int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
									  const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockBoundaryConditions<DustLorentzShockChargedDilute>(iv, consVar, geom);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<DustLorentzShockChargedDilute>::setCustomBoundaryConditionsFaceVar(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockFaceBoundaryConditions<DustLorentzShockChargedDilute, dir>(iv, dest, geom);
}

template <>
AMREX_GPU_HOST_DEVICE auto
DustSources<DustLorentzShockChargedBackreacting>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
										amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantStoppingTime<DustLorentzShockChargedBackreacting>();
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustLorentzShockChargedBackreacting>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	return constantChargeToMassRatio<DustLorentzShockChargedBackreacting>();
}

template <> void QuokkaSimulation<DustLorentzShockChargedBackreacting>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	setShockInitialConditions<DustLorentzShockChargedBackreacting>(grid_elem);
}

template <> void QuokkaSimulation<DustLorentzShockChargedBackreacting>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	setShockFaceVars<DustLorentzShockChargedBackreacting>(grid_elem);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<DustLorentzShockChargedBackreacting>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockBoundaryConditions<DustLorentzShockChargedBackreacting>(iv, consVar, geom);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<DustLorentzShockChargedBackreacting>::setCustomBoundaryConditionsFaceVar(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	setShockFaceBoundaryConditions<DustLorentzShockChargedBackreacting, dir>(iv, dest, geom);
}

auto problem_main() -> int
{
	bool write_csv = true;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);

	return runLowMachRegression(write_csv);
}
