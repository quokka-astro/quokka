/// \file testDustDampingMHDZeroBMixedStiff.cpp
/// \brief Dust drag damping test with MHD enabled, zero magnetic field, and mixed stopping times.
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string_view>
#include <vector>

namespace
{
constexpr double rho = 1.0;
constexpr double rho_dust1 = 1.0;
constexpr double rho_dust2 = 1.0;
constexpr double TS1 = 0.2;
constexpr double TS2 = 0.002;
constexpr double P_INITIAL = 1.0;
constexpr double OMEGA_DRAG = 1.0;
constexpr double STOP_TIME = 2.0;
constexpr double HISTORY_DT = 0.1;
constexpr double gas_gamma = 1.4;
constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

constexpr double gas_velocity_initial = 1.0;
constexpr double dust1_velocity_initial = 2.0;
constexpr double dust2_velocity_initial = 0.5;

struct AnalyticModeData {
	double v_com;
	double lambda1;
	double lambda2;
	double dust1_ratio1;
	double dust1_ratio2;
	double dust2_ratio1;
	double dust2_ratio2;
	double coeff1;
	double coeff2;
};

struct AnalyticState {
	double v_gas;
	double v_dust1;
	double v_dust2;
};

auto getAnalyticModeData() -> AnalyticModeData const &
{
	static AnalyticModeData const data = []() {
		double const alpha1 = 1.0 / TS1;
		double const alpha2 = 1.0 / TS2;
		double const epsilon1 = rho_dust1 / rho;
		double const epsilon2 = rho_dust2 / rho;
		double const total_mass = rho + rho_dust1 + rho_dust2;
		double const v_com = (rho * gas_velocity_initial + rho_dust1 * dust1_velocity_initial + rho_dust2 * dust2_velocity_initial) / total_mass;

		double const spectral_sum = (1.0 + epsilon1) * alpha1 + (1.0 + epsilon2) * alpha2;
		double const spectral_prod = alpha1 * alpha2 * (1.0 + epsilon1 + epsilon2);
		double const discriminant = std::max(0.0, spectral_sum * spectral_sum - 4.0 * spectral_prod);
		double const discriminant_sqrt = std::sqrt(discriminant);
		double const lambda1 = 0.5 * (-spectral_sum + discriminant_sqrt);
		double const lambda2 = 0.5 * (-spectral_sum - discriminant_sqrt);

		double const dust1_ratio1 = alpha1 / (alpha1 + lambda1);
		double const dust1_ratio2 = alpha1 / (alpha1 + lambda2);
		double const dust2_ratio1 = alpha2 / (alpha2 + lambda1);
		double const dust2_ratio2 = alpha2 / (alpha2 + lambda2);

		double const delta_v_g = gas_velocity_initial - v_com;
		double const delta_v_d1 = dust1_velocity_initial - v_com;
		double const coeff1 = (delta_v_d1 - delta_v_g * dust1_ratio2) / (dust1_ratio1 - dust1_ratio2);
		double const coeff2 = delta_v_g - coeff1;

		return AnalyticModeData{
		    .v_com = v_com,
		    .lambda1 = lambda1,
		    .lambda2 = lambda2,
		    .dust1_ratio1 = dust1_ratio1,
		    .dust1_ratio2 = dust1_ratio2,
		    .dust2_ratio1 = dust2_ratio1,
		    .dust2_ratio2 = dust2_ratio2,
		    .coeff1 = coeff1,
		    .coeff2 = coeff2,
		};
	}();

	return data;
}

auto analyticState(double t) -> AnalyticState
{
	AnalyticModeData const &mode = getAnalyticModeData();
	double const lambda1_exp = std::exp(mode.lambda1 * t);
	double const lambda2_exp = std::exp(mode.lambda2 * t);

	double const gas_mode = mode.coeff1 * lambda1_exp + mode.coeff2 * lambda2_exp;
	double const dust1_mode = mode.coeff1 * mode.dust1_ratio1 * lambda1_exp + mode.coeff2 * mode.dust1_ratio2 * lambda2_exp;
	double const dust2_mode = mode.coeff1 * mode.dust2_ratio1 * lambda1_exp + mode.coeff2 * mode.dust2_ratio2 * lambda2_exp;

	return AnalyticState{
	    .v_gas = mode.v_com + gas_mode,
	    .v_dust1 = mode.v_com + dust1_mode,
	    .v_dust2 = mode.v_com + dust2_mode,
	};
}

auto v_gas_analytic(double t) -> double { return analyticState(t).v_gas; }

auto v_dust1_analytic(double t) -> double { return analyticState(t).v_dust1; }

auto v_dust2_analytic(double t) -> double { return analyticState(t).v_dust2; }

auto E_gas_analytic(double t) -> double
{
	const int n_points = 4000;
	const double dt = t / n_points;
	double integral = 0.0;

	for (int i = 0; i < n_points; ++i) {
		double const t1 = i * dt;
		double const t2 = (i + 1) * dt;

		AnalyticState const state1 = analyticState(t1);
		AnalyticState const state2 = analyticState(t2);

		double const term1 =
		    (rho_dust1 * (state1.v_dust1 - state1.v_gas) / TS1 * state1.v_gas + rho_dust2 * (state1.v_dust2 - state1.v_gas) / TS2 * state1.v_gas +
		     OMEGA_DRAG *
			 (rho_dust1 * std::pow(state1.v_dust1 - state1.v_gas, 2) / TS1 + rho_dust2 * std::pow(state1.v_dust2 - state1.v_gas, 2) / TS2));

		double const term2 =
		    (rho_dust1 * (state2.v_dust1 - state2.v_gas) / TS1 * state2.v_gas + rho_dust2 * (state2.v_dust2 - state2.v_gas) / TS2 * state2.v_gas +
		     OMEGA_DRAG *
			 (rho_dust1 * std::pow(state2.v_dust1 - state2.v_gas, 2) / TS1 + rho_dust2 * std::pow(state2.v_dust2 - state2.v_gas, 2) / TS2));

		integral += 0.5 * (term1 + term2) * dt;
	}

	const double E_gas_initial = P_INITIAL / (gas_gamma - 1.0) + 0.5 * rho * gas_velocity_initial * gas_velocity_initial;
	return E_gas_initial + integral;
}
} // namespace

struct DustDampingMHDZeroBMixedStiff {
};

template <> struct SimulationData<DustDampingMHDZeroBMixedStiff> {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_vec_;
	std::vector<double> v_dust1_vec_;
	std::vector<double> v_dust2_vec_;
	std::vector<double> E_gas_vec_;
};

template <> struct quokka::EOS_Traits<DustDampingMHDZeroBMixedStiff> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = gas_gamma;
};

namespace
{
constexpr double Egas0 = P_INITIAL / (quokka::EOS_Traits<DustDampingMHDZeroBMixedStiff>::gamma - 1.0) + 0.5 * rho * gas_velocity_initial * gas_velocity_initial;
constexpr double Egas0_internal = P_INITIAL / (quokka::EOS_Traits<DustDampingMHDZeroBMixedStiff>::gamma - 1.0);
} // namespace

template <> struct Physics_Traits<DustDampingMHDZeroBMixedStiff> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 2;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingMHDZeroBMixedStiff>::ComputeReciprocalStoppingTime(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 2> alpha{};
	alpha[0] = 1.0 / TS1;
	alpha[1] = 1.0 / TS2;
	return alpha;
}

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingMHDZeroBMixedStiff>::ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 2> dimensionless_charge_to_mass_ratio{};
	dimensionless_charge_to_mass_ratio.fill(1.0);
	return dimensionless_charge_to_mass_ratio;
}

template <> void QuokkaSimulation<DustDampingMHDZeroBMixedStiff>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_fc = Physics_Indices<DustDampingMHDZeroBMixedStiff>::nvarPerDim_fc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}
	});
}

template <> void QuokkaSimulation<DustDampingMHDZeroBMixedStiff>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::internalEnergy_index) = Egas0_internal;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x1Momentum_index) = rho * gas_velocity_initial;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x3Momentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::dustDensity_index) = rho_dust1;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x1DustMomentum_index) = rho_dust1 * dust1_velocity_initial;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x2DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x3DustMomentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::dustDensity_index + numDustVars) = rho_dust2;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x1DustMomentum_index + numDustVars) = rho_dust2 * dust2_velocity_initial;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x2DustMomentum_index + numDustVars) = 0.0;
		state_cc(i, j, k, HydroSystem<DustDampingMHDZeroBMixedStiff>::x3DustMomentum_index + numDustVars) = 0.0;
	});
}

template <> void QuokkaSimulation<DustDampingMHDZeroBMixedStiff>::computeAfterTimestep()
{
	auto [_, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);

		const double density = values.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::density_index)[0];
		const double momentum_x = values.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::x1Momentum_index)[0];
		const double Egas_total = values.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::energy_index)[0];

		userData_.v_gas_vec_.push_back(momentum_x / density);
		userData_.E_gas_vec_.push_back(Egas_total);

		const double dust1_density = values.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::dustDensity_index)[0];
		const double dust1_momentum_x = values.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::x1DustMomentum_index)[0];
		userData_.v_dust1_vec_.push_back(dust1_momentum_x / dust1_density);

		const double dust2_density = values.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::dustDensity_index + numDustVars)[0];
		const double dust2_momentum_x = values.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::x1DustMomentum_index + numDustVars)[0];
		userData_.v_dust2_vec_.push_back(dust2_momentum_x / dust2_density);
	}
}

namespace
{
using ResolvedRkScheme = quokka::dust::ResolvedRkScheme;
using MixedStiffDustSources = DustSources<DustDampingMHDZeroBMixedStiff>;

constexpr double HALF_DECADE_FACTOR = 3.1622776601683795;
constexpr std::array<double, 17> SWEEP_DT_VALUES = {1.0e-4,
						    HALF_DECADE_FACTOR * 1.0e-4,
						    1.0e-3,
						    TS2,
						    HALF_DECADE_FACTOR * 1.0e-3,
						    1.0e-2,
						    HALF_DECADE_FACTOR * 1.0e-2,
						    1.0e-1,
						    TS1,
						    HALF_DECADE_FACTOR * 1.0e-1,
						    1.0,
						    HALF_DECADE_FACTOR,
						    1.0e1,
						    HALF_DECADE_FACTOR * 1.0e1,
						    1.0e2,
						    HALF_DECADE_FACTOR * 1.0e2,
						    1.0e3};

struct SchemeRunResult {
	ResolvedRkScheme scheme;
	SimulationData<DustDampingMHDZeroBMixedStiff> data;
	double rel_err_gas_vx;
	double rel_err_dust1_vx;
	double rel_err_dust2_vx;
	double rel_err_gas_E;
};

struct SchemeErrorTolerance {
	double rel_err_gas_vx;
	double rel_err_dust1_vx;
	double rel_err_dust2_vx;
	double rel_err_gas_E;
};

struct SweepSample {
	ResolvedRkScheme scheme;
	double requested_dt;
	double effective_dt;
	int step_count;
	double end_time;
	double velocity_error;
	bool used_resolved_branch;
};

struct NoResidualCorrectionState {
	MixedStiffDustSources::Vec3 gas_momentum;
	amrex::GpuArray<MixedStiffDustSources::Vec3, 2> dust_momentum;
	double gas_energy;
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

auto resolvedRkSchemeTolerance(ResolvedRkScheme scheme) -> SchemeErrorTolerance
{
	switch (scheme) {
		case ResolvedRkScheme::TP2025:
			return SchemeErrorTolerance{.rel_err_gas_vx = 3.0e-3, .rel_err_dust1_vx = 2.0e-4, .rel_err_dust2_vx = 3.0e-3, .rel_err_gas_E = 1.5e-3};
		case ResolvedRkScheme::GL4:
			return SchemeErrorTolerance{.rel_err_gas_vx = 2.0e-2, .rel_err_dust1_vx = 2.0e-4, .rel_err_dust2_vx = 2.0e-2, .rel_err_gas_E = 6.0e-3};
		case ResolvedRkScheme::Midpoint:
			return SchemeErrorTolerance{.rel_err_gas_vx = 6.0e-2, .rel_err_dust1_vx = 1.0e-3, .rel_err_dust2_vx = 6.0e-2, .rel_err_gas_E = 2.0e-2};
	}
	return SchemeErrorTolerance{
	    .rel_err_gas_vx = std::numeric_limits<double>::quiet_NaN(),
	    .rel_err_dust1_vx = std::numeric_limits<double>::quiet_NaN(),
	    .rel_err_dust2_vx = std::numeric_limits<double>::quiet_NaN(),
	    .rel_err_gas_E = std::numeric_limits<double>::quiet_NaN(),
	};
}

auto makePeriodicFaceBCs() -> amrex::Vector<amrex::BCRec>
{
	const int nvars_fc = Physics_Indices<DustDampingMHDZeroBMixedStiff>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}
	return BCs_fc;
}

auto runDustDampingSimulation(ResolvedRkScheme scheme, double constant_dt, int step_count) -> SimulationData<DustDampingMHDZeroBMixedStiff>
{
	const double CFL_number = 1000000.0; // large CFL number to avoid CFL violation
	amrex::ParmParse pp;
	pp.add("constant_dt", constant_dt);
	pp.add("stop_time", static_cast<double>(step_count) * constant_dt);
	pp.add("max_timesteps", step_count);
	pp.add("suppress_output", 1);
	pp.add("show_performance_hints", 0);

	auto BCs_cc = quokka::BC<DustDampingMHDZeroBMixedStiff>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	auto BCs_fc = makePeriodicFaceBCs();
	QuokkaSimulation<DustDampingMHDZeroBMixedStiff> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;
	sim.constantDt_ = constant_dt;
	sim.dustResolvedRkScheme_ = scheme;
	sim.dust_omega_drag_ = OMEGA_DRAG;
	sim.dust_omega_gyro_res_ = 0.0;
	sim.stopTime_ = static_cast<double>(step_count) * constant_dt;
	sim.maxTimesteps_ = step_count;

	sim.setInitialConditions();

	auto [_, val_ini] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.t_vec_.push_back(0.0);

		const double initial_density = val_ini.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::density_index)[0];
		const double initial_momentum_x = val_ini.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::x1Momentum_index)[0];
		const double initial_Egas_total = val_ini.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::energy_index)[0];
		sim.userData_.v_gas_vec_.push_back(initial_momentum_x / initial_density);
		sim.userData_.E_gas_vec_.push_back(initial_Egas_total);

		const double initial_dust1_density = val_ini.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::dustDensity_index)[0];
		const double initial_dust1_momentum_x = val_ini.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::x1DustMomentum_index)[0];
		sim.userData_.v_dust1_vec_.push_back(initial_dust1_momentum_x / initial_dust1_density);

		const double initial_dust2_density = val_ini.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::dustDensity_index + numDustVars)[0];
		const double initial_dust2_momentum_x = val_ini.at(HydroSystem<DustDampingMHDZeroBMixedStiff>::x1DustMomentum_index + numDustVars)[0];
		sim.userData_.v_dust2_vec_.push_back(initial_dust2_momentum_x / initial_dust2_density);
	}

	sim.evolve();
	return sim.userData_;
}

void advanceTp2025WithoutResidualCorrectionHalfStep(NoResidualCorrectionState &state, double source_dt, double full_dt)
{
	amrex::GpuArray<double, 2> const rho_d = {rho_dust1, rho_dust2};
	amrex::GpuArray<double, 2> const alpha = {1.0 / TS1, 1.0 / TS2};
	amrex::GpuArray<double, 2> const epsilon = {rho_dust1 / rho, rho_dust2 / rho};
	amrex::GpuArray<double, 2> const omega_L = {0.0, 0.0};
	amrex::GpuArray<MixedStiffDustSources::Vec3, 2> q_n{};
	MixedStiffDustSources::Vec3 const b_hat = MixedStiffDustSources::Vec3::Zero();

	for (int g = 0; g < 2; ++g) {
		q_n[g] = state.dust_momentum[g] - epsilon[g] * state.gas_momentum;
	}

	bool const resolved_branch = full_dt < TS1;
	auto const coefficients = MixedStiffDustSources::SelectGirkCoefficients(resolved_branch, ResolvedRkScheme::TP2025);
	amrex::GpuArray<MixedStiffDustSources::DustStageAffineOperators, 2> ops{};
	for (int g = 0; g < 2; ++g) {
		ops[g] =
		    MixedStiffDustSources::ComputeDustStageAffineOperators(alpha[g], omega_L[g], alpha[g], omega_L[g], epsilon[g], source_dt,
									   coefficients.gamma1, coefficients.gamma2, coefficients.beta1, coefficients.beta2);
	}

	auto const gas_stage = MixedStiffDustSources::SolveGasStageRates(ops, q_n, b_hat);
	amrex::GpuArray<MixedStiffDustSources::Vec3, 2> dust_rate1{};
	amrex::GpuArray<MixedStiffDustSources::Vec3, 2> dust_rate2{};
	amrex::GpuArray<MixedStiffDustSources::Vec3, 2> q1{};
	amrex::GpuArray<MixedStiffDustSources::Vec3, 2> q2{};
	for (int g = 0; g < 2; ++g) {
		dust_rate1[g] = ops[g].P1.apply(q_n[g], b_hat) + ops[g].X1.apply(gas_stage.k1, b_hat) + ops[g].Y1.apply(gas_stage.k2, b_hat);
		dust_rate2[g] = ops[g].P2.apply(q_n[g], b_hat) + ops[g].X2.apply(gas_stage.k1, b_hat) + ops[g].Y2.apply(gas_stage.k2, b_hat);
		auto const relative_rate1 = dust_rate1[g] - epsilon[g] * gas_stage.k1;
		auto const relative_rate2 = dust_rate2[g] - epsilon[g] * gas_stage.k2;
		q1[g] = q_n[g] + source_dt * (coefficients.gamma1 * relative_rate1 + coefficients.beta1 * relative_rate2);
		q2[g] = q_n[g] + source_dt * (coefficients.beta2 * relative_rate1 + coefficients.gamma2 * relative_rate2);
	}

	auto const gas_momentum_old = state.gas_momentum;
	state.gas_momentum += source_dt * (coefficients.b * gas_stage.k1 + (1.0 - coefficients.b) * gas_stage.k2);
	double physical_drag_heating = 0.0;
	for (int g = 0; g < 2; ++g) {
		state.dust_momentum[g] += source_dt * (coefficients.b * dust_rate1[g] + (1.0 - coefficients.b) * dust_rate2[g]);
		physical_drag_heating +=
		    source_dt / rho_d[g] * (coefficients.b * alpha[g] * q1[g].dot(q1[g]) + (1.0 - coefficients.b) * alpha[g] * q2[g].dot(q2[g]));
	}
	double const gas_work = (state.gas_momentum.dot(state.gas_momentum) - gas_momentum_old.dot(gas_momentum_old)) / (2.0 * rho);
	state.gas_energy += gas_work + physical_drag_heating;
}

auto reconstructTp2025WithoutResidualCorrection(SimulationData<DustDampingMHDZeroBMixedStiff> const &tp2025_data) -> std::vector<double>
{
	NoResidualCorrectionState state{
	    .gas_momentum = MixedStiffDustSources::Vec3{rho * gas_velocity_initial, 0.0, 0.0},
	    .dust_momentum = {MixedStiffDustSources::Vec3{rho_dust1 * dust1_velocity_initial, 0.0, 0.0},
			      MixedStiffDustSources::Vec3{rho_dust2 * dust2_velocity_initial, 0.0, 0.0}},
	    .gas_energy = Egas0,
	};
	std::vector<double> gas_energy{state.gas_energy};
	for (size_t i = 1; i < tp2025_data.t_vec_.size(); ++i) {
		double const full_dt = tp2025_data.t_vec_[i] - tp2025_data.t_vec_[i - 1];
		advanceTp2025WithoutResidualCorrectionHalfStep(state, 0.5 * full_dt, full_dt);
		advanceTp2025WithoutResidualCorrectionHalfStep(state, 0.5 * full_dt, full_dt);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(state.gas_momentum[0] / rho - tp2025_data.v_gas_vec_[i]) < 1.0e-12,
						 "TP2025 diagnostic reconstruction must preserve the gas momentum update.");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(state.dust_momentum[0][0] / rho_dust1 - tp2025_data.v_dust1_vec_[i]) < 1.0e-12,
						 "TP2025 diagnostic reconstruction must preserve the first dust momentum update.");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(state.dust_momentum[1][0] / rho_dust2 - tp2025_data.v_dust2_vec_[i]) < 1.0e-12,
						 "TP2025 diagnostic reconstruction must preserve the second dust momentum update.");
		gas_energy.push_back(state.gas_energy);
	}
	return gas_energy;
}

auto computeTimeAveragedVelocityError(SimulationData<DustDampingMHDZeroBMixedStiff> const &data) -> double
{
	double error_sum = 0.0;
	for (size_t i = 1; i < data.t_vec_.size(); ++i) {
		AnalyticState const exact = analyticState(data.t_vec_[i]);
		error_sum += std::abs((data.v_gas_vec_[i] - exact.v_gas) / exact.v_gas);
		error_sum += std::abs((data.v_dust1_vec_[i] - exact.v_dust1) / exact.v_dust1);
		error_sum += std::abs((data.v_dust2_vec_[i] - exact.v_dust2) / exact.v_dust2);
	}
	// Average the sum of the gas, dust-1, and dust-2 component-wise relative
	// velocity errors over all post-step samples. Do not divide the sum by three.
	return error_sum / static_cast<double>(data.t_vec_.size() - 1);
}

auto runTimestepSweep() -> std::vector<SweepSample>
{
	std::vector<SweepSample> samples;
	samples.reserve(resolved_rk_schemes.size() * SWEEP_DT_VALUES.size());
	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		for (double const requested_dt : SWEEP_DT_VALUES) {
			int const step_count = (requested_dt < TS1) ? static_cast<int>(std::ceil(TS1 / requested_dt)) : 1;
			auto const data = runDustDampingSimulation(scheme, requested_dt, step_count);
			double const effective_dt = data.t_vec_[1] - data.t_vec_[0];
			samples.push_back({
			    .scheme = scheme,
			    .requested_dt = requested_dt,
			    .effective_dt = effective_dt,
			    .step_count = step_count,
			    .end_time = data.t_vec_.back(),
			    .velocity_error = computeTimeAveragedVelocityError(data),
			    .used_resolved_branch = effective_dt < TS1,
			});
		}
	}
	return samples;
}

auto computeRunResult(ResolvedRkScheme scheme, SimulationData<DustDampingMHDZeroBMixedStiff> data) -> SchemeRunResult
{
	std::vector<double> const &t = data.t_vec_;
	std::vector<double> const &v_gas = data.v_gas_vec_;
	std::vector<double> const &v_dust1 = data.v_dust1_vec_;
	std::vector<double> const &v_dust2 = data.v_dust2_vec_;
	std::vector<double> const &E_gas = data.E_gas_vec_;

	std::vector<double> v_gas_exact(t.size());
	std::vector<double> v_dust1_exact(t.size());
	std::vector<double> v_dust2_exact(t.size());
	std::vector<double> E_gas_exact(t.size());

	for (size_t i = 0; i < t.size(); ++i) {
		v_gas_exact[i] = v_gas_analytic(t[i]);
		v_dust1_exact[i] = v_dust1_analytic(t[i]);
		v_dust2_exact[i] = v_dust2_analytic(t[i]);
		E_gas_exact[i] = E_gas_analytic(t[i]);
	}

	auto rel_err = [](const std::vector<double> &sim_vals, const std::vector<double> &exact_vals) {
		if ((sim_vals.size() != exact_vals.size()) || exact_vals.empty()) {
			return std::numeric_limits<double>::quiet_NaN();
		}
		double err = 0.0;
		double sol = 0.0;
		for (size_t i = 0; i < sim_vals.size(); ++i) {
			err += std::abs(sim_vals[i] - exact_vals[i]);
			sol += std::abs(exact_vals[i]);
		}
		return (sol > 0.0) ? (err / sol) : std::numeric_limits<double>::quiet_NaN();
	};

	double const rel_err_gas_vx = rel_err(v_gas, v_gas_exact);
	double const rel_err_dust1_vx = rel_err(v_dust1, v_dust1_exact);
	double const rel_err_dust2_vx = rel_err(v_dust2, v_dust2_exact);
	double const rel_err_gas_E = rel_err(E_gas, E_gas_exact);

	return SchemeRunResult{
	    .scheme = scheme,
	    .data = std::move(data),
	    .rel_err_gas_vx = rel_err_gas_vx,
	    .rel_err_dust1_vx = rel_err_dust1_vx,
	    .rel_err_dust2_vx = rel_err_dust2_vx,
	    .rel_err_gas_E = rel_err_gas_E,
	};
}

void writeHistoryCsv(const std::vector<SchemeRunResult> &runs, std::vector<double> const &tp2025_no_residual_correction)
{
	const size_t n_samples = runs.front().data.t_vec_.size();
	for (auto const &run : runs) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(run.data.t_vec_.size() == n_samples && run.data.v_gas_vec_.size() == n_samples &&
						     run.data.v_dust1_vec_.size() == n_samples && run.data.v_dust2_vec_.size() == n_samples &&
						     run.data.E_gas_vec_.size() == n_samples,
						 "DustDampingMHDZeroBMixedStiff histories must have equal lengths.");
	}

	std::ofstream file("dust_damping_mhd_zero_b_mixed_stiff_history.csv");
	file << std::setprecision(17);
	file << "t";
	for (auto const &run : runs) {
		std::string_view const slug = resolvedRkSchemeSlug(run.scheme);
		file << ",v_gas_" << slug;
	}
	file << ",v_gas_exact";
	for (auto const &run : runs) {
		std::string_view const slug = resolvedRkSchemeSlug(run.scheme);
		file << ",v_dust1_" << slug;
	}
	file << ",v_dust1_exact";
	for (auto const &run : runs) {
		std::string_view const slug = resolvedRkSchemeSlug(run.scheme);
		file << ",v_dust2_" << slug;
	}
	file << ",v_dust2_exact";
	for (auto const &run : runs) {
		std::string_view const slug = resolvedRkSchemeSlug(run.scheme);
		file << ",E_gas_" << slug;
	}
	file << ",E_gas_tp2025_no_residual_correction,E_gas_exact\n";

	for (size_t i = 0; i < n_samples; ++i) {
		double const t = runs.front().data.t_vec_[i];
		file << t;
		for (auto const &run : runs) {
			file << "," << run.data.v_gas_vec_[i];
		}
		file << "," << v_gas_analytic(t);
		for (auto const &run : runs) {
			file << "," << run.data.v_dust1_vec_[i];
		}
		file << "," << v_dust1_analytic(t);
		for (auto const &run : runs) {
			file << "," << run.data.v_dust2_vec_[i];
		}
		file << "," << v_dust2_analytic(t);
		for (auto const &run : runs) {
			file << "," << run.data.E_gas_vec_[i];
		}
		file << "," << tp2025_no_residual_correction[i] << "," << E_gas_analytic(t) << "\n";
	}
}

void writeExactCsv(const std::vector<SchemeRunResult> &runs)
{
	const size_t n_dense_points = 1000;
	const double t_max = runs.front().data.t_vec_.empty() ? 0.0 : runs.front().data.t_vec_.back();

	std::ofstream file("dust_damping_mhd_zero_b_mixed_stiff_exact.csv");
	file << std::setprecision(17);
	file << "t,v_gas_exact,v_dust1_exact,v_dust2_exact,E_gas_exact\n";
	for (size_t i = 0; i < n_dense_points; ++i) {
		double const t = t_max * static_cast<double>(i) / static_cast<double>(n_dense_points - 1);
		file << t << "," << v_gas_analytic(t) << "," << v_dust1_analytic(t) << "," << v_dust2_analytic(t) << "," << E_gas_analytic(t) << "\n";
	}
}

void writeSummaryCsv(const std::vector<SchemeRunResult> &runs)
{
	std::ofstream file("dust_damping_mhd_zero_b_mixed_stiff_summary.csv");
	file << std::setprecision(17);
	file << "scheme,rel_err_gas_vx,rel_err_dust1_vx,rel_err_dust2_vx,rel_err_gas_E\n";
	for (auto const &run : runs) {
		file << resolvedRkSchemeSlug(run.scheme) << "," << run.rel_err_gas_vx << "," << run.rel_err_dust1_vx << "," << run.rel_err_dust2_vx << ","
		     << run.rel_err_gas_E << "\n";
	}
}

void writeTimestepSweepCsv(std::vector<SweepSample> const &samples)
{
	std::ofstream file("dust_damping_mhd_zero_b_mixed_stiff_timestep_sweep.csv");
	file << std::setprecision(17);
	file << "scheme,requested_dt,effective_dt,step_count,end_time,velocity_error,used_resolved_branch,fast_stopping_time,branch_transition_dt\n";
	for (auto const &sample : samples) {
		file << resolvedRkSchemeSlug(sample.scheme) << "," << sample.requested_dt << "," << sample.effective_dt << "," << sample.step_count << ","
		     << sample.end_time << "," << sample.velocity_error << "," << (sample.used_resolved_branch ? 1 : 0) << "," << TS2 << "," << TS1 << "\n";
	}
}
} // namespace

auto problem_main() -> int
{
	bool write_csv = true;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);

	std::vector<SchemeRunResult> runs;
	runs.reserve(resolved_rk_schemes.size());
	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		int const step_count = static_cast<int>(std::lround(STOP_TIME / HISTORY_DT));
		runs.push_back(computeRunResult(scheme, runDustDampingSimulation(scheme, HISTORY_DT, step_count)));
	}
	std::vector<double> const tp2025_no_residual_correction = reconstructTp2025WithoutResidualCorrection(runs[0].data);
	std::vector<SweepSample> const sweep_samples = write_csv ? runTimestepSweep() : std::vector<SweepSample>{};

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		for (auto const &run : runs) {
			SchemeErrorTolerance const tol = resolvedRkSchemeTolerance(run.scheme);
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for gas vx    = " << run.rel_err_gas_vx
				       << "\n";
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for dust1 vx  = " << run.rel_err_dust1_vx
				       << "\n";
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for dust2 vx  = " << run.rel_err_dust2_vx
				       << "\n";
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for gas E     = " << run.rel_err_gas_E
				       << "\n";
			if (!std::isfinite(run.rel_err_gas_vx) || !std::isfinite(run.rel_err_dust1_vx) || !std::isfinite(run.rel_err_dust2_vx) ||
			    !std::isfinite(run.rel_err_gas_E) || (run.rel_err_gas_vx > tol.rel_err_gas_vx) || (run.rel_err_dust1_vx > tol.rel_err_dust1_vx) ||
			    (run.rel_err_dust2_vx > tol.rel_err_dust2_vx) || (run.rel_err_gas_E > tol.rel_err_gas_E)) {
				status = 1;
			}
		}

		auto const &tp2025 = runs[0];
		auto const &gl4 = runs[1];
		auto const &midpoint = runs[2];
		bool const gas_order_ok = (tp2025.rel_err_gas_vx < gl4.rel_err_gas_vx) && (gl4.rel_err_gas_vx < midpoint.rel_err_gas_vx);
		bool const dust2_order_ok = (tp2025.rel_err_dust2_vx < gl4.rel_err_dust2_vx) && (gl4.rel_err_dust2_vx < midpoint.rel_err_dust2_vx);
		bool const energy_order_ok = (tp2025.rel_err_gas_E < gl4.rel_err_gas_E) && (gl4.rel_err_gas_E < midpoint.rel_err_gas_E);
		if (!gas_order_ok || !dust2_order_ok || !energy_order_ok) {
			status = 1;
		}

		if (write_csv) {
			writeHistoryCsv(runs, tp2025_no_residual_correction);
			writeExactCsv(runs);
			writeSummaryCsv(runs);
			writeTimestepSweepCsv(sweep_samples);
		}
		amrex::Print() << "Finished.\n";
	}

	return status;
}
