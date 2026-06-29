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
AMREX_GPU_HOST_DEVICE auto
DustSources<DustDampingMHDZeroBMixedStiff>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
									  amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 2> alpha{};
	alpha[0] = 1.0 / TS1;
	alpha[1] = 1.0 / TS2;
	return alpha;
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustDampingMHDZeroBMixedStiff>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 2> charge_to_mass_ratio{};
	charge_to_mass_ratio.fill(1.0);
	return charge_to_mass_ratio;
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
			return SchemeErrorTolerance{.rel_err_gas_vx = 1.5e-2, .rel_err_dust1_vx = 2.0e-4, .rel_err_dust2_vx = 1.5e-2, .rel_err_gas_E = 5.0e-3};
		case ResolvedRkScheme::Midpoint:
			return SchemeErrorTolerance{.rel_err_gas_vx = 5.0e-2, .rel_err_dust1_vx = 1.0e-3, .rel_err_dust2_vx = 5.0e-2, .rel_err_gas_E = 2.0e-2};
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

auto runDustDampingSimulation(ResolvedRkScheme scheme) -> SimulationData<DustDampingMHDZeroBMixedStiff>
{
	const double CFL_number = 1000000.0; // large CFL number to avoid CFL violation

	auto BCs_cc = quokka::BC<DustDampingMHDZeroBMixedStiff>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);
	auto BCs_fc = makePeriodicFaceBCs();
	QuokkaSimulation<DustDampingMHDZeroBMixedStiff> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;
	sim.dustResolvedRkScheme_ = scheme;
	sim.dust_omega_drag_ = OMEGA_DRAG;
	sim.dust_omega_gyro_res_ = 0.0;

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

void writeHistoryCsv(const std::vector<SchemeRunResult> &runs)
{
	if (runs.empty()) {
		return;
	}

	size_t n_samples = runs.front().data.t_vec_.size();
	for (auto const &run : runs) {
		n_samples = std::min(n_samples, run.data.t_vec_.size());
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
	file << ",E_gas_exact\n";

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
		file << "," << E_gas_analytic(t) << "\n";
	}
}

void writeExactCsv(const std::vector<SchemeRunResult> &runs)
{
	if (runs.empty()) {
		return;
	}

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
} // namespace

auto problem_main() -> int
{
	bool write_csv = true;
	amrex::ParmParse const pp("problem");
	pp.query("write_csv", write_csv);

	std::vector<SchemeRunResult> runs;
	runs.reserve(resolved_rk_schemes.size());
	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		runs.push_back(computeRunResult(scheme, runDustDampingSimulation(scheme)));
	}

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

		auto const find_run = [&runs](ResolvedRkScheme scheme) {
			return std::find_if(runs.begin(), runs.end(), [scheme](auto const &run) { return run.scheme == scheme; });
		};
		auto const tp2025 = find_run(ResolvedRkScheme::TP2025);
		auto const gl4 = find_run(ResolvedRkScheme::GL4);
		auto const midpoint = find_run(ResolvedRkScheme::Midpoint);
		if ((tp2025 == runs.end()) || (gl4 == runs.end()) || (midpoint == runs.end())) {
			status = 1;
		} else {
			bool const gas_order_ok = (tp2025->rel_err_gas_vx < gl4->rel_err_gas_vx) && (gl4->rel_err_gas_vx < midpoint->rel_err_gas_vx);
			bool const dust2_order_ok = (tp2025->rel_err_dust2_vx < gl4->rel_err_dust2_vx) && (gl4->rel_err_dust2_vx < midpoint->rel_err_dust2_vx);
			bool const energy_order_ok = (tp2025->rel_err_gas_E < gl4->rel_err_gas_E) && (gl4->rel_err_gas_E < midpoint->rel_err_gas_E);
			if (!gas_order_ok || !dust2_order_ok || !energy_order_ok) {
				status = 1;
			}
		}

		if (write_csv) {
			writeHistoryCsv(runs);
			writeExactCsv(runs);
			writeSummaryCsv(runs);
		}
		amrex::Print() << "Finished.\n";
	}

	return status;
}
