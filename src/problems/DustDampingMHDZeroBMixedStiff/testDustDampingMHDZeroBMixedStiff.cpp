/// \file testDustDampingMHDZeroBMixedStiff.cpp
/// \brief Dust drag damping test with MHD enabled, zero magnetic field, and mixed stopping times.
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

namespace
{
constexpr double rho = 1.0;
constexpr double rho_dust1 = 1.0;
constexpr double rho_dust2 = 1.0;
constexpr double TS1 = 0.2;
constexpr double TS2 = 0.002;
constexpr double P_INITIAL = 1.0;
constexpr double OMEGA = 1.0;
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
	double const exp1 = std::exp(mode.lambda1 * t);
	double const exp2 = std::exp(mode.lambda2 * t);

	double const gas_mode = mode.coeff1 * exp1 + mode.coeff2 * exp2;
	double const dust1_mode = mode.coeff1 * mode.dust1_ratio1 * exp1 + mode.coeff2 * mode.dust1_ratio2 * exp2;
	double const dust2_mode = mode.coeff1 * mode.dust2_ratio1 * exp1 + mode.coeff2 * mode.dust2_ratio2 * exp2;

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
		     OMEGA * (rho_dust1 * std::pow(state1.v_dust1 - state1.v_gas, 2) / TS1 + rho_dust2 * std::pow(state1.v_dust2 - state1.v_gas, 2) / TS2));

		double const term2 =
		    (rho_dust1 * (state2.v_dust1 - state2.v_gas) / TS1 * state2.v_gas + rho_dust2 * (state2.v_dust2 - state2.v_gas) / TS2 * state2.v_gas +
		     OMEGA * (rho_dust1 * std::pow(state2.v_dust1 - state2.v_gas, 2) / TS1 + rho_dust2 * std::pow(state2.v_dust2 - state2.v_gas, 2) / TS2));

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

template <> struct Physics_Traits<DustDampingMHDZeroBMixedStiff> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 2;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
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

constexpr std::array<ResolvedRkScheme, 3> resolved_rk_schemes = {ResolvedRkScheme::TP2025, ResolvedRkScheme::GL4, ResolvedRkScheme::Midpoint};
constexpr std::array<char const *, 3> scheme_colors = {"C0", "C1", "C2"};
constexpr std::array<char const *, 3> scheme_markers = {"o", "s", "^"};

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
	sim.dust_omega_drag_ = 1.0;
	sim.dust_omega_magnetic_res_ = 0.0;

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
} // namespace

auto problem_main() -> int
{
	std::vector<SchemeRunResult> runs;
	runs.reserve(resolved_rk_schemes.size());
	for (ResolvedRkScheme const scheme : resolved_rk_schemes) {
		runs.push_back(computeRunResult(scheme, runDustDampingSimulation(scheme)));
	}

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const double rel_err_tol = 0.03;
		for (auto const &run : runs) {
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for gas vx    = " << run.rel_err_gas_vx
				       << "\n";
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for dust1 vx  = " << run.rel_err_dust1_vx
				       << "\n";
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for dust2 vx  = " << run.rel_err_dust2_vx
				       << "\n";
			amrex::Print() << "[" << quokka::dust::resolvedRkSchemeName(run.scheme) << "] Relative L1 norm for gas E     = " << run.rel_err_gas_E
				       << "\n";
			if (!std::isfinite(run.rel_err_gas_vx) || !std::isfinite(run.rel_err_dust1_vx) || !std::isfinite(run.rel_err_dust2_vx) ||
			    !std::isfinite(run.rel_err_gas_E) || (run.rel_err_gas_vx > rel_err_tol) || (run.rel_err_dust1_vx > rel_err_tol) ||
			    (run.rel_err_dust2_vx > rel_err_tol) || (run.rel_err_gas_E > rel_err_tol)) {
				status = 1;
			}
		}

#ifdef HAVE_PYTHON
		std::vector<double> const &t = runs.front().data.t_vec_;
		const size_t n_dense_points = 1000;
		std::vector<double> t_dense(n_dense_points);
		std::vector<double> v_gas_exact_dense(n_dense_points);
		std::vector<double> v_dust1_exact_dense(n_dense_points);
		std::vector<double> v_dust2_exact_dense(n_dense_points);
		std::vector<double> E_gas_exact_dense(n_dense_points);
		double const t_max = t.empty() ? 0.0 : t.back();
		for (size_t i = 0; i < n_dense_points; ++i) {
			t_dense[i] = t_max * static_cast<double>(i) / (n_dense_points - 1);
			v_gas_exact_dense[i] = v_gas_analytic(t_dense[i]);
			v_dust1_exact_dense[i] = v_dust1_analytic(t_dense[i]);
			v_dust2_exact_dense[i] = v_dust2_analytic(t_dense[i]);
			E_gas_exact_dense[i] = E_gas_analytic(t_dense[i]);
		}

		auto plotRunSeries = [&](auto const &series_accessor) {
			for (size_t idx = 0; idx < runs.size(); ++idx) {
				auto const &series = series_accessor(runs[idx].data);
				matplotlibcpp::plot(runs[idx].data.t_vec_, series,
						    {{"label", quokka::dust::resolvedRkSchemeName(runs[idx].scheme)},
						     {"color", scheme_colors[idx]},
						     {"linestyle", "-"},
						     {"marker", scheme_markers[idx]},
						     {"markersize", "3"}});
			}
		};

		matplotlibcpp::clf();
		matplotlibcpp::plot(t_dense, v_gas_exact_dense, {{"label", "analytic"}, {"color", "k"}, {"linestyle", "--"}});
		plotRunSeries([](auto const &data) -> std::vector<double> const & { return data.v_gas_vec_; });
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_g$)");
		matplotlibcpp::title("Gas Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_mhd_zero_b_mixed_stiff_gas_velocity.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::plot(t_dense, v_dust1_exact_dense, {{"label", "analytic"}, {"color", "k"}, {"linestyle", "--"}});
		plotRunSeries([](auto const &data) -> std::vector<double> const & { return data.v_dust1_vec_; });
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,1}$)");
		matplotlibcpp::title("Dust1 Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_mhd_zero_b_mixed_stiff_dust1_velocity.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::plot(t_dense, v_dust2_exact_dense, {{"label", "analytic"}, {"color", "k"}, {"linestyle", "--"}});
		plotRunSeries([](auto const &data) -> std::vector<double> const & { return data.v_dust2_vec_; });
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($v_{d,2}$)");
		matplotlibcpp::title("Dust2 Velocity Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_mhd_zero_b_mixed_stiff_dust2_velocity.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::plot(t_dense, E_gas_exact_dense, {{"label", "analytic"}, {"color", "k"}, {"linestyle", "--"}});
		plotRunSeries([](auto const &data) -> std::vector<double> const & { return data.E_gas_vec_; });
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("t");
		matplotlibcpp::ylabel(R"($E_g$)");
		matplotlibcpp::title("Gas Energy Evolution");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_damping_mhd_zero_b_mixed_stiff_gas_energy.pdf");
#endif
		amrex::Print() << "Finished.\n";
	}

	return status;
}
