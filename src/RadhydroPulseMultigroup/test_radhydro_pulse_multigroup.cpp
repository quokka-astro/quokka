//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_pulse.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radhydro_pulse_multigroup.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int isconst = 0;
#if 1
constexpr int n_groups_ = 1;
constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{0., inf};
#elif 1
constexpr int n_groups_ = 8;
constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1e15, 3.16e15, 1e16, 3.16e16, 1e17, 3.16e17, 1e18, 3.16e18, 1e19};
#else
constexpr int n_groups_ = 64;
constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1.00000000e+15, 1.15478198e+15, 1.33352143e+15, 1.53992653e+15,
       1.77827941e+15, 2.05352503e+15, 2.37137371e+15, 2.73841963e+15,
       3.16227766e+15, 3.65174127e+15, 4.21696503e+15, 4.86967525e+15,
       5.62341325e+15, 6.49381632e+15, 7.49894209e+15, 8.65964323e+15,
       1.00000000e+16, 1.15478198e+16, 1.33352143e+16, 1.53992653e+16,
       1.77827941e+16, 2.05352503e+16, 2.37137371e+16, 2.73841963e+16,
       3.16227766e+16, 3.65174127e+16, 4.21696503e+16, 4.86967525e+16,
       5.62341325e+16, 6.49381632e+16, 7.49894209e+16, 8.65964323e+16,
       1.00000000e+17, 1.15478198e+17, 1.33352143e+17, 1.53992653e+17,
       1.77827941e+17, 2.05352503e+17, 2.37137371e+17, 2.73841963e+17,
       3.16227766e+17, 3.65174127e+17, 4.21696503e+17, 4.86967525e+17,
       5.62341325e+17, 6.49381632e+17, 7.49894209e+17, 8.65964323e+17,
       1.00000000e+18, 1.15478198e+18, 1.33352143e+18, 1.53992653e+18,
       1.77827941e+18, 2.05352503e+18, 2.37137371e+18, 2.73841963e+18,
       3.16227766e+18, 3.65174127e+18, 4.21696503e+18, 4.86967525e+18,
       5.62341325e+18, 6.49381632e+18, 7.49894209e+18, 8.65964323e+18,
       1.00000000e+19};
#endif 

constexpr double kappa0 = 180.; // cm^2 g^-1
// constexpr double kappa0 = 0.0180; // cm^2 g^-1
constexpr double T0 = 1.0e7;	// K (temperature)
constexpr double T1 = 2.0e7;	// K (temperature)
constexpr double rho0 = 1.2;	// g cm^-3 (matter density)
constexpr double a_rad = C::a_rad;
constexpr double c = C::c_light; // speed of light (cgs)
constexpr double chat = c;
constexpr double width = 24.0; // cm, width of the pulse
constexpr double erad_floor = a_rad * T0 * T0 * T0 * T0 * 1.0e-12;
constexpr double mu = 2.33 * C::m_u;
constexpr double h_planck = C::hplanck;
constexpr double k_B = C::k_B;
constexpr double initial_time = 0.0;
constexpr double max_time = 2.4e-5;
// constexpr double v0 = 0.;      // non-advecting pulse
constexpr double v0 = 1.0e6; // advecting pulse, v0 = 2.0 * width / max_time;

constexpr double T_ref = T0;
constexpr double nu_ref = 1.0e18; // Hz
constexpr double coeff_ = h_planck * nu_ref / (k_B * T_ref);

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr bool compute_v_over_c_terms = true;
	static constexpr double energy_unit = C::hplanck;
	static constexpr amrex::GpuArray<double, Physics_Traits<PulseProblem>::nGroups + 1> radBoundaries = rad_boundaries_;
};

AMREX_GPU_HOST_DEVICE
auto compute_initial_Tgas(const double x) -> double
{
	// compute temperature profile for Gaussian radiation pulse
	const double sigma = width;
	return T0 + (T1 - T0) * std::exp(-x * x / (2.0 * sigma * sigma));
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_rho(const double x) -> double
{
	// compute density profile for Gaussian radiation pulse
	auto T = compute_initial_Tgas(x);
	return rho0 * T0 / T + (a_rad * mu / 3. / k_B) * (std::pow(T0, 4) / T - std::pow(T, 3));
}

AMREX_GPU_HOST_DEVICE 
auto compute_kappa(const double nu, const double Tgas) -> double
{
  // cm^-1
  auto T_ = Tgas / T_ref;
  auto nu_ = nu / nu_ref;
  return kappa0 * std::pow(T_, -0.5) * std::pow(nu_, -3) * (1.0 - std::exp(-coeff_ * nu_ / T_));
}

AMREX_GPU_HOST_DEVICE 
auto compute_repres_nu() -> quokka::valarray<double, n_groups_>
{
  // return the geometrical mean as the representative frequency for each group
  quokka::valarray<double, n_groups_> nu_rep{};
  if constexpr (n_groups_ == 1) {
    nu_rep[0] = nu_ref;
  } else {
    for (int g = 0; g < n_groups_; ++g) {
      nu_rep[g] = std::sqrt(rad_boundaries_[g] * rad_boundaries_[g + 1]);
    }
  }
  return nu_rep;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaPVec{};
  auto nu_rep = compute_repres_nu();
  for (int g = 0; g < nGroups_; ++g) {
    kappaPVec[g] = compute_kappa(nu_rep[g], Tgas) / rho;
  }
  if constexpr (isconst == 1) {
    kappaPVec.fillin(100.0);
  }
	return kappaPVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
  return ComputePlanckOpacity(rho, Tgas);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacityTempDerivative(const double rho, const double Tgas)
    -> quokka::valarray<double, nGroups_>
{
  quokka::valarray<double, nGroups_> opacity_deriv{};
  auto nu_rep = compute_repres_nu();
  auto T = Tgas / T0;
  for (int g = 0; g < nGroups_; ++g) {
    auto nu = nu_rep[g] / nu_ref;
    opacity_deriv[g] = 1. / T0 * kappa0 * (-0.5 * std::pow(T, -1.5) * (1. - std::exp(-coeff_ * nu / T)) - coeff_ * std::pow(T, -2.5) * std::exp(-coeff_ * nu / T));
    opacity_deriv[g] /= rho;
  }
  if constexpr (isconst == 1) {
    opacity_deriv.fillin(0.0);
  }
	return opacity_deriv;
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<PulseProblem>::ComputeEddingtonFactor(double /*f*/) -> double
{
	return (1. / 3.); // Eddington approximation
}

template <> void RadhydroSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	const auto radBoundaries_g = RadSystem_Traits<PulseProblem>::radBoundaries;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		const double Trad = compute_initial_Tgas(x - x0);
		const double rho = compute_exact_rho(x - x0);
		const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho, Trad);

    auto Erad_g = RadSystem<PulseProblem>::ComputeThermalRadiation(Trad, radBoundaries_g);

		for (int g = 0; g < Physics_Traits<PulseProblem>::nGroups; ++g) {
      // state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = (1. + 4. / 3. * (v0 * v0) / (c * c)) * Erad;
      state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
      state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 4. / 3. * v0 * Erad_g[g];
      state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
      state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
    }

		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a *linear* radiation diffusion problem, i.e.
	// parameters are chosen such that the radiation and gas temperatures
	// should be near equilibrium, and the opacity is chosen to go as
	// T^3, such that the radiation diffusion equation becomes linear in T.

	// This makes this problem a stringent test of the asymptotic-
	// preserving property of the computational method, since the
	// optical depth per cell at the peak of the temperature profile is
	// of order 10^5.

	// Problem parameters
	const long int max_timesteps = 1e8;
	// const double CFL_number = 0.8;
	const double CFL_number = 0.4;
	// const double CFL_number = 0.0008;
	// const int nx = 32;

	// const double max_dt = 2e-9;   // t_cr = 2 cm / cs = 7e-8 s
	const double max_dt = 1e-3; // t_cr = 2 cm / cs = 7e-8 s

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::foextrap); // extrapolate
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	RadhydroSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.cflNumber_ = CFL_number;
	sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> V0vec(nx);
	std::vector<double> rhogas(nx);
	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Erad_exact;

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		// const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index)[i];
    double Erad_t = 0.0;
    for (int g = 0; g < Physics_Traits<PulseProblem>::nGroups; ++g) {
      Erad_t += values.at(RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
    }
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<PulseProblem>::x1GasMomentum_index)[i] / rho_t;
		rhogas.at(i) = rho_t;
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t;
		Egas.at(i) = values.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho_t, Egas.at(i));
		Vgas.at(i) = v_t;
		V0vec.at(i) = v0;

		auto Trad_val = compute_initial_Tgas(x - x0);
		auto Erad_val = a_rad * std::pow(Trad_val, 4);
		xs_exact.push_back(x);
		Trad_exact.push_back(Trad_val);
		Erad_exact.push_back(Erad_val);
	}

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < xs.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Trad[i]);
		sol_norm += std::abs(Trad[i]);
	}
	const double error_tol = 0.002; // without advection, rel_error = 0.002275703823
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
  std::string t_str = fmt::format("t{:.5g}", initial_time + sim.tNew_[0]);
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	Trad_args["label"] = "radiation temperature";
	Trad_args["linestyle"] = "-.";
	Tgas_args["label"] = "gas temperature";
	Tgas_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (K)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", initial_time + sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_multigroup_temperature.pdf");

	// plot gas density profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> rho_args;
	rho_args["label"] = "gas density";
	rho_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, rhogas, rho_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("density (g cm^-3)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", initial_time + sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_multigroup_density.pdf");

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity";
	vgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("velocity (cm s^-1)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", initial_time + sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_multigroup_velocity.pdf");

#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
