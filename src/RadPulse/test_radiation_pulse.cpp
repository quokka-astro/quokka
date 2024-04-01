//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_pulse.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
/// References: Sekora & Stone (2010, JCoPh, 229, 6819)
///

#include "test_radiation_pulse.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// constexpr int max_steps_ = 3e2;
constexpr double kappa0 = 4.0e4;  // absorption coefficient
constexpr double T0 = 1.0;	  // 
constexpr double rho0 = 1.0;	  // 
constexpr double c = 10.0;
constexpr double chat = c;
constexpr double a_rad = 1.0;
// constexpr double erad_floor = a_rad * 1.0e-12;
constexpr double erad_floor = 0.0;
constexpr double diff_coeff = c / (3.0 * rho0 * kappa0);
constexpr double max_time_ = 0.13 * 0.13 / diff_coeff;

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};

AMREX_GPU_HOST_DEVICE
auto compute_exact_Erad(const double x, const double t) -> double
{
	// compute exact solution for Gaussian radiation pulse, assuming diffusion approximation
	return 1.0 / std::sqrt(160.0 * diff_coeff * t + 1) * std::exp(-40.0 * x * x / (160.0 * diff_coeff * t + 1.0));
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_Frad(const double x, const double t) -> double
{
	// compute exact solution for Gaussian radiation pulse, assuming diffusion approximation
	const auto erad = compute_exact_Erad(x, t);
	return 80. * x / (3. * kappa0) * erad;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaPVec{};
	for (int i = 0; i < nGroups_; ++i) {
		kappaPVec[i] = kappa0 / rho;
	}
	return kappaPVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
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

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		double Erad = compute_exact_Erad(x, 0.0);
		double Frad = compute_exact_Frad(x, 0.0);
		if (std::abs(x) > 0.5) {
			Erad = std::exp(-10.0);
			Frad = 0.0;
		}
		const double Trad = std::pow(Erad / a_rad, 1. / 4.);
		const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, Trad);

		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = Frad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = 0.;
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
	int max_timesteps;
	const double CFL_number = 5.0;
	// const int nx = 32;

	// read max_timesteps from input file
	amrex::ParmParse pp;
	pp.get("max_timesteps", max_timesteps);

	const double max_time = max_time_;

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(0, amrex::BCType::foextrap); // extrapolate
			BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadhydroSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	// sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();

	std::vector<double> xs(nx);
	std::vector<double> Erad0(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> T_initial(nx);
	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Erad_exact;
	std::vector<double> mtm(nx);
	std::vector<double> rho(nx);

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position0[i];
		xs.at(i) = x;
		Erad0.at(i) = values0.at(RadSystem<PulseProblem>::radEnergy_index)[i];

		rho.at(i) = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		mtm.at(i) = values.at(RadSystem<PulseProblem>::x1GasMomentum_index)[i];

		const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t;
		Egas.at(i) = values.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho0, Egas.at(i));

		auto Erad_val = compute_exact_Erad(x, sim.tNew_[0]);
		auto Trad_val = std::pow(Erad_val / a_rad, 1. / 4.);
		xs_exact.push_back(x);
		Trad_exact.push_back(Trad_val);
		Erad_exact.push_back(Erad_val);
	}

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < xs.size(); ++i) {
		// err_norm += std::abs(Erad[i] - Erad_exact[i]);
		// sol_norm += std::abs(Erad_exact[i]);
		err_norm += std::abs(Trad[i] - Tgas[i]);
		sol_norm += std::abs(Trad[i]);
	}
	const double error_tol = 1.0e-3;
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();

	// Initial pulse

	std::map<std::string, std::string> args;
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Tinit_args;
	std::map<std::string, std::string> Trad_exact_args;
	Tinit_args["label"] = "initial";
	Tinit_args["color"] = "grey";
	// Trad_args["label"] = "radiation";
	Trad_args["linestyle"] = "-";
	// Tgas_args["label"] = "gas";
	Tgas_args["linestyle"] = "--";
	// Trad_exact_args["label"] = "radiation (exact)";
	Trad_exact_args["linestyle"] = ":";

	matplotlibcpp::plot(xs, Erad0, Tinit_args);
	matplotlibcpp::plot(xs, Erad, Trad_args);
	// matplotlibcpp::plot(xs, Egas, Tgas_args);
	matplotlibcpp::plot(xs, Erad_exact, Trad_exact_args);

	// Plot radiation temperature

	args["linestyle"] = "-";
	args["color"] = "r";
	matplotlibcpp::plot(xs, Trad, args);
	args["linestyle"] = "--";
	args["color"] = "g";
	matplotlibcpp::plot(xs, Tgas, args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time sqrt(2 D t) = {:.4g}", std::sqrt(2.0 * diff_coeff * sim.tNew_[0])));
	matplotlibcpp::save("./radiation_pulse_temperature.pdf");

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity";
	vgas_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, mtm, vgas_args);
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("rho * v_x (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time sqrt(2 D t) = {:.4g}", std::sqrt(2.0 * diff_coeff * sim.tNew_[0])));
	matplotlibcpp::save("./radiation_pulse_velocity.pdf");
#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
