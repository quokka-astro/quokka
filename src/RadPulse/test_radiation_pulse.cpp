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
#include "fundamental_constants.H"
#include "physics_info.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double rho0 = 1.0; // cgs

// Jiang+2013
// constexpr double c = 514.4;
// constexpr double v = 1.0;
// constexpr double kappa0 = 4.0e1;  // absorption coefficient
// constexpr double nuSqr = 0.002;
// constexpr double max_time_ = 45.0 / v;

// Jiang+2013, dimensional
constexpr double c = C::c_light;
constexpr double vshift = c / 514.4;
constexpr double v = 0.0;
constexpr double kappa0 = 4.0e1; // absorption coefficient
constexpr double nuSqr = 0.002;
constexpr double max_time_ = 3 * 45.0 / vshift;

constexpr double chat = 0.1 * c;
constexpr double a_rad = C::a_rad;
constexpr double erad_floor = a_rad * 1.0e-13;
// constexpr double erad_floor = 0.0;
constexpr double diff_coeff = c / (3.0 * kappa0);

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = C::m_p;
	static constexpr double boltzmann_constant = C::k_B;
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
	const double base = std::exp(-nuSqr * 50.0 * 50.0);
	const double xhat = x - v * t;
	const double widthSqr = 4.0 * diff_coeff * t * nuSqr + 1;
	const double erad = 1.0 / std::sqrt(widthSqr) * std::exp(-nuSqr * std::pow(xhat, 2) / widthSqr);
	return std::max(base, erad);
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_Frad0(const double x) -> double
{
	// compute exact solution for Gaussian radiation pulse, assuming diffusion approximation
	auto erad = compute_exact_Erad(x, 0.0);
	return 2.0 * nuSqr * x / (3. * kappa0) * erad + 4.0 * v / (3. * c) * erad;
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
		const double Erad = compute_exact_Erad(x, 0.0);
		const double Frad = compute_exact_Frad0(x);
		const double Trad = std::pow(Erad / a_rad, 1. / 4.);
		const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, Trad);

		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = Frad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v * v;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = rho0 * v;
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
	int max_timesteps = 1e5;
	const double CFL_number = 0.8;
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
			// BCs_cc[n].setLo(0, amrex::BCType::foextrap); // extrapolate
			// BCs_cc[n].setHi(0, amrex::BCType::foextrap);
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	RadhydroSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = 1.e-2;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);

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
		amrex::Real const x = position[i];
		xs.at(i) = x;
		auto Erad0_val = compute_exact_Erad(x, 0.0);
		Erad0.at(i) = Erad0_val;

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

	// Plot radiation energy density
	std::map<std::string, std::string> args;
	args["label"] = "initial";
	args["color"] = "grey";
	args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Erad0, args);
	args["label"] = "numerical";
	args["color"] = "C0";
	args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Erad, args);
	args["label"] = "exact";
	args["color"] = "C1";
	args["linestyle"] = "-.";
	matplotlibcpp::plot(xs, Erad_exact, args);

	// save figure
	matplotlibcpp::yscale("log");
	matplotlibcpp::ylim(0.003, 2.0);
	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("Erad (dimensionless)");
	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("time sqrt(2 D t) = {:.4g}", std::sqrt(2.0 * diff_coeff * sim.tNew_[0])));
	matplotlibcpp::title(fmt::format("vshift t = {:.4g}", vshift * sim.tNew_[0]));
	matplotlibcpp::save("./radiation_pulse_Erad.pdf");

	// Plot velocity

	matplotlibcpp::clf();
	args["label"] = "velocity";
	args["color"] = "C0";
	args["linestyle"] = "-";
	matplotlibcpp::plot(xs, mtm, args);
	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("x momentum (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("vshift t = {:.4g}", vshift * sim.tNew_[0]));
	matplotlibcpp::save("./radiation_pulse_velocity.pdf");
#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
