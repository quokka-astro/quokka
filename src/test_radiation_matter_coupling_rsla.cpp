//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_matter_coupling.cpp
/// \brief Defines a test problem for radiation-matter coupling.
///

#include "test_radiation_matter_coupling_rsla.hpp"

auto main(int argc, char** argv) -> int
{
	// Initialization

	amrex::Initialize(argc, argv);

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_matter_coupling();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct CouplingProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

const double c = 2.99792458e10;	 // cgs
const double c_hat = 0.1*c;

// Su & Olson (1997) test problem
const double eps_SuOlson = 1.0;
const double a_rad = 7.5646e-15; // cgs
const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

template <>
auto RadSystem<CouplingProblem>::ComputeTgasFromEgas(const double rho,
						     const double Egas)
    -> double
{
	return std::pow(4.0 * Egas / alpha_SuOlson, 1. / 4.);
}

template <>
auto RadSystem<CouplingProblem>::ComputeEgasFromTgas(const double rho,
						     const double Tgas)
    -> double
{
	return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
}

template <>
auto RadSystem<CouplingProblem>::ComputeEgasTempDerivative(const double rho,
							   const double Tgas)
    -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	// However, for this problem, this must be of the form \alpha T^3
	// in order to obtain an exact solution to the problem.
	// The input parameter is the *temperature*, not Egas itself.

	return alpha_SuOlson * std::pow(Tgas, 3);
}

auto testproblem_radiation_matter_coupling() -> int
{
	// Problem parameters

	const int nx = 4;
	const double Lx = 1e5;
	const double CFL_number = 1.0;
	// const double constant_dt = 1.0e-11; // s
	// const double max_time = 1.0e-7;	    // s
	const double constant_dt = 1.0e-8; // s
	const double max_time = 1.0e-2;	   // s
	const int max_timesteps = 1e6;

	const double Erad = 1.0e12; // erg cm^-3
	const double Egas = 1.0e2;  // erg cm^-3
	const double rho = 1.0e-7;  // g cm^-3
	const auto initial_Trad = std::pow(Erad / a_rad, 1. / 4.);

	// Problem initialization

	RadSystem<CouplingProblem> rad_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	rad_system.c_light_ = c;
	rad_system.c_hat_ = c_hat;

	const double initial_Tgas = rad_system.ComputeTgasFromEgas(rho, Egas);
	const auto kappa =
	    RadSystem<CouplingProblem>::ComputeOpacity(rho, initial_Tgas);

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = Egas;
		rad_system.set_staticGasDensity(i) = rho;
	}

	const auto initial_Erad = rad_system.ComputeRadEnergy();
	const auto initial_Egas = rad_system.ComputeGasEnergy();
	const auto initial_Etot = initial_Erad + initial_Egas;

	amrex::Print() << "Initial radiation temperature = " << initial_Trad << "\n";
	amrex::Print() << "Initial gas temperature = " << initial_Tgas << "\n";

#if 0
	const double c_v =
	    rad_system.boltzmann_constant_ /
	    (rad_system.mean_molecular_mass_ * (rad_system.gamma_ - 1.0));
	const auto initial_Tgas = Egas / (rho * c_v);

	amrex::Print() << "Volumetric heat capacity c_v = " << rho * c_v << "\n";
#endif

	// Main time loop

	std::vector<double> t;
	std::vector<double> Trad;
	std::vector<double> Tgas;
	std::vector<double> Egas_v;

	for (int j = 0; j < max_timesteps; ++j) {

		const auto current_Erad = rad_system.ComputeRadEnergy();
		const auto current_Egas = rad_system.ComputeGasEnergy();
		const auto current_Etot = current_Erad + current_Egas;
		const auto Ediff = std::fabs(current_Etot - initial_Etot);

		if (rad_system.time() >= max_time) {
			amrex::Print() << "Timestep " << j
				  << "; t = " << rad_system.time() << "\n";
			amrex::Print() << "radiation energy = " << current_Erad
				  << "\n";
			amrex::Print() << "gas energy = " << current_Egas << "\n";
			amrex::Print() << "Total energy = " << current_Etot << "\n";
			amrex::Print() << "(Energy nonconservation = " << Ediff
				  << ")\n";
			amrex::Print() << "\n";

			break;
		}

		rad_system.AdvanceTimestep(constant_dt);

		t.push_back(rad_system.time());
		Trad.push_back(std::pow(rad_system.radEnergy(0 + nghost) /
					    rad_system.radiation_constant(),
					1. / 4.));

		auto Egas_i = rad_system.gasEnergy(0 + nghost);
		Tgas.push_back(rad_system.ComputeTgasFromEgas(rho, Egas_i));
		Egas_v.push_back(rad_system.gasEnergy(0 + nghost));
	}

	// Solve for asymptotically-exact solution (Gonzalez et al. 2007)
	// 	[original source for this test: Turner & Stone (2001)]
	const int nmax = t.size();
	//std::vector<double> t_exact(nmax);
	std::vector<double> Tgas_rsla_exact(nmax);
	std::vector<double> Tgas_exact(nmax);

	for (int n = 0; n < nmax; ++n) {
#if 0
		// Solution to Turner and Stone (2001) problem
		const double T_r = initial_Trad;
		const double T0 = initial_Tgas;

		const double T_gas =
		    (static_cast<double>(n + 1) / static_cast<double>(nmax)) *
			(T_r - T0) +
		    T0;

		const double term1 =
		    std::atan(T0 / T_r) - std::atan(T_gas / T_r);
		const double term2 = -std::log(T_r - T0) + std::log(T_r + T0) +
				     std::log(T_r - T_gas) -
				     std::log(T_r + T_gas);

		const double norm_fac =
		    (-kappa * rad_system.c_light_ *
		     rad_system.radiation_constant() / c_v) *
		    std::pow(T_r, 3);

		const double time_t = (0.5 * term1 + 0.25 * term2) / norm_fac;
#endif

		const double time_t = t.at(n);
		const double T0_4 = std::pow(initial_Tgas, 4);

		const double E0_rsla = ((c/c_hat)*Erad + Egas) / (a_rad + (c_hat/c) * alpha_SuOlson / 4.0);
		const double E0		 = (Erad + Egas) / (a_rad + alpha_SuOlson / 4.0);

		const double T4_rsla =
		    (T0_4 - (c_hat/c)*E0_rsla) * std::exp( -(4. / alpha_SuOlson) *
					   (a_rad + (c_hat/c) * alpha_SuOlson / 4.0) *
					   kappa * rho * c * time_t ) + (c_hat/c)*E0_rsla;

		const double T4 =
		    (T0_4 - E0) * std::exp( -(4. / alpha_SuOlson) *
					   (a_rad + alpha_SuOlson / 4.0) *
					   kappa * rho * c * time_t ) + E0;

		const double T_gas_rsla = std::pow(T4_rsla, 1. / 4.);
		const double T_gas = std::pow(T4, 1. / 4.);

		Tgas_rsla_exact.at(n) = T_gas_rsla;
		Tgas_exact.at(n) = T_gas;
	}

	// interpolate exact solution onto output timesteps
	//std::vector<double> Tgas_exact_interp(t.size());
	//interpolate_arrays(t.data(), Tgas_exact_interp.data(), t.size(),
	//		   t_exact.data(), Tgas_exact.data(), t_exact.size());

	// compute L1 error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < t.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Tgas_rsla_exact[i]);
		sol_norm += std::abs(Tgas_rsla_exact[i]);
	}
	const double rel_error = err_norm / sol_norm;
	const double error_tol = 1e-5;
	amrex::Print() << "relative L1 error norm = " << rel_error << std::endl;

	matplotlibcpp::clf();
	matplotlibcpp::yscale("log");
	matplotlibcpp::xscale("log");
	matplotlibcpp::ylim(0.1 * std::min(Tgas.front(), Trad.front()),
			    10.0 * std::max(Trad.back(), Tgas.back()));

	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature (numerical, RSLA)";
	matplotlibcpp::plot(t, Trad, Trad_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature (numerical, RSLA)";
	matplotlibcpp::plot(t, Tgas, Tgas_args);

	std::map<std::string, std::string> exactsolrsla_args;
	exactsolrsla_args["label"] = "gas temperature (exact, RSLA)";
	exactsolrsla_args["linestyle"] = "--";
	exactsolrsla_args["color"] = "black";
	matplotlibcpp::plot(t, Tgas_rsla_exact, exactsolrsla_args);

	std::map<std::string, std::string> exactsol_args;
	exactsol_args["label"] = "gas temperature (exact)";
	exactsol_args["linestyle"] = ":";
	exactsol_args["color"] = "black";
	matplotlibcpp::plot(t, Tgas_exact, exactsol_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("time t (s)");
	matplotlibcpp::ylabel("temperature T (K)");
	matplotlibcpp::title(fmt::format("dt = {:.4g}\nt = {:.4g}", constant_dt,
					 rad_system.time()));
	matplotlibcpp::save(fmt::format("./radcoupling_rsla.pdf"));

	matplotlibcpp::clf();

	std::vector<double> frac_err(t.size());
	for (int i = 0; i < t.size(); ++i) {
		frac_err.at(i) = Tgas_rsla_exact.at(i) / Tgas.at(i) - 1.0;
	}
	matplotlibcpp::plot(t, frac_err);
	matplotlibcpp::xlabel("time t (s)");
	matplotlibcpp::ylabel("fractional error in material temperature");
	matplotlibcpp::save(fmt::format("./radcoupling_rsla_fractional_error.pdf"));

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;

	int status = 0;
	if (rel_error > error_tol) {
		status = 1;
	}
	return status;
}
