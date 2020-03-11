//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_marshak.hpp"

void testproblem_radiation_marshak()
{
	// For this problem, you must do reconstruction in the reduced
	// flux, *not* the flux. Otherwise, F exceeds cE at sharp temperature
	// gradients.

	// Problem parameters

	const int max_timesteps = 10001;
	const double CFL_number = 0.1;
	const int nx = 1200;

	const double max_dtau = 1e-3;	  // dimensionless time
	const double initial_dtau = 1e-7; // dimensionless time
	const double max_tau = 0.1;	  // dimensionless time
	const double Lz = 10.0;		  // dimensionless length

	// Su & Olson (1997) parameters
	const double eps_SuOlson = 1.0;
	const double z0 = 0.5;	  // dimensionless length scale
	const double tau0 = 10.0; // dimensionless time scale

	const double rho = 1.0; // g cm^-3 (matter density)
	const double kappa = 1.0;
	const double T_hohlraum = 3.481334e6; // K [= 300 eV]

	// Problem initialization

	RadSystem rad_system(RadSystem::Nx = nx, RadSystem::Lx = 1.0,
			     RadSystem::CFL = CFL_number);

	auto nghost = rad_system.nghost();

	const double a_rad = rad_system.radiation_constant();
	const double c = rad_system.c_light();
	const double kelvin_to_eV = 8.617385e-5;

	const double chi = rho * kappa; // cm^-1 (total matter opacity)
	const double x0 = z0 / chi;	// cm
	const double Lx = Lz / chi;	// cm
	const double t0 = tau0 / (eps_SuOlson * c * chi);		  // s
	const double max_time = max_tau / (eps_SuOlson * c * chi);	  // s
	const double max_dt = max_dtau / (eps_SuOlson * c * chi);	  // s
	const double initial_dt = initial_dtau / (eps_SuOlson * c * chi); // s

	rad_system.set_lx(Lx);

	const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;
	const double Q = 1.0 / (2.0 * z0); // do NOT change this
	const double S = Q * (a_rad * std::pow(T_hohlraum, 4)); // erg cm^{-3}
	const auto initial_Egas =
	    1e-10 * (rho * alpha_SuOlson) * std::pow(T_hohlraum, 4);
	const auto initial_Erad = 1e-10 * (a_rad * std::pow(T_hohlraum, 4));

	rad_system.Erad_floor_ = initial_Erad;

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x =
		    Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		rad_system.set_radEnergy(i) = initial_Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = initial_Egas;
		rad_system.set_staticGasDensity(i) = rho;
		if (x < x0) {
			rad_system.set_radEnergySource(i) = S;
		}
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= max_time) {
			break;
		}

		if (rad_system.time() >= t0) {
			for (int i = nghost; i < nx + nghost; ++i) {
				rad_system.set_radEnergySource(i) = 0.0;
			}
		}

		const double this_dtMax = ((j == 0) ? initial_dt : max_dt);
		rad_system.AdvanceTimestepRK2(this_dtMax);
		// rad_system.AdvanceTimestepSDC(this_dtMax);

		std::cout << "Timestep " << j << "; t = " << rad_system.time()
			  << "; dt = " << rad_system.dt() << "\n";

		const auto Erad = rad_system.ComputeRadEnergy();
		const auto Egas = rad_system.ComputeGasEnergy();
		const auto Etot = Erad + Egas;
		const auto Ediff = std::fabs(Etot - Etot0);

		std::cout << "radiation energy = " << Erad << "\n";
		std::cout << "gas energy = " << Egas << "\n";
		std::cout << "Total energy = " << Etot << "\n";
		std::cout << "(Energy nonconservation = " << Ediff << ")\n";
		std::cout << "\n";
	}

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;

		const auto Erad_t = rad_system.radEnergy(i + nghost);
		Erad.at(i) = Erad_t / (a_rad * std::pow(T_hohlraum, 4));
		Trad.at(i) = kelvin_to_eV * std::pow(Erad_t / a_rad, 1. / 4.);

		const auto Egas_t = rad_system.gasEnergy(i + nghost);
		Egas.at(i) = Egas_t / (a_rad * std::pow(T_hohlraum, 4));
		Tgas.at(i) =
		    kelvin_to_eV *
		    std::pow(Egas_t / (rho * alpha_SuOlson), (1. / 4.));
	}

	matplotlibcpp::clf();
	matplotlibcpp::xlim(0.2, 8.0); // cm

	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature";
	matplotlibcpp::plot(xs, Trad, Trad_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (eV)");
	matplotlibcpp::title(fmt::format("time t = {:.4g}", rad_system.time()));
	matplotlibcpp::save("./marshak_wave_temperature.pdf");

	matplotlibcpp::clf();
	matplotlibcpp::xlim(0.2, 8.0); // cm

	std::map<std::string, std::string> Erad_args;
	Erad_args["label"] = "radiation energy density";
	matplotlibcpp::plot(xs, Erad, Erad_args);

	std::map<std::string, std::string> Egas_args;
	Egas_args["label"] = "gas energy density";
	matplotlibcpp::plot(xs, Egas, Egas_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("energy density (dimensionless)");
	matplotlibcpp::title(fmt::format(
	    "time ct = {:.4g}", rad_system.time() * (eps_SuOlson * c * chi)));
	matplotlibcpp::save("./marshak_wave.pdf");

	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::ylim(1e-3, 3.0);
	matplotlibcpp::save("./marshak_wave_loglog.pdf");

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}
