//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_classical_marshak.hpp"

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_classical_marshak();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

struct SuOlsonProblem {}; // dummy type to allow compile-type polymorphism via template specialization

template <>
void RadSystem<SuOlsonProblem>::FillGhostZones(array_t &cons)
{
	// Su & Olson (1996) boundary conditions
	const double T_H = 1.0; // Hohlraum temperature
	const double E_inc = radiation_constant_ * std::pow(T_H, 4);
	const double F_inc = c_light_ * E_inc / 4.0;

	// x1 left side boundary (Marshak)
	for (int i = 0; i < nghost_; ++i) {
		cons(radEnergy_index, i) = E_inc;
		cons(x1RadFlux_index, i) = F_inc;
	}

	// x1 right side boundary (reflecting)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) = cons(
		    radEnergy_index, (nghost_ + nx_) - (i - nx_ - nghost_ + 1));
		cons(x1RadFlux_index, i) =
		    -1.0 * cons(x1RadFlux_index,
				(nghost_ + nx_) - (i - nx_ - nghost_ + 1));
	}
}

auto testproblem_radiation_classical_marshak() -> int
{
	// For this problem, you must do reconstruction in the reduced
	// flux, *not* the flux. Otherwise, F exceeds cE at sharp temperature
	// gradients.

	// Problem parameters

	const int max_timesteps = 2e5;
	const double CFL_number = 0.4;
	const int nx = 1500;

	const double initial_dtau = 1e-9; // dimensionless time
	const double max_dtau = 1e-2;	  // dimensionless time
	const double max_tau = 10.0;	  // dimensionless time
	const double Lz = 100.0;	  // dimensionless length

	// Su & Olson (1997) parameters
	const double eps_SuOlson = 1.0;

	const double rho = 1.0; // g cm^-3 (matter density)
	const double kappa = 1.0;
	const double T_hohlraum = 1.0; // dimensionless
	// const double T_hohlraum_scaled = 3.481334e6; // K [= 300 eV]

	// Problem initialization

	RadSystem<SuOlsonProblem> rad_system(
	    {.nx = nx, .lx = Lz, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(1.0);
	rad_system.set_c_light(1.0);

	auto nghost = rad_system.nghost();

	const double a_rad = rad_system.radiation_constant();
	const double c = rad_system.c_light();

	std::cout << "radiation constant (code units) = " << a_rad << "\n";
	std::cout << "c_light (code units) = " << c << "\n";

	// const double kelvin_to_eV = 8.617385e-5;

	const double chi = rho * kappa; // cm^-1 (total matter opacity)
	const double Lx = Lz / chi;	// cm
	const double max_time = max_tau / (eps_SuOlson * c * chi);	  // s
	const double max_dt = max_dtau / (eps_SuOlson * c * chi);	  // s
	const double initial_dt = initial_dtau / (eps_SuOlson * c * chi); // s

	std::cout << "Lx = " << Lx << "\n";
	std::cout << "initial_dt = " << initial_dt << "\n";
	std::cout << "max_dt = " << max_dt << "\n";

	rad_system.set_lx(Lx);

	const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

	auto ComputeTgasFromEgas = [=](const double Eint) {
		return std::pow(4.0 * Eint / alpha_SuOlson, 1. / 4.);
	};

	auto ComputeEgasFromTgas = [=](const double Tgas) {
		return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
	};

	const auto initial_Egas = 1e-10 * ComputeEgasFromTgas(T_hohlraum);
	const auto initial_Erad = 1e-10 * (a_rad * std::pow(T_hohlraum, 4));

	rad_system.Erad_floor_ = initial_Erad;

	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = initial_Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = initial_Egas;
		rad_system.set_staticGasDensity(i) = rho;
		rad_system.set_radEnergySource(i) = 0.0;
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {

		if (rad_system.time() >= max_time) {
			std::cout << "Timestep " << j
				  << "; t = " << rad_system.time()
				  << "; dt = " << rad_system.dt() << "\n";

			const auto Erad = rad_system.ComputeRadEnergy();
			const auto Egas = rad_system.ComputeGasEnergy();
			const auto Etot = Erad + Egas;
			const auto Ediff = std::fabs(Etot - Etot0);

			std::cout << "radiation energy = " << Erad << "\n";
			std::cout << "gas energy = " << Egas << "\n";
			std::cout << "Total energy = " << Etot << "\n";
			std::cout << "(Energy nonconservation = " << Ediff
				  << ")\n";
			std::cout << "\n";

			break;
		}

		const double this_dtMax = ((j == 0) ? initial_dt : max_dt);
		rad_system.AdvanceTimestepRK2(this_dtMax);
	}

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = std::sqrt(3.0) * x;

		const auto Erad_t =
		    rad_system.radEnergy(i + nghost) / std::sqrt(3.);
		Erad.at(i) = Erad_t;
		Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);

		const auto Egas_t =
		    rad_system.gasEnergy(i + nghost) / std::sqrt(3.);
		Egas.at(i) = Egas_t;
		Tgas.at(i) = ComputeTgasFromEgas(Egas_t);
	}

	// read in exact solution

	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Tmat_exact;

	std::string filename = "../../extern/SuOlson/100pt_tau10p0.dat";
	std::ifstream fstream(filename, std::ios::in);
	assert( fstream.is_open() );

	std::string header;
	std::getline(fstream, header);

	for(std::string line; std::getline(fstream, line); ) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value; iss >> value; ) {
			values.push_back(value);
		}
		auto x_val = std::sqrt(3.0)*values.at(1);
		auto Trad_val = values.at(4);
		auto Tmat_val = values.at(5);

		xs_exact.push_back(x_val);
		Trad_exact.push_back(Trad_val);
		Tmat_exact.push_back(Tmat_val);
	}

	// compute error norm

	std::vector<double> Trad_interp(xs_exact.size());
	interpolate_arrays(xs_exact.data(), Trad_interp.data(), xs_exact.size(),
						xs.data(), Trad.data(), xs.size());

	double err_norm = 0.;
	double sol_norm = 0.;
	for(int i = 0; i < xs_exact.size(); ++i) {
		err_norm += std::pow( Trad_interp[i] - Trad_exact[i], 2 );
		sol_norm += std::pow( Trad_exact[i], 2 );
	}

	const double error_tol = 0.003;
	const double rel_error = err_norm / sol_norm;
	std::cout << "Relative L2 error norm = " << rel_error << std::endl;

	// plot results

	// temperature
	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature";
	matplotlibcpp::plot(xs, Trad, Trad_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::xlim(0.4, 100.); // dimensionless
	matplotlibcpp::ylim(0.0, 1.0);	// dimensionless
	matplotlibcpp::xscale("log");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", rad_system.time()));
	matplotlibcpp::save("./classical_marshak_wave_temperature.pdf");

	// energy density
	matplotlibcpp::clf();

	std::map<std::string, std::string> Erad_args;
	Erad_args["label"] = "Numerical solution";
	Erad_args["color"] = "black";
	matplotlibcpp::plot(xs, Erad, Erad_args);

	std::map<std::string, std::string> Egas_args;
	Egas_args["label"] = "gas energy density";
	Egas_args["color"] = "red";
	matplotlibcpp::plot(xs, Egas, Egas_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::xlim(0.4, 100.0); // cm
	matplotlibcpp::ylim(0.0, 1.0);
	matplotlibcpp::xscale("log");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format(
	    "time ct = {:.4g}", rad_system.time() * (eps_SuOlson * c * chi)));
	matplotlibcpp::save("./classical_marshak_wave.pdf");

	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::xlim(0.4, 100.0); // cm
	matplotlibcpp::ylim(1e-8, 1.3);
	matplotlibcpp::save("./classical_marshak_wave_loglog.pdf");

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
