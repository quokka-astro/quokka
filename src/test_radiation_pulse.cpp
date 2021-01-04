//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_pulse.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_pulse.hpp"

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_pulse();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

const double kappa = 1.0e4;
const double rho = 1.0;	       // g cm^-3 (matter density)
const double a_rad = 1.0e-5;
const double c = 1.0;
const double Erad_floor = 1e-10;

const double Lx = 1.0;	  // dimensionless length
const double x0 = Lx / 2.0;
const double initial_time = 1.0;

auto compute_exact_erad(const double x, const double t) -> double
{
	// compute exact solution for Gaussian radiation pulse
	// 		assuming diffusion approximation
	const double sigma = 0.025;
	const double D = c / (3.0*kappa);
	const double width_sq = (sigma*sigma + D*t);
	const double normfac = 1.0 / (2.0 * std::sqrt( M_PI * width_sq ));
	return normfac * std::exp( -(x*x) / (4.0*width_sq) );
}

auto compute_exact_flux(const double x, const double t) -> double
{
	// compute exact solution for Gaussian radiation pulse
	// 		assuming diffusion approximation
	const double sigma = 0.025;
	const double D = c / (3.0*kappa);
	const double width_sq = (sigma*sigma + D*t);
	const double normfac = 1.0 / (2.0 * std::sqrt( M_PI * width_sq ));
	return D * normfac * std::exp( -(x*x) / (4.0*width_sq) ) * x / (2.0*width_sq);
}

template <> void RadSystem<PulseProblem>::FillGhostZones(array_t &cons)
{
	double t = time_ + initial_time;

	// x1 left side boundary
	for (int i = 0; i < nghost_; ++i) {
		const double x = lx_ * ((i + 0.5) / static_cast<double>(nx_));
		const double Erad = compute_exact_erad(x - x0, t);
		const double Frad = std::min(compute_exact_flux(x - x0, t), -c*Erad);
		
		cons(radEnergy_index, i) = Erad;
		cons(x1RadFlux_index, i) = Frad;
	}

	// x1 right side boundary
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		const double x = lx_ * ((i + 0.5) / static_cast<double>(nx_));
		const double Erad = compute_exact_erad(x - x0, t);
		const double Frad = std::max(compute_exact_flux(x - x0, t), c*Erad);
		
		cons(radEnergy_index, i) = Erad;
		cons(x1RadFlux_index, i) = Frad;
	}
}

template <>
auto RadSystem<PulseProblem>::ComputeOpacity(const double rho,
					       const double Tgas) -> double
{
	return (kappa / rho);
}

auto testproblem_radiation_pulse() -> int
{
	// Problem parameters

	const int max_timesteps = 1e5;
	const double CFL_number = 0.8;
	const int nx = 100;

	const double initial_dt = 1e-5; // dimensionless time
	const double max_dt = 1e-4;	  // dimensionless time
	const double max_time = 10.0;	  // dimensionless time

	// Problem initialization

	std::vector<double> T_eq(nx);
	std::vector<double> Erad_initial(nx);
	std::vector<double> Frad_initial(nx);

	for (int i = 0; i < nx; ++i) {
		// initialize initial temperature
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		const double Erad = compute_exact_erad(x - x0, initial_time);
		const double Frad = compute_exact_flux(x - x0, initial_time);
		const double Trad = std::pow(Erad / a_rad, 1./4.);
		Erad_initial.at(i) = Erad;
		Frad_initial.at(i) = Frad;
		T_eq.at(i) = Trad;
	}

	RadSystem<PulseProblem> rad_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	rad_system.set_c_light(c);
	rad_system.set_lx(Lx);
	rad_system.Erad_floor_ = Erad_floor;
	rad_system.boltzmann_constant_ = (2./3.)*1.0e6;
	rad_system.mean_molecular_mass_ = 1.0;
	rad_system.gamma_ = 5./3.;

	const auto c_v = (rad_system.boltzmann_constant_ / rad_system.mean_molecular_mass_) /
			 (rad_system.gamma_ - 1.);

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = Erad_initial.at(i - nghost);
		rad_system.set_x1RadFlux(i) = Frad_initial.at(i - nghost);
		rad_system.set_radEnergySource(i) = 0.0;

		rad_system.set_gasEnergy(i) = rad_system.ComputeEgasFromTgas(rho, T_eq.at(i - nghost));
		rad_system.set_staticGasDensity(i) = rho;
		rad_system.set_x1GasMomentum(i) = 0.0;
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	std::cout << "radiation constant (code units) = " << a_rad << "\n";
	std::cout << "c_light (code units) = " << c << "\n";
	std::cout << "Lx = " << Lx << "\n";
	std::cout << "initial_dt = " << initial_dt << "\n";
	std::cout << "max_dt = " << max_dt << "\n";
	std::cout << "initial time = " << initial_time << std::endl;

	// Main time loop
	int j;
	for (j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= max_time) {
			break;
		}

		const double this_dtMax = ((j == 0) ? initial_dt : max_dt);
		rad_system.AdvanceTimestepRK2(this_dtMax);
	}

	std::cout << "Timestep " << j << "; t = " << rad_system.time()
		  << "; dt = " << rad_system.dt() << "\n";

	const auto Erad_tot = rad_system.ComputeRadEnergy();
	const auto Egas_tot = rad_system.ComputeGasEnergy();
	const auto Etot = Erad_tot + Egas_tot;
	const auto Ediff = std::fabs(Etot - Etot0);

	std::cout << "radiation energy = " << Erad_tot << "\n";
	std::cout << "gas energy = " << Egas_tot << "\n";
	std::cout << "Total energy = " << Etot << "\n";
	std::cout << "(Energy nonconservation = " << Ediff << ")\n";
	std::cout << "\n";


	// read out results

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> x1RadFlux(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;

		const auto Erad_t = rad_system.radEnergy(i + nghost);
		const auto Trad_t = std::pow(Erad_t / a_rad, 1./4.);
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t;
		Egas.at(i) = a_rad * rad_system.gasEnergy(i + nghost);
		Tgas.at(i) = rad_system.gasEnergy(i + nghost) / (rho * c_v);
		x1RadFlux.at(i) = rad_system.x1RadFlux(i + nghost);
	}

	// compute exact solution
	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Erad_exact;

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));

		auto Erad_val = compute_exact_erad(x - x0, initial_time + rad_system.time());
		auto Trad_val = std::pow(Erad_val / a_rad, 1./4.);

		xs_exact.push_back(x);
		Trad_exact.push_back(Trad_val);
		Erad_exact.push_back(Erad_val);
	}

	// compute error norm

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < xs.size(); ++i) {
		err_norm += std::abs(Trad[i] - Trad_exact[i]);
		sol_norm += std::abs(Trad_exact[i]);
	}

	const double error_tol = 0.001;
	const double rel_error = err_norm / sol_norm;
	std::cout << "Relative L1 error norm = " << rel_error << std::endl;

	// plot energy density

	std::map<std::string, std::string> Erad_args, Erad_exact_args, Erad_initial_args;
	Erad_args["label"] = "Numerical solution";
	Erad_args["color"] = "red";
	Erad_args["linestyle"] = ":";
	
	Erad_exact_args["label"] = "Exact solution";
	Erad_exact_args["color"] = "blue";

	Erad_initial_args["label"] = "initial condition";
	Erad_initial_args["color"] = "black";
	Erad_initial_args["linestyle"] = "-.";

	matplotlibcpp::plot(xs, Erad, Erad_args);
	matplotlibcpp::plot(xs_exact, Erad_exact, Erad_exact_args);
	matplotlibcpp::plot(xs, Erad_initial, Erad_initial_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", initial_time + rad_system.time() * c));
	matplotlibcpp::save("./radiation_pulse.pdf");

	matplotlibcpp::clf();

	std::map<std::string, std::string> Trad_args, Tgas_args;
	Trad_args["label"] = "radiation temperature";
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", initial_time + rad_system.time() * c));
	matplotlibcpp::save("./radiation_pulse_temperature.pdf");

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	int status = 0;

	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}

	return status;
}
