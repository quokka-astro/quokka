//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_pulse.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_pulse.hpp"

auto main(int argc, char** argv) -> int
{
	// Initialization

	amrex::Initialize(argc, argv);

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_pulse();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

const double kappa0 = 1.0e5;   // cm^-1 (opacity at temperature T0)
const double T0 = 1.0;		   // K (temperature)
const double rho = 1.0;	       // g cm^-3 (matter density)
const double a_rad = 4.0e-10;  // radiation constant == 4sigma_SB/c (dimensionless)
const double c = 1.0e8;	   // speed of light (dimensionless)
const double chat = 1.0e7;
const double Erad_floor = a_rad * std::pow(1.0e-5, 4);

const double Lx = 1.0;	  // dimensionless length
const double x0 = Lx / 2.0;
const double initial_time = 1.0e-8;

auto compute_exact_Trad(const double x, const double t) -> double
{
	// compute exact solution for Gaussian radiation pulse
	// 		assuming diffusion approximation
	const double sigma = 0.025;
	const double D = 4.0 * c * a_rad * std::pow(T0, 3) / (3.0 * kappa0);
	const double width_sq = (sigma*sigma + D*t);
	const double normfac = 1.0 / (2.0 * std::sqrt( M_PI * width_sq ));
	return 0.5 * normfac * std::exp( -(x*x) / (4.0*width_sq) );
}

template <> void RadSystem<PulseProblem>::FillGhostZones(array_t &cons)
{
	double t = time_ + initial_time;

	// x1 left side boundary
	for (int i = 0; i < nghost_; ++i) {
		const double Erad = Erad_floor_;
		const double Frad = 0.;
		
		cons(radEnergy_index, i) = Erad;
		cons(x1RadFlux_index, i) = Frad;
	}

	// x1 right side boundary
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		const double Erad = Erad_floor_;
		const double Frad = 0.;
		
		cons(radEnergy_index, i) = Erad;
		cons(x1RadFlux_index, i) = Frad;
	}
}

template <>
auto RadSystem<PulseProblem>::ComputeOpacity(const double rho,
					       const double Tgas) -> double
{
	return (kappa0 / rho) * std::max( std::pow(Tgas/T0, 3), 1.0 );
}

template <>
auto RadSystem<PulseProblem>::ComputeOpacityTempDerivative(const double rho,
							const double Tgas)
    -> double
{
	if (Tgas > 1.0) {
		return (kappa0 / rho) * (3.0/T0) * std::pow(Tgas/T0, 2);
	} else {
		return 0.;
	}
}

auto testproblem_radiation_pulse() -> int
{
	// This problem is a *linear* radiation diffusion problem, i.e.
	// parameters are chosen such that the radiation and gas temperatures
	// should be near equilibrium, and the opacity is chosen to go as
	// T^3, such that the radiation diffusion equation becomes linear in T.

	// This makes this problem a stringent test of the asymptotic-
	// preserving property of the computational method, since the
	// optical depth per cell at the peak of the temperature profile is
	// of order 10^5.

	// This problem cannot be run for a meaningful amount of time
	// compared to the diffusion time. (RSLA only makes the diffusion time
	// longer by a factor of c/c_hat and therefore does not help.)
	// However, this problem does test whether matter and radiation equilibrate
	// (as they should) and that numerical diffusion is not excessive.

	// Problem parameters

	const int max_timesteps = 3e5;
	const double CFL_number = 0.8;
	const int nx = 128;

	const double max_dt = 1e-3;	  	// dimensionless time
	const double max_time = 1.0e-4;	// dimensionless time

	// Problem initialization

	std::vector<double> T_initial(nx);
	std::vector<double> Erad_initial(nx);
	std::vector<double> Frad_initial(nx);

	for (int i = 0; i < nx; ++i) {
		// initialize initial temperature
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		const double Trad = compute_exact_Trad(x - x0, initial_time);
		const double Erad = a_rad * std::pow(Trad, 4);
		T_initial.at(i) = Trad;
		Erad_initial.at(i) = Erad;
	}

	RadSystem<PulseProblem> rad_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	rad_system.set_lx(Lx);
	rad_system.c_light_ = c;
	rad_system.c_hat_ = chat;
	rad_system.Erad_floor_ = Erad_floor;
	rad_system.boltzmann_constant_ = (2./3.);
	rad_system.mean_molecular_mass_ = 1.0;
	rad_system.gamma_ = 5./3.;

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = Erad_floor;
		rad_system.set_x1RadFlux(i) = 0.;
		rad_system.set_radEnergySource(i) = 0.0;

		rad_system.set_gasEnergy(i) = rad_system.ComputeEgasFromTgas(rho, T_initial.at(i - nghost));
		rad_system.set_staticGasDensity(i) = rho;
		rad_system.set_x1GasMomentum(i) = 0.0;
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	std::cout << "radiation constant (code units) = " << a_rad << "\n";
	std::cout << "c_light (code units) = " << c << "\n";
	std::cout << "Lx = " << Lx << "\n";
	std::cout << "max_dt = " << max_dt << "\n";
	std::cout << "initial time = " << initial_time << std::endl;
	std::cout << "initial gas energy = " << Egas0 << std::endl;
	std::cout << "initial radiation energy = " << Erad0 << "\n" << std::endl;

	// Main time loop
	int j;
	double dt_prev = NAN;
	for (j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= max_time) {
			break;
		}

		// Compute timestep
		const double dt_expand_fac = 1.2;
		const double computed_dt = rad_system.ComputeTimestep(max_dt);
		const double this_dt = std::min(computed_dt, dt_expand_fac*dt_prev);

		rad_system.AdvanceTimestepRK2(this_dt);
		dt_prev = this_dt;
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

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;

		const auto Erad_t = rad_system.radEnergy(i + nghost);
		const auto Trad_t = std::pow(Erad_t / a_rad, 1./4.);
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t;
		Egas.at(i) = rad_system.gasEnergy(i + nghost);
		Tgas.at(i) = rad_system.ComputeTgasFromEgas(rho, Egas.at(i));
	}

	// compute exact solution
	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Erad_exact;

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));

		auto Trad_val = compute_exact_Trad(x - x0, initial_time + rad_system.time());
		auto Erad_val = a_rad * std::pow(Trad_val, 4);

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

	const double error_tol = 0.005;
	const double rel_error = err_norm / sol_norm;
	std::cout << "Relative L1 error norm = " << rel_error << std::endl;

	// plot temperature
	matplotlibcpp::clf();

	std::map<std::string, std::string> Trad_args, Tgas_args, Tinit_args;
	Trad_args["label"] = "radiation temperature";
	Trad_args["linestyle"] = "-.";
	Tgas_args["label"] = "gas temperature";
	Tgas_args["linestyle"] = "--";
	Tinit_args["label"] = "initial temperature";
	Tinit_args["color"] = "grey";
	//Trad_exact_args["label"] = "radiation temperature (exact)";
	//Trad_exact_args["linestyle"] = ":";
	matplotlibcpp::plot(xs, T_initial, Tinit_args);
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	//matplotlibcpp::plot(xs, Trad_exact, Trad_exact_args);

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
