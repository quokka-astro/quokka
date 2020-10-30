//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radhydro_shock.cpp
/// \brief Defines a test problem for a radiative shock.
///

#include "test_radhydro_shock.hpp"

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radhydro_shock();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

struct ShockProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

const double a_rad = 1.0e-4;	// equal to P_0 in dimensionless units
const double sigma_a = 1.0e6;	// absorption cross section
const double Mach0 = 3.0;

const double c_s0 = 1.0; // adiabatic sound speed
const double c = sqrt(3.0*sigma_a) * c_s0; // dimensionless speed of light
//const double c = 1.0;
//const double c_s0 = 1.0 / sqrt(3.0*sigma_a);

//const double c = 100.0 * (Mach0 + c_s0); // old parameter value

const double kappa = sigma_a * (c_s0 / c);	// specific opacity
const double gamma_gas = (5./3.);
const double mu = gamma_gas; // mean molecular weight (required s.t. c_s0 == 1)
//const double k_B = 1.0; // dimensionless Boltzmann constant
const double k_B = std::pow(c_s0, 2);	// required to make temperature and sound speed consistent
const double c_v = k_B / (mu * (gamma_gas - 1.0));	// specific heat

const double T0 = 1.0;
const double rho0 = 1.0;
const double v0 = (Mach0 * c_s0);

const double T1 = 3.661912665809719;
const double rho1 = 3.0021676971081166;
const double v1 = (Mach0 * c_s0) * (rho0 / rho1);

const double Erad0 = a_rad * std::pow(T0, 4);
const double Egas0 = rho0 * c_v * T0;
const double Erad1 = a_rad * std::pow(T1, 4);
const double Egas1 = rho1 * c_v * T1;

template <>
auto RadSystem<ShockProblem>::ComputeOpacity(const double rho, const double Tgas)
    -> double
{
	return kappa;
}

//#if 0
template <>
auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double f) -> double
{
	return (1./3.);	// Eddington approximation
}
//#endif

template <> void RadSystem<ShockProblem>::FillGhostZones(array_t &cons)
{
	// x1 left side boundary (shock)
	for (int i = 0; i < nghost_; ++i) {
		cons(radEnergy_index, i) = Erad0;
		cons(x1RadFlux_index, i) = 0.;
	}

	// x1 right side boundary (shock)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) = Erad1;
		cons(x1RadFlux_index, i) = 0.;
	}
}

template <> void HydroSystem<ShockProblem>::FillGhostZones(array_t &cons)
{
	// x1 left side boundary (shock)
	const double xmom_L = cons(x1Momentum_index, nghost_);
	const double dens_L = cons(density_index, nghost_);
	const double vx_L = xmom_L / dens_L;

	for (int i = 0; i < nghost_; ++i) {
		cons(density_index, i) = rho0;
		cons(x1Momentum_index, i) = (xmom_L < (rho0*v0)) ? xmom_L : (rho0*v0);
		cons(energy_index, i) = Egas0 + 0.5*rho0*(v0*v0);
	}

	// x1 right side boundary (shock)
	const double xmom_R = cons(x1Momentum_index, nghost_ + nx_ - 1);
	const double dens_R = cons(density_index, nghost_ + nx_ - 1);
	const double vx_R = xmom_R / dens_R;

	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(density_index, i) = rho1;
		cons(x1Momentum_index, i) = (xmom_R > (rho1*v1)) ? xmom_R : (rho1*v1);
		cons(energy_index, i) = Egas1 + 0.5*rho1*(v1*v1);
	}
}

template <> void RadSystem<ShockProblem>::AdvanceTimestep(const double hydro_dt)
{
	// Subcycle to reach hydro_dt exactly in M substeps.
	auto advance_to_time = time_ + hydro_dt;
	int Nsubsteps = 0;
	while (time_ < advance_to_time) {
		auto dt_remaining = (advance_to_time - time_);
		auto dt_max = std::min(dtExpandFactor_ * dtPrev_, dt_remaining);
		auto dt_substep = ComputeTimestep(std::min(hydro_dt, dt_max));
		AdvanceTimestepRK2(dt_substep);
		++Nsubsteps;
	}
	std::cout << "\tAdvanced radiation subsystem with " << Nsubsteps << " substeps.\n";
	assert(time_ == advance_to_time); // NOLINT
}

auto testproblem_radhydro_shock() -> int
{
	// Problem parameters

	const int max_timesteps = 2e4;
	const double CFL_number = 0.8;
	const int nx = 512;
	const double Lx = 10.0 * (c/c_s0) / sigma_a;	// length

	const double initial_dtau = 1.0e-3;	// dimensionless time
	const double max_dtau = 1.0e-3;		// dimensionless time
	const double max_tau = 1.0 * (Lx/c_s0);		// dimensionless time

	const double max_time = max_tau / c_s0;
	const double max_dt = max_dtau / c_s0;
	const double initial_dt = initial_dtau / c_s0;

	// Problem initialization

	RadSystem<ShockProblem> rad_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	rad_system.set_c_light(c);
	rad_system.mean_molecular_mass_ = mu;
	rad_system.boltzmann_constant_ = k_B;
	rad_system.gamma_ = gamma_gas;

	HydroSystem<ShockProblem> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma_gas});

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));

		rad_system.set_radEnergySource(i) = 0.0;

		if (x < ((2./3.)*Lx)) {
			rad_system.set_radEnergy(i) = Erad0;
			rad_system.set_x1RadFlux(i) = 0.0;

			hydro_system.set_energy(i) = Egas0 + 0.5*rho0*(v0*v0);
			hydro_system.set_density(i) = rho0;
			hydro_system.set_x1Momentum(i) = rho0*v0;
		} else {
			rad_system.set_radEnergy(i) = Erad1;
			rad_system.set_x1RadFlux(i) = 0.0;

			hydro_system.set_energy(i) = Egas1 + 0.5*rho1*(v1*v1);
			hydro_system.set_density(i) = rho1;
			hydro_system.set_x1Momentum(i) = rho1*v1;
		
		}
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = hydro_system.ComputeEnergy();
	const auto Etot0 = Erad0 + Egas0;

	std::cout << "radiation constant (code units) = " << a_rad << "\n";
	std::cout << "c_light (code units) = " << c << "\n";
	std::cout << "Lx = " << Lx << "\n";
	std::cout << "initial_dt = " << initial_dt << "\n";
	std::cout << "max_dt = " << max_dt << "\n";
	std::cout << "max_time = " << max_time << "\n\n";

	// Main time loop
	int j;
	double dt_prev = std::numeric_limits<double>::max();
	const double dt_expand_factor = 1.1;
	for (j = 0; j < max_timesteps; ++j) {
		if (hydro_system.time() >= max_time) {
			break;
		}

		const double this_dtMax = ((j == 0) ? initial_dt : max_dt);

		// Fill ghost zones
		const auto all_cells = std::make_pair(0, hydro_system.dim1());
		hydro_system.FillGhostZones(hydro_system.consVar_);
		hydro_system.ConservedToPrimitive(hydro_system.consVar_,
						  all_cells);

		rad_system.FillGhostZones(rad_system.consVar_);
		rad_system.ConservedToPrimitive(rad_system.consVar_, all_cells);

		// Compute hydro timestep
		const double computed_dt = hydro_system.ComputeTimestep(this_dtMax);
		const double this_dt = std::min(computed_dt, dt_expand_factor*dt_prev);

		std::cout << "[timestep " << j << "] ";
		std::cout << "t = " << hydro_system.time() << "\tdt = " << this_dt << std::endl;

		// Advance hydro subsystem
		hydro_system.AdvanceTimestepRK2(this_dt);

		// Copy hydro vars into rad_system
		for (int i = nghost; i < (nx + nghost); ++i) {
			rad_system.set_staticGasDensity(i) = hydro_system.density(i);
			rad_system.set_x1GasMomentum(i) = hydro_system.x1Momentum(i);
			rad_system.set_gasEnergy(i) = hydro_system.energy(i);
		}

		// Advance radiation subsystem, subcycling if necessary
		rad_system.AdvanceTimestep(this_dt);

		// Copy updated hydro vars back into hydro_system
		for (int i = nghost; i < (nx + nghost); ++i) {
			hydro_system.set_x1Momentum(i) = rad_system.x1GasMomentum(i);
			hydro_system.set_energy(i) = rad_system.gasEnergy(i);
		}

		// Update previous timestep
		dt_prev = this_dt;

		std::cout << std::endl;
	}

	std::cout << "Timestep " << j << "; t = " << hydro_system.time()
		  << "; dt = " << hydro_system.dt() << "\n";

	const auto total_Erad = rad_system.ComputeRadEnergy();
	const auto total_Egas = hydro_system.ComputeEnergy();
	const auto total_E = total_Erad + total_Egas;
	const auto Ediff = std::fabs(total_E - Etot0);

	std::cout << "radiation energy = " << total_Erad << "\n";
	std::cout << "gas energy = " << total_Egas << "\n";
	std::cout << "Total energy = " << total_E << "\n";
	std::cout << "(Energy nonconservation = " << Ediff << ")\n";
	std::cout << "\n";

	// read output variables

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Frad_over_c(nx);
	std::vector<double> Egas(nx);
	std::vector<double> x1GasMomentum(nx);
	std::vector<double> x1RadFlux(nx);
	std::vector<double> gasDensity(nx);
	std::vector<double> gasVelocity(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;

		const auto Erad_t = rad_system.radEnergy(i + nghost);
		Erad.at(i) = Erad_t / a_rad;	// scale by P_0
		Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);

		const auto Etot_t = rad_system.gasEnergy(i + nghost);
		const auto Frad = rad_system.x1RadFlux(i + nghost);
		const auto rho = rad_system.staticGasDensity(i + nghost);
		const auto x1GasMom = rad_system.x1GasMomentum(i + nghost);
		const auto Ekin = (x1GasMom*x1GasMom) / (2.0*rho);
		const auto Egas_t = Etot_t - Ekin;

		Egas.at(i) = Egas_t;
		Tgas.at(i) = rad_system.ComputeTgasFromEgas(rho, Egas_t);

		x1GasMomentum.at(i) = x1GasMom;
		x1RadFlux.at(i) = Frad;
		Frad_over_c.at(i) = Frad;

		gasDensity.at(i) = rho;
		gasVelocity.at(i) = (x1GasMom / rho) / c_s0;
	}

	// read in exact solution

	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Tmat_exact;
	std::vector<double> Frad_over_c_exact;

	std::string filename = "../../extern/LowrieEdwards/shock.txt";
	std::ifstream fstream(filename, std::ios::in);

	const double error_tol = 0.003;
	double rel_error = NAN;
	if(fstream.is_open()) {

		std::string header;
		std::getline(fstream, header);

		for (std::string line; std::getline(fstream, line);) {
			std::istringstream iss(line);
			std::vector<double> values;

			for (double value; iss >> value;) {
				values.push_back(value);
			}
			auto x_val = values.at(0);
			auto Tmat_val = values.at(3);
			auto Trad_val = values.at(4);
			auto Frad_over_c_val = values.at(5);

			if ((x_val > 0.0) && (x_val < Lx)) {
				xs_exact.push_back(x_val);
				Tmat_exact.push_back(Tmat_val);
				Trad_exact.push_back(Trad_val);
				Frad_over_c_exact.push_back(Frad_over_c_val);
			}
			//std::cout << "solution " << x_val << "\t" << Tmat_val << "\t" << Trad_val << std::endl;
		}

		// compute error norm

		std::vector<double> Trad_interp(xs_exact.size());
		std::cout << "xs min/max = " << xs[0] << ", " << xs[xs.size()-1] << std::endl;
		std::cout << "xs_exact min/max = " << xs_exact[0] << ", " << xs_exact[xs_exact.size()-1] << std::endl;

		interpolate_arrays(xs_exact.data(), Trad_interp.data(), xs_exact.size(),
			   xs.data(), Trad.data(), xs.size());

		double err_norm = 0.;
		double sol_norm = 0.;
		for (int i = 0; i < xs_exact.size(); ++i) {
			err_norm += std::abs(Trad_interp[i] - Trad_exact[i]);
			sol_norm += std::abs(Trad_exact[i]);
		}

		rel_error = err_norm / sol_norm;
		std::cout << "Error norm = " << err_norm << std::endl;
		std::cout << "Solution norm = " << sol_norm << std::endl;
		std::cout << "Relative L1 error norm = " << rel_error << std::endl;
	}

	// plot results

	// temperature
	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "Trad";
	Trad_args["color"] = "black";
	matplotlibcpp::plot(xs, Trad, Trad_args);

	if(fstream.is_open()) {
		std::map<std::string, std::string> Trad_exact_args;
		Trad_exact_args["label"] = "Trad (diffusion ODE)";
		Trad_exact_args["color"] = "black";
		Trad_exact_args["linestyle"] = "dashed";
		matplotlibcpp::plot(xs_exact, Trad_exact, Trad_exact_args);
	}

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "Tmat";
	Tgas_args["color"] = "red";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);

	if(fstream.is_open()) {
		std::map<std::string, std::string> Tgas_exact_args;
		Tgas_exact_args["label"] = "Tmat (diffusion ODE)";
		Tgas_exact_args["color"] = "red";
		Tgas_exact_args["linestyle"] = "dashed";
		matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);
	}

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", hydro_system.time()));
	matplotlibcpp::save("./radshock_temperature.pdf");

	// radiation flux
	std::map<std::string, std::string> gasmom_args, radmom_args, Frad_exact_args;
	//gasmom_args["label"] = "gas momentum density";
	radmom_args["label"] = "Frad";
	Frad_exact_args["label"] = "Frad (diffusion ODE)";

	matplotlibcpp::clf();
	//matplotlibcpp::plot(xs, x1GasMomentum, gasmom_args);
	matplotlibcpp::plot(xs, Frad_over_c, radmom_args);
	if(fstream.is_open()) {
		matplotlibcpp::plot(xs_exact, Frad_over_c_exact, Frad_exact_args);
	}
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("radiation flux (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::save("./radshock_flux.pdf");

	// gas density
	std::map<std::string, std::string> gasdens_args, gasvx_args;
	gasdens_args["label"] = "gas density";
	gasdens_args["color"] = "black";
	gasvx_args["label"] = "gas velocity";
	gasvx_args["color"] = "blue";
	gasvx_args["linestyle"] = "dashed";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, gasDensity, gasdens_args);
	matplotlibcpp::plot(xs, gasVelocity, gasvx_args);
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("mass density (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::save("./radshock_gasdensity.pdf");

	// energy density
	matplotlibcpp::clf();

	std::map<std::string, std::string> Erad_args;
	Erad_args["label"] = "radiation energy density";
	Erad_args["color"] = "black";
	matplotlibcpp::plot(xs, Erad, Erad_args);

	std::map<std::string, std::string> Egas_args;
	Egas_args["label"] = "gas energy density";
	Egas_args["color"] = "red";
	matplotlibcpp::plot(xs, Egas, Egas_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(
	    fmt::format("time t = {:.4g}", hydro_system.time()));
	matplotlibcpp::save("./radshock_energy.pdf");

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}

	return status;
}
