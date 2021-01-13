//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radhydro_shock.cpp
/// \brief Defines a test problem for a radiative shock.
///

#include "test_radhydro_shock.hpp"

auto main(int argc, char** argv) -> int
{
	// Initialization

	amrex::Initialize(argc, argv);

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radhydro_shock();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct ShockProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// parameters taken from Section 9.5 of Skinner et al. (2019)
// [The Astrophysical Journal Supplement Series, 241:7 (27pp), 2019 March]

constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;	 // cm s^-1
constexpr double k_B = 1.380658e-16; // erg K^-1
constexpr double m_H = 1.6726231e-24; // mass of hydrogen atom [g]

constexpr double P0 = 1.0e-4;	// equal to P_0 in dimensionless units
constexpr double sigma_a = 1.0e6;	// absorption cross section
constexpr double Mach0 = 3.0;
constexpr double c_s0 = 1.73e7; // adiabatic sound speed [cm s^-1]

constexpr double kappa = 577.0;	// "opacity" == rho*kappa [cm^-1] (!!)
constexpr double gamma_gas = (5./3.);
constexpr double mu = m_H; // mean molecular weight [grams]
constexpr double c_v = k_B / (mu * (gamma_gas - 1.0));	// specific heat [erg g^-1 K^-1]

constexpr double T0 = 2.18e6; // K
constexpr double rho0 = 5.69; // g cm^-3
constexpr double v0 = 5.19e7; // cm s^-1

constexpr double T1 = 7.98e6; // K
constexpr double rho1 = 17.1; // g cm^-3
constexpr double v1 = 1.73e7; // cm s^-1

constexpr double chat = 10.0*(v0 + c_s0); // reduced speed of light

const double Erad0 = a_rad * std::pow(T0, 4); // erg cm^-3
constexpr double Egas0 = rho0 * c_v * T0; // erg cm^-3
const double Erad1 = a_rad * std::pow(T1, 4); // erg cm^-3
constexpr double Egas1 = rho1 * c_v * T1; // erg cm^-3

template <>
auto RadSystem<ShockProblem>::ComputeOpacity(const double rho, const double Tgas)
    -> double
{
	return (kappa / rho);
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

	for (int i = 0; i < nghost_; ++i) {
		cons(density_index, i) = rho0;
		cons(x1Momentum_index, i) = (xmom_L < (rho0*v0)) ? xmom_L : (rho0*v0);
		cons(energy_index, i) = Egas0 + 0.5*rho0*(v0*v0);
	}

	// x1 right side boundary (shock)
	const double xmom_R = cons(x1Momentum_index, nghost_ + nx_ - 1);

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
	//amrex::Print() << "\tAdvanced radiation subsystem with " << Nsubsteps << " substeps.\n";
	assert(time_ == advance_to_time); // NOLINT
}

auto testproblem_radhydro_shock() -> int
{
	// Problem parameters

	const int max_timesteps = 2e4;
	const double CFL_number = 0.2;
	const int nx = 256;
	const double Lx = 0.01575; // cm

	const double initial_dtau = 1.0e-3;	  // dimensionless time
	const double max_dtau = 1.0e-3;		  // dimensionless time

	const double initial_dt = initial_dtau / c_s0;
	const double max_dt = max_dtau / c_s0;
	const double max_time = 9.08e-10; // s

	// Problem initialization

	RadSystem<ShockProblem> rad_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	//rad_system.set_c_light(c);
	rad_system.c_light_ = c;
	rad_system.c_hat_ = chat;
	rad_system.mean_molecular_mass_ = mu;
	rad_system.boltzmann_constant_ = k_B;
	rad_system.gamma_ = gamma_gas;

	HydroSystem<ShockProblem> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma_gas});

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));

		rad_system.set_radEnergySource(i) = 0.0;

		constexpr double shock_position = 0.0132 / 0.01575;
		if (x < (shock_position * Lx)) {
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
	const auto Etot0 = (c/chat)*Erad0 + Egas0;

	amrex::Print() << "radiation constant (code units) = " << a_rad << "\n";
	amrex::Print() << "c_light (code units) = " << c << "\n";
	amrex::Print() << "Lx = " << Lx << "\n";
	amrex::Print() << "initial_dt = " << initial_dt << "\n";
	amrex::Print() << "max_dt = " << max_dt << "\n";
	amrex::Print() << "max_time = " << max_time << "\n\n";

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

		//amrex::Print() << "[timestep " << j << "] ";
		//amrex::Print() << "t = " << hydro_system.time() << "\tdt = " << this_dt << std::endl;

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

		//amrex::Print() << std::endl;
	}

	amrex::Print() << "Timestep " << j << "; t = " << hydro_system.time()
		  << "; dt = " << hydro_system.dt() << "\n";

	const auto total_Erad = rad_system.ComputeRadEnergy();
	const auto total_Egas = hydro_system.ComputeEnergy();
	const auto total_E = (c/chat)*total_Erad + total_Egas;
	const auto Ediff = std::fabs(total_E - Etot0);

	amrex::Print() << "radiation energy = " << total_Erad << "\n";
	amrex::Print() << "gas energy = " << total_Egas << "\n";
	amrex::Print() << "Total energy = " << total_E << "\n";
	amrex::Print() << "(Energy nonconservation = " << Ediff << ")\n";
	amrex::Print() << "\n";

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
		xs.at(i) = x; // cm

		const auto Erad_t = rad_system.radEnergy(i + nghost);
		Erad.at(i) = Erad_t / a_rad;	// scale by P_0
		Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.) / T0; // dimensionless

		const auto Etot_t = rad_system.gasEnergy(i + nghost);
		const auto Frad = rad_system.x1RadFlux(i + nghost);
		const auto rho = rad_system.staticGasDensity(i + nghost);
		const auto x1GasMom = rad_system.x1GasMomentum(i + nghost);
		const auto Ekin = (x1GasMom*x1GasMom) / (2.0*rho);
		const auto Egas_t = Etot_t - Ekin;

		Egas.at(i) = Egas_t;
		Tgas.at(i) = rad_system.ComputeTgasFromEgas(rho, Egas_t) / T0; // dimensionless

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

	std::string filename = "../extern/LowrieEdwards/shock.txt";
	std::ifstream fstream(filename, std::ios::in);

	const double error_tol = 0.005;
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
			auto x_val = values.at(0); // cm
			auto Tmat_val = values.at(3); // dimensionless
			auto Trad_val = values.at(4); // dimensionless
			auto Frad_over_c_val = values.at(5);

			if ((x_val > 0.0) && (x_val < Lx)) {
				xs_exact.push_back(x_val);
				Tmat_exact.push_back(Tmat_val);
				Trad_exact.push_back(Trad_val);
				Frad_over_c_exact.push_back(Frad_over_c_val);
			}
		}

		// compute error norm

		std::vector<double> Trad_interp(xs_exact.size());
		amrex::Print() << "xs min/max = " << xs[0] << ", " << xs[xs.size()-1] << std::endl;
		amrex::Print() << "xs_exact min/max = " << xs_exact[0] << ", " << xs_exact[xs_exact.size()-1] << std::endl;

		interpolate_arrays(xs_exact.data(), Trad_interp.data(), xs_exact.size(),
			   xs.data(), Trad.data(), xs.size());

		double err_norm = 0.;
		double sol_norm = 0.;
		for (int i = 0; i < xs_exact.size(); ++i) {
			err_norm += std::abs(Trad_interp[i] - Trad_exact[i]);
			sol_norm += std::abs(Trad_exact[i]);
		}

		rel_error = err_norm / sol_norm;
		amrex::Print() << "Error norm = " << err_norm << std::endl;
		amrex::Print() << "Solution norm = " << sol_norm << std::endl;
		amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;
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

	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", hydro_system.time()));
	matplotlibcpp::save("./radshock_cgs_temperature.pdf");

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
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("mass density (g/cc)");
	matplotlibcpp::legend();
	matplotlibcpp::save("./radshock_cgs_gasdensity.pdf");

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;

	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}

	return status;
}
