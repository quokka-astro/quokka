//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_marshak_asymptotic.hpp"

auto main(int argc, char **argv) -> int
{
	// Initialization
	amrex::Initialize(argc, argv);

	int result = 0;

	{ // objects must be destroyed before amrex::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_marshak_cgs();

	} // destructors must be called before amrex::finalize()
	amrex::Finalize();

	return result;
}

struct SuOlsonProblemCgs {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double kappa = 300.0;		      // cm^-1 (opacity)
constexpr double rho = 2.0879373766122384;    // g cm^-3 (matter density)
constexpr double T_hohlraum = 1.1604448449e7; // K (1 keV)
// const double kelvin_to_eV = 8.617385e-5;
constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;  // cm s^-1

template <> void RadSystem<SuOlsonProblemCgs>::FillGhostZones(array_t &cons)
{
	// This boundary condition is only first-order accurate!
	// (Not quite good enough for simulating an optically-thick Marshak wave;
	//  the temperature becomes T_H in the first zone, rather than just at the face.)

	// Su & Olson (1996) boundary conditions
	const double T_H = T_hohlraum;
	const double E_inc = radiation_constant_ * std::pow(T_H, 4);
	const double c = c_light_;
	// const double F_inc = c * E_inc / 4.0;

	const double E_0 = cons(radEnergy_index, nghost_);
	const double E_1 = cons(radEnergy_index, nghost_ + 1);
	const double F_0 = cons(x1RadFlux_index, nghost_);
	const double F_1 = cons(x1RadFlux_index, nghost_ + 1);

	// use PPM stencil at interface to solve for F_rad in the ghost zones
	//const double F_bdry_PPM = 0.5 * c * E_inc - (7. / 12.) * (c * E_0 + 2.0 * F_0) +
	//			  (1. / 12.) * (c * E_1 + 2.0 * F_1);

	const double F_bdry_PLM = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * F_0);

	const double F_bdry = F_bdry_PLM;

	assert(std::abs(F_bdry / (c * E_inc)) < 1.0);

	// x1 left side boundary (Marshak)
	for (int i = 0; i < nghost_; ++i) {
		cons(radEnergy_index, i) = E_inc;
		cons(x1RadFlux_index, i) = F_bdry;
	}

	// x1 right side boundary (outflow)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) =
		    cons(radEnergy_index, (nghost_ + nx_) - (i - nx_ - nghost_ + 1));
		cons(x1RadFlux_index, i) =
		    cons(x1RadFlux_index, (nghost_ + nx_) - (i - nx_ - nghost_ + 1));
	}
}

template <>
auto RadSystem<SuOlsonProblemCgs>::ComputeOpacity(const double rho, const double Tgas) -> double
{
	auto sigma = kappa * std::pow(Tgas / T_hohlraum, -3); // cm^-1
	return (sigma / rho);				      // cm^2 g^-1
}

template <>
auto RadSystem<SuOlsonProblemCgs>::ComputeOpacityTempDerivative(const double rho, const double Tgas)
    -> double
{
	auto sigma_dT = (-3.0 * kappa / Tgas) * std::pow(Tgas / T_hohlraum, -3); // cm^-1
	return (sigma_dT / rho);
}

template <>
auto RadSystem<SuOlsonProblemCgs>::ComputeEddingtonFactor(double f) -> double
{
	return (1./3.);	// Eddington approximation
}

auto testproblem_radiation_marshak_cgs() -> int
{
	// This problem tests whether the numerical scheme is asymptotic preserving.
	// This requires both a spatial discretization *and* a temporal discretization
	// that have the asymptotic-preserving property. Operator splitting the transport and source
	// terms can give a splitting error that is arbitrarily large in the asymptotic limit! [SDC
	// (or a semi-implicit predictor-corrector method) is REQUIRED for a correct solution.]

	// For discussion of the asymptotic-preserving property:
	// 1. R.G. McClarren, R.B. Lowrie, The effects of slope limiting on asymptotic-preserving
	//     numerical methods for hyperbolic conservation laws, Journal of Computational Physics 227
	//     (2008) 9711–9726.
	// 2. R.G. McClarren, T.M. Evans, R.B. Lowrie, J.D. Densmore, Semi-implicit time integration
	//     for PN thermal radiative transfer, Journal of Computational Physics 227 (2008)
	//     7561-7586.

	// Note: PLM (w/ asymptotic correction in Riemann solver) does a good job, but not quite as
	// good as linear DG on this problem. There are some 'stair-stepping' artifacts that appear
	// with PLM at low resolution that don't appear when using DG.
	// -- Hopefully, this will be solved with a larger stencil (e.g. PPM).

	// Problem parameters

	const int max_timesteps = 3e5;
	const double CFL_number = 0.9;
	const int nx = 100; // [18 == matches resolution of McClarren & Lowrie (2008)]

	const double initial_dt = 5.0e-12; // s
	const double max_dt = 5.0e-12;	   // s
	const double max_time = 10.0e-9;   // s
	const double Lx = 0.66;		   // cm

	const double dx = Lx / nx;	    // cm
	const double tau_cell = kappa * dx; // dimensionless optical depth

	// Problem initialization

	RadSystem<SuOlsonProblemCgs> rad_system({.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	rad_system.set_c_light(c);
	rad_system.set_lx(Lx);

	const double c_v = (rad_system.boltzmann_constant_ / rad_system.mean_molecular_mass_) /
			   (rad_system.gamma_ - 1.);

	const double T_initial = 1.0e-9 * T_hohlraum; // K
	const double initial_Egas = rad_system.ComputeEgasFromTgas(rho, T_initial);
	const double initial_Erad = a_rad * std::pow(T_initial, 4);
	const double T_floor = T_initial;
	rad_system.Erad_floor_ = a_rad * std::pow(T_floor, 4);

	// enforce Marshak boundary condition in Riemann solver
	rad_system.T_marshak_left_ = T_hohlraum;
	rad_system.do_marshak_left_boundary_ = true;

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = initial_Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = initial_Egas;
		rad_system.set_staticGasDensity(i) = rho;
		rad_system.set_x1GasMomentum(i) = 0.0;
		rad_system.set_radEnergySource(i) = 0.0;
	}

	const double Erad0 = rad_system.ComputeRadEnergy();
	const double Egas0 = rad_system.ComputeGasEnergy();
	const double Etot0 = Erad0 + Egas0;

	std::cout << "radiation constant (code units) = " << a_rad << "\n";
	std::cout << "c_light (code units) = " << c << "\n";
	std::cout << "Lx = " << Lx << "\n";
	std::cout << "tau_cell = " << tau_cell << "\n";
	std::cout << "rho * c_v = " << rho * c_v << "\n";
	std::cout << "initial_dt = " << initial_dt << "\n";
	std::cout << "max_dt = " << max_dt << "\n";
	std::cout << "max_time = " << max_time << std::endl;

	// Main time loop
	double dt_prev = NAN;
	int j = NAN;
	for (j = 0; j < max_timesteps; ++j) {

		if (rad_system.time() >= max_time) {
			break;
		}

		const double this_dtMax = ((j == 0) ? initial_dt : max_dt);
		// Compute timestep
		const double dt_expand_fac = 1.2;
		const double computed_dt = rad_system.ComputeTimestep(this_dtMax);
		const double this_dt = std::min(computed_dt, dt_expand_fac * dt_prev);

		amrex::Print() << "cycle " << j << " t = " << rad_system.time();
		amrex::Print() << " dt = " << this_dt << "\n";
		rad_system.AdvanceTimestepSDC2(this_dt);
		dt_prev = this_dt;
	}

	std::cout << "Timestep " << j << "; t = " << rad_system.time()
		  << "; dt = " << rad_system.dt() << "\n";

	const double Erad1 = rad_system.ComputeRadEnergy();
	const double Egas1 = rad_system.ComputeGasEnergy();
	const double Etot1 = Erad1 + Egas1;
	const double Ediff = std::fabs(Etot1 - Etot0);

	std::cout << "radiation energy = " << Erad1 << "\n";
	std::cout << "gas energy = " << Egas1 << "\n";
	std::cout << "Total energy = " << Etot1 << "\n";
	std::cout << "(Energy nonconservation = " << Ediff << ")\n";
	std::cout << "\n";

	// compare against diffusion solution

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> x1GasMomentum(nx);
	std::vector<double> x1RadFlux(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;

		const double Erad_t = rad_system.radEnergy(i + nghost);
		Erad.at(i) = Erad_t;
		Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.) / T_hohlraum;

		const double Etot_t = rad_system.gasEnergy(i + nghost);
		const double rho = rad_system.staticGasDensity(i + nghost);
		const double x1GasMom = rad_system.x1GasMomentum(i + nghost);
		const double Ekin = (x1GasMom * x1GasMom) / (2.0 * rho);

		const double Egas_t = (Etot_t - Ekin);
		Egas.at(i) = Egas_t;
		Tgas.at(i) = rad_system.ComputeTgasFromEgas(rho, Egas_t) / T_hohlraum;

		x1GasMomentum.at(i) = rad_system.x1GasMomentum(i + nghost);
		x1RadFlux.at(i) = rad_system.x1RadFlux(i + nghost);
	}

	// read in exact solution

	std::vector<double> xs_exact;
	std::vector<double> Tmat_exact;

	std::string filename = "../extern/marshak_similarity.csv";
	std::ifstream fstream(filename, std::ios::in);
	assert(fstream.is_open());

	std::string header;
	std::getline(fstream, header);

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value; iss >> value;) {
			values.push_back(value);
		}
		auto x_val = values.at(0);
		auto Tmat_val = values.at(1);

		xs_exact.push_back(x_val);
		Tmat_exact.push_back(Tmat_val);
		//amrex::Print() << x_val << " " << Tmat_val << "\n";
	}

	// plot results

	// material temperature
	matplotlibcpp::clf();

	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Tgas_exact_args;
	Tgas_args["label"] = "gas temperature";
	Tgas_args["marker"] = ".";
	Tgas_exact_args["label"] = "gas temperature (exact)";
	Tgas_exact_args["marker"] = "x";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);

	matplotlibcpp::ylim(0.0, 1.0);	// keV
	matplotlibcpp::xlim(0.0, 0.55); // cm
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (keV)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", rad_system.time()));
	matplotlibcpp::save("./marshak_wave_asymptotic_gastemperature.pdf");

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	int status = 0;
	return status;
}
