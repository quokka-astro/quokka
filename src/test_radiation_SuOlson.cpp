//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_SuOlson.hpp"

auto main(int argc, char** argv) -> int
{
	// Initialization

	amrex::Initialize(argc, argv);

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_marshak();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct MarshakProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// Su & Olson (1997) parameters
const double eps_SuOlson = 1.0;
const double kappa = 1.0;
const double rho = 1.0;	       // g cm^-3 (matter density)
const double T_hohlraum = 1.0; // dimensionless
// const double T_hohlraum_scaled = 3.481334e6; // K [= 300 eV]
// const double kelvin_to_eV = 8.617385e-5;
const double a_rad = 1.0;
const double c = 1.0;
const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

template <>
auto RadSystem<MarshakProblem>::ComputeTgasFromEgas(const double rho,
						    const double Egas) -> double
{
	return std::pow(4.0 * Egas / alpha_SuOlson, 1. / 4.);
}

template <>
auto RadSystem<MarshakProblem>::ComputeEgasFromTgas(const double rho,
						    const double Tgas) -> double
{
	return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
}

template <>
auto RadSystem<MarshakProblem>::ComputeEgasTempDerivative(const double rho,
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

auto testproblem_radiation_marshak() -> int
{
	// For this problem, you must do reconstruction in the reduced
	// flux, *not* the flux. Otherwise, F exceeds cE at sharp temperature
	// gradients.

	// Problem parameters

	const int max_timesteps = 12000;
	const double CFL_number = 0.4;
	const int nx = 1500;
	const double Lz = 30.0; // dimensionless length

	const double initial_dtau = 1e-9; // dimensionless time
	const double max_dtau = 1e-2;	  // dimensionless time
	// const double max_tau = 3.16228;	  // dimensionless time
	const double max_tau = 10.0; // dimensionless time

	// Su & Olson (1997) parameters
	const double z0 = 0.5;		     // dimensionless length scale
	const double tau0 = 10.0;	     // dimensionless time scale
	const double Q = (1.0 / (2.0 * z0)); // do NOT change this

	const double rho = 1.0; // g cm^-3 (matter density)
	const double kappa = 1.0;

	// Problem initialization

	RadSystem<MarshakProblem> rad_system(
	    {.nx = nx, .lx = Lz, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(1.0);
	rad_system.set_c_light(1.0);

	auto nghost = rad_system.nghost();

	const double a_rad = rad_system.radiation_constant();
	const double c = rad_system.c_light();
	const double chi = rho * kappa; // cm^-1 (total matter opacity)
	const double x0 = z0 / chi;	// cm
	const double Lx = Lz / chi;	// cm
	const double t0 = tau0 / (eps_SuOlson * c * chi);		  // s
	const double max_time = max_tau / (eps_SuOlson * c * chi);	  // s
	const double max_dt = max_dtau / (eps_SuOlson * c * chi);	  // s
	const double initial_dt = initial_dtau / (eps_SuOlson * c * chi); // s

	amrex::Print() << "radiation constant (code units) = " << a_rad << "\n";
	amrex::Print() << "c_light (code units) = " << c << "\n";
	amrex::Print() << "Lx = " << Lx << "\n";
	amrex::Print() << "initial_dt = " << initial_dt << "\n";
	amrex::Print() << "max_dt = " << max_dt << "\n";

	rad_system.set_lx(Lx);

	const auto initial_Egas =
	    1e-10 * rad_system.ComputeEgasFromTgas(rho, T_hohlraum);
	const auto initial_Erad = 1e-10 * (a_rad * std::pow(T_hohlraum, 4));

	const double S = Q * (a_rad * std::pow(T_hohlraum, 4)); // erg cm^{-3}

	rad_system.Erad_floor_ = initial_Erad;

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double xl = Lx * ((idx_value) / static_cast<double>(nx));
		const double xr = Lx * ((idx_value + 1.0) / static_cast<double>(nx));

		rad_system.set_radEnergy(i) = initial_Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = initial_Egas;
		rad_system.set_staticGasDensity(i) = rho;

		double vol_frac = 0.0;
		if ((xl < x0) && (xr <= x0)) {
			vol_frac = 1.0;
		} else if ((xl < x0) && (xr > x0)) {
			vol_frac = (x0 - xl) / (xr - xl);
			assert(vol_frac > 0.0); // NOLINT
		}

		rad_system.set_radEnergySource(i) = S * vol_frac;
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {

		if (rad_system.time() >= max_time) {
			amrex::Print() << "Timestep " << j
				  << "; t = " << rad_system.time()
				  << "; dt = " << rad_system.dt() << "\n";

			const auto Erad = rad_system.ComputeRadEnergy();
			const auto Egas = rad_system.ComputeGasEnergy();
			const auto Etot = Erad + Egas;
			const auto Ediff = std::fabs(Etot - Etot0);
			const auto Eadded = x0 * rad_system.time() * (c * S);

			amrex::Print() << "radiation energy = " << Erad << "\n";
			amrex::Print() << "gas energy = " << Egas << "\n";
			amrex::Print() << "Total energy = " << Etot << "\n";
			amrex::Print() << "(Energy nonconservation = " << Ediff
				  << ")\n";
			amrex::Print() << "Injected energy = " << Eadded << "\n";
			amrex::Print() << "\n";

			break;
		}

		if (rad_system.time() >= t0) {
			for (int i = nghost; i < nx + nghost; ++i) {
				rad_system.set_radEnergySource(i) = 0.0;
			}
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
		xs.at(i) = x;

		const auto Erad_t = rad_system.radEnergy(i + nghost);

		Erad.at(i) = Erad_t;
		Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);

		const auto Etot_t = rad_system.gasEnergy(i + nghost);
		const auto rho = rad_system.staticGasDensity(i + nghost);
		const auto x1GasMom = rad_system.x1GasMomentum(i + nghost);
		const auto Ekin = (x1GasMom*x1GasMom) / (2.0*rho);
		const auto Egas_t = Etot_t - Ekin;

		Egas.at(i) = Egas_t;
		Tgas.at(i) = rad_system.ComputeTgasFromEgas(rho, Egas_t);
	}

	std::vector<double> xs_exact = {
	    0.01, 0.1, 0.17783, 0.31623, 0.45,	  0.5,	  0.56234,
	    0.75, 1.0, 1.33352, 1.77828, 3.16228, 5.62341};

	std::vector<double> Erad_diffusion_exact_0p1 = {
	    0.09403, 0.09326, 0.09128, 0.08230, 0.06086, 0.04766, 0.03171,
	    0.00755, 0.00064, 0.,      0.,	0.,	 0.};
	std::vector<double> Erad_transport_exact_0p1 = {
	    0.09531, 0.09531, 0.09532, 0.09529, 0.08823, 0.04765, 0.00375,
	    0.,	     0.,      0.,      0.,	0.,	 0.};

	std::vector<double> Erad_diffusion_exact_1p0 = {
	    0.50359, 0.49716, 0.48302, 0.43743, 0.36656, 0.33271, 0.29029,
	    0.18879, 0.10150, 0.04060, 0.01011, 0.00003, 0.};
	std::vector<double> Erad_transport_exact_1p0 = {
	    0.64308, 0.63585, 0.61958, 0.56187, 0.44711, 0.35801, 0.25374,
	    0.11430, 0.03648, 0.00291, 0.,	0.,	 0.};

	std::vector<double> Egas_transport_exact_1p0 = {
	    0.27126, 0.26839, 0.26261, 0.23978, 0.18826, 0.14187, 0.08838,
	    0.03014, 0.00625, 0.00017, 0.,	0.,	 0.};

	std::vector<double> Erad_diffusion_exact_3p1 = {
	    0.95968, 0.95049, 0.93036, 0.86638, 0.76956, 0.72433,
	    0.66672, 0.51507, 0.35810, 0.21309, 0.10047, 0.00634};
	std::vector<double> Erad_transport_exact_3p1 = {
	    1.20052, 1.18869, 1.16190, 1.07175, 0.90951, 0.79902,
	    0.66678, 0.44675, 0.27540, 0.14531, 0.05968, 0.00123};

	std::vector<double> Erad_diffusion_exact_10p0 = {
	    1.86585, 1.85424, 1.82889, 1.74866, 1.62824, 1.57237, 1.50024,
	    1.29758, 1.06011, 0.79696, 0.52980, 0.12187, 0.00445};
	std::vector<double> Erad_transport_exact_10p0 = {
	    2.23575, 2.21944, 2.18344, 2.06448, 1.86072, 1.73178, 1.57496,
	    1.27398, 0.98782, 0.70822, 0.45016, 0.09673, 0.00375};

	std::vector<double> Egas_transport_exact_10p0 = {
	    2.11186, 2.09585, 2.06052, 1.94365, 1.74291, 1.61536, 1.46027,
	    1.16591, 0.88992, 0.62521, 0.38688, 0.07642, 0.00253};

	std::vector<double> Trad_exact_10(Erad_transport_exact_10p0);
	std::vector<double> Trad_exact_1(Erad_transport_exact_1p0);

	std::vector<double> Tgas_exact_10(Egas_transport_exact_10p0);
	std::vector<double> Tgas_exact_1(Egas_transport_exact_10p0);

	for (int i = 0; i < xs_exact.size(); ++i) {
		Trad_exact_10.at(i) =
		    std::pow(Erad_transport_exact_10p0.at(i) / a_rad, 1. / 4.);
		Trad_exact_1.at(i) =
		    std::pow(Erad_transport_exact_1p0.at(i) / a_rad, 1. / 4.);

		Tgas_exact_10.at(i) = rad_system.ComputeTgasFromEgas(
		    rho, Egas_transport_exact_10p0.at(i));
		Tgas_exact_1.at(i) = rad_system.ComputeTgasFromEgas(
		    rho, Egas_transport_exact_1p0.at(i));
	}

	// interpolate numerical solution onto exact solution tabulated points

	std::vector<double> Tgas_numerical_interp(xs_exact.size());
	interpolate_arrays(xs_exact.data(), Tgas_numerical_interp.data(),
			   xs_exact.size(), xs.data(), Tgas.data(), xs.size());

	// compute L2 error norm

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < xs_exact.size(); ++i) {
		err_norm +=
		    std::abs(Tgas_numerical_interp[i] - Tgas_exact_10[i]);
		sol_norm += std::abs(Tgas_exact_10[i]);
	}
	const double rel_error = err_norm / sol_norm;
	const double error_tol =
	    0.03; // this will not agree to better than this, due to not being
		  // able to capture fEdd < 1/3 behavior
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	// Plot solution

	matplotlibcpp::clf();
	matplotlibcpp::xlim(0.2, 8.0); // cm

	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature";
	matplotlibcpp::plot(xs, Trad, Trad_args);

	std::map<std::string, std::string> Trad_exact10_args;
	Trad_exact10_args["label"] = "radiation temperature (exact)";
	Trad_exact10_args["marker"] = ".";
	Trad_exact10_args["linestyle"] = "none";
	Trad_exact10_args["color"] = "black";
	matplotlibcpp::plot(xs_exact, Trad_exact_10, Trad_exact10_args);

	std::map<std::string, std::string> Trad_exact1_args;
	Trad_exact1_args["label"] = "radiation temperature (exact)";
	Trad_exact1_args["marker"] = ".";
	Trad_exact1_args["linestyle"] = "none";
	Trad_exact1_args["color"] = "black";
	// matplotlibcpp::plot(xs_exact, Trad_exact_1, Trad_exact1_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);

	std::map<std::string, std::string> Tgas_exact10_args;
	Tgas_exact10_args["label"] = "gas temperature (exact)";
	Tgas_exact10_args["marker"] = "*";
	Tgas_exact10_args["linestyle"] = "none";
	Tgas_exact10_args["color"] = "black";
	matplotlibcpp::plot(xs_exact, Tgas_exact_10, Tgas_exact10_args);

	std::map<std::string, std::string> Tgas_exact1_args;
	Tgas_exact1_args["label"] = "gas temperature (exact)";
	Tgas_exact1_args["marker"] = "*";
	Tgas_exact1_args["linestyle"] = "none";
	Tgas_exact1_args["color"] = "black";
	// matplotlibcpp::plot(xs_exact, Tgas_exact_1, Tgas_exact1_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::title(fmt::format("time t = {:.4g}", rad_system.time()));
	matplotlibcpp::xlim(0.1, 30.0); // cm
	// matplotlibcpp::ylim(0.0, 1.3);	// dimensionless
	matplotlibcpp::xscale("log");
	matplotlibcpp::save("./SuOlsonTest_temperature.pdf");

	matplotlibcpp::clf();

	std::map<std::string, std::string> Erad_args;
	Erad_args["label"] = "Numerical solution";
	Erad_args["color"] = "black";
	matplotlibcpp::plot(xs, Erad, Erad_args);

	std::map<std::string, std::string> diffusion_args;
	diffusion_args["label"] = "diffusion solution (exact)";
	diffusion_args["color"] = "gray";
	diffusion_args["linestyle"] = "dashed";
	diffusion_args["marker"] = ".";
	matplotlibcpp::plot(xs_exact, Erad_diffusion_exact_10p0,
			    diffusion_args);

	std::map<std::string, std::string> transport_args;
	transport_args["label"] = "transport solution (exact)";
	transport_args["color"] = "red";
	transport_args["linestyle"] = "none";
	transport_args["marker"] = "*";
	matplotlibcpp::plot(xs_exact, Erad_transport_exact_10p0,
			    transport_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::title(fmt::format(
	    "time ct = {:.4g}", rad_system.time() * (eps_SuOlson * c * chi)));
	matplotlibcpp::xlim(0.0, 3.0); // cm
	//	matplotlibcpp::ylim(0.0, 2.3);
	matplotlibcpp::save("./SuOlsonTest.pdf");

	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::xlim(0.2, 8.0); // cm
	matplotlibcpp::ylim(1e-3, 3.0);
	matplotlibcpp::save("./SuOlsonTest_loglog.pdf");

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;

	int status = 0;
	if (rel_error > error_tol) {
		status = 1;
	}
	return status;
}
