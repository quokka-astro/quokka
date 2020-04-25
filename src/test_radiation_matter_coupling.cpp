//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_matter_coupling.cpp
/// \brief Defines a test problem for radiation-matter coupling.
///

#include "test_radiation_matter_coupling.hpp"

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		testproblem_radiation_matter_coupling();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

void testproblem_radiation_matter_coupling()
{
	// Problem parameters

	const int nx = 4;
	const double Lx = 1.0;
	const double CFL_number = 0.4;
	// const double constant_dt = 1.0e-11; // s
	// const double max_time = 1.0e-7;	    // s
	const double constant_dt = 1.0e-9; // s
	const double max_time = 1.0e-4;	   // s

	const int max_timesteps = 1e7;

	// Problem initialization

	RadSystem<AthenaArray<double>> rad_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	auto nghost = rad_system.nghost();

	const double Erad = 1.0e12; // erg cm^-3
	const double Egas = 1.0e2;  // erg cm^-3
	const double rho = 1.0e-7;  // g cm^-3

	// Su & Olson (1997) test problem
	const double eps_SuOlson = 1.0;
	const double a_rad = rad_system.radiation_constant_;
	const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

	auto ComputeTgasFromEgas = [=](const double Eint) {
		return std::pow(4.0 * Eint / alpha_SuOlson, 1. / 4.);
	};

	auto ComputeEgasFromTgas = [=](const double Tgas) {
		return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
	};

	const double initial_Tgas = ComputeTgasFromEgas(Egas);
	std::cout << "Initial Tgas = " << initial_Tgas << "\n";

#if 0
	const double c_v =
	    rad_system.boltzmann_constant_ /
	    (rad_system.mean_molecular_mass_ * (rad_system.gamma_ - 1.0));
	const auto initial_Tgas = Egas / (rho * c_v);

	std::cout << "Volumetric heat capacity c_v = " << rho * c_v << "\n";
#endif

	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = Egas;
		rad_system.set_staticGasDensity(i) = rho;
	}

	std::vector<double> t;
	std::vector<double> Trad;
	std::vector<double> Tgas;
	std::vector<double> Egas_v;

	const auto initial_Erad = rad_system.ComputeRadEnergy();
	const auto initial_Egas = rad_system.ComputeGasEnergy();
	const auto initial_Etot = initial_Erad + initial_Egas;

	const auto initial_Trad =
	    std::pow(Erad / rad_system.radiation_constant(), 1. / 4.);
	const auto kappa =
	    RadSystem<AthenaArray<double>>::ComputeOpacity(rho, initial_Tgas);

	std::cout << "Initial radiation temperature = " << initial_Trad << "\n";
	std::cout << "Initial gas temperature = " << initial_Tgas << "\n";

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {

		const auto current_Erad = rad_system.ComputeRadEnergy();
		const auto current_Egas = rad_system.ComputeGasEnergy();
		const auto current_Etot = current_Erad + current_Egas;
		const auto Ediff = std::fabs(current_Etot - initial_Etot);

		if (rad_system.time() >= max_time) {
			std::cout << "Timestep " << j
				  << "; t = " << rad_system.time() << "\n";
			std::cout << "radiation energy = " << current_Erad
				  << "\n";
			std::cout << "gas energy = " << current_Egas << "\n";
			std::cout << "Total energy = " << current_Etot << "\n";
			std::cout << "(Energy nonconservation = " << Ediff
				  << ")\n";
			std::cout << "\n";

			break;
		}

		rad_system.AdvanceTimestep(constant_dt);

		t.push_back(rad_system.time());
		Trad.push_back(std::pow(rad_system.radEnergy(0 + nghost) /
					    rad_system.radiation_constant(),
					1. / 4.));

		auto Egas_i = rad_system.gasEnergy(0 + nghost);
		Tgas.push_back(ComputeTgasFromEgas(Egas_i));
		Egas_v.push_back(rad_system.gasEnergy(0 + nghost));
	}

	// Solve for asymptotically-exact solution (Gonzalez et al. 2007)
	const int nmax = t.size();
	std::vector<double> t_exact(nmax);
	std::vector<double> Tgas_exact(nmax);

	for (int n = 0; n < nmax; ++n) {
#if 0
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
		const double arad = rad_system.radiation_constant();
		const double c = rad_system.c_light_;
		const double E0 = (Erad + Egas) / (arad + alpha_SuOlson / 4.0);
		const double T0_4 = std::pow(initial_Tgas, 4);

		const double T4 =
		    (T0_4 - E0) * std::exp(-(4. / alpha_SuOlson) *
					   (arad + alpha_SuOlson / 4.0) *
					   kappa * rho * c * time_t) +
		    E0;

		const double T_gas = std::pow(T4, 1. / 4.);

		t_exact.at(n) = (time_t);
		Tgas_exact.at(n) = (T_gas);
	}

	matplotlibcpp::clf();
	matplotlibcpp::yscale("log");
	matplotlibcpp::xscale("log");
	matplotlibcpp::ylim(0.1 * std::min(Tgas.front(), Trad.front()),
			    10.0 * std::max(Trad.back(), Tgas.back()));

	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature";
	matplotlibcpp::plot(t, Trad, Trad_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(t, Tgas, Tgas_args);

	std::map<std::string, std::string> exactsol_args;
	exactsol_args["label"] = "gas temperature (exact)";
	exactsol_args["linestyle"] = "--";
	exactsol_args["color"] = "black";
	matplotlibcpp::plot(t_exact, Tgas_exact, exactsol_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("time t (s)");
	matplotlibcpp::ylabel("temperature T (K)");
	matplotlibcpp::title(fmt::format("dt = {:.4g}\nt = {:.4g}", constant_dt,
					 rad_system.time()));
	matplotlibcpp::save(fmt::format("./radcoupling.pdf"));

	matplotlibcpp::clf();

	std::vector<double> frac_err(t.size());
	for (int i = 0; i < t.size(); ++i) {
		frac_err.at(i) = Tgas_exact.at(i) / Tgas.at(i) - 1.0;
		// std::cout << Tgas.at(i) << "\t" << Tgas_exact.at(i) <<
		// std::endl;
	}
	matplotlibcpp::plot(t, frac_err);
	matplotlibcpp::xlabel("time t (s)");
	matplotlibcpp::ylabel("fractional error in material temperature");
	matplotlibcpp::save(fmt::format("./radcoupling_fractional_error.pdf"));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}

#define LIKELY_IN_CACHE_SIZE 8

/** @brief find index of a sorted array such that arr[i] <= key < arr[i + 1].
 *
 * If an starting index guess is in-range, the array values around this
 * index are first checked.  This allows for repeated calls for well-ordered
 * keys (a very common case) to use the previous index as a very good guess.
 *
 * If the guess value is not useful, bisection of the array is used to
 * find the index.  If there is no such index, the return values are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @param guess initial guess of index
 * @return index
 */
static int64_t binary_search_with_guess(const double key, const double *arr,
					int64_t len, int64_t guess)
{
	int64_t imin = 0;
	int64_t imax = len;

	/* Handle keys outside of the arr range first */
	if (key > arr[len - 1]) {
		return len;
	} else if (key < arr[0]) {
		return -1;
	}

	/*
	 * If len <= 4 use linear search.
	 * From above we know key >= arr[0] when we start.
	 */
	if (len <= 4) {
		int64_t i;

		for (i = 1; i < len && key >= arr[i]; ++i)
			;
		return i - 1;
	}

	if (guess > len - 3) {
		guess = len - 3;
	}
	if (guess < 1) {
		guess = 1;
	}

	/* check most likely values: guess - 1, guess, guess + 1 */
	if (key < arr[guess]) {
		if (key < arr[guess - 1]) {
			imax = guess - 1;
			/* last attempt to restrict search to items in cache */
			if (guess > LIKELY_IN_CACHE_SIZE &&
			    key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
				imin = guess - LIKELY_IN_CACHE_SIZE;
			}
		} else {
			/* key >= arr[guess - 1] */
			return guess - 1;
		}
	} else {
		/* key >= arr[guess] */
		if (key < arr[guess + 1]) {
			return guess;
		} else {
			/* key >= arr[guess + 1] */
			if (key < arr[guess + 2]) {
				return guess + 1;
			} else {
				/* key >= arr[guess + 2] */
				imin = guess + 2;
				/* last attempt to restrict search to items in
				 * cache */
				if (guess < len - LIKELY_IN_CACHE_SIZE - 1 &&
				    key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
					imax = guess + LIKELY_IN_CACHE_SIZE;
				}
			}
		}
	}

	/* finally, find index by bisection */
	while (imin < imax) {
		const npy_intp imid = imin + ((imax - imin) >> 1);
		if (key >= arr[imid]) {
			imin = imid + 1;
		} else {
			imax = imid;
		}
	}
	return imin - 1;
}

#undef LIKELY_IN_CACHE_SIZE
