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

const double kappa = 1.0;
const double rho = 1.0;	       // g cm^-3 (matter density)
const double a_rad = 1.0;
const double c = 1.0;
const double T_floor = 1e-5;

template <> void RadSystem<PulseProblem>::FillGhostZones(array_t &cons)
{
	// Fill boundary conditions with exact solution

	const double T_H = T_floor;
	const double E_rad = radiation_constant_ * std::pow(T_H, 4);
	const double F_rad = 0.0;

	// x1 left side boundary (Neumann)
	for (int i = 0; i < nghost_; ++i) {
		cons(radEnergy_index, i) = E_rad;
		cons(x1RadFlux_index, i) = -F_rad;
		cons(gasEnergy_index, i) = ComputeEgasFromTgas(rho, T_H);
		cons(x1GasMomentum_index, i) = 0.0;
	}

	// x1 right side boundary (Neumann)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) = E_rad;
		cons(x1RadFlux_index, i) = F_rad;
		cons(gasEnergy_index, i) = ComputeEgasFromTgas(rho, T_H);
		cons(x1GasMomentum_index, i) = 0.0;
	}
}

template <>
auto RadSystem<PulseProblem>::ComputeOpacity(const double rho,
					       const double Tgas) -> double
{
	return kappa;
}

template <>
void RadSystem<PulseProblem>::AddSourceTerms(array_t &cons,
					  std::pair<int, int> range)
{
	// Compute *equilibrium* temperature, then
	// Set radiation and gas to equilibrium temperature

	for (int i = range.first; i < range.second; ++i) {
		const double a_rad = radiation_constant_;

		// load fluid properties
		const double rho = cons(gasDensity_index, i);
		const double Egastot0 = cons(gasEnergy_index, i);
		const double x1GasMom0 = cons(x1GasMomentum_index, i);

		const double vx0 = x1GasMom0 / rho;
		const double vsq0 = vx0*vx0;			// N.B. modify for 3d
		const double Ekin0 = 0.5*rho*vsq0;
		const double Egas0 = Egastot0 - Ekin0;

		// load radiation energy
		const double Erad0 = cons(radEnergy_index, i);

		assert(Egas0 > 0.0); // NOLINT
		assert(Erad0 > 0.0); // NOLINT

		const double Etot0 = Egas0 + Erad0;

		// find equilibrium temperature (nonlinear, must solve via iteration)

		// BEGIN NEWTON-RAPHSON LOOP
		double F = NAN;
		double dF_dT = NAN;
		double eta = NAN;

		double Egas_guess = Egas0;
		double Erad_guess = Erad0;
		double Teq_guess = RadSystem<problem_t>::ComputeTgasFromEgas(rho, Egas_guess);
		const double T_floor = 1e-10;
		const double resid_tol = 1e-10;
		const int maxIter = 200;
		int n = 0;
		for (n = 0; n < maxIter; ++n) {

			// compute material temperature
			T_gas = RadSystem<problem_t>::ComputeTgasFromEgas(rho, Egas_guess);

			// compute emissivity
			fourPiB_over_c = a_rad * std::pow(T_gas, 4);

			// compute derivatives w/r/t T_gas
			const double dB_dTgas = (4.0 * fourPiB) / T_gas;

			// compute residuals
			F = (fourPiB_over_c + Erad_guess) - Etot0;

			// check if converged
			if (std::abs(F / Etot0) < resid_tol) {
				break;
			}

			// compute Jacobian
			const double C_v =
			    RadSystem<problem_t>::ComputeEgasTempDerivative(rho, T_gas);

			dF_dT = dB_dTgas + C_v;

			// Update variables
			const double delta_Teq = (-F / dF_dT);
			Teq_guess += delta_Teq;

			const double newEgas = RadSystem<problem_t>::ComputeEgasFromTgas(Teq_guess);
			const double newErad = a_rad * std::pow(Teq_guess, 4);
			Egas_guess = newEgas;
			Erad_guess = newErad;

		} // END NEWTON-RAPHSON LOOP

		assert(std::abs(F / Etot0) < resid_tol); // NOLINT

		assert(Erad_guess > 0.0); // NOLINT
		assert(Egas_guess > 0.0); // NOLINT

		// store new radiation energy
		cons(radEnergy_index, i) = Erad_guess;
		cons(gasEnergy_index, i) = Egas_guess + Ekin0;
	}
}

auto compute_exact_solution(const double x, const double t) -> double
{
	// compute exact solution for Gaussian radiation pulse
	// 		assuming diffusion approximation


}

auto testproblem_radiation_pulse() -> int
{
	// Problem parameters

	const int max_timesteps = 2e5;
	const double CFL_number = 0.4;
	const int nx = 100;

	const double initial_dt = 1e-6; // dimensionless time
	const double max_dt = 1e-3;	  // dimensionless time
	const double max_time = 1.0;	  // dimensionless time
	const double Lz = 1.0;	  // dimensionless length

	// Problem initialization

	std::vector<double> T_eq(nx);
	for (int i = 0; i < nx; ++i) {
		// initialize initial temperature

	}

	RadSystem<PulseProblem> rad_system(
	    {.nx = nx, .lx = Lz, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	rad_system.set_c_light(c);
	rad_system.set_lx(Lx);
	rad_system.Erad_floor_ = a_rad * std::pow(T_floor, 4);

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = (a_rad * std::pow(T_eq.at(i), 4));
		rad_system.set_x1RadFlux(i) = 0.0;

		rad_system.set_gasEnergy(i) = rad_system.ComputeEgasFromTgas(rho, T_eq.at(i));;
		rad_system.set_staticGasDensity(i) = rho;
		rad_system.set_x1GasMomentum(i) = 0.0;

		rad_system.set_radEnergySource(i) = 0.0;
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	std::cout << "radiation constant (code units) = " << a_rad << "\n";
	std::cout << "c_light (code units) = " << c << "\n";
	std::cout << "Lx = " << Lx << "\n";
	std::cout << "initial_dt = " << initial_dt << "\n";
	std::cout << "max_dt = " << max_dt << "\n";

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

	const auto Erad = rad_system.ComputeRadEnergy();
	const auto Egas = rad_system.ComputeGasEnergy();
	const auto Etot = Erad + Egas;
	const auto Ediff = std::fabs(Etot - Etot0);

	std::cout << "radiation energy = " << Erad << "\n";
	std::cout << "gas energy = " << Egas << "\n";
	std::cout << "Total energy = " << Etot << "\n";
	std::cout << "(Energy nonconservation = " << Ediff << ")\n";
	std::cout << "\n";


	// read out results

	std::vector<double> xs(nx);
	std::vector<double> Erad(nx);
	std::vector<double> x1RadFlux(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;

		const auto Erad_t = rad_system.radEnergy(i + nghost);
		Erad.at(i) = Erad_t;
		x1RadFlux.at(i) = rad_system.x1RadFlux(i + nghost);
	}

	// compute exact solution

	std::vector<double> xs_exact(nx);
	std::vector<double> Erad_exact(nx);

	for (int i = 0; i < xs_exact.size(); ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));

		auto x_val = x;
		auto Erad_val = ();

		xs_exact.push_back(x_val);
		Erad_exact.push_back(Erad_val);
	}

	// compute error norm

	std::vector<double> Erad_interp(xs_exact.size());
	interpolate_arrays(xs_exact.data(), Erad_interp.data(), xs_exact.size(),
			   xs.data(), Erad.data(), xs.size());

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < xs_exact.size(); ++i) {
		err_norm += std::abs(Erad_interp[i] - Erad_exact[i]);
		sol_norm += std::abs(Erad_exact[i]);
	}

	const double error_tol = 0.001;
	const double rel_error = err_norm / sol_norm;
	std::cout << "Relative L1 error norm = " << rel_error << std::endl;

	// plot energy density

	std::map<std::string, std::string> Erad_args, Erad_exact_args;
	Erad_args["label"] = "Numerical solution";
	Erad_args["color"] = "black";
	Erad_exact_args["label"] = "Exact solution";
	Erad_exact_args["color"] = "blue";

	matplotlibcpp::plot(xs, Erad, Erad_args);
	matplotlibcpp::plot(xs_exact, Erad_exact, Erad_exact_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", rad_system.time() * c));
	matplotlibcpp::save("./radiation_pulse.pdf");

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
