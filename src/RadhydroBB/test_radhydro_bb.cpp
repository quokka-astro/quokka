/// \file test_radhydro_bb.cpp
/// \brief Defines a test problem for blackbody spectrum in a uniform advecting medium.
///

#include <cmath>
#include <unordered_map>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"

#include "ArrayUtil.hpp"
#include "fextract.hpp"
#include "interpolate.hpp"
#include "radiation_system.hpp"

// #include "AMReX_BC_TYPES.H"
#include "AMReX_IntVect.H"
#include "AMReX_Print.H"
// #include "RadhydroSimulation.hpp"
// #include "fextract.hpp"
#include "physics_info.hpp"
// #include "radiation_system.hpp"
#include "test_radhydro_bb.hpp"

static constexpr bool export_csv = true;

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int n_groups_ = 50;

constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_ = []() constexpr {
	if constexpr (n_groups_ == 1) {
		return amrex::GpuArray<double, 2>{0.0, inf};
	} else if constexpr (n_groups_ == 4) { // from 1e-3 to 1e2
		return amrex::GpuArray<double, 5>{1.0e-4, 1.0e-3, 3.0, 1.0e2, 1.0e3};
	} else if constexpr (n_groups_ == 7) {
		return amrex::GpuArray<double, 8>{1.0e-4, 1.0e-3, 1.0e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.0e3};
	} else if constexpr (n_groups_ == 50) {
		return amrex::GpuArray<double, 51>{
		    1.00000000e-03, 1.25892541e-03, 1.58489319e-03, 1.99526231e-03, 2.51188643e-03, 3.16227766e-03, 3.98107171e-03, 5.01187234e-03,
		    6.30957344e-03, 7.94328235e-03, 1.00000000e-02, 1.25892541e-02, 1.58489319e-02, 1.99526231e-02, 2.51188643e-02, 3.16227766e-02,
		    3.98107171e-02, 5.01187234e-02, 6.30957344e-02, 7.94328235e-02, 1.00000000e-01, 1.25892541e-01, 1.58489319e-01, 1.99526231e-01,
		    2.51188643e-01, 3.16227766e-01, 3.98107171e-01, 5.01187234e-01, 6.30957344e-01, 7.94328235e-01, 1.00000000e+00, 1.25892541e+00,
		    1.58489319e+00, 1.99526231e+00, 2.51188643e+00, 3.16227766e+00, 3.98107171e+00, 5.01187234e+00, 6.30957344e+00, 7.94328235e+00,
		    1.00000000e+01, 1.25892541e+01, 1.58489319e+01, 1.99526231e+01, 2.51188643e+01, 3.16227766e+01, 3.98107171e+01, 5.01187234e+01,
		    6.30957344e+01, 7.94328235e+01, 1.00000000e+02};
	}
}();

constexpr double c = 1.0e8;
// model 0
// constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 1e-4 * c;
// constexpr double kappa0 = 1.0e4; // dx = 1, tau = kappa0 * dx = 1e4
// constexpr double chat = 1.0e7;
// model 1
// constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 1e-4 * c;
// constexpr double kappa0 = 1.0e4; // dx = 1, tau = kappa0 * dx = 1e4
// constexpr double chat = 1.0e8;
// model 2
// constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 1e-2 * c;
// constexpr double kappa0 = 1.0e5;
// constexpr double chat = 1.0e8;
// model 3
constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 0.0;
// constexpr double v0 = 1e-2 * c;
// constexpr double v0 = 0.3 * c;
constexpr double v0 = 0.001 * c;
constexpr double kappa0 = 1.0e5;
constexpr double chat = c;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.0;
constexpr double k_B = 1.0;

constexpr double nu_unit = 1.0;
constexpr double T_equilibrium = 0.768032502191;

// static diffusion, beta = 1e-4, tau_cell = kappa0 * dx = 100, beta tau_cell = 1e-2
// constexpr double kappa0 = 100.; // cm^2 g^-1
// constexpr double v0 = 1.0e-4 * c; // advecting pulse

// dynamic diffusion, beta = 1e-3, tau = kappa0 * dx = 1e5, beta tau = 100
// constexpr double max_time = 10.0 / v0;
// constexpr double max_time = 1000.0 / (c * rho0 * kappa0); // dt >> 1 / (c * chi)
constexpr double max_time = 10.0 / (1e-2 * c);

constexpr double Erad0 = a_rad * T0 * T0 * T0 * T0;
constexpr double Erad_beta2 = (1. + 4. / 3. * (v0 * v0) / (c * c)) * Erad0;
constexpr double erad_floor = a_rad * 1e-30;

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = beta_order_;
	static constexpr double energy_unit = nu_unit;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	// static constexpr OpacityModel opacity_model = OpacityModel::user;
	// static constexpr OpacityModel opacity_model = OpacityModel::PPL_fixed_slope;
	// static constexpr OpacityModel opacity_model = OpacityModel::PPL_fixed_slope_with_transport;
	// static constexpr OpacityModel opacity_model = OpacityModel::PPL_free_slope;
	static constexpr OpacityModel opacity_model = OpacityModel::PPL_free_slope_with_delta_terms;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<PulseProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[1][i] = kappa0;
	}
	return exponents_and_values;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaPVec{};
	for (int i = 0; i < nGroups_; ++i) {
		kappaPVec[i] = kappa0;
	}
	return kappaPVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_bb(const double nu, const double T) -> double
{
	double const x = nu_unit * nu / (k_B * T);
	double const coeff = nu_unit / (k_B * T);
	double const planck_integral = std::pow(x, 3) / (std::exp(x) - 1.0);
	return coeff * planck_integral / (std::pow(PI, 4) / 15.0) * (a_rad * std::pow(T, 4));
}

// template <>
// template <typename ArrayType>
// AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeRadQuantityExponents(ArrayType const & /*quant*/,
// 									     amrex::GpuArray<double, nGroups_ + 1> const & /*boundaries*/)
//     -> amrex::GpuArray<double, nGroups_>
// {
// 	amrex::GpuArray<double, nGroups_> exponents{};
// 	for (int g = 0; g < nGroups_; ++g) {
// 		exponents[g] = -1.0;
// 	}
// 	return exponents;
// }

template <> void RadhydroSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, T0);

	double erad = NAN;
	double frad = NAN;
	if constexpr (beta_order_ == 0) {
		erad = Erad0;
		frad = 0.0;
	} else if constexpr (beta_order_ == 1) {
		erad = Erad0;
		frad = 4. / 3. * v0 * Erad0;
	} else if constexpr (beta_order_ == 2) {
		erad = Erad_beta2;
		frad = 4. / 3. * v0 * Erad0;
	} else { // beta_order_ == 3
		erad = Erad_beta2;
		frad = 4. / 3. * v0 * Erad0 * (1. + (v0 * v0) / (c * c));
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < n_groups_; ++g) {
			state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = erad_floor;
			state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

// #define LIKELY_IN_CACHE_SIZE 8
// AMREX_GPU_HOST_DEVICE
// int64_t binary_search_with_guess(const double key, const double *arr, int64_t len, int64_t guess)
// {
// 	int64_t imin = 0;
// 	int64_t imax = len;

// 	/* Handle keys outside of the arr range first */
// 	if (key > arr[len - 1]) {
// 		return len;
// 	} else if (key < arr[0]) {
// 		return -1;
// 	}

// 	/*
// 	 * If len <= 4 use linear search.
// 	 * From above we know key >= arr[0] when we start.
// 	 */
// 	if (len <= 4) {
// 		int64_t i;

// 		for (i = 1; i < len && key >= arr[i]; ++i)
// 			;
// 		return i - 1;
// 	}

// 	if (guess > len - 3) {
// 		guess = len - 3;
// 	}
// 	if (guess < 1) {
// 		guess = 1;
// 	}

// 	/* check most likely values: guess - 1, guess, guess + 1 */
// 	if (key < arr[guess]) {
// 		if (key < arr[guess - 1]) {
// 			imax = guess - 1;
// 			/* last attempt to restrict search to items in cache */
// 			if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
// 				imin = guess - LIKELY_IN_CACHE_SIZE;
// 			}
// 		} else {
// 			/* key >= arr[guess - 1] */
// 			return guess - 1;
// 		}
// 	} else {
// 		/* key >= arr[guess] */
// 		if (key < arr[guess + 1]) {
// 			return guess;
// 		} else {
// 			/* key >= arr[guess + 1] */
// 			if (key < arr[guess + 2]) {
// 				return guess + 1;
// 			} else {
// 				/* key >= arr[guess + 2] */
// 				imin = guess + 2;
// 				/* last attempt to restrict search to items in
// 				 * cache */
// 				if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
// 					imax = guess + LIKELY_IN_CACHE_SIZE;
// 				}
// 			}
// 		}
// 	}

// 	/* finally, find index by bisection */
// 	while (imin < imax) {
// 		const int64_t imid = imin + ((imax - imin) >> 1);
// 		if (key >= arr[imid]) {
// 			imin = imid + 1;
// 		} else {
// 			imax = imid;
// 		}
// 	}
// 	return imin - 1;
// }

// #undef LIKELY_IN_CACHE_SIZE

// AMREX_GPU_HOST_DEVICE
// void interpolate_arrays(double *x, double *y, int len, double *arr_x, double *arr_y, int arr_len)
// {
// 	/* Note: arr_x must be sorted in ascending order,
// 		and arr_len must be >= 3. */

// 	int64_t j = 0;
// 	for (int i = 0; i < len; i++) {
// 		j = binary_search_with_guess(x[i], arr_x, arr_len, j);

// 		if (j == -1) {
// 			y[i] = NAN;
// 		} else if (j == arr_len) {
// 			y[i] = NAN;
// 		} else if (j == arr_len - 1) {
// 			y[i] = arr_y[j];
// 		} else if (x[i] == arr_x[j]) { // avoid roundoff error
// 			y[i] = arr_y[j];
// 		} else {
// 			const double slope = (arr_y[j + 1] - arr_y[j]) / (arr_x[j + 1] - arr_x[j]);
// 			y[i] = slope * (x[i] - arr_x[j]) + arr_y[j];
// 		}
// 		// assert(!std::isnan(y[i]));
// 	}
// }

auto problem_main() -> int
{
	// This problem is a *linear* radiation diffusion problem, i.e.
	// parameters are chosen such that the radiation and gas temperatures
	// should be near equilibrium, and the opacity is chosen to go as
	// T^3, such that the radiation diffusion equation becomes linear in T.

	// This makes this problem a stringent test of the asymptotic-
	// preserving property of the computational method, since the
	// optical depth per cell at the peak of the temperature profile is
	// of order 10^5.

	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 8.0;

	const double max_dt = 1.0;

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	RadhydroSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> Vgas_exact(nx);
	std::vector<double> rhogas(nx);
	std::vector<double> Trad_exact{};
	std::vector<double> Tgas_exact{};
	std::vector<double> Erad_exact{};

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		// const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index)[i];
		double Erad_t = 0.0;
		for (int g = 0; g < n_groups_; ++g) {
			Erad_t += values.at(RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<PulseProblem>::x1GasMomentum_index)[i] / rho_t;
		rhogas.at(i) = rho_t;
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t / T0;
		Egas.at(i) = values.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho_t, Egas.at(i)) / T0;
		Tgas_exact.push_back(1.0);
		Vgas.at(i) = v_t;
		Vgas_exact.at(i) = v0;

		auto const Erad_val = a_rad * std::pow(T0, 4);
		Trad_exact.push_back(T_equilibrium);
		Erad_exact.push_back(Erad_val);
	}

	// compute spectrum
	std::vector<double> spec{};	// spectrum density at the end, Erad / bin_width
	std::vector<double> E_r{};	// spectrum density at the end, Erad
	std::vector<double> F_r_spec{}; // flux at the end, Frad
	std::vector<double> bin_center{};
	int const ii = 3; // a random grid
	for (int g = 0; g < n_groups_; ++g) {
		bin_center.push_back(std::sqrt(rad_boundaries_[g] * rad_boundaries_[g + 1]));
		const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[ii];
		const auto Frad_t = values.at(RadSystem<PulseProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g)[ii];
		const double bin_width = rad_boundaries_[g + 1] - rad_boundaries_[g];
		E_r.push_back(Erad_t);
		F_r_spec.push_back(Frad_t / bin_width);
		spec.push_back(Erad_t / bin_width);
	}

	// Read in exact solution
	std::vector<double> nu_exact;
	std::vector<double> Fnu_exact;
	std::vector<double> Enu_exact;

	std::string const filename = "../extern/BB_doppler.csv";
	std::ifstream fstream(filename, std::ios::in);

	double err_norm = 0.;
	double sol_norm = 0.;
	std::string header;
	std::getline(fstream, header);

	nu_exact.push_back(0.0);
	Fnu_exact.push_back(0.0);
	Enu_exact.push_back(0.0);
	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values_line;
		std::string value;

		while (std::getline(iss, value, ',')) {
			values_line.push_back(std::stod(value));
		}
		double const nu_val = values_line.at(0);  // dimensionless
		double const Fnu_val = values_line.at(1); // dimensionless
		nu_exact.push_back(nu_val);
		Fnu_exact.push_back(Fnu_val);
		double const enu = compute_exact_bb(nu_val, T_equilibrium);
		Enu_exact.push_back(enu);
	}

	// compute error norm

	std::vector<double> F_interp(n_groups_);
	interpolate_arrays(bin_center.data(), F_interp.data(), n_groups_, nu_exact.data(), Fnu_exact.data(), static_cast<int>(nu_exact.size()));

	for (int g = 0; g < n_groups_; ++g) {
		double f_int = 0.0;
		if (!std::isnan(F_interp[g])) {
			f_int = F_interp[g];
		}
		err_norm += std::abs(F_r_spec[g] - f_int);
		sol_norm += std::abs(f_int);
	}

	const double error_tol = 0.1;
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Tradexact_args;
	Trad_args["label"] = "T_rad (numerical)";
	Trad_args["linestyle"] = "-";
	Tradexact_args["label"] = "T (exact)";
	Tradexact_args["linestyle"] = "--";
	Tgas_args["label"] = "T_gas (numerical)";
	Tgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Trad_exact, Tradexact_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	// if constexpr (beta_order_ == 1) {
	// 	matplotlibcpp::ylim(1.0 - 1.0e-7, 1.0 + 1.0e-7);
	// }
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./adv_temp.pdf");

	// plot spectrum
	matplotlibcpp::clf();
	std::unordered_map<std::string, std::string> spec_args;
	spec_args["label"] = "spectrum";
	spec_args["color"] = "C0";
	matplotlibcpp::scatter(bin_center, spec, 10.0, spec_args);
	std::map<std::string, std::string> spec_exact_args;
	spec_exact_args["label"] = "spectrum (exact)";
	spec_exact_args["linestyle"] = "-";
	spec_exact_args["color"] = "C1";
	matplotlibcpp::plot(nu_exact, Enu_exact, spec_exact_args);
	// log-log
	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::ylim(1.0e-8, 1.0e0);
	matplotlibcpp::xlabel("frequency (dimensionless)");
	matplotlibcpp::ylabel("spectrum density (dimensionless)");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./adv_spectrum.pdf");

	// plot flux spectrum
	matplotlibcpp::clf();
	std::unordered_map<std::string, std::string> Frad_args;
	Frad_args["label"] = "flux";
	Frad_args["color"] = "C0";
	matplotlibcpp::scatter(bin_center, F_r_spec, 10.0, Frad_args);
	std::map<std::string, std::string> Frad_exact_args;
	Frad_exact_args["label"] = "flux (exact)";
	Frad_exact_args["linestyle"] = "-";
	Frad_exact_args["color"] = "C1";
	matplotlibcpp::plot(nu_exact, Fnu_exact, Frad_exact_args);
	// log-log
	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::ylim(1.0e-3, 1.0e5);
	matplotlibcpp::xlabel("frequency (dimensionless)");
	matplotlibcpp::ylabel("flux density (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./adv_flux_spectrum.pdf");
#endif

	if (export_csv) {
		std::ofstream file;
		file.open("adv_spectrum.csv");
		file << "nu_Left, E_r (erg/cm^3/Hz)\n";
		for (int g = 0; g < n_groups_; ++g) {
			file << std::scientific << std::setprecision(12) << rad_boundaries_[g] << "," << spec[g] << "\n";
		}
		file.close();

		file.open("adv_flux_spectrum.csv");
		file << "nu_Left, F_r (erg/cm^2/s/Hz)\n";
		for (int g = 0; g < n_groups_; ++g) {
			file << std::scientific << std::setprecision(12) << rad_boundaries_[g] << "," << F_r_spec[g] << "\n";
		}
		file.close();

		file.open("adv_temp.csv");
		file << "xs,Tgas,Trad\n";
		for (size_t i = 0; i < xs.size(); ++i) {
			file << std::scientific << std::setprecision(12) << xs[i] << "," << Tgas[i] << "," << Trad[i] << "\n";
		}
		file.close();
	}

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
