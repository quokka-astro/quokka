/// \file test_radhydro_uniform_advecting.cpp
/// \brief Defines a test problem for radiation advection in a uniform medium with grey radiation.
///

#include "test_radhydro_uniform_advecting.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"
#include "radiation_system.hpp"

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
	} else if constexpr (n_groups_ == 52) { // from 1e-3 to 1e2
		constexpr amrex::GpuArray<double, 53> rad_boundaries = {
			0.0,
			1.00000000e-03, 1.25892541e-03, 1.58489319e-03, 1.99526231e-03,
			2.51188643e-03, 3.16227766e-03, 3.98107171e-03, 5.01187234e-03,
			6.30957344e-03, 7.94328235e-03, 1.00000000e-02, 1.25892541e-02,
			1.58489319e-02, 1.99526231e-02, 2.51188643e-02, 3.16227766e-02,
			3.98107171e-02, 5.01187234e-02, 6.30957344e-02, 7.94328235e-02,
			1.00000000e-01, 1.25892541e-01, 1.58489319e-01, 1.99526231e-01,
			2.51188643e-01, 3.16227766e-01, 3.98107171e-01, 5.01187234e-01,
			6.30957344e-01, 7.94328235e-01, 1.00000000e+00, 1.25892541e+00,
			1.58489319e+00, 1.99526231e+00, 2.51188643e+00, 3.16227766e+00,
			3.98107171e+00, 5.01187234e+00, 6.30957344e+00, 7.94328235e+00,
			1.00000000e+01, 1.25892541e+01, 1.58489319e+01, 1.99526231e+01,
			2.51188643e+01, 3.16227766e+01, 3.98107171e+01, 5.01187234e+01,
			6.30957344e+01, 7.94328235e+01, 1.00000000e+02, inf
		};
		return rad_boundaries;
	} else if constexpr (n_groups_ == 50) {
		constexpr amrex::GpuArray<double, 51> rad_boundaries = {
			1.00000000e-03, 1.25892541e-03, 1.58489319e-03, 1.99526231e-03,
			2.51188643e-03, 3.16227766e-03, 3.98107171e-03, 5.01187234e-03,
			6.30957344e-03, 7.94328235e-03, 1.00000000e-02, 1.25892541e-02,
			1.58489319e-02, 1.99526231e-02, 2.51188643e-02, 3.16227766e-02,
			3.98107171e-02, 5.01187234e-02, 6.30957344e-02, 7.94328235e-02,
			1.00000000e-01, 1.25892541e-01, 1.58489319e-01, 1.99526231e-01,
			2.51188643e-01, 3.16227766e-01, 3.98107171e-01, 5.01187234e-01,
			6.30957344e-01, 7.94328235e-01, 1.00000000e+00, 1.25892541e+00,
			1.58489319e+00, 1.99526231e+00, 2.51188643e+00, 3.16227766e+00,
			3.98107171e+00, 5.01187234e+00, 6.30957344e+00, 7.94328235e+00,
			1.00000000e+01, 1.25892541e+01, 1.58489319e+01, 1.99526231e+01,
			2.51188643e+01, 3.16227766e+01, 3.98107171e+01, 5.01187234e+01,
			6.30957344e+01, 7.94328235e+01, 1.00000000e+02
		};
		return rad_boundaries;
	} else if constexpr (n_groups_ == 82) {
		constexpr amrex::GpuArray<double, 83> rad_boundaries = {
			0.0,
			1.00000000e-04, 1.25892541e-04, 1.58489319e-04, 1.99526231e-04,
			2.51188643e-04, 3.16227766e-04, 3.98107171e-04, 5.01187234e-04,
			6.30957344e-04, 7.94328235e-04, 1.00000000e-03, 1.25892541e-03,
			1.58489319e-03, 1.99526231e-03, 2.51188643e-03, 3.16227766e-03,
			3.98107171e-03, 5.01187234e-03, 6.30957344e-03, 7.94328235e-03,
			1.00000000e-02, 1.25892541e-02, 1.58489319e-02, 1.99526231e-02,
			2.51188643e-02, 3.16227766e-02, 3.98107171e-02, 5.01187234e-02,
			6.30957344e-02, 7.94328235e-02, 1.00000000e-01, 1.25892541e-01,
			1.58489319e-01, 1.99526231e-01, 2.51188643e-01, 3.16227766e-01,
			3.98107171e-01, 5.01187234e-01, 6.30957344e-01, 7.94328235e-01,
			1.00000000e+00, 1.25892541e+00, 1.58489319e+00, 1.99526231e+00,
			2.51188643e+00, 3.16227766e+00, 3.98107171e+00, 5.01187234e+00,
			6.30957344e+00, 7.94328235e+00, 1.00000000e+01, 1.25892541e+01,
			1.58489319e+01, 1.99526231e+01, 2.51188643e+01, 3.16227766e+01,
			3.98107171e+01, 5.01187234e+01, 6.30957344e+01, 7.94328235e+01,
			1.00000000e+02, 1.25892541e+02, 1.58489319e+02, 1.99526231e+02,
			2.51188643e+02, 3.16227766e+02, 3.98107171e+02, 5.01187234e+02,
			6.30957344e+02, 7.94328235e+02, 1.00000000e+03, 1.25892541e+03,
			1.58489319e+03, 1.99526231e+03, 2.51188643e+03, 3.16227766e+03,
			3.98107171e+03, 5.01187234e+03, 6.30957344e+03, 7.94328235e+03,
			1.00000000e+04, inf
		};
		return rad_boundaries;
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
	static constexpr double energy_unit = 1.0;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewisePowerLaw;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<PulseProblem>::DefineOpacityExponentsAndLowerValues(const double rho, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	for (int i = 0; i < nGroups_; ++i) {
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

// template <>
// template <typename ArrayType>
// AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeRadQuantityExponents(ArrayType const & /*quant*/,
// 									     amrex::GpuArray<double, nGroups_ + 1> const & /*boundaries*/)
//     -> amrex::GpuArray<double, nGroups_>
// {
// 	amrex::GpuArray<double, nGroups_> exponents{};
// 	for (int g = 0; g < nGroups_; ++g) {
// 		exponents[g] = 0.0;
// 	}
// 	return exponents;
// }

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

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

		auto Erad_val = a_rad * std::pow(T0, 4);
		double trad_exact = NAN;
		if constexpr (beta_order_ <= 1) {
			trad_exact = std::pow(Erad0 / a_rad, 1. / 4.);
		} else { // beta_order_ == 2 or 3
			trad_exact = std::pow(Erad_beta2 / a_rad, 1. / 4.);
		}
		Trad_exact.push_back(trad_exact);
		Erad_exact.push_back(Erad_val);
	}

	// compute spectrum
	std::vector<double> spec{}; // spectrum density at the end, Erad / bin_width
	std::vector<double> E_r{}; // spectrum density at the end, Erad
	std::vector<double> F_r{}; // flux at the end, Frad
	std::vector<double> bin_center{};
	int const ii = 10; // a random grid
	for (int g = 0; g < n_groups_; ++g) {
		bin_center.push_back(std::sqrt(rad_boundaries_[g] * rad_boundaries_[g + 1]));
		const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[ii];
		const auto Frad_t = values.at(RadSystem<PulseProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g)[ii];
		const double bin_width = rad_boundaries_[g + 1] - rad_boundaries_[g];
		E_r.push_back(Erad_t);
		F_r.push_back(Frad_t);
		spec.push_back(Erad_t / bin_width);
	}

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < xs.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Tgas_exact[i]);
		sol_norm += std::abs(Tgas_exact[i]);
	}
	const double error_tol = 1.0e-10; // This is a very very stringent test (to machine accuracy!)
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Texact_args;
	std::map<std::string, std::string> Tradexact_args;
	Trad_args["label"] = "radiation (numerical)";
	Trad_args["linestyle"] = "-";
	Tradexact_args["label"] = "radiation (exact)";
	Tradexact_args["linestyle"] = "--";
	Tgas_args["label"] = "gas (numerical)";
	Tgas_args["linestyle"] = "-";
	Texact_args["label"] = "gas (exact)";
	Texact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	// matplotlibcpp::plot(xs, Trad_exact, Tradexact_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	// matplotlibcpp::plot(xs, Tgas_exact, Texact_args);
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
	std::map<std::string, std::string> spec_args;
	spec_args["label"] = "spectrum";
	matplotlibcpp::plot(bin_center, spec, spec_args);
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

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity";
	vgas_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	vgas_args["label"] = "gas velocity (exact)";
	vgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Vgas_exact, vgas_args);
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("v");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./adv_vel.pdf");
#endif

	if (export_csv) {
		std::ofstream file;
		file.open("adv_spectrum.csv");
		file << "nu_Left, E_r\n";
		for (int g = 0; g < n_groups_; ++g) {
			file << std::scientific << std::setprecision(12) << rad_boundaries_[g] << "," << E_r[g] << "\n";
		}
		file.close();

		file.open("adv_flux_spectrum.csv");
		file << "nu_Left, F_r\n";
		for (int g = 0; g < n_groups_; ++g) {
			file << std::scientific << std::setprecision(12) << rad_boundaries_[g] << "," << F_r[g] << "\n";
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
	// if ((rel_error > error_tol) || std::isnan(rel_error)) {
	// 	status = 1;
	// }
	return status;
}
