//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak_asymptotic.cpp
/// \brief Defines a test problem for radiation in the asymptotic diffusion regime.
///

#include "AMReX_BLassert.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "matplotlibcpp.h"
#include "radiation_system.hpp"
#include "test_radiation_marshak_asymptotic.hpp"
#include <ios>

struct SuOlsonProblemCgs {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int opacity_model_ = 1; // 0 = user, 1 = piecewise power-law

constexpr int n_groups_ = 6;
constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {0.3e12, 0.3e14, 0.6e14, 0.9e14, 1.2e14, 1.5e14, 1.5e16};

constexpr int max_step_ = 1e6;

constexpr double kappa = 1000.0; // cm^2 g^-1 (opacity)
constexpr double rho0 = 1.0e-3; // g cm^-3
constexpr double T_initial = 300.0; // K
constexpr double T_L = 1000.0; // K
constexpr double T_R = 300.0; // K
constexpr double rho_C_V = 1.0e-3; // erg g^-1 cm^-3 K^-1
constexpr double c_v = rho_C_V / rho0;
constexpr double mu = 1.0 / (5. / 3. - 1.) * C::k_B / c_v;

constexpr double a_rad = radiation_constant_cgs_;
constexpr double Erad_floor_ = a_rad * T_initial * T_initial * T_initial * T_initial * 1e-20;

template <> struct quokka::EOS_Traits<SuOlsonProblemCgs> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<SuOlsonProblemCgs> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_; // number of radiation groups
};

template <> struct RadSystem_Traits<SuOlsonProblemCgs> {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::hplanck; // set boundary unit to Hz
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = group_edges_;
	static constexpr OpacityModel opacityModel = static_cast<OpacityModel>(opacity_model_); // 0: user, 1: piecewisePowerLaw
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	for (int i = 0; i < nGroups_; ++i) {
		exponents_and_values[1][i] = kappa;
	}
	return exponents_and_values;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaPVec{};
	for (int i = 0; i < nGroups_; ++i) {
		kappaPVec[i] = kappa;
	}
	return kappaPVec;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeFluxMeanOpacity(const double rho, const double Tgas)
    -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeEddingtonFactor(double /*f*/) -> double
{
	return (1. / 3.); // Eddington approximation
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<SuOlsonProblemCgs>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
							      int /*numcomp*/, amrex::GeometryData const & geom, const amrex::Real /*time*/,
							      const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int j = 0;
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	auto const radBoundaries_g = RadSystem<SuOlsonProblemCgs>::radBoundaries_;

	if (i < lo[0] || i >= hi[0]) {
		double T_H = NAN;
		if (i < lo[0]) {
			T_H = T_L;
		} else {
			T_H = T_R;
		}

		auto Erad_g = RadSystem<SuOlsonProblemCgs>::ComputeThermalRadiation(T_H, radBoundaries_g);
		const double Egas = quokka::EOS<SuOlsonProblemCgs>::ComputeEintFromTgas(rho0, T_initial);

		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}

		// gas boundary conditions are the same on both sides
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho0;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
	}
}

template <> void RadhydroSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto radBoundaries_g = RadSystem_Traits<SuOlsonProblemCgs>::radBoundaries;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double Egas = quokka::EOS<SuOlsonProblemCgs>::ComputeEintFromTgas(rho0, T_initial);
		// const double Erad = a_rad * std::pow(T_initial, 4);
		auto Erad_g = RadSystem<SuOlsonProblemCgs>::ComputeThermalRadiation(T_initial, radBoundaries_g);

		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index) = Erad_g;
		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index) = 0;
		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index) = 0;
		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index) = 0;
		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem tests whether the numerical scheme is asymptotic preserving.
	// This requires both a spatial discretization *and* a temporal discretization
	// that have the asymptotic-preserving property. Operator splitting the
	// transport and source terms can give a splitting error that is arbitrarily
	// large in the asymptotic limit! A fully implicit method or a semi-implicit
	// predictor-corrector method [2] similar to SDC is REQUIRED for a correct solution.
	//
	// For discussion of the asymptotic-preserving property, see [1] and [2]. For
	// a discussion of the exact, self-similar solution to this problem, see [3].
	// Note that when used with an SDC time integrator, PLM (w/ asymptotic correction
	// in Riemann solver) does a good job, but not quite as good as linear DG on this
	// problem. There are some 'stair-stepping' artifacts that appear with PLM at low
	// resolution that do not appear when using DG. This is likely the "wide stencil"
	// issue discussed in [4].
	//
	// 1. R.G. McClarren, R.B. Lowrie, The effects of slope limiting on asymptotic-preserving
	//     numerical methods for hyperbolic conservation laws, Journal of
	//     Computational Physics 227 (2008) 9711–9726.
	// 2. R.G. McClarren, T.M. Evans, R.B. Lowrie, J.D. Densmore, Semi-implicit time integration
	//     for PN thermal radiative transfer, Journal of Computational Physics 227
	//     (2008) 7561-7586.
	// 3. Y. Zel'dovich, Y. Raizer, Physics of Shock Waves and High-Temperature Hydrodynamic
	//     Phenomena (1964), Ch. X.: Thermal Waves.
	// 4. Lowrie, R. B. and Morel, J. E., Issues with high-resolution Godunov methods for
	//     radiation hydrodynamics, Journal of Quantitative Spectroscopy and
	//     Radiative Transfer, 69, 475–489, 2001.

	// Problem parameters
	const int max_timesteps = max_step_;
	const double CFL_number = 0.8;
	// const double initial_dt = 5.0e-12; // s
	const double max_dt = 1.0;	   // s
	const double max_time = 1.36e-7;

	constexpr int nvars = RadSystem<SuOlsonProblemCgs>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0,
				amrex::BCType::ext_dir);     // custom (Marshak) x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	RadhydroSimulation<SuOlsonProblemCgs> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	// sim.initDt_ = initial_dt;
	sim.maxDt_ = max_dt;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compare against diffusion solution
	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	// define a vector of n_groups_ vectors
	std::vector<std::vector<double>> Trad_g(n_groups_);

	for (int i = 0; i < nx; ++i) {
		double Erad_t = 0.;
		// const double Erad_t = values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index)[i];
		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			Erad_t += values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		const double Egas_t = values.at(RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index)[i];
		const double rho = values.at(RadSystem<SuOlsonProblemCgs>::gasDensity_index)[i];
		amrex::Real const x = position[i];
		xs.at(i) = x;
		Tgas.at(i) = quokka::EOS<SuOlsonProblemCgs>::ComputeTgasFromEint(rho, Egas_t);
		Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);
		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			auto Erad_g = values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
			Trad_g[g].push_back(std::pow(Erad_g / a_rad, 1. / 4.));
		}
	}

	// // read in exact solution

	// std::vector<double> xs_exact;
	// std::vector<double> Tmat_exact;

	// std::string filename = "../extern/marshak_similarity.csv";
	// std::ifstream fstream(filename, std::ios::in);
	// AMREX_ALWAYS_ASSERT(fstream.is_open());

	// std::string header;
	// std::getline(fstream, header);

	// for (std::string line; std::getline(fstream, line);) {
	// 	std::istringstream iss(line);
	// 	std::vector<double> values;

	// 	for (double value = NAN; iss >> value;) {
	// 		values.push_back(value);
	// 	}
	// 	auto x_val = values.at(0);
	// 	auto Tmat_val = values.at(1);

	// 	xs_exact.push_back(x_val);
	// 	Tmat_exact.push_back(Tmat_val);
	// }

	// // compute error norm

	// // interpolate numerical solution onto exact tabulated solution
	// std::vector<double> Tmat_interp(xs_exact.size());
	// interpolate_arrays(xs_exact.data(), Tmat_interp.data(), static_cast<int>(xs_exact.size()), xs.data(), Tgas.data(), static_cast<int>(xs.size()));

	// double err_norm = 0.;
	// double sol_norm = 0.;
	// for (size_t i = 0; i < xs_exact.size(); ++i) {
	// 	err_norm += std::abs(Tmat_interp[i] - Tmat_exact[i]);
	// 	sol_norm += std::abs(Tmat_exact[i]);
	// }

	// const double error_tol = 0.09;
	// const double rel_error = err_norm / sol_norm;
	// amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	// save data to file
	std::ofstream fstream;
	fstream.open("marshak_wave_Vaytet.csv");
	fstream << "# x,Tgas,Trad, Trad_1, Trad_2, Trad_3, Trad_4, Trad_5, Trad_6";
	for (int i = 0; i < nx; ++i) {
		fstream << std::endl;
		fstream << std::scientific << std::setprecision(14) << xs[i] << ", " << Tgas[i] << ", " << Trad[i] << ", " << Trad_g[0][i] << ", " << Trad_g[1][i] << ", " << Trad_g[2][i] << ", " << Trad_g[3][i] << ", " << Trad_g[4][i] << ", " << Trad_g[5][i];
	}
	fstream.close();

#ifdef HAVE_PYTHON
	// plot results
	matplotlibcpp::clf();
	std::map<std::string, std::string> args;
	args["label"] = "gas";
	args["linestyle"] = "-";
	args["color"] = "r";
	matplotlibcpp::plot(xs, Tgas, args);
	args["label"] = "radiation";
	args["linestyle"] = "-";
	args["color"] = "g";
	// args["marker"] = "x";
	matplotlibcpp::plot(xs, Trad, args);

	for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
		std::map<std::string, std::string> Trad_g_args;
		Trad_g_args["label"] = fmt::format("group {}", g);
		Trad_g_args["linestyle"] = "-";
		Trad_g_args["color"] = "C" + std::to_string(g + 2);
		matplotlibcpp::plot(xs, Trad_g[g], Trad_g_args);
	}

	// Tgas_exact_args["label"] = "gas temperature (exact)";
	// Tgas_exact_args["color"] = "C0";
	// // Tgas_exact_args["marker"] = "x";
	// matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);

	matplotlibcpp::xlim(0.0, 12.0); // cm
	matplotlibcpp::ylim(0.0, 1000.0);	// K
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (K)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./marshak_wave_Vaytet.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	int status = 0;
	// if ((rel_error > error_tol) || std::isnan(rel_error)) {
	// 	status = 1;
	// }
	return status;
}
