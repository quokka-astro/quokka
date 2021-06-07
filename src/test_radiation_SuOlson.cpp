//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_SuOlson.hpp"
#include "AMReX_ParallelDescriptor.H"

auto main(int argc, char **argv) -> int
{
	// Initialization (copied from ExaWind)

	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {
		amrex::ParmParse pp("amrex");
		// Set the defaults so that we throw an exception instead of attempting
		// to generate backtrace files. However, if the user has explicitly set
		// these options in their input files respect those settings.
		if (!pp.contains("throw_exception")) {
			pp.add("throw_exception", 1);
		}
		if (!pp.contains("signal_handling")) {
			pp.add("signal_handling", 0);
		}
	});

	int result = 0;

	{ // objects must be destroyed before amrex::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_marshak();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct MarshakProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// Su & Olson (1997) parameters
constexpr double eps_SuOlson = 1.0;
constexpr double kappa = 1.0;
constexpr double rho = 1.0;	   // g cm^-3 (matter density)
constexpr double T_hohlraum = 1.0; // dimensionless
constexpr double x0 = 0.5;	   // dimensionless length scale
constexpr double t0 = 10.0;	   // dimensionless time scale

constexpr double a_rad = 1.0;
constexpr double c = 1.0;
constexpr double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

template <> struct RadSystem_Traits<MarshakProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = c;
	static constexpr double radiation_constant = a_rad;
	static constexpr double mean_molecular_mass = 1.0;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = 0.;
	static constexpr double T_marshak_left = 0.;
	static constexpr bool do_marshak_left_boundary = false;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeTgasFromEgas(const double rho, const double Egas) -> double
{
	return std::pow(4.0 * Egas / alpha_SuOlson, 1. / 4.);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeEgasFromTgas(const double rho, const double Tgas) -> double
{
	return (alpha_SuOlson / 4.0) * (Tgas*Tgas*Tgas*Tgas);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeEgasTempDerivative(const double rho, const double Tgas)
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

const auto initial_Egas = 1e-10 * RadSystem<MarshakProblem>::ComputeEgasFromTgas(rho, T_hohlraum);
const auto initial_Erad = 1e-10 * (a_rad * (T_hohlraum*T_hohlraum*T_hohlraum*T_hohlraum));

template <>
void RadSystem<MarshakProblem>::SetRadEnergySource(array_t &radEnergySource,
						   amrex::Box const &indexRange,
						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						   amrex::Real time)
{

	const double Q = (1.0 / (2.0 * x0));			// do NOT change this
	const double S = Q * (a_rad * std::pow(T_hohlraum, 4)); // erg cm^{-3}

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const xl = (i + Real(0.)) * dx[0];
		amrex::Real const xr = (i + Real(1.)) * dx[0];

		double vol_frac = 0.0;
		if ((xl < x0) && (xr <= x0)) {
			vol_frac = 1.0;
		} else if ((xl < x0) && (xr > x0)) {
			vol_frac = (x0 - xl) / (xr - xl);
			assert(vol_frac > 0.0); // NOLINT
		}

		amrex::Real src = 0.;
		if (time < t0) {
			src = S * vol_frac;
		}
		radEnergySource(i, j, k) = src;
	});
}

template <> void RadiationSimulation<MarshakProblem>::setInitialConditions()
{
	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			state(i, j, k, RadSystem<MarshakProblem>::radEnergy_index) = initial_Erad;
			state(i, j, k, RadSystem<MarshakProblem>::x1RadFlux_index) = 0;
			state(i, j, k, RadSystem<MarshakProblem>::x2RadFlux_index) = 0;
			state(i, j, k, RadSystem<MarshakProblem>::x3RadFlux_index) = 0;

			state(i, j, k, RadSystem<MarshakProblem>::gasEnergy_index) = initial_Egas;
			state(i, j, k, RadSystem<MarshakProblem>::gasDensity_index) = rho;
			state(i, j, k, RadSystem<MarshakProblem>::x1GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<MarshakProblem>::x2GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<MarshakProblem>::x3GasMomentum_index) = 0.;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

auto testproblem_radiation_marshak() -> int
{
	// For this problem, you must do reconstruction in the reduced
	// flux, *not* the flux. Otherwise, F exceeds cE at sharp temperature
	// gradients.

	// Problem parameters
	const int nx = 1500;
	const double Lx = 30.0; // dimensionless length
	const int max_timesteps = 12000;
	const double CFL_number = 0.4;
	//const double initial_dt = 1e-9; // dimensionless time
	//const double max_dt = 1e-2;	// dimensionless time

	// const double max_time = 3.16228;	  // dimensionless time
	const double max_time = 10.0; // dimensionless time

	// Problem initialization
	amrex::IntVect gridDims{AMREX_D_DECL(nx, 4, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))}, // NOLINT
	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(1.0), amrex::Real(1.0))}}; // NOLINT

	constexpr int nvars = 9;

	auto isNormalComp = [=] (int n, int dim) {
		if ((n == RadSystem<MarshakProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<MarshakProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<MarshakProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if(isNormalComp(n, i)) {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
				boundaryConditions[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
				boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);				
			}
		}
	}

	RadiationSimulation<MarshakProblem> sim(gridDims, boxSize, boundaryConditions, nvars);
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.outputAtInterval_ = false;
	sim.plotfileInterval_ = 100; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// copy all FABs to a local FAB across the entire domain
	amrex::BoxArray localBoxes(sim.domain_);
	amrex::DistributionMapping localDistribution(localBoxes, 1);
	amrex::MultiFab state_final(localBoxes, localDistribution, sim.ncomp_, 0);
	amrex::MultiFab state_exact_local(localBoxes, localDistribution, sim.ncomp_, 0);
	state_final.ParallelCopy(sim.state_new_);
	auto const &state_final_array = state_final.array(0);

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> Erad(nx);
		std::vector<double> Egas(nx);

		for (int i = 0; i < nx; ++i) {
			xs.at(i) = Lx * ((i + 0.5) / static_cast<double>(nx));
			const auto Erad_t = state_final_array(i, 0, 0, RadSystem<MarshakProblem>::radEnergy_index);
			const auto Etot_t = state_final_array(i, 0, 0, RadSystem<MarshakProblem>::gasEnergy_index);
			const auto rho = state_final_array(i, 0, 0, RadSystem<MarshakProblem>::gasDensity_index);
			const auto x1GasMom = state_final_array(i, 0, 0, RadSystem<MarshakProblem>::x1GasMomentum_index);

			Erad.at(i) = Erad_t;
			Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);
			const auto Ekin = (x1GasMom * x1GasMom) / (2.0 * rho);
			const auto Egas_t = Etot_t - Ekin;
			Egas.at(i) = Egas_t;
			Tgas.at(i) = RadSystem<MarshakProblem>::ComputeTgasFromEgas(rho, Egas_t);
		}

		std::vector<double> xs_exact = {0.01,	 0.1,	  0.17783, 0.31623, 0.45,
						0.5,	 0.56234, 0.75,	   1.0,	    1.33352,
						1.77828, 3.16228, 5.62341};

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

		std::vector<double> Erad_diffusion_exact_3p1 = {0.95968, 0.95049, 0.93036, 0.86638,
								0.76956, 0.72433, 0.66672, 0.51507,
								0.35810, 0.21309, 0.10047, 0.00634};
		std::vector<double> Erad_transport_exact_3p1 = {1.20052, 1.18869, 1.16190, 1.07175,
								0.90951, 0.79902, 0.66678, 0.44675,
								0.27540, 0.14531, 0.05968, 0.00123};

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

			Tgas_exact_10.at(i) =
			    RadSystem<MarshakProblem>::ComputeTgasFromEgas(rho, Egas_transport_exact_10p0.at(i));
			Tgas_exact_1.at(i) =
			    RadSystem<MarshakProblem>::ComputeTgasFromEgas(rho, Egas_transport_exact_1p0.at(i));
		}

		// interpolate numerical solution onto exact solution tabulated points

		std::vector<double> Tgas_numerical_interp(xs_exact.size());
		interpolate_arrays(xs_exact.data(), Tgas_numerical_interp.data(), xs_exact.size(),
				   xs.data(), Tgas.data(), xs.size());

		// compute L2 error norm

		double err_norm = 0.;
		double sol_norm = 0.;
		for (int i = 0; i < xs_exact.size(); ++i) {
			err_norm += std::abs(Tgas_numerical_interp[i] - Tgas_exact_10[i]);
			sol_norm += std::abs(Tgas_exact_10[i]);
		}
		const double rel_error = err_norm / sol_norm;
		const double error_tol = 0.03; // this will not agree to better than this, due to
					       // not being able to capture fEdd < 1/3 behavior
		amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;
		if (rel_error > error_tol) {
			status = 1;
		}

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
		matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNow_));
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
		matplotlibcpp::plot(xs_exact, Erad_diffusion_exact_10p0, diffusion_args);

		std::map<std::string, std::string> transport_args;
		transport_args["label"] = "transport solution (exact)";
		transport_args["color"] = "red";
		transport_args["linestyle"] = "none";
		transport_args["marker"] = "*";
		matplotlibcpp::plot(xs_exact, Erad_transport_exact_10p0, transport_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x (dimensionless)");
		matplotlibcpp::ylabel("radiation energy density (dimensionless)");
		matplotlibcpp::title(
		    fmt::format("time ct = {:.4g}", sim.tNow_ * (eps_SuOlson * c * rho * kappa)));
		matplotlibcpp::xlim(0.0, 3.0); // cm
		//	matplotlibcpp::ylim(0.0, 2.3);
		matplotlibcpp::save("./SuOlsonTest.pdf");

		matplotlibcpp::xscale("log");
		matplotlibcpp::yscale("log");
		matplotlibcpp::xlim(0.2, 8.0); // cm
		matplotlibcpp::ylim(1e-3, 3.0);
		matplotlibcpp::save("./SuOlsonTest_loglog.pdf");
	}

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;

	return status;
}
