//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak_hot.cpp
/// \brief Defines a test Marshak wave problem with extremely high specific heat capacity in radiation.
///

#include "test_radiation_marshak_hot.hpp"
#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"

struct StreamingProblem {
};

constexpr double initial_Erad = 1.0e-5;
constexpr double erad_floor = 1.0e-15;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	 // speed of light
constexpr double chat = 1.0;	 // reduced speed of light
constexpr double kappa0 = 1.0e4; // opacity
constexpr double rho = 1.0;
constexpr double a_rad = 1.0e15;
constexpr double EradL = a_rad;

template <> struct quokka::EOS_Traits<StreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StreamingProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
};

template <> struct RadSystem_Traits<StreamingProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// calculate radEnergyFractions
	quokka::valarray<amrex::Real, Physics_Traits<StreamingProblem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<StreamingProblem>::nGroups;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad0 * radEnergyFractions[g];
			state_cc(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x3GasMomentum_index) = 0.;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<StreamingProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
							     int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
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

	// calculate radEnergyFractions
	quokka::valarray<amrex::Real, Physics_Traits<StreamingProblem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<StreamingProblem>::nGroups;
	}

	if (i < lo[0]) {
		// streaming inflow boundary
		const double Erad = EradL;
		const double Frad = c * Erad;

		// multigroup radiation
		// x1 left side boundary (Marshak)
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad * radEnergyFractions[g];
			consVar(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frad * radEnergyFractions[g];
			consVar(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	} else if (i >= hi[0]) {
		// right-side boundary -- constant
		const double Erad = initial_Erad;
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			auto const Erad_g = Erad * radEnergyFractions[g];
			consVar(i, j, k, RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g;
			consVar(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	}

	// gas boundary conditions are the same everywhere
	const double Egas = initial_Egas;
	consVar(i, j, k, RadSystem<StreamingProblem>::gasEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<StreamingProblem>::gasDensity_index) = rho;
	consVar(i, j, k, RadSystem<StreamingProblem>::gasInternalEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<StreamingProblem>::x1GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<StreamingProblem>::x2GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<StreamingProblem>::x3GasMomentum_index) = 0.;
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1e-2;
	const double tmax = 1.0;
	const int max_timesteps = 5000;

	// Boundary conditions
	constexpr int nvars = RadSystem<StreamingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<StreamingProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compute error norm
	std::vector<double> erad(nx);
	std::vector<double> erad_exact(nx);
	std::vector<double> T(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		double erad_sim = 0.0;
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			erad_sim += values.at(RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		erad.at(i) = erad_sim;
		const double e_gas = values.at(RadSystem<StreamingProblem>::gasInternalEnergy_index)[i];
		T.at(i) = quokka::EOS<StreamingProblem>::ComputeTgasFromEint(rho, e_gas);

		// compute exact solution
		const double tau = kappa0 * rho * x;
		erad_exact.at(i) = (x <= chat * tmax) ? EradL * std::exp(-tau) : 0.0;
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(erad[i] - erad_exact[i]);
		sol_norm += std::abs(erad_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.02;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	// matplotlibcpp::ylim(0.0, 1.1);

	std::map<std::string, std::string> erad_args;
	std::map<std::string, std::string> erad_exact_args;
	erad_args["label"] = "numerical solution";
	erad_exact_args["label"] = "exact solution";
	erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, erad, erad_args);
	matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./radiation_marshak_hot_Erad.pdf");

	// plot temperature
	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, T);
	matplotlibcpp::title("Temperature");
	matplotlibcpp::save("./radiation_marshak_hot_temperature.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	// return status;
	return 0;
}
