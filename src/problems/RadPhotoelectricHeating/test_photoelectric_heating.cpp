//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_photoelectric_heating.cpp
/// \brief Defines a test problem for photoelectric heating in the free-streaming regime.
///

#include "test_photoelectric_heating.hpp"
#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"

struct Problem {
};

const bool with_opacity = 1;
constexpr bool enable_dust_ = 1;
constexpr bool enable_pe_ = 0;
constexpr int n_group_ = 2;
constexpr amrex::GpuArray<double, n_group_ + 1> rad_boundary_ = []() constexpr {
	if constexpr (n_group_ == 2) {
		return amrex::GpuArray<double, 3>{1.0e-3, 3.0, 100.};
	} else if constexpr (n_group_ == 1) {
		return amrex::GpuArray<double, 2>{0.0, inf};
	}
}();

constexpr double erad_floor = 1.0e-12;
constexpr double initial_Egas = 1.0e-12;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = c;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double kappa1 = 1.0; // opacity
constexpr double rho0 = 1.0;
// constexpr double mu = 1.0e-5; // such that CV = 3/2 * 1/mu = 1.5e5
constexpr double mu = 1.0;
constexpr double k_B = 1.0;

template <> struct quokka::EOS_Traits<Problem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<Problem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_group_; // number of radiation groups
};

template <> struct RadSystem_Traits<Problem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = k_B;
	static constexpr amrex::GpuArray<double, Physics_Traits<Problem>::nGroups + 1> radBoundaries = rad_boundary_;
	static constexpr OpacityModel opacity_model = []() constexpr {
		if constexpr (n_group_ > 1) {
			return OpacityModel::piecewise_constant_opacity;
		} else {
			return OpacityModel::single_group;
		}
	}();
	static constexpr bool enable_dust_gas_thermal_coupling_model = enable_dust_;
};

// template <> struct ISM_Traits<Problem> {
// 	static constexpr bool enable_photoelectric_heating = enable_pe_;
// 	static constexpr bool enable_line_cooling = false;
// };

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<Problem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	if (with_opacity) {
		return kappa1;
	}
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<Problem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<Problem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho,
							     const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		if (with_opacity) {
			exponents_and_values[1][i] = kappa1;
		} else {
			exponents_and_values[1][i] = kappa0;
		}
	}
	// exponents_and_values[1][0] = kappa0;
	// exponents_and_values[1][1] = kappa1;
	return exponents_and_values;
}

// template <>
// AMREX_GPU_HOST_DEVICE auto
// RadSystem<Problem>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/, amrex::Real const num_density) -> amrex::Real
// {
// 	return 1.5 * k_B * num_density; // = C_V * num_density
// }

template <> void QuokkaSimulation<Problem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = erad_floor;
	const auto Egas0 = initial_Egas;

	// calculate radEnergyFractions
	quokka::valarray<amrex::Real, Physics_Traits<Problem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<Problem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<Problem>::nGroups;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<Problem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<Problem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad0 * radEnergyFractions[g];
			state_cc(i, j, k, RadSystem<Problem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<Problem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<Problem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<Problem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<Problem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<Problem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<Problem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<Problem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<Problem>::x3GasMomentum_index) = 0.;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<Problem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
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

	if (i < lo[0]) {
		// streaming inflow boundary
		quokka::valarray<double, 2> Erad{};
		for (int g = 0; g < n_group_; ++g) {
			if (g == n_group_ - 1) {
				Erad[g] = 1.0;
			} else {
				Erad[g] = erad_floor;
			}
		}
		const auto Frad = c * Erad;

		// multigroup radiation
		// x1 left side boundary (Marshak)
		for (int g = 0; g < Physics_Traits<Problem>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<Problem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad[g];
			consVar(i, j, k, RadSystem<Problem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frad[g];
			consVar(i, j, k, RadSystem<Problem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<Problem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	} else if (i >= hi[0]) {
		// right-side boundary -- constant
		for (int g = 0; g < Physics_Traits<Problem>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<Problem>::radEnergy_index + Physics_NumVars::numRadVars * g) = erad_floor;
			consVar(i, j, k, RadSystem<Problem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<Problem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<Problem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	}

	// gas boundary conditions are the same everywhere
	const double Egas = initial_Egas;
	consVar(i, j, k, RadSystem<Problem>::gasEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<Problem>::gasDensity_index) = rho0;
	consVar(i, j, k, RadSystem<Problem>::gasInternalEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<Problem>::x1GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<Problem>::x2GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<Problem>::x3GasMomentum_index) = 0.;
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.4;
	const double dt_max = 1e-2;
	const double tmax = 0.5;
	const int max_timesteps = 5000;

	// Boundary conditions
	constexpr int nvars = RadSystem<Problem>::nvar_;
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
	QuokkaSimulation<Problem> sim(BCs_cc);

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
	std::vector<double> Tgas(nx);
	std::vector<double> Tgas_exact(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		if (with_opacity) {
			erad_exact.at(i) = (x <= chat * tmax) ? 1.0 * std::exp(- x) : 0.0;
		} else {
			erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
		}
		double erad_sim = 0.0;
		for (int g = 0; g < Physics_Traits<Problem>::nGroups; ++g) {
			erad_sim += values.at(RadSystem<Problem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		erad.at(i) = erad_sim;
		const double Egas_t = values.at(RadSystem<Problem>::gasInternalEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<Problem>::ComputeTgasFromEint(rho0, Egas_t);
		// Tgas_exact.at(i) = (x <= chat * tmax) ? std::exp(-x) * (tmax - x / c) : 0.0;
		if (with_opacity) {
			Tgas_exact.at(i) = erad_exact.at(i) * (tmax - x / c);
		} else {
			Tgas_exact.at(i) = 0.0;
		}
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		if (!with_opacity) {
			err_norm += std::abs(Tgas[i] - Tgas_exact[i]);
			sol_norm += std::abs(Tgas_exact[i]);
		}
		err_norm += std::abs(erad[i] - erad_exact[i]);
		sol_norm += std::abs(erad_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 1.0e-2;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

	if (with_opacity && enable_dust_) {
		status = 0;
	}

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 1.1);

	std::map<std::string, std::string> erad_args;
	std::map<std::string, std::string> erad_exact_args;
	erad_exact_args["label"] = "exact solution";
	erad_exact_args["linestyle"] = "-";
	erad_exact_args["color"] = "k";
	erad_args["label"] = "numerical solution";
	erad_args["linestyle"] = "--";
	erad_args["color"] = "C1";
	matplotlibcpp::plot(xs, erad_exact, erad_exact_args);
	matplotlibcpp::plot(xs, erad, erad_args);

	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("FUV energy density (dimensionless)");
	matplotlibcpp::ylim(-0.1, 1.1);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("photoelectric heating at t = {:.1f}", sim.tNew_[0]));
	matplotlibcpp::save("./pe_Erad.pdf");

	// Plot temperature
	matplotlibcpp::clf();
	// matplotlibcpp::ylim(0.0, 1.1);

	std::map<std::string, std::string> temp_args;
	std::map<std::string, std::string> temp_exact_args;
	temp_exact_args["label"] = "exact solution";
	temp_exact_args["linestyle"] = "-";
	temp_exact_args["color"] = "k";
	temp_args["label"] = "numerical solution";
	temp_args["linestyle"] = "--";
	temp_args["color"] = "C1";
	matplotlibcpp::plot(xs, Tgas_exact, temp_exact_args);
	matplotlibcpp::plot(xs, Tgas, temp_args);

	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("T (dimensionless)");
	matplotlibcpp::ylim(-0.1, 1.1);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("photoelectric heating at t = {:.1f}", sim.tNew_[0]));
	matplotlibcpp::save("./pe_temperature.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}
