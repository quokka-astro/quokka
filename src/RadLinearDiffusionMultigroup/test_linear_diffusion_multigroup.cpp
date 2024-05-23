//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///


// step 1: make it CGS unit


#include <cmath>
#include <limits>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

#include "fextract.hpp"
#include "physics_info.hpp"
#include "radiation_system.hpp"
#include "test_linear_diffusion_multigroup.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct TheProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int max_timesteps_ = 3;

// nu_0 = 1.0
// u_0 = 1.0
constexpr double a_rad = 1.0;
constexpr double c = 1.7320508075688772;  // = sqrt(3.0)
constexpr double h = c * c * c / (8.0 * M_PI); // B_0 = 8 * pi * h / c^3 = 1
constexpr double k_B = h;
constexpr double C_v = 1.0;
constexpr double mu = 3.0 / 2.0 * k_B;    // C_v = 3/2 k_B / mu = 1, so mu = 3/2 * k_B
constexpr double kappa_0 = 1.0 / 1.7320508075688772;  // = 1.0 / sqrt(3.0)

constexpr double rho0 = 1.0;
constexpr double x_max = 4.0;
constexpr double x0 = 0.5;
constexpr double t1 = 1.0;
constexpr double T0 = 1.0;
constexpr double T_f = 0.1;
constexpr double T_floor = 1.0e-20 * T0;
constexpr double Egas_floor = C_v * rho0 * T_floor;
constexpr double Erad_floor_ = 1.0e-20;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 8; // number of radiation groups
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = c;
	static constexpr double radiation_constant = a_rad;
	static constexpr double mean_molecular_mass = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = Erad_floor_;
	// static constexpr bool compute_v_over_c_terms = true;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = h;
	static constexpr amrex::GpuArray<double, Physics_Traits<TheProblem>::nGroups + 1> radBoundaries{0.0, 0.0057179440500000015, 0.017974864931786093, 0.044248663379038114, 0.10056888372676295, 0.22129627784088032, 0.4800861689243623, 1.0348252835920335, 2.2239578422629616};
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeThermalRadiation(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<double, nGroups_> B_g{};
  double nu_g = NAN;
  const double coeff = 1.0;   // = B0 * h / k_B
	for (int g = 0; g < nGroups_; ++g) {
    if (g == 0) {
      nu_g = 0.5 * RadSystem_Traits<TheProblem>::radBoundaries[1];
    } else {
      // take the geometrical mean
      nu_g = std::sqrt(RadSystem_Traits<TheProblem>::radBoundaries[g] * RadSystem_Traits<TheProblem>::radBoundaries[g + 1]);
    }
    auto tmp = coeff * std::pow(nu_g, 3);
    auto frac = std::exp(- boundaries[g] / T_f) - std::exp(- boundaries[g + 1] / T_f);
    B_g[g] = tmp * frac * temperature;
	}
	return B_g;   // = a_r * T^4 = 4 * pi * B / c
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaVec{};
  double nu_g = NAN;
	for (int g = 0; g < nGroups_; ++g) {
    if (g == 0) {
      nu_g = 0.5 * RadSystem_Traits<TheProblem>::radBoundaries[1];
    } else {
      // take the geometrical mean
      nu_g = std::sqrt(RadSystem_Traits<TheProblem>::radBoundaries[g] * RadSystem_Traits<TheProblem>::radBoundaries[g + 1]);
    }
    auto kappa = kappa_0 * std::pow(nu_g, -3);
		kappaVec[g] = kappa / rho;
	}
	return kappaVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeEnergyMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <> void RadhydroSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];

    auto Erad = Erad_floor_;
    double T = std::abs(x) > x0 ? T_floor : T0;
    auto Egas = quokka::EOS<TheProblem>::ComputeEintFromTgas(rho0, T);
    
		for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad;
			state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}

		state_cc(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters

	const int max_timesteps = max_timesteps_;
	const double CFL_number = 100.;
	const double initial_dt = 0.005;
	const double max_dt = 0.005;
	const double max_time = 1.0;

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<TheProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	// Boundary conditions
	constexpr int nvars = RadSystem<TheProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		if (isNormalComp(n, 0)) {
			BCs_cc[n].setLo(0, amrex::BCType::reflect_odd);
		} else {
			BCs_cc[n].setLo(0, amrex::BCType::reflect_even);
		}
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {	    // x2- and x3- directions
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	RadhydroSimulation<TheProblem> sim(BCs_cc);

	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.initDt_ = initial_dt;
	sim.maxDt_ = max_dt;
	sim.plotfileInterval_ = -1;

	// evolve
	sim.setInitialConditions();
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	int nx = static_cast<int>(position.size());

	// compare with exact solution
	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> Erad(nx);
		std::vector<double> Egas(nx);

		for (int i = 0; i < nx; ++i) {
			xs.at(i) = position[i];
      double Erad_t = 0.;
      for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
        Erad_t += values.at(RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
      }
      Erad.at(i) = Erad_t;
			const auto rho = values.at(RadSystem<TheProblem>::gasDensity_index)[i];
      const auto Egas_t = values.at(RadSystem<TheProblem>::gasEnergy_index)[i];
      const auto Tgas_t = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Egas_t);
			Egas.at(i) = Egas_t;
			Tgas.at(i) = Tgas_t;
		}

#ifdef HAVE_PYTHON
		// Plot solution

		matplotlibcpp::clf();

		std::map<std::string, std::string> args;
		args["label"] = "Egas";
		matplotlibcpp::plot(xs, Egas, args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x (dimensionless)");
		matplotlibcpp::ylabel("E");
		matplotlibcpp::title(fmt::format("ct = {:.4g}", c * sim.tNew_[0]));
		// matplotlibcpp::xlim(0.0, 2. * x0);
		// matplotlibcpp::ylim(0.0, 1.3);	// dimensionless
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(fmt::format("./LinearDiffusionMP_Egas_step{:d}.pdf", max_timesteps_));

    // temperature
    matplotlibcpp::clf();
    args["label"] = "gas temperature";
    matplotlibcpp::plot(xs, Tgas, args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x (dimensionless)");
		matplotlibcpp::ylabel("T");
		matplotlibcpp::title(fmt::format("ct = {:.4g}", c * sim.tNew_[0]));
		// matplotlibcpp::xlim(0.0, 2. * x0);
		// matplotlibcpp::ylim(0.0, 1.3);	// dimensionless
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(fmt::format("./LinearDiffusionMP_Tgas_step{:d}.pdf", max_timesteps_));

    matplotlibcpp::clf();
    args["label"] = "Erad";
    matplotlibcpp::plot(xs, Erad, args);
    matplotlibcpp::legend();
    matplotlibcpp::xlabel("length x (dimensionless)");
    matplotlibcpp::ylabel("Erad");
    matplotlibcpp::title(fmt::format("ct = {:.4g}", c * sim.tNew_[0]));
    // matplotlibcpp::xlim(0.0, 2. * x0);
    // matplotlibcpp::ylim(0.0, 1.3);	// dimensionless
    matplotlibcpp::tight_layout();
    matplotlibcpp::save(fmt::format("./LinearDiffusionMP_Erad_step{:d}.pdf", max_timesteps_));
#endif
	}

	return status;
}
