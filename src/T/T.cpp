//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_force.cpp
/// \brief Defines a test problem for radiation force terms.
///

#include <cstdint>
#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_REAL.H"

#include "ArrayUtil.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "fmt/format.h"
#include "fundamental_constants.H"
#include "hydro_system.hpp"
#include "interpolate.hpp"
#include "physics_info.hpp"
#include "radiation_system.hpp"
#include "T.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct TheProblem {
};

constexpr double kappa0 = 1.0;	     // cm^2 g^-1
constexpr double mu = 2.33 * C::m_u; // g
constexpr double k_B = C::k_B;
constexpr double gamma_gas = 5. / 3.;    // isothermal gas EOS
constexpr double a_rad = C::a_rad;
constexpr double T0 = 1.0e7; // K (temperature)
constexpr double T1 = 2.0e7; // K (temperature)
constexpr double Erad0 = a_rad * T0 * T0 * T0 * T0; // erg cm^-3
constexpr double erad_floor = 1.0e-20 * Erad0;

// constexpr double rho0 = 1.2;
constexpr double rho0 = 1.0e-20;
constexpr double Tgas0 = 1.0e-20;
constexpr double c = C::c_light; // speed of light (cgs)
constexpr double chat = c;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gamma = gamma_gas;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	// number of radiation groups
	static constexpr int nGroups = 1;
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/)
    -> quokka::valarray<double, Physics_Traits<TheProblem>::nGroups>
{
	quokka::valarray<double, Physics_Traits<TheProblem>::nGroups> kappaPVec{};
	for (int g = 0; g < nGroups_; ++g) {
		kappaPVec[g] = kappa0;
	}
	return kappaPVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/)
    -> quokka::valarray<double, Physics_Traits<TheProblem>::nGroups>
{
	quokka::valarray<double, Physics_Traits<TheProblem>::nGroups> kappaFVec{};
	for (int g = 0; g < nGroups_; ++g) {
		kappaFVec[g] = kappa0;
	}
	return kappaFVec;
}

template <> void RadhydroSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// // extract variables required from the geom object
	// const amrex::Box &indexRange = grid_elem.indexRange_;
	// const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
  amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
    auto const r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));

		double rho = 0.0;
		double Trad = 0.0;
		if (r < 0.2) {
			rho = rho0;
			Trad = T0;
		} else {
			rho = 0.1 * rho0;
			Trad = 0.1 * T0;
		}
		// Trad = T0;
		// const double Egas = quokka::EOS<TheProblem>::ComputeEintFromTgas(rho0, T0);
		const double Egas = quokka::EOS<TheProblem>::ComputeEintFromTgas(rho0, Tgas0);
		const double Erad = a_rad * std::pow(Trad, 4);

		// for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
			int g = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad;
			state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		// }

		state_cc(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 128;
	constexpr double CFL_number = 0.4;
	double max_dt = 1.0e10;
	double tmax = 1.0e-8;
	int max_timesteps = 10;

	// Boundary conditions
	constexpr int nvars = RadSystem<TheProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// periodic
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Read max_dt from parameter file
	amrex::ParmParse const pp;
	pp.query("max_dt", max_dt);
	pp.query("tmax", tmax);
	pp.query("max_timesteps", max_timesteps);

	// Problem initialization
	RadhydroSimulation<TheProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.reconstructionOrder_ = 3;	       // PPM
	sim.stopTime_ = tmax;
	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	// sim.plotfileInterval_ = -1;
	sim.maxDt_ = max_dt;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	auto [position2y, values2y] = fextract(sim.state_new_cc_[0], sim.Geom(0), 1, 0.0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> rho_arr(nx);
	std::vector<double> v_arr(nx);
	std::vector<double> T_arr(nx);
	std::vector<double> P_arr(nx);
	std::vector<double> Erad_arr(nx);

	for (int i = 0; i < nx; ++i) {
		xs.at(i) = position[i];

		double const rho = values.at(RadSystem<TheProblem>::gasDensity_index)[i];
		double const x1GasMom = values.at(RadSystem<TheProblem>::x1GasMomentum_index)[i];
		double const vx = x1GasMom / rho;
		double const Egas = values.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
		double const Tgas = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Egas);
		double const Erad = values.at(RadSystem<TheProblem>::radEnergy_index)[i];

		rho_arr.at(i) = rho;
		v_arr.at(i) = vx;
		T_arr.at(i) = Tgas;
		P_arr.at(i) = quokka::EOS<TheProblem>::ComputePressure(rho, Egas);
		// P_arr.at(i) = 2. / 3. * Egas;
		Erad_arr.at(i) = Erad;
	}

	// compute error norm
	const int ny = static_cast<int>(position2y.size());
	std::vector<double> xsy(ny);
	std::vector<double> rho_arry(ny);
	std::vector<double> v_arry(ny);
	std::vector<double> T_arry(ny);
	std::vector<double> P_arry(ny);
	std::vector<double> Erad_arry(ny);

	for (int i = 0; i < ny; ++i) {
		xsy.at(i) = position2y[i];

		double const rho = values2y.at(RadSystem<TheProblem>::gasDensity_index)[i];
		double const x1GasMom = values2y.at(RadSystem<TheProblem>::x1GasMomentum_index)[i];
		double const vx = x1GasMom / rho;
		double const Egas = values2y.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
		double const Tgas = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Egas);
		double const Erad = values2y.at(RadSystem<TheProblem>::radEnergy_index)[i];

		rho_arry.at(i) = rho;
		v_arry.at(i) = vx;
		T_arry.at(i) = Tgas;
		P_arry.at(i) = quokka::EOS<TheProblem>::ComputePressure(rho, Egas);
		// P_arry.at(i) = 2. / 3. * Egas;
		Erad_arry.at(i) = Erad;
	}

#ifdef HAVE_PYTHON
	// plot velocity
	int const s = nx / 64; // stride
	std::map<std::string, std::string> args;
	args["label"] = "vx";
	args["color"] = "C0";
	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, v_arr, args);
	args["label"] = "vy";
	args["linestyle"] = "--";
	args["color"] = "C1";
	matplotlibcpp::plot(xsy, v_arry, args);
	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("vx");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./T_vel_t{:.4g}.pdf", sim.tNew_[0]));

	// plot density
	matplotlibcpp::clf();
	args["label"] = "rho along x";
	args["linestyle"] = "-";
	args["color"] = "C0";
	matplotlibcpp::plot(xs, rho_arr, args);
	args["label"] = "rho along y";
	args["linestyle"] = "--";
	args["color"] = "C1";
	matplotlibcpp::plot(xsy, rho_arry, args);
	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("rho");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./T_rho_t{:.4g}.pdf", sim.tNew_[0]));

	// plot temperature
	matplotlibcpp::clf();
	args["label"] = "T along x";
	args["linestyle"] = "-";
	args["color"] = "C0";
	matplotlibcpp::plot(xs, T_arr, args);
	args["label"] = "T along y";
	args["linestyle"] = "--";
	args["color"] = "C1";
	matplotlibcpp::plot(xsy, T_arry, args);
	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("T");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./T_T_t{:.4g}.pdf", sim.tNew_[0]));
	
	// plot pressure
	matplotlibcpp::clf();
	args["label"] = "P along x";
	args["linestyle"] = "-";
	args["color"] = "C0";
	matplotlibcpp::plot(xs, P_arr, args);
	args["label"] = "P along y";
	args["linestyle"] = "--";
	args["color"] = "C1";
	matplotlibcpp::plot(xsy, P_arry, args);
	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("P");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./T_P_t{:.4g}.pdf", sim.tNew_[0]));
	
	// plot Erad
	matplotlibcpp::clf();
	args["label"] = "Erad along x";
	args["linestyle"] = "-";
	args["color"] = "C0";
	matplotlibcpp::plot(xs, Erad_arr, args);
	args["label"] = "Erad along y";
	args["linestyle"] = "--";
	args["color"] = "C1";
	matplotlibcpp::plot(xsy, Erad_arry, args);
	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("Erad");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./T_Erad_t{:.4g}.pdf", sim.tNew_[0]));

#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
