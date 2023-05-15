//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radhydro_wave.cpp
/// \brief Defines a test problem for a linear radiation-hydro wave.
///

#include <complex>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "EOS.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_radhydro_wave.hpp"

struct WaveProblem {
};

template <> struct Physics_Traits<WaveProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};

template <> struct quokka::EOS_Traits<WaveProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = 1.0; // dimensionless
	static constexpr double boltzmann_constant = 1.0;    // dimensionless
};

constexpr double rho0 = 1.0;   // background density
constexpr double v0 = 0.;      // background velocity
constexpr double T0 = 1.0;     // background pressure
constexpr double Erad0 = 1.0;  // background radiation energy density
constexpr double Frad0 = 0.;   // background radiation flux
constexpr double amp = 1.0e-6; // perturbation amplitude

// set the dimensionless speed of light

// set the ratio of radiation pressure to thermal pressure

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real x_L = prob_lo[0] + (i + static_cast<amrex::Real>(0.0)) * dx[0];
	const amrex::Real x_R = prob_lo[0] + (i + static_cast<amrex::Real>(1.0)) * dx[0];
	const amrex::Real A = amp;

	// right eigenvector of radiation wave
	const quokka::valarray<std::complex<double>, 5> R = {-6.04108e-10 + 4.60307e-9 * 1j, 9.43323e-7 + 0.0000261512 * 1j, 0.0290768 + 0.093306 * 1j,
							     0.853528, 0.511764 - 0.00603545 * 1j};
	const quokka::valarray<std::complex<double>, 5> P_0 = {rho0, v0, T0, Erad0, Frad0};
	const quokka::valarray<std::complex<double>, 5> dU = (A * R / (2.0 * M_PI * dx[0])) * (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

	double rho = P_0[0] + dU[0];
	double vx = P_0[1] + dU[1];
	double Tgas = P_0[2] + dU[2];

	double Eint = quokka::EOS<WaveProblem>::ComputeEintFromTgas(rho, Tgas);

	// gas vars
	state(i, j, k, HydroSystem<WaveProblem>::density_index) = rho;
	state(i, j, k, HydroSystem<WaveProblem>::x1Momentum_index) = rho * vx;
	state(i, j, k, HydroSystem<WaveProblem>::x2Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::x3Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::energy_index) = Eint + 0.5 * rho * vx;
	state(i, j, k, HydroSystem<WaveProblem>::internalEnergy_index) = Eint;
	
	// rad vars
	state(i, j, k, RadSystem<WaveProblem>::radEnergy_index) = NAN;
	state(i, j, k, RadSystem<WaveProblem>::x1RadFlux_index) = NAN;
	state(i, j, k, RadSystem<WaveProblem>::x2RadFlux_index) = 0;
	state(i, j, k, RadSystem<WaveProblem>::x3RadFlux_index) = 0;
}

template <> void RadhydroSimulation<WaveProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused components with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo);
	});
}

auto problem_main() -> int
{
	// Problem initialization
	const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}
	RadhydroSimulation<WaveProblem> sim(BCs_cc);

	// set initial conditions
	sim.setInitialConditions();
	auto [pos_exact, val_exact] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);

	// Main time loop
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	int nx = static_cast<int>(position.size());
	std::vector<double> xs = position;

	// compute error norm
	amrex::Real err_sq = 0.;
	for (int n = 0; n < RadhydroSimulation<WaveProblem>::ncompHydro_; ++n) {
		if (n == HydroSystem<WaveProblem>::internalEnergy_index) {
			continue;
		}
		amrex::Real dU_k = 0.;
		for (int i = 0; i < nx; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = val_exact.at(n)[i];
			const amrex::Real U_k1 = values.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx);
		}
		// ε = || Δ U || = [&sum_k (Δ Uk)2]^{1/2}
		err_sq += dU_k * dU_k;
	}
	const amrex::Real epsilon = std::sqrt(err_sq);
	amrex::Print() << "rms of component-wise L1 error norms = " << epsilon << std::endl;

#ifdef HAVE_PYTHON
	// plot results
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// extract values
		std::vector<double> d(nx);
		std::vector<double> vx(nx);
		std::vector<double> T(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real rho = values.at(HydroSystem<WaveProblem>::density_index)[i];
			amrex::Real xmom = values.at(HydroSystem<WaveProblem>::x1Momentum_index)[i];
			amrex::Real Egas = values.at(HydroSystem<WaveProblem>::energy_index)[i];

			amrex::Real xvel = xmom / rho;
			amrex::Real Eint = Egas - xmom * xmom / (2.0 * rho);
			amrex::Real Tgas = quokka::EOS<WaveProblem>::ComputeTgasFromEint(rho, Eint);

			d.at(i) = (rho - rho0) / amp;
			vx.at(i) = (xvel - v0) / amp;
			T.at(i) = (Tgas - T0) / amp;
		}

		std::vector<double> density_exact(nx);
		std::vector<double> velocity_exact(nx);
		std::vector<double> temperature_exact(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real rho = val_exact.at(HydroSystem<WaveProblem>::density_index)[i];
			amrex::Real xmom = val_exact.at(HydroSystem<WaveProblem>::x1Momentum_index)[i];
			amrex::Real Egas = val_exact.at(HydroSystem<WaveProblem>::energy_index)[i];

			amrex::Real xvel = xmom / rho;
			amrex::Real Eint = Egas - xmom * xmom / (2.0 * rho);
			amrex::Real Tgas = quokka::EOS<WaveProblem>::ComputeTgasFromEint(rho, Eint);

			density_exact.at(i) = (rho - rho0) / amp;
			velocity_exact.at(i) = (xvel - v0) / amp;
			temperature_exact.at(i) = (Tgas - T0) / amp;
		}

		// Plot results
		amrex::Real const t = sim.tNew_[0];

		std::map<std::string, std::string> d_args;
		std::map<std::string, std::string> dinit_args;
		std::map<std::string, std::string> dexact_args;
		d_args["label"] = "density";
		dinit_args["label"] = "density (initial)";

		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, d, d_args);
		matplotlibcpp::plot(xs, density_exact, dinit_args);
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("t = {:.4f}", t));
		matplotlibcpp::save(fmt::format("./radhydro_density_{:.4f}.pdf", t));

		std::map<std::string, std::string> T_args;
		std::map<std::string, std::string> Tinit_args;
		std::map<std::string, std::string> Texact_args;
		T_args["label"] = "temperature";
		Tinit_args["label"] = "temperature (initial)";

		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, T, T_args);
		matplotlibcpp::plot(xs, temperature_exact, Tinit_args);
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("t = {:.4f}", t));
		matplotlibcpp::save(fmt::format("./radhydro_temperature_{:.4f}.pdf", t));

		std::map<std::string, std::string> v_args;
		std::map<std::string, std::string> vinit_args;
		std::map<std::string, std::string> vexact_args;
		v_args["label"] = "velocity";
		vinit_args["label"] = "velocity (initial)";

		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, vx, v_args);
		matplotlibcpp::plot(xs, velocity_exact, vinit_args);
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("t = {:.4f}", t));
		matplotlibcpp::save(fmt::format("./radhydro_velocity_{:.4f}.pdf", t));
	}
#endif

	const double err_tol = 1.0e-8; // for Nx = 100
	int status = 0;
	if (epsilon > err_tol) {
		status = 1;
	}

	return status;
}
