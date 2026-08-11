//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroWaveConvergence.cpp
/// \brief Defines a Richardson convergence test for the linear hydro wave.
///

#include "hydro/hydro_system.hpp"
#include <format>
#include <limits>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include "util/richardson.hpp"

struct WaveProblem {
};

template <> struct quokka::EOS_Traits<WaveProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<WaveProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr ViscosityModel viscosity_model = ViscosityModel::constant; // shear/bulk default to 0; no-op unless set
};

constexpr double rho0 = 1.0;					    // background density
constexpr double P0 = 1.0 / quokka::EOS_Traits<WaveProblem>::gamma; // background pressure
constexpr double v0 = 0.;					    // background velocity
constexpr double amp = 1.0e-6;					    // perturbation amplitude

// viscous decay rate for the wave analytic solution: (4/3*shear + bulk)*k^2/(2*rho0), for the single
// hardcoded mode k=2*pi below; zero (no decay) unless hydro.shear_viscosity/hydro.bulk_viscosity are set
AMREX_GPU_MANAGED double viscous_decay_rate = 0.0; // NOLINT

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real time)
{
	const amrex::Real x_L = prob_lo[0] + (i + static_cast<amrex::Real>(0.0)) * dx[0];
	const amrex::Real x_R = prob_lo[0] + (i + static_cast<amrex::Real>(1.0)) * dx[0];
	const amrex::Real A = amp * std::exp(-viscous_decay_rate * time);

	const quokka::valarray<double, 3> R = {1.0, -1.0, 1.5}; // right eigenvector of sound wave
	const quokka::valarray<double, 3> U_0 = {rho0, rho0 * v0, P0 / (quokka::EOS_Traits<WaveProblem>::gamma - 1.0) + 0.5 * rho0 * std::pow(v0, 2)};
	const quokka::valarray<double, 3> dU = (A * R / (2.0 * M_PI * dx[0])) * (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

	double const rho = U_0[0] + dU[0];
	double const xmom = U_0[1] + dU[1];
	double const Etot = U_0[2] + dU[2];
	double const Eint = Etot - 0.5 * (xmom * xmom) / rho;

	state(i, j, k, HydroSystem<WaveProblem>::density_index) = rho;
	state(i, j, k, HydroSystem<WaveProblem>::x1Momentum_index) = xmom;
	state(i, j, k, HydroSystem<WaveProblem>::x2Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::x3Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::energy_index) = Etot;
	state(i, j, k, HydroSystem<WaveProblem>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<WaveProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused components with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, 0.0);
	});
}

// Sets viscous_decay_rate from hydro.shear_viscosity/hydro.bulk_viscosity. Zero (no decay) when both
// are absent, recovering the ideal wave solution. Shared by the convergence sweep (runWaveTest) and
// the fixed-resolution run_sim path.
void configureViscousParameters()
{
	double shearViscosity = 0.0;
	double bulkViscosity = 0.0;
	amrex::ParmParse const hpp("hydro");
	hpp.query("shear_viscosity", shearViscosity);
	hpp.query("bulk_viscosity", bulkViscosity);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(shearViscosity >= 0.0 && bulkViscosity >= 0.0, "hydro.shear_viscosity and hydro.bulk_viscosity must be non-negative.");
	constexpr double k_magn = 2.0 * M_PI; // single hardcoded mode, box length 1
	viscous_decay_rate = (4.0 / 3.0 * shearViscosity + bulkViscosity) * k_magn * k_magn / (2.0 * rho0);
	if (shearViscosity > 0.0 || bulkViscosity > 0.0) {
		amrex::Print() << "Hydro wave (viscous): decay_rate=" << viscous_decay_rate << "\n";
	}
}

// fills every cell of mf with the analytic wave solution at the given time
void fillWaveSolutionState(amrex::MultiFab &mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real time)
{
	const int ncomp = mf.nComp();
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				state(i, j, k, n) = 0; // fill unused components with zeros
			}
			computeWaveSolution(i, j, k, state, dx, prob_lo, time);
		});
	}
}

auto runWaveTest(int nx, int ny, int nz) -> double
{
	configureViscousParameters();

	// Problem parameters
	const double CFL_number = 0.1;
	const double max_time = 1.0;
	const int max_timesteps = std::max(20000, nx * 100);

	// Problem initialization
	const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, ny, nz};
	pp.add("max_level", 0);
	if (!pp.contains("blocking_factor")) {
		pp.add("blocking_factor", 8);
	}
	if (!pp.contains("max_grid_size")) {
		pp.add("max_grid_size", 128);
	}
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {0.0, 0.0, 0.0};
	amrex::Vector<double> const prob_hi = {1.0, 1.0, 1.0};
	amrex::Vector<int> const is_periodic = {1, 1, 1};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	QuokkaSimulation<WaveProblem> sim(BCs_cc);

	sim.cflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;

	// set initial conditions
	sim.setInitialConditions();

	// Main time loop
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	int const nx_final = static_cast<int>(position.size());

	// analytic solution at the final simulation time; equals the t=0 state only when viscous_decay_rate == 0
	amrex::MultiFab exactState(sim.boxArray(0), sim.DistributionMap(0), QuokkaSimulation<WaveProblem>::nvars_, 0);
	fillWaveSolutionState(exactState, sim.geom[0].CellSizeArray(), sim.geom[0].ProbLoArray(), sim.tNew_[0]);
	auto [pos_exact, val_exact] = fextract(exactState, sim.geom[0], 0, 0.5);

	// compute error norm
	amrex::Real err_sq = 0.;
	for (int n = 0; n < QuokkaSimulation<WaveProblem>::nvars_; ++n) {
		if (n == HydroSystem<WaveProblem>::internalEnergy_index) {
			continue;
		}
		amrex::Real dU_k = 0.;
		for (int i = 0; i < nx_final; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = val_exact.at(n)[i];
			const amrex::Real U_k1 = values.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx_final);
		}
		// ε = || Δ U || = [&sum_k (Δ Uk)2]^{1/2}
		err_sq += dU_k * dU_k;
	}
	amrex::Real epsilon = std::sqrt(err_sq);
	// fextract only gathers the full comparison to the IO processor; broadcast so every rank agrees
	amrex::ParallelDescriptor::Bcast(&epsilon, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	return epsilon;
}

auto problem_main() -> int
{
	bool run_convergence = true;
	bool run_sim = false;
	double error_tol = 1.0e-8;
	{
		amrex::ParmParse const pp("setup");
		pp.query("run_convergence", run_convergence);
		pp.query("run_sim", run_sim);
		pp.query("error_tol", error_tol);
	}

	int status = 0;

	if (run_sim) {
		configureViscousParameters();

		const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
		amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
		for (int n = 0; n < ncomp_cc; ++n) {
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			}
		}

		QuokkaSimulation<WaveProblem> sim(BCs_cc);
		sim.cflNumber_ = 0.3;
		sim.setInitialConditions();

		sim.evolve();

		auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
		const int nx_final = static_cast<int>(position.size());

		// analytic solution at the final simulation time; equals the t=0 state only when viscous_decay_rate == 0
		amrex::MultiFab exactState(sim.boxArray(0), sim.DistributionMap(0), QuokkaSimulation<WaveProblem>::nvars_, 0);
		fillWaveSolutionState(exactState, sim.geom[0].CellSizeArray(), sim.geom[0].ProbLoArray(), sim.tNew_[0]);
		auto [pos_exact, val_exact] = fextract(exactState, sim.geom[0], 0, 0.5);

		amrex::Real err_sq = 0.;
		for (int n = 0; n < QuokkaSimulation<WaveProblem>::nvars_; ++n) {
			if (n == HydroSystem<WaveProblem>::internalEnergy_index) {
				continue;
			}
			amrex::Real dU_k = 0.;
			for (int i = 0; i < nx_final; ++i) {
				const amrex::Real U_k0 = val_exact.at(n)[i];
				const amrex::Real U_k1 = values.at(n)[i];
				dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx_final);
			}
			err_sq += dU_k * dU_k;
		}
		amrex::Real epsilon = std::sqrt(err_sq);
		// fextract only gathers the full comparison to the IO processor; broadcast so every rank agrees
		amrex::ParallelDescriptor::Bcast(&epsilon, 1, amrex::ParallelDescriptor::IOProcessorNumber());

		amrex::Print() << std::format("\nrun_sim error norm = {:.6e}  (tol = {:.6e})\n", static_cast<double>(epsilon), error_tol);
		if (epsilon > error_tol) {
			status = 1;
		}
	}

	if (run_convergence) {
		quokka::richardson::applyQuietDefaults();

		quokka::richardson::Parameters params{};
		params.machine_precision_target = 2.0e-11;
		params.nx_initial = 128;
		params.nx_max = 2048;
		{
			amrex::ParmParse const pp("setup");
			pp.query("nx_start", params.nx_initial);
			pp.query("nx_max", params.nx_max);
			pp.query("machine_precision_target", params.machine_precision_target);
			pp.query("refine_n_dims", params.refine_n_dims);
		}
		params.expected_rate = 2.0;
		params.tolerance = 0.3;
		params.test_name = "Hydro Wave";
		params.csv_filename = "hydro_wave_convergence.csv";

		if (quokka::richardson::run(params, [](int nx, int ny, int nz) { return runWaveTest(nx, ny, nz); }) != 0) {
			status = 1;
		}
	}

	return status;
}
