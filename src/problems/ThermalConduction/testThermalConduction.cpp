//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConduction.cpp
/// \brief Defines a test problem for thermal conduction.
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <cmath>
#include <fstream>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/richardson.hpp"

/** Thermal conduction test problem
The problem set up tests two types of conduction- spitzer and constant, both of which use different test problems.
These test problems have their own initial condition and analytic reference solutions. The conduction types are:
  - "constant": kappa = const. Initial condition is a smooth Gaussian temperature profile and so is the reference solution
	which has a diffusion constant that increases with time. The problem has also been set up to test AMR.
  - "spitzer": kappa = kappa0*T^2.5. Initial condition is the Pattle (1959) self-similar
    solution evaluated at t=spitzer_t_start_frac*stopTime_. The reference solution is the same Pattle profile evaluated at
    t=tNew_[0]+spitzer_t_start_frac*stopTime_. This test estimates the error across different resolutions and compares the slopes against
	unity. Most of the error comes from around the edges of the smooth solution which drop to 0 at a certain radius.
Physical parameters for the test problem are chosen to satisfy t_hydro / t_conduction >> 1, so that the gas does not have time to move
and the energy evolution is purely due to conduction. */

constexpr double Eint0 = 2.505e-8; // "constant": Gaussian peak. "spitzer": peak at the reference resolution nx_ref (both equivalent to T = 2.e8 K)
constexpr double Efloor = 5.674216387016754e-11; // equivalent to T = 2.e6 K
const double rho0 = 0.1;			 // 1/cm^3
constexpr double Lref = 7.714e+17;		 // quarter box length, fixes region of refinement
constexpr double sigma = 2.410685615625e+17;	 // "constant" only: width of the initial Gaussian, in cm (amr2-branch value)
constexpr double D = 4.396303164750053e+28;	 // "constant" only: fixed diffusion coefficient for the Gaussian solution, in cm^2/s (amr2-branch value)
constexpr int nx_ref = 128; // "spitzer" only: resolution at which Eint0 is the deposited peak value (matches inputs/ThermalConduction.toml)
constexpr double dx0_ref = 4.0 * Lref / nx_ref;
constexpr double M0 = (Eint0 - Efloor) * 2.0 * dx0_ref; // Normalization
constexpr double spitzer_t_start_frac = 0.5;
struct ThermalConductionProblem {};

template <> struct quokka::EOS_Traits<ThermalConductionProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
};

namespace
{
// Note that even in 3D the reference solution is for dimension =1 because of the set up
constexpr amrex::Real pattle_q = 2.5; // conductivity exponent: kappa(T) = kappa0 * T^pattle_q (2.5 for Spitzer)
struct ExactSolutionParams {
	bool isSpitzer = false;
	amrex::Real sigma2_t = 0.0; // "constant" only
	amrex::Real A = 0.0;	    // "spitzer" only: dEint/dT for this EOS
	amrex::Real r1 = 0.0;	    // "spitzer" only: front position at time t
	amrex::Real Tscale = 0.0;   // "spitzer" only: amplitude scale at time t
};

auto computeExactSolutionParams(bool isSpitzer, amrex::Real rho, amrex::Real kappa0, amrex::Real t) -> ExactSolutionParams
{
	ExactSolutionParams p;
	p.isSpitzer = isSpitzer;
	if (isSpitzer) {
		const amrex::Real A = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, 1.0); // A = mu * mp/rho/kb
		const amrex::Real D0 = kappa0 / A;							    // D(T) = D0 * T^pattle_q
		const amrex::Real Q0 = M0 / A;
		const amrex::Real Gamma_num = std::tgamma(1.0 / pattle_q + 1.5);
		const amrex::Real Gamma_den = std::tgamma(1.0 / pattle_q + 1.0);
		const amrex::Real r0 = (Q0 / std::sqrt(M_PI)) * Gamma_num / Gamma_den;
		const amrex::Real t0 = pattle_q * r0 * r0 / (2.0 * (pattle_q + 2.0) * D0);
		p.A = A;
		p.r1 = r0 * std::pow(t / t0, 1.0 / (pattle_q + 2.0));
		p.Tscale = std::pow(t / t0, -1.0 / (pattle_q + 2.0));
	} else {
		// Exact Gaussian diffusion solution
		p.sigma2_t = sigma * sigma + 2.0 * D * t;
	}
	return p;
}

AMREX_GPU_HOST_DEVICE auto evalExactEint(ExactSolutionParams const &p, amrex::Real rho, amrex::Real xlow, amrex::Real xhigh, amrex::Real dx) -> amrex::Real
{
	amrex::Real Eint = Efloor;
	if (p.isSpitzer) {
		// Pattle solution: zero-background self-similar profile, compactly supported within |x| <= r1.
		// Efloor is only a numerical representability floor outside the front, not part of the analytic solution.
		const amrex::Real x = 0.5 * (xlow + xhigh);
		if (std::abs(x) <= p.r1) {
			const amrex::Real base = 1.0 - (x / p.r1) * (x / p.r1);
			const amrex::Real T = std::pow(base, 1.0 / pattle_q) * p.Tscale;
			Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T);
		}
	} else {
		// Gaussian temperature profile
		const amrex::Real erfx_low = std::erf(xlow / std::sqrt(2.0 * p.sigma2_t));
		const amrex::Real erfx_high = std::erf(xhigh / std::sqrt(2.0 * p.sigma2_t));
		Eint += Eint0 * (sigma * std::sqrt(M_PI / 2.0)) * (erfx_high - erfx_low) / dx;
	}
	return Eint;
}
} // namespace

template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const bool isSpitzer = (conductionType_ == "spitzer");
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3

	const ExactSolutionParams params =
	    computeExactSolutionParams(isSpitzer, rho, electronConductionKappa0_, isSpitzer ? spitzer_t_start_frac * stopTime_ : 0.0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real xlow = prob_lo[0] + i * dx[0];
		const amrex::Real xhigh = prob_lo[0] + (i + 1) * dx[0];
		const amrex::Real Eint = evalExactEint(params, rho, xlow, xhigh, dx[0]);

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<ThermalConductionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for testing AMR on Gaussian problem
	const double refine_Lmax = Lref;

	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real y0 = 0.0;
		amrex::Real y1 = 1.0;
		amrex::Real z0 = 0.0;
		amrex::Real z1 = 1.0;

#if AMREX_SPACEDIM >= 2
		y0 = prob_lo[1] + (j * dx[1]);
		y1 = prob_lo[1] + ((j + 1) * dx[1]);
#endif
#if AMREX_SPACEDIM == 3
		z0 = prob_lo[2] + (k * dx[2]);
		z1 = prob_lo[2] + ((k + 1) * dx[2]);
#endif

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			bool const in_region = (std::abs(x) < refine_Lmax);

			amrex::ignore_unused(y, z);

			if (in_region) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};

		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfPointInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <>
void QuokkaSimulation<ThermalConductionProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real t = tNew_[0];
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3
	const bool isSpitzer = (conductionType_ == "spitzer");

	const ExactSolutionParams params =
	    computeExactSolutionParams(isSpitzer, rho, electronConductionKappa0_, isSpitzer ? (t + spitzer_t_start_frac * stopTime_) : t);

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const xlow = prob_lo[0] + i * dx[0];
			amrex::Real const xhigh = prob_lo[0] + (i + 1) * dx[0];
			amrex::Real const Eint_exact = evalExactEint(params, rho, xlow, xhigh, dx[0]);

			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index) = 0.0;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = 0.;
		});
	}
}

auto runConductionTest(int nx, int /*ny*/, int /*nz*/, int max_level = 0) -> double
{
	double max_time = 0.0;

	amrex::ParmParse pp_root;
	pp_root.query("stop_time", max_time);

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, nx, nx};
	pp.add("max_level", max_level);
	pp.addarr("n_cell", ncells);
	if (max_level > 0) {
		pp_root.add("amr_interpolation_method", 3);
	}

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {-1.5428e18, -1.5428e18, -1.5428e18};
	amrex::Vector<double> const prob_hi = {1.5428e+18, 1.5428e+18, 1.5428e+18};
	amrex::Vector<int> const is_periodic = {0, 0, 0};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	// Setup boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);

	sim.cflNumber_ = 0.3;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();
	return sim.computeErrorNorm();
}

auto problem_main() -> int
{
	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}
	// Problem initialization
	const QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);

	bool passed = false;

	if (sim.conductionType_ == "spitzer") {
		amrex::Vector<int> const resolutions = {32, 64, 128};
		amrex::Vector<double> errors;
		for (int nx : resolutions) {
			double const error = runConductionTest(nx, nx, nx);
			errors.push_back(error);
			amrex::Print() << std::format("nx = {:4d}  error norm = {:.6e}\n", nx, error);
		}

		// Best-fit slope of log(error) vs log(Nx) via ordinary least squares.
		double sum_x = 0.0;
		double sum_y = 0.0;
		double sum_xx = 0.0;
		double sum_xy = 0.0;
		int const n = static_cast<int>(resolutions.size());
		for (int i = 0; i < n; ++i) {
			double const log_nx = std::log(static_cast<double>(resolutions[i]));
			double const log_err = std::log(errors[i]);
			sum_x += log_nx;
			sum_y += log_err;
			sum_xx += log_nx * log_nx;
			sum_xy += log_nx * log_err;
		}
		double const mean_x = sum_x / n;
		double const mean_y = sum_y / n;
		double const slope = (sum_xy - n * mean_x * mean_y) / (sum_xx - n * mean_x * mean_x);
		double const intercept = mean_y - slope * mean_x;
		amrex::Print() << std::format("\nBest-fit line: log(error) = {:.4f} * log(Nx) + {:.4f}\n", slope, intercept);

		// error ~ Nx^slope for Pattle IC
		amrex::Print() << std::format("Spitzer conduction convergence: slope = {:.4f} (-1 expected, converging faster is fine)\n", slope);
		passed = slope <= -0.9;
	} else if (sim.conductionType_ == "constant") {
		// Single-resolution check against the full resolution study
		constexpr int nx = 32;
		constexpr int max_level = 1;
		double const error_norm = runConductionTest(nx, nx, nx, max_level);
		constexpr amrex::Real estimated_error = (AMREX_SPACEDIM == 1) ? 9.2430e-04 : 1.0318e-03;
		amrex::Real const delta = std::abs(error_norm - estimated_error) / estimated_error;

		amrex::Print() << std::format("nx = {:4d}  error norm = {:.6e} (expected = {:.6e})\n", nx, error_norm, estimated_error);
		passed = (delta <= 1.e-04 || error_norm < estimated_error);
	} else {
		amrex::Print() << "\nconduction.conduction_type must be \"spitzer\" or \"constant\"\n";
		return 1;
	}

	if (passed) {
		amrex::Print() << "\n✓ Thermal conduction test PASSED\n";
		return 0;
	}
	amrex::Print() << "\n✗ Thermal conduction test FAILED\n";
	return 1;
}
