//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConductionPattle.cpp
/// \brief Defines a test problem for Spitzer thermal conduction (kappa = kappa0*T^2.5) with a Pattle IC.
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include <cmath>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

/** Spitzer thermal conduction test problem  with Pattle IC
kappa = kappa0*T^2.5. Initial condition is the Pattle (1959) self-similar solution evaluated at
t = spitzer_t_start. The reference solution is the same Pattle profile evaluated at t = tNew_[0] + spitzer_t_start.
This test estimates the error across different resolutions and compares the convergence slope against unity.
Most of the error comes from around the edges of the smooth solution, which drop to 0 at a certain radius.
Physical parameters for the test problem are chosen to satisfy t_hydro / t_conduction >> 1, so that the gas does
not have time to move and the energy evolution is purely due to conduction. */

constexpr double Eint0 = 2.505e-8;   // peak Eint at the reference resolution nx_ref (equivalent to T = 2.e8 K)
constexpr double Efloor = 2.505e-11; // numerical representability floor outside the front, equivalent to T = 2.e6 K
const double rho0 = 0.1;	     // 1/cm^3
constexpr double Lref = 7.714e+17;   // quarter box length, fixes region of refinement
constexpr int nx_ref = 128;	     // resolution at which Eint0 is the deposited peak value (matches inputs/ThermalConductionPattle.toml)
constexpr double dx0_ref = 4.0 * Lref / nx_ref;
constexpr double M0 = (Eint0 - Efloor) * 2.0 * dx0_ref; // Normalization
constexpr double spitzer_t_start = 330471.1321990738;	// initial time at which the IC/reference Pattle solution is evaluated
constexpr amrex::Real pattle_q = 2.5;			// conductivity exponent: kappa(T) = kappa0 * T^pattle_q (2.5 for Spitzer)
struct ThermalConductionPattleProblem {};

template <> struct quokka::EOS_Traits<ThermalConductionPattleProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionPattleProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionPattleProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
};

namespace
{
// Estimate the Pattle (1959) self-similar solution for Eint in the cell [xlow, xhigh] at time t.
// Note that even in 3D the reference solution is for dimension = 1 because of the problem set up.
AMREX_GPU_HOST_DEVICE auto computePattleSolution(amrex::Real rho, amrex::Real kappa0, amrex::Real t, amrex::Real xlow, amrex::Real xhigh) -> amrex::Real
{
	const amrex::Real A = quokka::EOS<ThermalConductionPattleProblem>::ComputeEintFromTgas(rho, 1.0); // A = mu * mp/rho/kb
	const amrex::Real D0 = kappa0 / A;								  // D(T) = D0 * T^pattle_q
	const amrex::Real Q0 = M0 / A;
	const amrex::Real Gamma_num = std::tgamma(1.0 / pattle_q + 1.5);
	const amrex::Real Gamma_den = std::tgamma(1.0 / pattle_q + 1.0);
	const amrex::Real r0 = (Q0 / std::sqrt(M_PI)) * Gamma_num / Gamma_den;
	const amrex::Real t0 = pattle_q * r0 * r0 / (2.0 * (pattle_q + 2.0) * D0);
	const amrex::Real r1 = r0 * std::pow(t / t0, 1.0 / (pattle_q + 2.0));
	const amrex::Real Tscale = std::pow(t / t0, -1.0 / (pattle_q + 2.0));

	// Pattle solution: zero-background self-similar profile, compactly supported within |x| <= r1.
	// Efloor is only a numerical representability floor outside the front, not part of the analytic solution.
	amrex::Real Eint = Efloor;
	const amrex::Real x = 0.5 * (xlow + xhigh);
	if (std::abs(x) <= r1) {
		const amrex::Real base = 1.0 - (x / r1) * (x / r1);
		const amrex::Real T = std::pow(base, 1.0 / pattle_q) * Tscale;
		Eint = quokka::EOS<ThermalConductionPattleProblem>::ComputeEintFromTgas(rho, T);
	}
	return Eint;
}
} // namespace

template <> void QuokkaSimulation<ThermalConductionPattleProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3
	const amrex::Real kappa0 = electronConductionKappa0_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real xlow = prob_lo[0] + i * dx[0];
		const amrex::Real xhigh = prob_lo[0] + (i + 1) * dx[0];
		const amrex::Real Eint = computePattleSolution(rho, kappa0, spitzer_t_start, xlow, xhigh);

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionPattleProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionPattleProblem>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<ThermalConductionPattleProblem>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<ThermalConductionPattleProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for testing AMR near the Pattle front
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
void QuokkaSimulation<ThermalConductionPattleProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
										amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3
	const amrex::Real kappa0 = electronConductionKappa0_;
	const amrex::Real t = tNew_[0] + spitzer_t_start;

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const xlow = prob_lo[0] + i * dx[0];
			amrex::Real const xhigh = prob_lo[0] + (i + 1) * dx[0];
			amrex::Real const Eint_exact = computePattleSolution(rho, kappa0, t, xlow, xhigh);

			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			stateExact(i, j, k, HydroSystem<ThermalConductionPattleProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ThermalConductionPattleProblem>::energy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionPattleProblem>::internalEnergy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionPattleProblem>::x1Momentum_index) = 0.0;
			stateExact(i, j, k, HydroSystem<ThermalConductionPattleProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ThermalConductionPattleProblem>::x3Momentum_index) = 0.;
		});
	}
	amrex::Gpu::streamSynchronize();
}

auto runConductionTest(int nx) -> double
{
	constexpr double max_time = 660942.2643981476;
	constexpr int max_level = 0;

	// Set grid dimensions using AMReX parameter system (ny = nz = 8 for 3D)
	amrex::ParmParse pp("amr");
#if AMREX_SPACEDIM == 3
	amrex::Vector<int> const ncells = {nx, 8, 8};
	pp.add("blocking_factor_y", 8);
	pp.add("blocking_factor_z", 8);
#else
	amrex::Vector<int> const ncells = {nx, nx, nx};
#endif
	pp.add("max_level", max_level);
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {-1.5428e18, -1.5428e18, -1.5428e18};
	amrex::Vector<double> const prob_hi = {1.5428e+18, 1.5428e+18, 1.5428e+18};
	amrex::Vector<int> const is_periodic = {0, 0, 0};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	// Setup boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionPattleProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionPattleProblem> sim(BCs_cc);

	sim.cflNumber_ = 0.3;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();
	return sim.computeErrorNorm();
}

template <>
void QuokkaSimulation<ThermalConductionPattleProblem>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
									 amrex::MultiFab const &state_cc,
									 amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<ThermalConductionPattleProblem>::density_index);
				Real const Eint = HydroSystem<ThermalConductionPattleProblem>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				Real const Tgas = quokka::EOS<ThermalConductionPattleProblem>::ComputeTgasFromEint(rho, Eint);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	amrex::Vector<int> const resolutions = {32, 64, 128};
	amrex::Vector<double> errors;
	for (int nx : resolutions) {
		double const error = runConductionTest(nx);
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

	constexpr double expectedRate = 1.0;
	constexpr double tolerance = 0.0;
	constexpr double passThreshold = -(expectedRate - tolerance);
	amrex::Print() << std::format(
	    "Spitzer+Pattle conduction convergence: slope = {:.4f} ({:.1f} expected, converging faster is fine, pass threshold = {:.4f})\n", slope,
	    -expectedRate, passThreshold);
	bool const passed = slope <= passThreshold;

	if (passed) {
		amrex::Print() << "\n✓ Thermal conduction (spitzer, Pattle) test PASSED\n";
		return 0;
	}
	amrex::Print() << "\n✗ Thermal conduction (spitzer, Pattle) test FAILED\n";
	return 1;
}
