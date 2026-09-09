//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConductionConstant.cpp
/// \brief Defines a test problem for constant-conductivity thermal conduction (kappa = const).
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

/** Constant-conductivity thermal conduction test problem
kappa = const. Initial condition is a smooth Gaussian temperature profile, and the reference solution is the
same Gaussian profile with a diffusion width that grows with time (exact linear-diffusion solution). This test
runs at a single resolution, with one level of refinement active (tests AMR), and compares the resulting error
norm against a pre-computed reference value.
Physical parameters for the test problem are chosen to satisfy t_hydro / t_conduction >> 1, so that the gas does
not have time to move and the energy evolution is purely due to conduction. */

constexpr double Eint0 = 2.505e-8;   // Gaussian peak (equivalent to T = 2.e8 K)
constexpr double Efloor = 2.505e-11; // equivalent to T = 2.e6 K
const double rho0 = 0.1;	      // 1/cm^3
constexpr double Lref = 7.714e+17;   // quarter box length, fixes region of refinement
constexpr double sigma = 2.410685615625e+17; // width of the initial Gaussian, in cm (amr2-branch value)
constexpr double D = 4.396303164750053e+28;  // fixed diffusion coefficient for the Gaussian solution, in cm^2/s (amr2-branch value)
struct ThermalConductionConstantProblem {};

template <> struct quokka::EOS_Traits<ThermalConductionConstantProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionConstantProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionConstantProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
};

namespace
{
struct ExactSolutionParams {
	amrex::Real sigma2_t = 0.0;
};

auto computeExactSolutionParams(amrex::Real t) -> ExactSolutionParams
{
	ExactSolutionParams p;
	p.sigma2_t = sigma * sigma + 2.0 * D * t;
	return p;
}

AMREX_GPU_HOST_DEVICE auto evalExactEint(ExactSolutionParams const &p, amrex::Real xlow, amrex::Real xhigh, amrex::Real dx) -> amrex::Real
{
	amrex::Real Eint = Efloor;
	const amrex::Real erfx_low = std::erf(xlow / std::sqrt(2.0 * p.sigma2_t));
	const amrex::Real erfx_high = std::erf(xhigh / std::sqrt(2.0 * p.sigma2_t));
	Eint += Eint0 * (sigma * std::sqrt(M_PI / 2.0)) * (erfx_high - erfx_low) / dx;
	return Eint;
}
} // namespace

template <> void QuokkaSimulation<ThermalConductionConstantProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3

	const ExactSolutionParams params = computeExactSolutionParams(/*t=*/0.0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real xlow = prob_lo[0] + i * dx[0];
		const amrex::Real xhigh = prob_lo[0] + (i + 1) * dx[0];
		const amrex::Real Eint = evalExactEint(params, xlow, xhigh, dx[0]);

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionConstantProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionConstantProblem>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<ThermalConductionConstantProblem>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<ThermalConductionConstantProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for testing AMR on the Gaussian problem
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
void QuokkaSimulation<ThermalConductionConstantProblem>::computeReferenceSolution(amrex::MultiFab &ref,
										   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
										   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real t = tNew_[0];
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3

	const ExactSolutionParams params = computeExactSolutionParams(t);

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const xlow = prob_lo[0] + i * dx[0];
			amrex::Real const xhigh = prob_lo[0] + (i + 1) * dx[0];
			amrex::Real const Eint_exact = evalExactEint(params, xlow, xhigh, dx[0]);

			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			stateExact(i, j, k, HydroSystem<ThermalConductionConstantProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ThermalConductionConstantProblem>::energy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionConstantProblem>::internalEnergy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionConstantProblem>::x1Momentum_index) = 0.0;
			stateExact(i, j, k, HydroSystem<ThermalConductionConstantProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ThermalConductionConstantProblem>::x3Momentum_index) = 0.;
		});
	}
	amrex::Gpu::streamSynchronize();
}

auto runConductionTest(int nx, int max_level) -> double
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
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionConstantProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionConstantProblem> sim(BCs_cc);

	sim.cflNumber_ = 0.3;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();
	return sim.computeErrorNorm();
}

template <>
void QuokkaSimulation<ThermalConductionConstantProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
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
				Real const rho = state(i, j, k, HydroSystem<ThermalConductionConstantProblem>::density_index);
				Real const Eint = HydroSystem<ThermalConductionConstantProblem>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				Real const Tgas = quokka::EOS<ThermalConductionConstantProblem>::ComputeTgasFromEint(rho, Eint);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	// Single-resolution check, with one level of refinement active, against a pre-computed reference error norm.
	constexpr int nx = 32;
	constexpr int max_level = 1;
	double const error_norm = runConductionTest(nx, max_level);
	constexpr amrex::Real estimated_error = (AMREX_SPACEDIM == 1) ? 9.2430e-04 : 1.0318e-03;
	amrex::Real const delta = std::abs(error_norm - estimated_error) / estimated_error;

	amrex::Print() << std::format("nx = {:4d}  error norm = {:.6e} (expected = {:.6e})\n", nx, error_norm, estimated_error);
	bool const passed = (delta <= 1.e-04 || error_norm < estimated_error);

	if (passed) {
		amrex::Print() << "\n✓ Thermal conduction (constant) test PASSED\n";
		return 0;
	}
	amrex::Print() << "\n✗ Thermal conduction (constant) test FAILED\n";
	return 1;
}
