//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include "test_advection2d.hpp"
#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_BoxArray.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AdvectionSimulation.hpp"
#include <csignal>
#include <limits>

struct SquareProblem {
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
exactSolutionAtIndex(int i, int j, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo,
		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi,
		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) -> amrex::Real
{
	amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
	amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
	amrex::Real const x0 = prob_lo[0] + amrex::Real(0.5) * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + amrex::Real(0.5) * (prob_hi[1] - prob_lo[1]);
	amrex::Real rho = 0.;

	if ((std::abs(x - x0) < 0.1) && (std::abs(y - y0) < 0.1)) {
		rho = 1.;
	}
	return rho;
}

template <> void AdvectionSimulation<SquareProblem>::setInitialConditions()
{
	auto prob_lo = simGeometry_.ProbLoArray();
	auto prob_hi = simGeometry_.ProbHiArray();
	auto dx = dx_;

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(
		    indexRange, ncomp_, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
			    state(i, j, k, n) = exactSolutionAtIndex(i, j, prob_lo, prob_hi, dx);
		    });
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

void ComputeExactSolution(amrex::Array4<amrex::Real> const &exact_arr, amrex::Box const &indexRange,
			  const int nvars,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_hi,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		exact_arr(i, j, k, n) = exactSolutionAtIndex(i, j, prob_lo, prob_hi, dx);
	});
}

template <> void AdvectionSimulation<SquareProblem>::computeAfterTimestep()
{
	// copy all FABs to a local FAB across the entire domain
	amrex::BoxArray localBoxes(domain_);
	amrex::DistributionMapping localDistribution(localBoxes, 1);
	amrex::MultiFab state_mf(localBoxes, localDistribution, ncomp_, 0);
	state_mf.ParallelCopy(state_new_);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &state = state_mf.array(0);
		auto const prob_lo = simGeometry_.ProbLoArray();
		auto dx = dx_;

		amrex::Long asymmetry = 0;
		auto nx = nx_[0];
		auto ny = nx_[1];
		auto nz = nx_[2];
		auto ncomp = ncomp_;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				for (int k = 0; k < nz; ++k) {
					amrex::Real const x =
					    prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
					amrex::Real const y =
					    prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
					for (int n = 0; n < ncomp; ++n) {
						const amrex::Real comp_upper = state(i, j, k, n);
						// reflect across x/y diagonal
						const amrex::Real comp_lower = state(j, i, k, n);
						const amrex::Real average =
						    std::fabs(comp_upper + comp_lower);
						const amrex::Real residual =
						    std::abs(comp_upper - comp_lower) / average;

						if (comp_upper != comp_lower) {
							amrex::Print()
							    << i << ", " << j << ", " << k << ", "
							    << n << ", " << comp_upper << ", "
							    << comp_lower << " " << residual
							    << "\n";
							amrex::Print() << "x = " << x << "\n";
							amrex::Print() << "y = " << y << "\n";
							asymmetry++;
							AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
							    false, "x/y not symmetric!");
						}
					}
				}
			}
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(asymmetry == 0, "x/y not symmetric!");
	}
}

auto problem_main() -> int
{
	// check that we are in strict IEEE 754 mode
	// (If we are, then the results should be symmetric [about the diagonal of the grid] not
	// only to machine epsilon but to every last digit! Yes, this *is* possible [in 2D].)
	AMREX_ALWAYS_ASSERT(std::numeric_limits<double>::is_iec559);

	// Problem parameters
	const int nx = 100;
	const double Lx = 1.0;
	const double advection_velocity = 1.0; // same for x- and y- directions
	const double CFL_number = 0.4;
	const double max_time = 1.0;
	const double max_dt = 1.0e-3;
	const int max_timesteps = 1e4;
	const int nvars = 1; // only density

	amrex::IntVect gridDims{AMREX_D_DECL(nx, nx, 4)};

	amrex::RealBox boxSize{{AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
			       {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Lx), amrex::Real(1.0))}};

	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
			boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	AdvectionSimulation<SquareProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.maxDt_ = max_dt;
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;

	sim.advectionVx_ = advection_velocity;
	sim.advectionVy_ = advection_velocity;

	// set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Compute reference solution (at t=1 it is equal to the initial conditions)
	amrex::MultiFab state_exact(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				    sim.nghost_);

	for (amrex::MFIter iter(sim.state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = state_exact.array(iter);
		auto const prob_lo = sim.simGeometry_.ProbLoArray();
		auto const prob_hi = sim.simGeometry_.ProbHiArray();
		auto dx = sim.dx_;
		ComputeExactSolution(stateExact, indexRange, sim.ncomp_, prob_lo, prob_hi, dx);
	}

	// Compute error norm
	amrex::MultiFab residual(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				 sim.nghost_);
	amrex::MultiFab::Copy(residual, state_exact, 0, 0, sim.ncomp_, sim.nghost_);
	amrex::MultiFab::Saxpy(residual, -1., sim.state_new_, 0, 0, sim.ncomp_, sim.nghost_);

	double min_rel_error = std::numeric_limits<double>::max();
	for (int comp = 0; comp < sim.ncomp_; ++comp) {
		const auto sol_norm = state_exact.norm1(comp);
		const auto err_norm = residual.norm1(comp);
		const double rel_error = err_norm / sol_norm;
		min_rel_error = std::min(min_rel_error, rel_error);
		amrex::Print() << "Relative L1 error norm (comp " << comp << ") = " << rel_error
			       << "\n";
	}

	const double err_tol = 0.25;
	int status = 0;
	if (min_rel_error > err_tol) {
		status = 1;
	}

	return status;
}
