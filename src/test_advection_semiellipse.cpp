//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include "test_advection_semiellipse.hpp"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AdvectionSimulation.hpp"
#include "hyperbolic_system.hpp"
#include "linear_advection.hpp"
#include <vector>

auto main(int argc, char **argv) -> int
{
	// Initialization
	// amrex::Initialize(argc, argv);

	// copied from Exa-wind:
	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {
		amrex::ParmParse pp("amrex");
		// Set the defaults so that we throw an exception instead of attempting
		// to generate backtrace files. However, if the user has explicitly set
		// these options in their input files respect those settings.
		if (!pp.contains("throw_exception")) {
			pp.add("throw_exception", 1);
		}
		if (!pp.contains("signal_handling")) {
			pp.add("signal_handling", 0);
		}
	});

	int result = 0;

	{ // objects must be destroyed before amrex::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_advection();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct SawtoothProblem {
};

template <> void AdvectionSimulation<SawtoothProblem>::setInitialConditions()
{
	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_old_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			auto x = (0.5 + static_cast<double>(i)) / nx_;
			double dens = 0.0;
			if (std::abs(x - 0.2) <= 0.15) {
				dens = std::sqrt(1.0 - std::pow((x - 0.2) / 0.15, 2));
			}
			state(i, j, k, 0) = dens;
		});
	}
}

void ComputeExactSolution(amrex::Array4<amrex::Real> const &exact_arr, amrex::Box const &indexRange,
			  const double nx)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		auto x = (0.5 + static_cast<double>(i)) / nx;
		double dens = 0.0;
		if (std::abs(x - 0.2) <= 0.15) {
			dens = std::sqrt(1.0 - std::pow((x - 0.2) / 0.15, 2));
		}
		exact_arr(i, j, k, 0) = dens;
	});
}

auto testproblem_advection() -> int
{
	// Based on
	// https://www.mathematik.uni-dortmund.de/~kuzmin/fcttvd.pdf
	// Section 6.2: Convection of a semi-ellipse

	// Problem parameters

	const int nx = 400;
	const double Lx = 1.0;
	const double advection_velocity = 1.0;
	const double CFL_number = 0.3;
	const double max_time = 1.0;
	const double max_dt = 1e-4;
	const int max_timesteps = 1e4;
	const int nvars = 1; // only density

	const double atol = 1e-10; //< absolute tolerance for mass conservation

	// Problem initialization
	AdvectionSimulation<SawtoothProblem> sim;

	// run simulation
	sim.evolve();

	// Compute reference solution
	amrex::MultiFab state_exact(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				    sim.nghost_);

	for (amrex::MFIter iter(sim.state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = state_exact.array(iter);
		auto const &stateNew = sim.state_new_.const_array(iter);
		ComputeExactSolution(stateExact, indexRange, sim.nx_);
	}

	// Compute error norm
	const int this_comp = 0;
	const auto sol_norm = state_exact.norm1(this_comp);
	amrex::MultiFab::Saxpy(state_exact, -1., sim.state_new_, this_comp, this_comp, sim.ncomp_, sim.nghost_);
	const auto err_norm = state_exact.norm1(this_comp);
	const double rel_error = err_norm / sol_norm;

	amrex::Print() << "L1 solution norm = " << sol_norm << std::endl;
	amrex::Print() << "L1 error norm = " << err_norm << std::endl;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	const double err_tol = 0.015;
	int status = 0;
	if (rel_error > err_tol) {
		status = 1;
	}
	return status;
}
