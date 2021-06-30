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
#include "AMReX_ParmParse.H"
#include "AdvectionSimulation.hpp"
#include "hyperbolic_system.hpp"
#include "linear_advection.hpp"
#include "test_radhydro_shock_cgs.hpp"
#include <limits>
#include <vector>

struct SemiellipseProblem {
};

template <> void AdvectionSimulation<SemiellipseProblem>::setInitialConditions()
{
	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);
		auto const nx = nx_[0];

		amrex::ParallelFor(indexRange, ncomp_, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
			auto x = (0.5 + static_cast<double>(i)) / nx;
			double dens = 0.0;
			if (std::abs(x - 0.2) <= 0.15) {
				dens = std::sqrt(1.0 - std::pow((x - 0.2) / 0.15, 2));
			}
			state(i, j, k, n) = dens;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

void ComputeExactSolution(amrex::Array4<amrex::Real> const &exact_arr, amrex::Box const &indexRange,
			  const int nvars, const int nx)
{
	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		auto x = (0.5 + static_cast<double>(i)) / nx;
		double dens = 0.0;
		if (std::abs(x - 0.2) <= 0.15) {
			dens = std::sqrt(1.0 - std::pow((x - 0.2) / 0.15, 2));
		}
		exact_arr(i, j, k, n) = dens;
	});
}

auto problem_main() -> int
{
	// Based on
	// https://www.mathematik.uni-dortmund.de/~kuzmin/fcttvd.pdf
	// Section 6.2: Convection of a semi-ellipse

	// Problem parameters
	const int nx = 400;
	const double Lx = 1.0;
	const double advection_velocity = 1.0;
	const double max_time = 1.0;
	const double max_dt = 1e-4;
	const int max_timesteps = 1e4;
	
	const int nvars = 1; // only density

	amrex::IntVect gridDims{AMREX_D_DECL(nx, 4, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(1.0), amrex::Real(1.0))}};

	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
			boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	AdvectionSimulation<SemiellipseProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.maxDt_ = max_dt;
	sim.advectionVx_ = advection_velocity;
	sim.advectionVy_ = 0.;
	sim.advectionVz_ = 0.;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Compute reference solution
	amrex::MultiFab state_exact(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				    sim.nghost_);

	for (amrex::MFIter iter(sim.state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = state_exact.array(iter);
		ComputeExactSolution(stateExact, indexRange, sim.ncomp_, sim.nx_[0]);
	}

	// Compute error norm
	amrex::MultiFab residual(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				 sim.nghost_);
	amrex::MultiFab::Copy(residual, state_exact, 0, 0, sim.ncomp_, sim.nghost_);
	amrex::MultiFab::Saxpy(residual, -1., sim.state_new_, 0, 0, sim.ncomp_, sim.nghost_);
	
	double min_rel_error = std::numeric_limits<double>::max();
	for(int comp = 0; comp < sim.ncomp_; ++comp) {
		const auto sol_norm = state_exact.norm1(comp);
		const auto err_norm = residual.norm1(comp);
		const double rel_error = err_norm / sol_norm;
		min_rel_error = std::min(min_rel_error, rel_error);
		amrex::Print() << "Relative L1 error norm (comp " << comp << ") = " << rel_error << "\n";
	}

	const double err_tol = 0.015;
	int status = 0;
	if (min_rel_error > err_tol) {
		status = 1;
	}

	// copy all FABs to a local FAB across the entire domain
	amrex::BoxArray localBoxes(sim.domain_);
	amrex::DistributionMapping localDistribution(localBoxes, 1);
	amrex::MultiFab state_final(localBoxes, localDistribution, sim.ncomp_, 0);
	amrex::MultiFab state_exact_local(localBoxes, localDistribution, sim.ncomp_, 0);
	state_final.ParallelCopy(sim.state_new_);
	state_exact_local.ParallelCopy(state_exact);
	auto const &state_final_array = state_final.array(0);
	auto const &state_exact_array = state_exact_local.array(0);

	// plot solution
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> d_final(sim.nx_[0]);
		std::vector<double> d_initial(sim.nx_[0]);
		std::vector<double> x(sim.nx_[0]);

		for (int i = 0; i < sim.nx_[0]; ++i) {
			x.at(i) = (static_cast<double>(i) + 0.5) / sim.nx_[0];
			d_final.at(i) = state_final_array(i, 0, 0);
			d_initial.at(i) = state_exact_array(i, 0, 0);
		}

		// Plot results
		std::map<std::string, std::string> d_initial_args;
		std::map<std::string, std::string> d_final_args;
		d_initial_args["label"] = "density (initial)";
		d_final_args["label"] = "density (final)";

		matplotlibcpp::plot(x, d_initial, d_initial_args);
		matplotlibcpp::plot(x, d_final, d_final_args);
		matplotlibcpp::legend();
		matplotlibcpp::save(std::string("./advection_semiellipse.pdf"));
	}

	return status;
}
