//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include "test_advection.hpp"
#include "AMReX_Algorithm.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BoxArray.H"
#include "AMReX_Config.H"
#include "AMReX_CoordSys.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AdvectionSimulation.hpp"

struct SawtoothProblem {
};

template <> void AdvectionSimulation<SawtoothProblem>::setInitialConditionsAtLevel(int level)
{
	auto const &prob_lo = geom[level].ProbLoArray();
	auto const &dx = geom[level].CellSizeArray();
	auto const y_length = geom[level].ProbLength(1); // y-axis

	for (amrex::MFIter iter(state_old_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_[level].array(iter);

		amrex::ParallelFor(indexRange, ncomp_,
				   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
						amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
						auto value = std::fmod(y + 0.5*y_length, y_length);
						state(i, j, k, n) = value;
				   });
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

void ComputeExactSolution(amrex::Array4<amrex::Real> const &exact_arr, amrex::Box const &indexRange,
			  const int nvars, const int /*nx*/, const int ny)
{
	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		//auto value = static_cast<double>((i + nx / 2) % nx) / nx;
		auto value = static_cast<double>((j + ny / 2) % ny) / ny;
		exact_arr(i, j, k, n) = value;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	const int nx = 400;
	const double Lx = 1.0;
	const double advection_velocity = 1.0;
	const double CFL_number = 0.4;
	const double max_time = 1.0;
	const double max_dt = 1.0e-4;
	const int max_timesteps = 1e4;
	const int nvars = 1; // only density

	//amrex::IntVect gridDims{AMREX_D_DECL(nx, 4, 4)};
	amrex::IntVect gridDims{AMREX_D_DECL(4, nx, 4)};

	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
	    {AMREX_D_DECL(amrex::Real(1.0), amrex::Real(Lx), amrex::Real(1.0))}};
//	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(1.0), amrex::Real(1.0))}};

	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
			boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	AdvectionSimulation<SawtoothProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.maxDt_ = max_dt;
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;

	sim.advectionVx_ = 1.0;
	sim.advectionVy_ = advection_velocity;

	// set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	int status = 0;
#if 0
	// Compute reference solution
	amrex::MultiFab state_exact(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				    sim.nghost_);

	for (amrex::MFIter iter(sim.state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = state_exact.array(iter);
		ComputeExactSolution(stateExact, indexRange, sim.ncomp_, sim.nx_[0], sim.nx_[1]);
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

	const double err_tol = 0.015;
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
		auto N = sim.nx_[1];
		std::vector<double> d_final(N);
		std::vector<double> d_initial(N);
		std::vector<double> x(N);

		for (int i = 0; i < N; ++i) {
			x.at(i) = (static_cast<double>(i) + 0.5) / N;
//			d_final.at(i) = state_final_array(i, 0, 0);
//			d_initial.at(i) = state_exact_array(i, 0, 0);
			d_final.at(i) = state_final_array(0, i, 0);
			d_initial.at(i) = state_exact_array(0, i, 0);
		}

		// Plot results
		std::map<std::string, std::string> d_initial_args;
		std::map<std::string, std::string> d_final_args;
		d_initial_args["label"] = "density (initial)";
		d_final_args["label"] = "density (final)";

		matplotlibcpp::plot(x, d_initial, d_initial_args);
		matplotlibcpp::plot(x, d_final, d_final_args);
		matplotlibcpp::legend();
		matplotlibcpp::save(std::string("./advection.pdf"));
	}
#endif

	return status;
}
