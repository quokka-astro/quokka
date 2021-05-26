//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro_contact.hpp"
#include "AMReX_BLassert.H"
#include "AMReX_ParmParse.H"
#include "HydroSimulation.hpp"
#include "hydro_system.hpp"


auto main(int argc, char **argv) -> int
{
	// Initialization (copied from ExaWind)

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

		result = testproblem_hydro_contact();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct ContactProblem {
};

template <>
struct EOS_Traits<ContactProblem>
{
	static constexpr double gamma = 1.4;
};
constexpr double v_contact = 0.0; // contact wave velocity

template <> void HydroSimulation<ContactProblem>::setInitialConditions()
{
	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = simGeometry_.CellSizeArray();
    amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = simGeometry_.ProbLoArray();

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    		amrex::Real const x = prob_lo[0] + (i+Real(0.5)) * dx[0];

			double vx = NAN;
			double rho = NAN;
			double P = NAN;

			if (x < 0.5) {
				rho = 1.4;
				vx = v_contact;
				P = 1.0;
			} else {
				rho = 1.0;
				vx = v_contact;
				P = 1.0;
			}
			AMREX_ASSERT(!std::isnan(vx));
			AMREX_ASSERT(!std::isnan(rho));
			AMREX_ASSERT(!std::isnan(P));

			const auto gamma = HydroSystem<ContactProblem>::gamma_;
			state(i, j, k, HydroSystem<ContactProblem>::density_index) = rho;
			state(i, j, k, HydroSystem<ContactProblem>::x1Momentum_index) = rho * vx;
			state(i, j, k, HydroSystem<ContactProblem>::x2Momentum_index) = 0.;
			state(i, j, k, HydroSystem<ContactProblem>::x3Momentum_index) = 0.;
			state(i, j, k, HydroSystem<ContactProblem>::energy_index) =
			    P / (gamma - 1.) + 0.5 * rho * (vx * vx);
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

void ComputeExactSolution(amrex::Array4<amrex::Real> const &exact_arr, amrex::Box const &indexRange,
			amrex::GpuArray<Real, AMREX_SPACEDIM> dx, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    	amrex::Real const x = prob_lo[0] + (i+Real(0.5)) * dx[0];

		double vx = NAN;
		double rho = NAN;
		double P = NAN;

		if (x < 0.5) {
			rho = 1.4;
			vx = v_contact;
			P = 1.0;
		} else {
			rho = 1.0;
			vx = v_contact;
			P = 1.0;
		}

		const auto gamma = HydroSystem<ContactProblem>::gamma_;
		exact_arr(i, j, k, HydroSystem<ContactProblem>::density_index) = rho;
		exact_arr(i, j, k, HydroSystem<ContactProblem>::x1Momentum_index) = rho * vx;
		exact_arr(i, j, k, HydroSystem<ContactProblem>::x2Momentum_index) = 0.;
		exact_arr(i, j, k, HydroSystem<ContactProblem>::x3Momentum_index) = 0.;
		exact_arr(i, j, k, HydroSystem<ContactProblem>::energy_index) =
		    P / (gamma - 1.) + 0.5 * rho * (vx * vx);
	});
}

auto testproblem_hydro_contact() -> int
{
	// Problem parameters
	const int nx = 100;
	const double Lx = 1.0;
	const int nvars = 5; // Euler equations

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
	HydroSimulation<ContactProblem> sim(gridDims, boxSize, boundaryConditions);

	sim.stopTime_ = 2.0;
	sim.cflNumber_ = 0.8;
	sim.maxTimesteps_ = 2000;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Compute reference solution
	amrex::MultiFab state_exact(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				    sim.nghost_);
	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = sim.simGeometry_.CellSizeArray();
    amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = sim.simGeometry_.ProbLoArray();

	for (amrex::MFIter iter(sim.state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = state_exact.array(iter);
		ComputeExactSolution(stateExact, indexRange, dx, prob_lo);
	}

	// Compute error norm
	const int this_comp = 0;
	const auto sol_norm = state_exact.norm1(this_comp);
	amrex::MultiFab residual(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				 sim.nghost_);
	amrex::MultiFab::Copy(residual, state_exact, 0, 0, sim.ncomp_, sim.nghost_);
	amrex::MultiFab::Saxpy(residual, -1., sim.state_new_, this_comp, this_comp, sim.ncomp_,
			       sim.nghost_);
	const auto err_norm = residual.norm1(this_comp);
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	// For a stationary isolated contact wave using the HLLC solver,
	// the error should be *exactly* (i.e., to *every* digit) zero.
	// [See Section 10.7 and Figure 10.20 of Toro (1998).]
	const double error_tol = 0.0; // this is not a typo
	int status = 0;
	if (rel_error > error_tol) {
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

	// Plot results
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> x(nx);
		std::vector<double> d_final(nx);
		std::vector<double> vx_final(nx);
		std::vector<double> P_final(nx);
		std::vector<double> density_exact(nx);
		std::vector<double> pressure_exact(nx);
		std::vector<double> velocity_exact(nx);

		for (int i = 0; i < nx; ++i) {
    		amrex::Real const this_x = prob_lo[0] + (i+Real(0.5)) * dx[0];

			const auto rho = state_exact_array(i, 0, 0, HydroSystem<ContactProblem>::density_index);
			const auto xmom = state_exact_array(i, 0, 0, HydroSystem<ContactProblem>::x1Momentum_index);
			const auto E = state_exact_array(i, 0, 0, HydroSystem<ContactProblem>::energy_index);

			const auto vx = xmom/rho;
			const auto Eint = E - 0.5*rho*(vx*vx);
			const auto P = (HydroSystem<ContactProblem>::gamma_ - 1.) * Eint;

			x.push_back(this_x);
			density_exact.push_back(rho);
			pressure_exact.push_back(P);
			velocity_exact.push_back(vx);
		}

		for (int i = 0; i < nx; ++i) {
			const auto rho = state_final_array(i, 0, 0, HydroSystem<ContactProblem>::density_index);
			const auto xmom = state_final_array(i, 0, 0, HydroSystem<ContactProblem>::x1Momentum_index);
			const auto E = state_final_array(i, 0, 0, HydroSystem<ContactProblem>::energy_index);

			const auto vx = xmom/rho;
			const auto Eint = E - 0.5*rho*(vx*vx);
			const auto P = (HydroSystem<ContactProblem>::gamma_ - 1.) * Eint;

			d_final.push_back(rho);
			vx_final.push_back(vx);
			P_final.push_back(P);	
		}

		std::unordered_map<std::string, std::string> d_args;
		std::map<std::string, std::string> dexact_args;
		d_args["label"] = "density";
		d_args["color"] = "black";
		dexact_args["label"] = "density (exact solution)";

		matplotlibcpp::scatter(x, d_final, 10.0, d_args);
		matplotlibcpp::plot(x, density_exact, dexact_args);

		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNow_));
		matplotlibcpp::save("./hydro_contact.pdf");
	}

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}