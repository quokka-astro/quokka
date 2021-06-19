//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include "test_hydro3d_blast.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
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

		result = testproblem_hydro_sedov();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct SedovProblem {
};

template <> struct EOS_Traits<SedovProblem> {
	static constexpr double gamma = 5. / 3.;
};

template <> void HydroSimulation<SedovProblem>::setInitialConditions()
{
	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = simGeometry_.CellSizeArray();
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = simGeometry_.ProbLoArray();
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = simGeometry_.ProbHiArray();

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			amrex::Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];
			amrex::Real const z = prob_lo[2] + (k + Real(0.5)) * dx[2];
			amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) +
							std::pow(z - z0, 2));

			double vx = 0.;
			double vy = 0.;
			double vz = 0.;
			double rho = 1.0;
			double P = NAN;

			if (r < 0.1) { // inside sphere
				P = 10.;
			} else {
				P = 0.1;
			}

			AMREX_ASSERT(!std::isnan(vx));
			AMREX_ASSERT(!std::isnan(vy));
			AMREX_ASSERT(!std::isnan(vz));
			AMREX_ASSERT(!std::isnan(rho));
			AMREX_ASSERT(!std::isnan(P));

			const auto v_sq = vx * vx + vy * vy + vz * vz;
			const auto gamma = HydroSystem<SedovProblem>::gamma_;

			state(i, j, k, HydroSystem<SedovProblem>::density_index) = rho;
			state(i, j, k, HydroSystem<SedovProblem>::x1Momentum_index) = rho * vx;
			state(i, j, k, HydroSystem<SedovProblem>::x2Momentum_index) = rho * vy;
			state(i, j, k, HydroSystem<SedovProblem>::x3Momentum_index) = rho * vz;
			state(i, j, k, HydroSystem<SedovProblem>::energy_index) =
			    P / (gamma - 1.) + 0.5 * rho * v_sq;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

auto testproblem_hydro_sedov() -> int
{
	// Problem parameters
	const int nvars = 5; // Euler equations

	amrex::IntVect gridDims{AMREX_D_DECL(512, 512, 512)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
	    {AMREX_D_DECL(amrex::Real(1.0), amrex::Real(1.0), amrex::Real(1.0))}};

	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<SedovProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<SedovProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<SedovProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
				boundaryConditions[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
				boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	HydroSimulation<SedovProblem> sim(gridDims, boxSize, boundaryConditions);

	sim.stopTime_ = 0.1;
	sim.cflNumber_ = 0.3; // *must* be less than 1/3 in 3D!
	sim.maxTimesteps_ = 5000;
	sim.plotfileInterval_ = 100;
	sim.outputAtInterval_ = true;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}