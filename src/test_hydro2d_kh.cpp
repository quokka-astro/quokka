//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro2d_kh.hpp"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Random.H"
#include "AMReX_RandomEngine.H"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "test_radhydro_shock_cgs.hpp"
#include <csignal>

struct KelvinHelmholzProblem {
};

template <> struct EOS_Traits<KelvinHelmholzProblem> {
	static constexpr double gamma = 1.4;
};

template <> void RadhydroSimulation<KelvinHelmholzProblem>::setInitialConditions()
{
	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = simGeometry_.CellSizeArray();
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = simGeometry_.ProbLoArray();
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = simGeometry_.ProbHiArray();

	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const amp = 0.01;

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelForRNG(
		    indexRange, [=] AMREX_GPU_DEVICE (int i, int j, int k,
						     amrex::RandomEngine const &engine) noexcept
			{
			    amrex::Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];

			    // drawn from [0.0, 1.0)
			    amrex::Real const rand_x = amrex::Random(engine) - 0.5;
			    amrex::Real const rand_y = amrex::Random(engine) - 0.5;

			    double vx = 0.;
			    double vy = 0.;
			    double vz = 0.;
			    double rho = 1.0;
			    double P = 2.5;

			    if (std::abs(y - y0) <= 0.25) {
				    rho = 2.0;
				    vx = 0.5 + amp * rand_x;
				    vy = amp * rand_y;
			    } else {
				    rho = 1.0;
				    vx = -0.5 + amp * rand_x;
				    vy = amp * rand_y;
			    }

			    AMREX_ASSERT(!std::isnan(vx));
			    AMREX_ASSERT(!std::isnan(vy));
			    AMREX_ASSERT(!std::isnan(vz));
			    AMREX_ASSERT(!std::isnan(rho));
			    AMREX_ASSERT(!std::isnan(P));

			    const auto v_sq = vx * vx + vy * vy + vz * vz;
			    const auto gamma = HydroSystem<KelvinHelmholzProblem>::gamma_;

			    state(i, j, k, HydroSystem<KelvinHelmholzProblem>::density_index) = rho;
			    state(i, j, k, HydroSystem<KelvinHelmholzProblem>::x1Momentum_index) =
				rho * vx;
			    state(i, j, k, HydroSystem<KelvinHelmholzProblem>::x2Momentum_index) =
				rho * vy;
			    state(i, j, k, HydroSystem<KelvinHelmholzProblem>::x3Momentum_index) =
				rho * vz;
			    state(i, j, k, HydroSystem<KelvinHelmholzProblem>::energy_index) =
				P / (gamma - 1.) + 0.5 * rho * v_sq;
		    });
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

auto problem_main() -> int
{
	// Problem parameters
	amrex::IntVect gridDims{AMREX_D_DECL(1024, 1024, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
	    {AMREX_D_DECL(amrex::Real(1.0), amrex::Real(1.0), amrex::Real(1.0))}};

	const int nvars = RadhydroSimulation<KelvinHelmholzProblem>::nvarTotal_;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
			boundaryConditions[n].setHi(i, amrex::BCType::int_dir); // periodic
		}
	}

	// Problem initialization
	RadhydroSimulation<KelvinHelmholzProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.is_hydro_enabled_ = true;
	sim.is_radiation_enabled_ = false;
	sim.stopTime_ = 5.0;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 40000;
	sim.plotfileInterval_ = 100;
	sim.outputAtInterval_ = true;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Finished." << std::endl;
	}
	return 0;
}