//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro2d_rm.hpp"
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

		result = testproblem_hydro_rm();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct RichtmeyerMeshkovProblem {
};

template <> struct EOS_Traits<RichtmeyerMeshkovProblem> {
	static constexpr double gamma = 1.4;
};

namespace quokka
{
template <>
AMREX_GPU_HOST_DEVICE auto CheckSymmetryFluxes<RichtmeyerMeshkovProblem>(
    amrex::Array4<const amrex::Real> const &arr1, amrex::Array4<const amrex::Real> const &arr2,
    amrex::Box const &indexRange, const int ncomp, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
    -> bool
{
#ifdef DEBUG_SYMMETRY
	amrex::Long asymmetry = 0;
	amrex::GpuArray<int, 3> prob_lo = indexRange.loVect3d();
	auto nx = indexRange.hiVect3d()[0] + 1;
	auto ny = indexRange.hiVect3d()[1] + 1;
	auto nz = indexRange.hiVect3d()[2] + 1;
	AMREX_ASSERT(prob_lo[0] == 0);
	AMREX_ASSERT(prob_lo[1] == 0);
	AMREX_ASSERT(prob_lo[2] == 0);

	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			for (int k = 0; k < nz; ++k) {
				amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
				amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];

				for (int n = 0; n < ncomp; ++n) {
					const amrex::Real comp_upper = arr1(i, j, k, n);

					// reflect across x/y diagonal
					int n_lower = n;
					if (n == HydroSystem<
						     RichtmeyerMeshkovProblem>::x1Momentum_index) {
						n_lower = HydroSystem<
						    RichtmeyerMeshkovProblem>::x2Momentum_index;
					} else if (n == HydroSystem<RichtmeyerMeshkovProblem>::
							    x2Momentum_index) {
						n_lower = HydroSystem<
						    RichtmeyerMeshkovProblem>::x1Momentum_index;
					}

					amrex::Real comp_lower = arr2(j, i, k, n_lower);

					const amrex::Real average =
					    std::fabs(comp_upper + comp_lower);
					const amrex::Real residual =
					    std::abs(comp_upper - comp_lower) / average;

					if (comp_upper != comp_lower) {
#ifndef AMREX_USE_GPU
						amrex::Print()
						    << i << ", " << j << ", " << k << ", " << n
						    << ", " << comp_upper << ", " << comp_lower
						    << " " << residual << "\n";
						amrex::Print() << "x = " << x << "\n";
						amrex::Print() << "y = " << y << "\n";
#endif
						asymmetry++;
						AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
						    false,
						    "[CheckSymmetryFluxes] x/y not symmetric!");
					}
				}
			}
		}
	}
#endif // DEBUG_SYMMETRY
	return true;
}
} // namespace quokka

template <> void HydroSimulation<RichtmeyerMeshkovProblem>::computeAfterTimestep()
{
#ifdef DEBUG_SYMMETRY
	// this code does not actually work with Nranks > 1 ...

	// copy all FABs to a local FAB across the entire domain
	amrex::BoxArray localBoxes(domain_);
	amrex::DistributionMapping localDistribution(localBoxes, 1);
	amrex::MultiFab state_mf(localBoxes, localDistribution, ncomp_, 0);
	state_mf.ParallelCopy(state_new_);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &state = state_mf.array(0);
		const auto *prob_lo = simGeometry_.ProbLo();
		auto dx = dx_;

		amrex::Long asymmetry = 0;
		auto nx = nx_;
		auto ny = ny_;
		auto nz = nz_;
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
						int n_lower = n;
						if (n == HydroSystem<RichtmeyerMeshkovProblem>::
							     x1Momentum_index) {
							n_lower =
							    HydroSystem<RichtmeyerMeshkovProblem>::
								x2Momentum_index;
						} else if (n ==
							   HydroSystem<RichtmeyerMeshkovProblem>::
							       x2Momentum_index) {
							n_lower =
							    HydroSystem<RichtmeyerMeshkovProblem>::
								x1Momentum_index;
						}

						amrex::Real comp_lower = state(j, i, k, n_lower);

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
#endif
}

template <> void HydroSimulation<RichtmeyerMeshkovProblem>::setInitialConditions()
{
	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = simGeometry_.CellSizeArray();
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = simGeometry_.ProbLoArray();
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = simGeometry_.ProbHiArray();

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			amrex::Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];

			double vx = 0.;
			double vy = 0.;
			double vz = 0.;
			double rho = NAN;
			double P = NAN;

			if ((x + y) > 0.15) {
				P = 1.0;
				rho = 1.0;
			} else {
				P = 0.14;
				rho = 0.125;
			}

			AMREX_ASSERT(!std::isnan(vx));
			AMREX_ASSERT(!std::isnan(vy));
			AMREX_ASSERT(!std::isnan(vz));
			AMREX_ASSERT(!std::isnan(rho));
			AMREX_ASSERT(!std::isnan(P));

			const auto v_sq = vx * vx + vy * vy + vz * vz;
			const auto gamma = HydroSystem<RichtmeyerMeshkovProblem>::gamma_;

			state(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::density_index) = rho;
			state(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index) =
			    rho * vx;
			state(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index) =
			    rho * vy;
			state(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x3Momentum_index) =
			    rho * vz;
			state(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::energy_index) =
			    P / (gamma - 1.) + 0.5 * rho * v_sq;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

auto testproblem_hydro_rm() -> int
{
	// Problem parameters
	const int nvars = 5; // Euler equations

	amrex::IntVect gridDims{AMREX_D_DECL(512, 512, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
	    {AMREX_D_DECL(amrex::Real(0.3), amrex::Real(0.3), amrex::Real(1.0))}};

	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<RichtmeyerMeshkovProblem>::x3Momentum_index) && (dim == 2)) {
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
	HydroSimulation<RichtmeyerMeshkovProblem> sim(gridDims, boxSize, boundaryConditions);

	sim.stopTime_ = 2.5;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 40000;
	sim.plotfileInterval_ = 25;
	sim.outputAtInterval_ = true;

	// initialize
	sim.setInitialConditions();
	//sim.computeAfterTimestep();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}