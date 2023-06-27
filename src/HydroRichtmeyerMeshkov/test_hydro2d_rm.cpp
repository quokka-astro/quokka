//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "test_hydro2d_rm.hpp"

struct RichtmeyerMeshkovProblem {
};

template <> struct quokka::EOS_Traits<RichtmeyerMeshkovProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = quokka::boltzmann_constant_cgs;
	static constexpr double hydrogen_mass_code_units = quokka::hydrogen_mass_cgs;
};

template <> struct HydroSystem_Traits<RichtmeyerMeshkovProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<RichtmeyerMeshkovProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};

// #define DEBUG_SYMMETRY
template <> void RadhydroSimulation<RichtmeyerMeshkovProblem>::computeAfterTimestep()
{
#ifdef DEBUG_SYMMETRY
	// this code does not actually work with Nranks > 1 ...
	const int ncomp_cc = Physics_Indices<RichtmeyerMeshkovProblem>::nvarTotal_cc;

	// copy all FABs to a local FAB across the entire domain
	amrex::BoxArray localBoxes(domain_);
	amrex::DistributionMapping localDistribution(localBoxes, 1);
	amrex::MultiFab state_mf(localBoxes, localDistribution, ncomp_cc, 0);
	state_mf.ParallelCopy(state_new_cc_);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &state = state_mf.array(0);
		auto const prob_lo = simGeometry_.ProbLoArray();
		auto dx = dx_;

		amrex::Long asymmetry = 0;
		auto nx = nx_;
		auto ny = ny_;
		auto nz = nz_;
		auto ncomp = ncomp_cc;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				for (int k = 0; k < nz; ++k) {
					amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
					amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
					for (int n = 0; n < ncomp; ++n) {
						const amrex::Real comp_upper = state(i, j, k, n);

						// reflect across x/y diagonal
						int n_lower = n;
						if (n == HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index) {
							n_lower = HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index;
						} else if (n == HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index) {
							n_lower = HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index;
						}

						amrex::Real comp_lower = state(j, i, k, n_lower);

						const amrex::Real average = std::fabs(comp_upper + comp_lower);
						const amrex::Real residual = std::abs(comp_upper - comp_lower) / average;

						if (comp_upper != comp_lower) {
							amrex::Print() << i << ", " << j << ", " << k << ", " << n << ", " << comp_upper << ", " << comp_lower
								       << " " << residual << "\n";
							amrex::Print() << "x = " << x << "\n";
							amrex::Print() << "y = " << y << "\n";
							asymmetry++;
							AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, "x/y not symmetric!");
						}
					}
				}
			}
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(asymmetry == 0, "x/y not symmetric!");
	}
#endif
}

template <> void RadhydroSimulation<RichtmeyerMeshkovProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];

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
		const auto gamma = quokka::EOS_Traits<RichtmeyerMeshkovProblem>::gamma;

		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * v_sq;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// amrex::IntVect gridDims{AMREX_D_DECL(1024, 1024, 4)};
	// amrex::RealBox boxSize{
	//    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
	//    {AMREX_D_DECL(amrex::Real(0.3), amrex::Real(0.3), amrex::Real(1.0))}};

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

	const int ncomp_cc = Physics_Indices<RichtmeyerMeshkovProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	RadhydroSimulation<RichtmeyerMeshkovProblem> sim(BCs_cc);

	sim.stopTime_ = 2.5;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 50000;
	sim.plotfileInterval_ = 100;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
