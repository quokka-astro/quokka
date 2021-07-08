//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro2d_blast.hpp"
#include "AMReX_Array.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_TagBox.H"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "test_radhydro_shock_cgs.hpp"

struct BlastProblem {
};

template <> struct EOS_Traits<BlastProblem> {
	static constexpr double gamma = 5. / 3.;
};

template <> void RadhydroSimulation<BlastProblem>::setInitialConditionsAtLevel(int lev)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);

	for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		// const amrex::Box &indexRange = iter.fabbox(); // INCLUDES ghost zones
		auto const &state = state_new_[lev].array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
			amrex::Real const R = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

			double vx = 0.;
			double vy = 0.;
			double vz = 0.;
			double rho = 1.0;
			double P = NAN;

			if (R < 0.1) { // inside circle
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
			const auto gamma = HydroSystem<BlastProblem>::gamma_;

			state(i, j, k, HydroSystem<BlastProblem>::density_index) = rho;
			state(i, j, k, HydroSystem<BlastProblem>::x1Momentum_index) = rho * vx;
			state(i, j, k, HydroSystem<BlastProblem>::x2Momentum_index) = rho * vy;
			state(i, j, k, HydroSystem<BlastProblem>::x3Momentum_index) = rho * vz;
			state(i, j, k, HydroSystem<BlastProblem>::energy_index) =
			    P / (gamma - 1.) + 0.5 * rho * v_sq;

			//#if 0
			// initialize radiation variables just in case
			state(i, j, k, RadSystem<BlastProblem>::radEnergy_index) = 0;
			state(i, j, k, RadSystem<BlastProblem>::x1RadFlux_index) = 0;
			state(i, j, k, RadSystem<BlastProblem>::x2RadFlux_index) = 0;
			state(i, j, k, RadSystem<BlastProblem>::x3RadFlux_index) = 0;
			//#endif
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

template <>
void RadhydroSimulation<BlastProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags,
						amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2; // gradient refinement threshold

	for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			int const n = HydroSystem<BlastProblem>::density_index;
			amrex::Real const rho = state(i, j, k, n);

			amrex::Real const del_x =
			    std::abs(state(i + 1, j, k, n) - state(i - 1, j, k, n)) / 2.0;
			amrex::Real const del_y =
			    std::abs(state(i, j + 1, k, n) - state(i, j - 1, k, n)) / 2.0;
			amrex::Real const gradient_indicator = std::max(del_x, del_y) / rho;

			if (gradient_indicator > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// Problem parameters
	amrex::IntVect gridDims{AMREX_D_DECL(400, 600, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
	    {AMREX_D_DECL(amrex::Real(1.0), amrex::Real(1.5), amrex::Real(1.0))}};

	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<BlastProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<BlastProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<BlastProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	constexpr bool reflecting_boundary = false;

	const int nvars = RadhydroSimulation<BlastProblem>::nvarTotal_;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (reflecting_boundary) {
				if (isNormalComp(n, i)) {
					boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
					boundaryConditions[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
					boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);
				}
			} else {
				// periodic
				boundaryConditions[n].setLo(i, amrex::BCType::int_dir);
				boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
			}
		}
	}

	// Problem initialization
	RadhydroSimulation<BlastProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.is_hydro_enabled_ = true;
	sim.is_radiation_enabled_ = false;
	sim.stopTime_ = 1.5;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 20000;
	sim.plotfileInterval_ = 1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}