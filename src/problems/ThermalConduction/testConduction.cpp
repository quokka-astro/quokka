//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroBlast3D.cpp
/// \brief Defines a test problem for a 3D explosion.
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <fstream>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

struct ThermalConductionProblem {
};


bool test_passes = false; // if one of the energy checks fails, set to false. NOLINT

template <> struct quokka::EOS_Traits<ThermalConductionProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// declare global variables
const double rho = 1.0;	   // g cm^-3
double E_blast = 0.851072; // ergs. NOLINT


template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a ThermalConduction test problem using parameters from
	

	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		static_assert(!simulate_full_box, "single-cell initialization is only "
						  "implemented for octant symmetry!");
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];

		if(x < 0.0) {
			state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = 1.0;
		} else {
			state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = 0.1;
		}

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index);
	});
}

auto problem_main() -> int
{
	const int max_timesteps = 10000;
	const double CFL_number = 0.4;
	const double max_time = 5.0e-10; // s

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<TophatProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::foextrap);  // left x1 -- Marshak
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // right x1 -- extrapolate
		// for (int i = 1; i < AMREX_SPACEDIM; ++i) {
		// 	if (isNormalComp(n, i)) { // reflect lower
		// 		BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
		// 	} else {
		// 		BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
		// 	}
		// 	// extrapolate upper
		// 	BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		// }
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 2; // PLM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = 20;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
