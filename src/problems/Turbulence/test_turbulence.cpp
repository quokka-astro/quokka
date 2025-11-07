//
// Created by rowan on 8/11/24.
//

#include "test_turbulence.hpp"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "turbulence/TurbulentDriving.hpp"

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"

struct BasicTurbulence {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Physics_Traits<BasicTurbulence> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr bool is_driving_enabled = true;
	static constexpr bool is_self_gravity_enabled = false;

	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct quokka::EOS_Traits<BasicTurbulence> {
	static constexpr double gamma = 1.4;
	static constexpr double cs_isothermal = 1.0; // dimensionless
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> void QuokkaSimulation<BasicTurbulence>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = 1;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = 1;
		Real const Egas = Eint;
		Real const scalar_density = 0;

		state_cc(i, j, k, HydroSystem<BasicTurbulence>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<BasicTurbulence>::scalar0_index) = scalar_density;
	});
}

auto problem_main() -> int
{

	// Sets all variables to be periodic
	const int ncomp_cc = Physics_Indices<BasicTurbulence>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<BasicTurbulence> sim(BCs_cc);

	sim.stopTime_ = 6;
	sim.plotfileInterval_ = 300;
	sim.maxTimesteps_ = 100000;
	sim.setInitialConditions();

	// Main time loop
	sim.evolve();
	return 1;
}
