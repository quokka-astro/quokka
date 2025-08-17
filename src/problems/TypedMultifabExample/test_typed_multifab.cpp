//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
// \file test_typed_multifab.cpp
// \brief Defines a test problem to demonstrate TypedMultifab usage
//==============================================================================

#include <cmath>
#include <string>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "main.hpp"
#include "util/TypedMultifab.hpp"
#include "util/TypeList.hpp"
#include "util/VariableTypes.hpp"

// Define conserved variable types
namespace Conserved {
VARIABLE(hydro, density);
VARIABLE(hydro, momentum_x);
VARIABLE(hydro, momentum_y);
VARIABLE(hydro, momentum_z);
VARIABLE(hydro, energy);
VARIABLE(hydro, scalar_1);
VARIABLE(hydro, scalar_2);
} // namespace Conserved

// Define primitive variable types
namespace Primitive {
VARIABLE(hydro, density);
VARIABLE(hydro, velocity_x);
VARIABLE(hydro, velocity_y);
VARIABLE(hydro, velocity_z);
VARIABLE(hydro, pressure);
VARIABLE(hydro, scalar_1);
VARIABLE(hydro, scalar_2);
} // namespace Primitive

// Define TypeLists for different variable groups
using ConservedTypeList = quokka::TypeList<Conserved::density, Conserved::momentum_x, 
					    Conserved::momentum_y, Conserved::momentum_z,
					    Conserved::energy, Conserved::scalar_1, 
					    Conserved::scalar_2>;

using PrimitiveTypeList = quokka::TypeList<Primitive::density, Primitive::velocity_x,
					    Primitive::velocity_y, Primitive::velocity_z,
					    Primitive::pressure, Primitive::scalar_1,
					    Primitive::scalar_2>;

// Subset type lists
using ConservedHydroOnly = quokka::TypeList<Conserved::density, Conserved::momentum_x,
					     Conserved::momentum_y, Conserved::momentum_z,
					     Conserved::energy>;

using ScalarsOnly = quokka::TypeList<Conserved::scalar_1, Conserved::scalar_2>;

template <> struct HydroSystem_Traits<TypedMultifabExample> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<TypedMultifabExample> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr int numMassScalars = 2;
	static constexpr int numPassiveScalars = numMassScalars;
	static constexpr int nGroups = 1;
};

template <> struct SimulationData<TypedMultifabExample> {
	// nothing
};

template <> void QuokkaSimulation<TypedMultifabExample>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state = grid_elem.array_;
	
	const auto &ba = grid_elem.ba_;
	const auto &dm = grid_elem.dm_;
	const int nghost = grid_elem.nghost_;
	
	// Create a TypedMultifab with all conserved variables
	quokka::TypedMultifab<ConservedTypeList> conserved_mf(ba, dm, nghost);
	
	// Create a TypedMultifab with only hydro variables (subset)
	quokka::TypedMultifab<ConservedHydroOnly> hydro_mf(ba, dm, nghost, conserved_mf);
	
	// Create a TypedMultifab with only scalars (another subset)
	quokka::TypedMultifab<ScalarsOnly> scalar_mf(ba, dm, nghost, conserved_mf);
	
	// Create a combined TypedMultifab from multiple sources without deep copy
	using CombinedList = quokka::TypeListCat_t<ConservedHydroOnly, ScalarsOnly>;
	auto combined_mf = quokka::makeTypedMultifab<CombinedList>(ba, dm, nghost, hydro_mf, scalar_mf);
	
	// Print component names to demonstrate functionality
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "TypedMultifab Example:\n";
		amrex::Print() << "======================\n";
		
		amrex::Print() << "\nConserved variables:\n";
		for (const auto& name : conserved_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}
		
		amrex::Print() << "\nHydro-only subset:\n";
		for (const auto& name : hydro_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}
		
		amrex::Print() << "\nScalars-only subset:\n"; 
		for (const auto& name : scalar_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}
		
		amrex::Print() << "\nCombined from subsets (no deep copy):\n";
		for (const auto& name : combined_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}
		
		amrex::Print() << "\nVerifying no deep copies occurred:\n";
		amrex::Print() << "  Hydro density MultiFab ptr: " << &hydro_mf.getMultiFab<Conserved::density>() << "\n";
		amrex::Print() << "  Combined density MultiFab ptr: " << &combined_mf.getMultiFab<Conserved::density>() << "\n";
		amrex::Print() << "  Same pointer? " << (&hydro_mf.getMultiFab<Conserved::density>() == 
							       &combined_mf.getMultiFab<Conserved::density>() ? "YES" : "NO") << "\n";
	}
	
	// Set initial conditions using standard arrays (for compatibility)
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double rho = 1.0;
		const double vx = 0.1;
		const double vy = 0.0;
		const double vz = 0.0;
		const double P = 1.0;
		
		const double vx2 = vx * vx;
		const double vy2 = vy * vy;
		const double vz2 = vz * vz;
		const double vsq = vx2 + vy2 + vz2;
		const double e_kin = 0.5 * rho * vsq;
		const double e_int = P / (HydroSystem<TypedMultifabExample>::gamma_ - 1.0);
		const double E_gas = e_kin + e_int;
		
		state(i, j, k, HydroSystem<TypedMultifabExample>::density_index) = rho;
		state(i, j, k, HydroSystem<TypedMultifabExample>::x1Momentum_index) = rho * vx;
		state(i, j, k, HydroSystem<TypedMultifabExample>::x2Momentum_index) = rho * vy;
		state(i, j, k, HydroSystem<TypedMultifabExample>::x3Momentum_index) = rho * vz;
		state(i, j, k, HydroSystem<TypedMultifabExample>::energy_index) = E_gas;
		state(i, j, k, HydroSystem<TypedMultifabExample>::scalar0_index) = 0.5;
		state(i, j, k, HydroSystem<TypedMultifabExample>::scalar0_index + 1) = 0.3;
	});
}

template <> void QuokkaSimulation<TypedMultifabExample>::computeAfterTimestep()
{
	// Example of using TypedMultifab with MFIter
	if (amrex::ParallelDescriptor::IOProcessor() && timestep_ == 1) {
		amrex::Print() << "\nDemonstrating TypedMultifab access with MFIter:\n";
		
		for (int level = 0; level <= finest_level; ++level) {
			const auto &ba = grids[level];
			const auto &dm = dmap[level];
			const int nghost = 0;
			
			// Create a TypedMultifab wrapping the state
			quokka::TypedMultifab<ConservedTypeList> typed_state(ba, dm, nghost);
			
			// Access components in a type-safe way
			for (amrex::MFIter mfi(typed_state.getMultiFab<Conserved::density>()); mfi.isValid(); ++mfi) {
				const amrex::Box& bx = mfi.validbox();
				
				// Get typed arrays
				auto density_arr = typed_state.array<Conserved::density>(mfi);
				auto momx_arr = typed_state.array<Conserved::momentum_x>(mfi);
				auto energy_arr = typed_state.array<Conserved::energy>(mfi);
				
				// Example: could perform operations with type safety
				// No manual indexing needed!
			}
		}
		
		amrex::Print() << "TypedMultifab demonstration complete.\n";
	}
}

auto problem_main() -> int
{
	amrex::Initialize();
	
	{
		const int ncomp_cc = Physics_Indices<TypedMultifabExample>::nvarTotal_cc;
		amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
		for (int n = 0; n < ncomp_cc; ++n) {
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			}
		}
		
		QuokkaSimulation<TypedMultifabExample> sim(BCs_cc);
		sim.setInitialConditions();
		sim.evolve();
	}
	
	amrex::Finalize();
	return 0;
}