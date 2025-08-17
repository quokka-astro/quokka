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
#include "util/TypeList.hpp"
#include "util/TypedMultifab.hpp"
#include "util/VariableTypes.hpp"

// Define conserved variable types
namespace Conserved
{
VARIABLE(hydro, density);
VARIABLE(hydro, momentum_x);
VARIABLE(hydro, momentum_y);
VARIABLE(hydro, momentum_z);
VARIABLE(hydro, energy);

// Define multi-component scalar variable with 10 components
constexpr int NumScalars = 10;
MULTI_VARIABLE(hydro, scalar, NumScalars);
} // namespace Conserved

// Define problem-specific strong type aliases for scalar components
namespace ProblemSpecific
{
// Create strong type aliases for specific scalar components
COMPONENT_ALIAS(Conserved::scalar, 0, temperature);
COMPONENT_ALIAS(Conserved::scalar, 1, metallicity);
COMPONENT_ALIAS(Conserved::scalar, 2, electron_fraction);
COMPONENT_ALIAS(Conserved::scalar, 3, tracer_A);
COMPONENT_ALIAS(Conserved::scalar, 4, tracer_B);
// Components 5-9 remain unnamed but accessible as scalar<5>, scalar<6>, etc.
} // namespace ProblemSpecific

// Define primitive variable types
namespace Primitive
{
VARIABLE(hydro, density);
VARIABLE(hydro, velocity_x);
VARIABLE(hydro, velocity_y);
VARIABLE(hydro, velocity_z);
VARIABLE(hydro, pressure);
VARIABLE(hydro, scalar_1);
VARIABLE(hydro, scalar_2);
} // namespace Primitive

// Define TypeLists for different variable groups
// Use TypeListCat to combine hydro variables with expanded scalar components
using ConservedHydroOnly = quokka::TypeList<Conserved::density, Conserved::momentum_x, Conserved::momentum_y, Conserved::momentum_z, Conserved::energy>;
using ConservedScalarsExpanded = quokka::ExpandMultiVariable_t<Conserved::scalar_all>;
using ConservedTypeList = quokka::TypeListCat_t<ConservedHydroOnly, ConservedScalarsExpanded>;

using PrimitiveTypeList = quokka::TypeList<Primitive::density, Primitive::velocity_x, Primitive::velocity_y, Primitive::velocity_z, Primitive::pressure,
					   Primitive::scalar_1, Primitive::scalar_2>;

// Subset type lists - demonstrate selecting specific scalar components
using ScalarsOnly = quokka::TypeList<Conserved::scalar<0>, Conserved::scalar<1>, Conserved::scalar<2>>;

struct TypedMultifabExample {
}; // dummy problem tag

template <> struct HydroSystem_Traits<TypedMultifabExample> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<TypedMultifabExample> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr int numMassScalars = Conserved::NumScalars;
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
	// BA/DM are automatically extracted from conserved_mf
	quokka::TypedMultifab<ConservedHydroOnly> hydro_mf(conserved_mf);

	// Create a TypedMultifab with only scalars (another subset)
	quokka::TypedMultifab<ScalarsOnly> scalar_mf(conserved_mf);

	// Create a combined TypedMultifab from multiple sources without deep copy
	// BA/DM are automatically extracted from the first source
	using CombinedList = quokka::TypeListCat_t<ConservedHydroOnly, ScalarsOnly>;
	auto combined_mf = quokka::makeTypedMultifab<CombinedList>(hydro_mf, scalar_mf);

	// Print component names to demonstrate functionality
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "TypedMultifab Example:\n";
		amrex::Print() << "======================\n";

		amrex::Print() << "\nConserved variables (including " << Conserved::NumScalars << " scalar components):\n";
		for (const auto &name : conserved_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}

		amrex::Print() << "\nHydro-only subset:\n";
		for (const auto &name : hydro_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}

		amrex::Print() << "\nScalars-only subset (first 3 scalar components):\n";
		for (const auto &name : scalar_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}

		amrex::Print() << "\nCombined from subsets (no deep copy):\n";
		for (const auto &name : combined_mf.component_names()) {
			amrex::Print() << "  - " << name << "\n";
		}

		amrex::Print() << "\nDemonstrating strong type access:\n";
		amrex::Print() << "  - Temperature component name: " << ProblemSpecific::temperature::name() << "\n";
		amrex::Print() << "  - Metallicity component name: " << ProblemSpecific::metallicity::name() << "\n";
		amrex::Print() << "  - Electron fraction component name: " << ProblemSpecific::electron_fraction::name() << "\n";

		amrex::Print() << "\nVerifying no deep copies occurred:\n";
		amrex::Print() << "  Hydro density MultiFab ptr: " << &hydro_mf.getMultiFab<Conserved::density>() << "\n";
		amrex::Print() << "  Combined density MultiFab ptr: " << &combined_mf.getMultiFab<Conserved::density>() << "\n";
		amrex::Print() << "  Same pointer? "
			       << (&hydro_mf.getMultiFab<Conserved::density>() == &combined_mf.getMultiFab<Conserved::density>() ? "YES" : "NO") << "\n";
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
		
		// Initialize all scalar components
		for (int n = 0; n < Conserved::NumScalars; ++n) {
			state(i, j, k, HydroSystem<TypedMultifabExample>::scalar0_index + n) = 0.1 * (n + 1);
		}
	});
}

template <> void QuokkaSimulation<TypedMultifabExample>::computeAfterTimestep()
{
	// Demonstrate gradual migration approach
	if (amrex::ParallelDescriptor::IOProcessor() && timestep_ == 1) {
		amrex::Print() << "\nDemonstrating TypedMultifab with gradual migration:\n";

		for (int level = 0; level <= finest_level; ++level) {
			const auto &ba = grids[level];
			const auto &dm = dmap[level];
			const int nghost = nghost_cc_;

			// Option 1: Direct usage (without migration infrastructure)
			// Create typed view of our current state
			quokka::TypedMultifab<ConservedTypeList> typed_state(ba, dm, nghost, state_new_cc_[level]);

			// Option 2: Using migration infrastructure
			// Sync typed views with existing MultiFabs
			syncTypedMultiFabs<ConservedTypeList>(level);
			
			// Get typed state through migration interface
			auto* typed_states = getTypedStateNew<ConservedTypeList>();
			if (typed_states != nullptr && typed_states->size() > level) {
				auto& typed_state_migration = (*typed_states)[level];
				
				// Both approaches provide the same functionality
				for (amrex::MFIter mfi(state_new_cc_[level]); mfi.isValid(); ++mfi) {
					const auto &bx = mfi.validbox();
					
					// Access through direct typed view
					auto density_arr1 = typed_state.array<Conserved::density>(mfi);
					
					// Access through migration interface
					auto density_arr2 = typed_state_migration.array<Conserved::density>(mfi);
					
					// Both arrays point to the same data (zero-copy)
					amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
						// Verify they're the same
						AMREX_ALWAYS_ASSERT(density_arr1(i, j, k) == density_arr2(i, j, k));
					});
				}
			}

			// Access components in a type-safe way
			for (amrex::MFIter mfi(state_new_cc_[level]); mfi.isValid(); ++mfi) {
				const amrex::Box &bx = mfi.validbox();

				// Get typed arrays for hydro variables
				auto density_arr = typed_state.array<Conserved::density>(mfi);
				auto momx_arr = typed_state.array<Conserved::momentum_x>(mfi);
				auto energy_arr = typed_state.array<Conserved::energy>(mfi);

				// Access multi-component scalars using strong types
				auto temp_arr = typed_state.array<ProblemSpecific::temperature>(mfi);
				auto metal_arr = typed_state.array<ProblemSpecific::metallicity>(mfi);
				auto efrac_arr = typed_state.array<ProblemSpecific::electron_fraction>(mfi);

				// Access unnamed scalar components directly
				auto scalar5_arr = typed_state.array<Conserved::scalar<5>>(mfi);
				auto scalar9_arr = typed_state.array<Conserved::scalar<9>>(mfi);

				// Example: could perform operations with type safety
				// No manual indexing needed!
			}
			
			// Demonstrate creating subsets
			// BA/DM are automatically extracted from typed_state
			quokka::TypedMultifab<ScalarsOnly> scalars_only(typed_state);
			
			// Access just the scalars
			for (amrex::MFIter mfi(state_new_cc_[level]); mfi.isValid(); ++mfi) {
				auto metal_arr = scalars_only.array<ProblemSpecific::metallicity>(mfi);
				// Use metal_arr...
				amrex::ignore_unused(metal_arr);
			}

			// Test AMReX interoperability
			amrex::Print() << "\n=== Testing AMReX interoperability ===\n";
			amrex::MultiFab &density_mf = typed_state.getMultiFab<Conserved::density>();
			const int density_comp = typed_state.getComponentIndex<Conserved::density>();
			amrex::Print() << "Density component index within its MultiFab: " << density_comp << "\n";

			// Test contiguous component iterator
			amrex::Print() << "\n=== Testing contiguous component iterator ===\n";
			const auto component_groups = typed_state.getContiguousComponentGroups();
			amrex::Print() << "Number of contiguous component groups: " << component_groups.size() << "\n";
			
			for (size_t g = 0; g < component_groups.size(); ++g) {
				const auto &group = component_groups[g];
				amrex::Print() << "\nGroup " << g << ": " << group.num_comp << " contiguous components starting at index " 
				               << group.start_comp << "\n";
				amrex::Print() << "Components in this group:\n";
				for (const auto &name : group.component_names) {
					amrex::Print() << "  - " << name << "\n";
				}
				
				// Create alias MultiFab for this group
				amrex::MultiFab alias_mf = quokka::TypedMultifab<ConservedTypeList>::makeAliasMultiFab(group);
				
				// Apply a generic operation to all components in the group
				for (amrex::MFIter mfi(alias_mf); mfi.isValid(); ++mfi) {
					const amrex::Box &bx = mfi.validbox();
					auto const &arr = alias_mf.array(mfi);
					
					// Example: Apply simple reconstruction-like operation
					// In real code, this would be reconstruction, limiting, etc.
					amrex::ParallelFor(bx, group.num_comp,
					[=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
						// Simple example: compute average with neighbors
						if (i > 0 && i < bx.hiVect()[0]) {
							const amrex::Real left = arr(i-1, j, k, n);
							const amrex::Real center = arr(i, j, k, n);
							const amrex::Real right = arr(i+1, j, k, n);
							// In real reconstruction, we'd store this somewhere
							const amrex::Real avg = (left + center + right) / 3.0;
							amrex::ignore_unused(avg);
						}
					});
				}
			}
		}

		amrex::Print() << "TypedMultifab with gradual migration demonstration complete.\n";
	}
}

auto problem_main(int argc, char** argv) -> int
{
	amrex::Initialize(argc, argv);

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