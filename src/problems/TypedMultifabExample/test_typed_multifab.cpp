//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_typed_multifab.cpp
/// \brief Example demonstrating the use of TypedMultifab

#include "AMReX_Arena.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/TypedMultifab.hpp"
#include "util/valarray.hpp"
#include <cmath>

// Problem-specific includes
struct TypedMultifabExample {
};

// Define strongly-typed variables
namespace Conserved
{
VARIABLE(example, density);
VARIABLE(example, momentum_x);
VARIABLE(example, momentum_y);
VARIABLE(example, momentum_z);
VARIABLE(example, energy);
VARIABLE(example, scalar_a);
VARIABLE(example, scalar_b);
} // namespace Conserved

// Define TypeList for our problem
using ConsTypeList = quokka::TypeList<Conserved::density, Conserved::momentum_x, Conserved::momentum_y, Conserved::momentum_z, Conserved::energy,
				       Conserved::scalar_a, Conserved::scalar_b>;

template <> struct Physics_Traits<TypedMultifabExample> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 2; // scalar_a and scalar_b
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};

template <> struct HydroSystem_Traits<TypedMultifabExample> {
	static constexpr bool reconstruct_eint = true;
};

template <> void QuokkaSimulation<TypedMultifabExample>::setInitialConditionsOnGrid(quokka::grid_elem_t &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// Example: Create a TypedMultifab
	amrex::BoxArray ba(indexRange);
	amrex::DistributionMapping dm(ba);
	auto typedMF = quokka::makeTypedMultifab<ConsTypeList>(ba, dm, 0);

	// Fill the typed multifab
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const rho = 1.0;
		amrex::Real const vx = 0.1;
		amrex::Real const vy = 0.0;
		amrex::Real const vz = 0.0;
		amrex::Real const P = 1.0;

		// Access using strongly-typed interface
		auto const &arrays = typedMF.get().arrays();

		// Set conserved variables using type-safe access
		arrays[typedMF.comp<Conserved::density>()](i, j, k) = rho;
		arrays[typedMF.comp<Conserved::momentum_x>()](i, j, k) = rho * vx;
		arrays[typedMF.comp<Conserved::momentum_y>()](i, j, k) = rho * vy;
		arrays[typedMF.comp<Conserved::momentum_z>()](i, j, k) = rho * vz;

		// Set energy (assuming gamma = 5/3)
		amrex::Real const Eint = P / (5.0 / 3.0 - 1.0);
		amrex::Real const Ekin = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		arrays[typedMF.comp<Conserved::energy>()](i, j, k) = Eint + Ekin;

		// Set passive scalars
		arrays[typedMF.comp<Conserved::scalar_a>()](i, j, k) = 0.5;
		arrays[typedMF.comp<Conserved::scalar_b>()](i, j, k) = 0.3;

		// Copy to the regular state array (for compatibility)
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::energy_index) = Eint + Ekin;
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::scalar0_index) = 0.5;
		state_cc(i, j, k, HydroSystem<TypedMultifabExample>::scalar0_index + 1) = 0.3;
	});

	// Demonstrate component name access
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "TypedMultifab component names:\n";
		auto const &names = typedMF.componentNames();
		for (int i = 0; i < static_cast<int>(names.size()); ++i) {
			amrex::Print() << "  Component " << i << ": " << names[i] << "\n";
		}
	}
}

template <> void QuokkaSimulation<TypedMultifabExample>::createInitialParticles()
{
	// No particles in this example
}

template <> void QuokkaSimulation<TypedMultifabExample>::computeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int dcomp) const
{
	// No derived variables in this example
}

template <> void QuokkaSimulation<TypedMultifabExample>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// No error estimation in this example
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<TypedMultifabExample>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<TypedMultifabExample>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<TypedMultifabExample>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<TypedMultifabExample>::nvarTotal_cc;
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

	QuokkaSimulation<TypedMultifabExample> sim(BCs_cc);
	sim.setInitialConditions();
	sim.evolve();

	const int status = 0;
	return status;
}