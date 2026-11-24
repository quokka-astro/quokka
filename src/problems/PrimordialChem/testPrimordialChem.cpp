// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testPrimordialChem.cpp
/// \brief Defines a test problem for primordial chemistry.
///
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "util/BC.hpp"
#include <array>
#include <format>
#include <random>
#include <string>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "hydro/hydro_system.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "networks/primordial_chem/network.hpp"
#include "radiation/radiation_system.hpp"

#include "chemistry/Chemistry.hpp"
#include "microphysics/burn_state.hpp"

using amrex::Real;

struct PrimordialChemTest {}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Microphysics_Traits<PrimordialChemTest> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::PrimordialChem;
	static constexpr bool chemistry = true;
	using Tableau = quokka::math::rosenbrock::Rodas5P;
};

using Network = Microphysics_Traits<PrimordialChemTest>::Network;
using amrex::literals::operator""_rt;

template <> struct Physics_Traits<PrimordialChemTest> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = Network::NumSpec;	     // number of chemical species
	static constexpr int numPassiveScalars = numMassScalars + 0; // we only have mass scalars
};

template <> struct SimulationData<PrimordialChemTest> {
	amrex::Real temperature;
	/// the initial number density of each species [cm^-3] (primordial_chem.primary_species_<n + 1>)
	std::array<amrex::Real, Network::NumSpec> primary_species{};
};

template <> void QuokkaSimulation<PrimordialChemTest>::preCalculateInitialConditions()
{
	// parmparse species and temperature
	amrex::ParmParse const pp("primordial_chem");
	userData_.temperature = 1e1;
	pp.query("temperature", userData_.temperature);

	// defaults: all electrons (species 0), as before the port; the input keys number the species from 1
	userData_.primary_species.fill(0.0_rt);
	userData_.primary_species[Network::E] = 1.0_rt;
	for (int n = 0; n < Network::NumSpec; ++n) {
		pp.query(("primary_species_" + std::to_string(n + 1)), userData_.primary_species[n]);
	}
}

template <> void QuokkaSimulation<PrimordialChemTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	quokka::microphysics::BurnState<Network::NumSpec> state;

	const std::array<Real, Network::NumSpec> numdens = userData_.primary_species;

	state.T = userData_.temperature;

	// find the density in g/cm^3
	Real rhotot = 0.0_rt;
	for (int n = 0; n < Network::NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * Network::SpeciesMass()[n];
	}

	state.rho = rhotot;

	// normalize -- just in case

	std::array<Real, Network::NumSpec> mfracs = {-1.0};
	Real msum = 0.0_rt;
	for (int n = 0; n < Network::NumSpec; ++n) {
		mfracs[n] = state.xn[n] * Network::SpeciesMass()[n] / rhotot;
		msum += mfracs[n];
	}

	for (int n = 0; n < Network::NumSpec; ++n) {
		mfracs[n] /= msum;
		// use the normalized mass fractions to obtain the corresponding number densities
		state.xn[n] = mfracs[n] * rhotot / Network::SpeciesMass()[n];
	}

	// the initial specific internal energy
	state.e = quokka::EOSMultigamma<PrimordialChemTest>::EnergyFromTemperature(state.T, state.xn);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = state.rho; // g cm^-3

		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		// the EOS gives the specific internal energy, so multiply it by rho for Quokka
		Real const Eint = rho * state.e;

		static_assert(!Physics_Traits<PrimordialChemTest>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
		Real const Egas = quokka::EOS<PrimordialChemTest>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint, 0.0);

		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::x3Momentum_index) = zmom;

		for (int nn = 0; nn < Network::NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<PrimordialChemTest>::scalar0_index + nn) =
			    mfracs[nn] * rho; // we use partial densities and not mass fractions
		}
	});
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.25;
	const double max_time = 5e16; // > 1 Gyr
	const int max_timesteps = 5;

	QuokkaSimulation<PrimordialChemTest> sim;

	// Standard PPM gives unphysically enormous temperatures when used for
	// this problem (e.g., ~1e14 K or higher), but can be fixed by
	// reconstructing the temperature instead of the pressure
	sim.reconstructionOrder_ = 3; // PLM

	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = 100;
	sim.checkpointInterval_ = -1;

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Cleanup and exit
	int const status = 0;
	return status;
}
