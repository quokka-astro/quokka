//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_primordial_chem.cpp
/// \brief Defines a test problem for primordial chemistry (microphysics).
///
#include <random>
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
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_primordial_chem.hpp"

#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"

using amrex::Real;

struct PrimordialChemTest {
}; // dummy type to allow compile-type polymorphism via template specialization

// Currently, microphysics uses its own EOS, and this one below is used by hydro. Need to only have one EOS at some point.
template <> struct quokka::EOS_Traits<PrimordialChemTest> {
	static constexpr double gamma = 5. / 3.; // default value
	static constexpr double mean_molecular_weight = quokka::hydrogen_mass_cgs;
	static constexpr double boltzmann_constant = quokka::boltzmann_constant_cgs;
};

template <> struct Physics_Traits<PrimordialChemTest> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false; // in the future, this could point to microphysics, and set to true
	static constexpr int numPassiveScalars = 14;	    // number of chemical species
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};

// constexpr double Tgas0 = 6000.;	   // K
// constexpr double rho0 = 0.6 * m_H; // g cm^-3

template <> void RadhydroSimulation<PrimordialChemTest>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	Real const Lx = (prob_hi[0] - prob_lo[0]);
	Real const Ly = (prob_hi[1] - prob_lo[1]);
	Real const Lz = (prob_hi[2] - prob_lo[2]);

	init_extern_parameters();

	burn_t state;

	Real numdens[NumSpec] = {-1.0};

	for (int n = 1; n <= NumSpec; ++n) {
		switch (n) {

			case 1:
				numdens[n - 1] = primary_species_1;
				break;
			case 2:
				numdens[n - 1] = primary_species_2;
				break;
			case 3:
				numdens[n - 1] = primary_species_3;
				break;
			case 4:
				numdens[n - 1] = primary_species_4;
				break;
			case 5:
				numdens[n - 1] = primary_species_5;
				break;
			case 6:
				numdens[n - 1] = primary_species_6;
				break;
			case 7:
				numdens[n - 1] = primary_species_7;
				break;
			case 8:
				numdens[n - 1] = primary_species_8;
				break;
			case 9:
				numdens[n - 1] = primary_species_9;
				break;
			case 10:
				numdens[n - 1] = primary_species_10;
				break;
			case 11:
				numdens[n - 1] = primary_species_11;
				break;
			case 12:
				numdens[n - 1] = primary_species_12;
				break;
			case 13:
				numdens[n - 1] = primary_species_13;
				break;
			case 14:
				numdens[n - 1] = primary_species_14;
				break;
		}
	}

	state.T = temperature;

	// find the density in g/cm^3
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n]; // spmasses contains the masses of all species, defined in EOS
	}

	state.rho = rhotot;

	// normalize -- just in case

	Real mfracs[NumSpec] = {-1.0};
	Real msum = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		mfracs[n] = state.xn[n] * spmasses[n] / rhotot;
		msum += mfracs[n];
	}

	for (int n = 0; n < NumSpec; ++n) {
		mfracs[n] /= msum;
		// use the normalized mass fractions to obtain the corresponding number densities
		state.xn[n] = mfracs[n] * rhotot / spmasses[n];
	}

	// call the EOS to set initial internal energy e
	eos(eos_input_rt, state);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
		Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];
		Real const z = prob_lo[2] + (k + Real(0.5)) * dx[2];

		Real rho = state.rho; // g cm^-3

		Real xmom = 0;
		Real ymom = 0;
		Real zmom = 0;
		Real Eint = state.e;

		Real const Egas = RadSystem<PrimordialChemTest>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<PrimordialChemTest>::x3Momentum_index) = zmom;

		for (int nn = 0; nn < NumSpec; ++nn){
			state_cc(i, j, k, HydroSystem<PrimordialChemTest>::scalar0_index + nn) = mfracs[nn];
		}
	});
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.25;
	const double max_time = 5e16; // > 1 Gyr
	const int max_timesteps = 5;

	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<PrimordialChemTest>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::foextrap); // extrapolate
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		BCs_cc[n].setLo(1, amrex::BCType::foextrap);
		BCs_cc[n].setHi(1, amrex::BCType::foextrap);
#if AMREX_SPACEDIM == 3
		BCs_cc[n].setLo(2, amrex::BCType::foextrap);
		BCs_cc[n].setHi(2, amrex::BCType::foextrap);
#endif
	}

	RadhydroSimulation<PrimordialChemTest> sim(BCs_cc);

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
	int status = 0;
	return status;
}
