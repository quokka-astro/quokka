//==============================================================================
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testSmallScaleDynamo.cpp
/// \brief Small Scale Dynamo simulation.
///

#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include "turbulence/TurbulentDriving.hpp"
#include "util/BC.hpp"

#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include <cmath>

struct SmallScaleDynamo {};

// isothermal EOS: pressure = cs^2 * rho (no thermal energy equation)
template <> struct quokka::EOS_Traits<SmallScaleDynamo> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless sound speed
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<SmallScaleDynamo> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 0;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr amrex::Real gravitational_constant = 1.0;
};

// isothermal EOS has no internal energy to reconstruct; pressure computed directly from rho
template <> struct HydroSystem_Traits<SmallScaleDynamo> {
	static constexpr bool reconstruct_eint = false;
};

// uniform density, zero velocity; turbulent driving (configured in input file) stirs the gas
template <> void QuokkaSimulation<SmallScaleDynamo>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::density_index) = 1.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::energy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::internalEnergy_index) = 0.0;
	});
}

// zero initial magnetic field; a seed field can be added here to study dynamo growth
template <> void QuokkaSimulation<SmallScaleDynamo>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<SmallScaleDynamo>::mhdFirstIndex) = 0.0;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<SmallScaleDynamo>::mhdFirstIndex) = 0.0;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<SmallScaleDynamo>::mhdFirstIndex) = 0.0;
		}
	});
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<SmallScaleDynamo>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<SmallScaleDynamo>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<SmallScaleDynamo> sim(BCs_cc, BCs_fc);

	sim.setInitialConditions();

	sim.evolve();

	return 0;
}
