//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testBinaryOrbitCIC.cpp
/// \brief Defines a test problem for a binary orbit with particle-only
///        to test the particle-only timestepping (no HD or radiation active).
///

#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"

#include <AMReX_Math.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <iomanip>

struct BinaryCICTimestep {
};

template <> struct Particle_Traits<BinaryCICTimestep> : DefaultParticleTraits {  
    static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct Physics_Traits<BinaryCICTimestep> :DefaultPhysicsTraits {
    static constexpr bool is_hydro_enabled        = false;
	static constexpr bool is_radiation_enabled    = false;
	static constexpr bool is_self_gravity_enabled = true;  
    static constexpr int nGroups                  = 0;
	static constexpr int numMassScalars           = 0;
	static constexpr int numPassiveScalars        = 0;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};


template <> void QuokkaSimulation<BinaryCICTimestep>::createInitialCICParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4;
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("../inputs/BinaryCICTimestep_particles.txt", nreal_extra, nullptr);
}

auto problem_main() -> int {
	int status = 0;
	QuokkaSimulation<BinaryCICTimestep> sim;
	sim.initialize();
	sim.readParameters();
	sim.setInitialConditions();
	sim.evolve();

	return status;
}


