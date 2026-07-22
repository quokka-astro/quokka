#ifndef STAR_PARTICLE_INDICES_H_
#define STAR_PARTICLE_INDICES_H_

#include "AMReX_Enum.H"

namespace quokka
{

// Star particle real-component layout.
// Every Star particle has StarParticleFixedComps fixed scalars followed by
// nGroups luminosity slots and model-defined extras (nExtraReal / nExtraInt).
AMREX_ENUM(StarParticleDataIdx, // NOLINT
	   mass,		// Mass of the particle
	   vx,			// Velocity in x direction
	   vy,			// Velocity in y direction
	   vz,			// Velocity in z direction
	   birth_time,		// Simulation time when the particle formed
	   death_time,		// Simulation time when the particle becomes inactive (-1 if unset)
	   amx,			// Angular momentum in x direction (accreted from gas)
	   amy,			// Angular momentum in y direction (accreted from gas)
	   amz,			// Angular momentum in z direction (accreted from gas)
	   mdeut,		// Mass of gas that still contains deuterium
	   n,			// Polytropic index
	   mdot,		// Current mass accretion rate (set by the accretion module)
	   radius,		// Stellar radius (set by the stellar-evolution model)
	   lum			// Base index for luminosity components (MUST be last; expands to lum_0, lum_1, ... for nGroups)
);

constexpr int StarParticleMassIdx = static_cast<int>(StarParticleDataIdx::mass);
constexpr int StarParticleVxIdx = static_cast<int>(StarParticleDataIdx::vx);
constexpr int StarParticleVyIdx = static_cast<int>(StarParticleDataIdx::vy);
constexpr int StarParticleVzIdx = static_cast<int>(StarParticleDataIdx::vz);
constexpr int StarParticleBirthTimeIdx = static_cast<int>(StarParticleDataIdx::birth_time);
constexpr int StarParticleDeathTimeIdx = static_cast<int>(StarParticleDataIdx::death_time);
constexpr int StarParticleAmxIdx = static_cast<int>(StarParticleDataIdx::amx);
constexpr int StarParticleAmyIdx = static_cast<int>(StarParticleDataIdx::amy);
constexpr int StarParticleAmzIdx = static_cast<int>(StarParticleDataIdx::amz);
constexpr int StarParticleMdeutIdx = static_cast<int>(StarParticleDataIdx::mdeut);
constexpr int StarParticleNIdx = static_cast<int>(StarParticleDataIdx::n);
constexpr int StarParticleMdotIdx = static_cast<int>(StarParticleDataIdx::mdot);
constexpr int StarParticleRadiusIdx = static_cast<int>(StarParticleDataIdx::radius);
constexpr int StarParticleLumIdx = static_cast<int>(StarParticleDataIdx::lum);

static_assert(static_cast<int>(StarParticleDataIdx::lum) == 13);

// Number of fixed components before the variable-length sections (nGroups + model extras).
constexpr int StarParticleFixedComps = 13;

// Star particle integer-component layout.
// burnState tracks the deuterium-burning state machine (see quokka::burningState);
// a model may declare additional int components via nExtraInt.
AMREX_ENUM(StarParticleIntIdx, // NOLINT
	   burnState	       // Nuclear burning state
);

constexpr int StarParticleBurnStateIdx = static_cast<int>(StarParticleIntIdx::burnState);

// Number of fixed integer components before any model-defined extras.
constexpr int StarParticleFixedIntComps = 1;

} // namespace quokka

#endif // STAR_PARTICLE_INDICES_H_
