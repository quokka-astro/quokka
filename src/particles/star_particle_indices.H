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
	   vx,			// Velocity x
	   vy,			// Velocity y
	   vz,			// Velocity z
	   birth_time,		// Simulation time when the particle was created
	   mdot,		// Current mass accretion rate (set by the accretion module)
	   radius,		// Stellar radius (set by the stellar-evolution model)
	   lum			// Base index for luminosity components (MUST be last; expands to lum_0, lum_1, ... for nGroups)
);

constexpr int StarParticleMassIdx = static_cast<int>(StarParticleDataIdx::mass);
constexpr int StarParticleVxIdx = static_cast<int>(StarParticleDataIdx::vx);
constexpr int StarParticleVyIdx = static_cast<int>(StarParticleDataIdx::vy);
constexpr int StarParticleVzIdx = static_cast<int>(StarParticleDataIdx::vz);
constexpr int StarParticleBirthTimeIdx = static_cast<int>(StarParticleDataIdx::birth_time);
constexpr int StarParticleMdotIdx = static_cast<int>(StarParticleDataIdx::mdot);
constexpr int StarParticleRadiusIdx = static_cast<int>(StarParticleDataIdx::radius);
constexpr int StarParticleLumIdx = static_cast<int>(StarParticleDataIdx::lum);

static_assert(static_cast<int>(StarParticleDataIdx::lum) == 7);

// Number of fixed components before the variable-length sections (nGroups + extras).
constexpr int StarParticleFixedComps = 7;

} // namespace quokka

#endif // STAR_PARTICLE_INDICES_H_
