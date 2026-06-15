#ifndef STARPARTICLE_RADIATION_HPP_
#define STARPARTICLE_RADIATION_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "particles/particle_radiation.hpp" // LuminosityGpuConstTables
#include "particles/particle_types.hpp"     // StarParticle*Idx, ParticleType, Particle_Traits

#if AMREX_SPACEDIM == 3

namespace quokka
{

// Framework dispatcher for per-particle stellar-evolution updates.
// Reads the particle's mass and accretion rate, calls the model selected by
// Particle_Traits<problem_t>::stellar_model, and stores the resulting radius and luminosity.
class StellarUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateStellarProperties(ParticleType &p, amrex::Real /*current_time*/, amrex::Real dt,
										LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		using Model = typename Particle_Traits<problem_t>::stellar_model;

		const amrex::Real mass = p.rdata(StarParticleMassIdx);
		const amrex::Real mdot = p.rdata(StarParticleMdotIdx);

		amrex::Real radius_val = 0.0;
		amrex::Real lum_val = 0.0;
		Model::evolve(mass, mdot, dt, radius_val, lum_val);

		p.rdata(StarParticleRadiusIdx) = radius_val;
		p.rdata(StarParticleLumIdx) = lum_val; // first luminosity slot (nGroups==1 for the toy model)
	}
};

} // namespace quokka

#endif // AMREX_SPACEDIM == 3

#endif // STARPARTICLE_RADIATION_HPP_
