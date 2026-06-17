#ifndef STARPARTICLE_RADIATION_HPP_
#define STARPARTICLE_RADIATION_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "particles/particle_radiation.hpp" // LuminosityGpuConstTables
#include "particles/particle_types.hpp"	    // StarParticle*Idx, ParticleType, Particle_Traits

#if AMREX_SPACEDIM == 3

namespace quokka
{

// Framework dispatcher for per-particle stellar-evolution updates.
// Passes the full particle real-data array to the model so it can read and
// modify any component (mass, radius, luminosity groups, etc.).
class StellarUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateStellarProperties(ParticleType &p, amrex::Real /*current_time*/, amrex::Real dt,
										LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		using Model = typename Particle_Traits<problem_t>::stellar_model;
		Model::evolve(&p.rdata(0), Nout, dt);
	}
};

} // namespace quokka

#endif // AMREX_SPACEDIM == 3

#endif // STARPARTICLE_RADIATION_HPP_
