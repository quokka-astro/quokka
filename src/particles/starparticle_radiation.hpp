#ifndef STARPARTICLE_RADIATION_HPP_
#define STARPARTICLE_RADIATION_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "particles/particle_types.hpp" // StarParticle*Idx, ParticleType, Particle_Traits

#if AMREX_SPACEDIM == 3

namespace quokka
{

// Framework dispatcher for per-particle stellar-evolution updates.
// Passes the full particle real- and integer-data arrays to the model so
// it can read and modify any component.
class StellarUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateStellarProperties(ParticleType &p, amrex::Real /*current_time*/, amrex::Real dt) noexcept
	{
		using Model = typename Particle_Traits<problem_t>::stellar_model;
		if constexpr (ParticleType::NInt > 0) {
			Model::evolve(&p.rdata(0), &p.idata(0), Nout, dt);
		} else {
			Model::evolve(&p.rdata(0), nullptr, Nout, dt);
		}
	}
};

} // namespace quokka

#endif // AMREX_SPACEDIM == 3

#endif // STARPARTICLE_RADIATION_HPP_
