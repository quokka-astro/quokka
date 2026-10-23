#ifndef QUOKKA_PARTICLE_TRACER_HPP_
#define QUOKKA_PARTICLE_TRACER_HPP_

#include "AMReX_TracerParticle_mod_K.H"
#include "particles/particle_container.hpp"

namespace quokka
{
// Midpoint advection retains the velocity sampled at the midpoint for output.
inline void AdvectTracers(ParticleContainer &particles, amrex::MultiFab *umac, int lev, amrex::Real dt)
{
	const auto layout = particles.Layout();
	AMREX_ALWAYS_ASSERT(particles.Schema().Population() == ParticleType::Tracer);
	AMREX_ASSERT(particles.OK(lev, lev, umac[0].nGrow() - 1));
	const auto plo = particles.Geom(lev).ProbLoArray();
	const auto dxi = particles.Geom(lev).InvCellSizeArray();
	amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> remapped;
	amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> velocity;
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		AMREX_ASSERT(umac[d].nGrowVect().allGE(1));
		AMREX_ASSERT(!umac[d].contains_nan());
		velocity[d] = &umac[d];
		if (!particles.OnSameGrids(lev, umac[0])) {
			const int ng = umac[d].nGrow();
			remapped[d] = std::make_unique<amrex::MultiFab>(amrex::convert(particles.ParticleBoxArray(lev), amrex::IntVect::TheDimensionVector(d)),
									particles.ParticleDistributionMap(lev), umac[d].nComp(), ng);
			velocity[d] = remapped[d].get();
			velocity[d]->ParallelCopy(umac[d], 0, 0, umac[d].nComp(), ng, ng);
		}
	}
	for (ParticleContainer::ParIterType pti(particles, lev); pti.isValid(); ++pti) {
		auto &tile = pti.GetParticleTile();
		const auto data = tile.getParticleTileData();
		amrex::GpuArray<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> arrays;
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			arrays[d] = (*velocity[d])[pti.index()].const_array();
		}
		amrex::ParallelFor(tile.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
			ParticleView p(data, i, layout);
			if (!p.id().is_valid()) {
				return;
			}
			amrex::GpuArray<amrex::ParticleReal, AMREX_SPACEDIM> oldPosition;
			amrex::GpuArray<amrex::ParticleReal, AMREX_SPACEDIM> v;
			amrex::mac_interpolate(p, plo, dxi, arrays, v.data());
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				oldPosition[d] = p.pos(d);
				p.pos(d) += static_cast<amrex::ParticleReal>(amrex::ParticleReal(0.5) * dt * v[d]);
			}
			amrex::mac_interpolate(p, plo, dxi, arrays, v.data());
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				p.pos(d) = oldPosition[d] + static_cast<amrex::ParticleReal>(dt * v[d]);
				p.rdata(layout.velocity + d) = v[d];
			}
		});
	}
	amrex::Gpu::streamSynchronize();
}
} // namespace quokka
#endif
