#ifndef SINKPARTICLES_HPP_ // NOLINT
#define SINKPARTICLES_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file SinkParticles.hpp
/// \brief Implements the particle container for gravitationally-interacting particles
///

#include "AMReX.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParIter.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_Particles.H"
#include "AMReX_ParticleContainer.H"
#include "AMReX_NeighborParticles.H"

namespace quokka_sinkparticle
{

enum ParticleDataIdx {ParticleMassIdx = 0, ParticleVxIdx, ParticleVyIdx, ParticleVzIdx, ParticleAMxIdx, ParticleAMyIdx, ParticleAMzIdx};
constexpr int SinkParticleRealComps = 7; // mass vx vy vz amx amy amz

using SinkParticleContainer = amrex::AmrParticleContainer<SinkParticleRealComps>;
using SinkParticleIterator = amrex::ParIter<SinkParticleRealComps>;
using NeighborSinkParticleContainer = amrex::NeighborParticleContainer<SinkParticleRealComps, 0>;
  //  using NeighborSinkParticleContainer = amrex::NeighborParticleContainer<SinkParticleRealComps, 0>(gdb, R_merge);

struct SinkDeposition {
	amrex::Real Gconst{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const SinkParticleContainer::ParticleType &p, amrex::Array4<amrex::Real> const &rho,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const SinkParticleContainer::ParticleType &part, int comp) {
					      return 4.0 * M_PI * Gconst * part.rdata(comp); // weight by 4 pi G
				      });
	}
};

  //template<typename problem_t> class NeighborSinkParticleContainer
  //  : public amrex::NeighborParticleContainer<7, 0>
  //{
  
  //public:
  
  //  NeighborSinkParticleContainer(amrex::ParGDBBase * gdb, int R_merge)
  //    : amrex::NeighborParticleContainer<7, 0>(gdb, R_merge)
  //  {}
  //};
    
} // namespace quokka_sinkparticle

#endif // SINKPARTICLES_HPP_
