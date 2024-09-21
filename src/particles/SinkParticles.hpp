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
using NeighborSinkParticleContainer = amrex::NeighborParticleContainer<SinkParticleRealComps, 10>;
  
struct SinkParticleStruct
{
  Real pos[3];
  Real mass;
  Real vx, vy, vz;
  Real amx, amy, amz;
  long id;
  int cpu;
};

  void combine_particles(SinkParticleContainer& particles,
			  NeighborSinkParticleContainer& neighbor_particles,
			  std::vector<quokka_sinkparticle::SinkParticleContainer::ParticleType>& combined_particles)
  {
    // Add particles from the main SinkParticleContainer
    for (auto it = particles.begin(); it != particles.end(); ++it)
      {
	combined_particles.push_back(*it);
      }

    // Add particles from the NeighborSinkPartilcleContainer.
    for (auto it = neighbor_particles.begin(); it != neighbor_particles.end(); ++it)
      {
	combined_particles.push_back(*it);
      }
  }

  void convert_to_array_of_structs(const std::vector<quokka_sinkparticle::SinkParticleContainer::ParticleType>& combined_particles, std::vector<SinkParticleStruct>& particle_array)
  {
    for (const auto& p : combined_particles)
      {
	SinkParticleStruct ps;
	ps.pos[0] = p.pos(0);
	ps.pos[1] = p.pos(1);
	ps.pos[2] = p.pos(2);
        ps.vx = p.rdata(1);
        ps.vy = p.rdata(2);
        ps.vz = p.rdata(3);
        ps.amx = p.rdata(4);
        ps.amy = p.rdata(5);
        ps.amz = p.rdata(6);
	ps.id = p.id();
	ps.cpu = p.cpu();
	particle_array.push_back(ps);
      }
  }

void merge_particles(std::vector<SinkParticleStruct>& combined_particles, Real merge_distance, std::unordered_map<long, bool>& merged_map, std::vector<SinkParticleStruct>& local_particles)
{
    // Sort combined_particles by mass in descending order
    std::sort(combined_particles.begin(), combined_particles.end(),
              [](const SinkParticleStruct& a, const SinkParticleStruct& b)
              {
                  return a.mass > b.mass;
              });

    std::vector<SinkParticleStruct> merged_particles;

    for (const auto& particle : combined_particles)
    {
        if (merged_map[particle.id]) continue; // Skip if already merged

        SinkParticleStruct merged_particle = particle;
        bool is_local = (particle.cpu == ParallelDescriptor::MyProc());

        for (auto it = local_particles.begin(); it != local_particles.end();)
        {
            if (merged_map[it->id]) 
            {
                ++it;
                continue; // Skip if already merged
            }

            Real dist = std::sqrt((merged_particle.pos[0] - it->pos[0]) * (merged_particle.pos[0] - it->pos[0]) +
                                  (merged_particle.pos[1] - it->pos[1]) * (merged_particle.pos[1] - it->pos[1]) +
                                  (merged_particle.pos[2] - it->pos[2]) * (merged_particle.pos[2] - it->pos[2]));
            if (dist < merge_distance)
            {
                // Merge particles
	        merged_particle.pos[0] = (merged_particle.pos[0]*merged_particle.mass+ it->pos[0]*it->mass) / (merged_particle.mass+it->mass);
	        merged_particle.pos[1] = (merged_particle.pos[1]*merged_particle.mass+ it->pos[1]*it->mass) / (merged_particle.mass+it->mass);
	        merged_particle.pos[2] = (merged_particle.pos[2]*merged_particle.mass+ it->pos[2]*it->mass) / (merged_particle.mass+it->mass);
                merged_particle.mass = merged_particle.mass + it->mass;
                merged_particle.vx = merged_particle.vx*merged_particle.mass + it->vx*it->mass/(merged_particle.mass+it->mass);
                merged_particle.vy = merged_particle.vy*merged_particle.mass + it->vy*it->mass/(merged_particle.mass+it->mass);
                merged_particle.vz = merged_particle.vz*merged_particle.mass + it->vz*it->mass/(merged_particle.mass+it->mass);
                merged_particle.amx = merged_particle.amx + it->amx;
                merged_particle.amy = merged_particle.amy + it->amy;
                merged_particle.amz = merged_particle.amz + it->amz;

                // Mark the particle as merged
                merged_map[it->id] = true;
                it = local_particles.erase(it); // Remove merged particle from local_particles
            }
            else
            {
                ++it;
            }
        }

	if (is_local)
        {
            merged_particles.push_back(merged_particle);
        }	
    }

    local_particles = std::move(merged_particles);
}
  
void update_particle_container(ParticleContainer<7, 0, 0, 0>& particles, const std::vector<SinkParticleStruct>& local_particles)
{
    // Clear the existing particles in the ParticleContainer
    particles.clearParticles();

    // Add the new local_particles to the ParticleContainer
    for (const auto& ps : local_particles)
    {
        SinkParticleType p;
        p.id() = ps.id;
        p.cpu() = ps.cpu;
        p.pos(0) = ps.pos[0];
        p.pos(1) = ps.pos[1];
        p.pos(2) = ps.pos[2];
        p.rdata(0) = ps.mass;
        p.rdata(1) = ps.mx;
        p.rdata(2) = ps.my;
        p.rdata(3) = ps.mz;
        p.rdata(4) = ps.amx;
        p.rdata(5) = ps.amy;
        p.rdata(6) = ps.amz;
        particles.push_back(p);
    }

    // Redistribute particles to ensure correct placement
    particles.Redistribute();
}
    
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
  
} // namespace quokka_sinkparticle

#endif // SINKPARTICLES_HPP_
