#ifndef SINK_PARTICLES_HPP_ // NOLINT
#define SINK_PARTICLES_HPP_
//==============================================================================
// ...
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file sink_particles.hpp
/// \brief Define a class for sink particles calculations.

#include <limits>

#include "AMReX.H"
#include "AMReX_AmrCore.H"

namespace quokka::sink_particles
{

  template <typename problem_t> void createSinkParticles(int lev, amrex::MultiFab &mf)
  {
    BL_PROFILE("createSinkParticles()")
      for (amrex::MFIter iter(mf(lev)); iter.isValid(); ++iter) {
	 const amrex::Box &indexRange = iter.validbox();
	 auto const &state = mf.array(iter);
	 auto const &nsubsteps = nsubstepsMF.array(iter);
	 const Real gamma = quokka::EOS_Traits<problem_t>::gamma;
	 
	 amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	   const Real Egas = state(i, j, k, HydroSystem<problem_t>::energy_index);
	   if (Egas <= 0) {
	       c = C_ISO;
	     }
	   else {
	     const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
	     const Real x1Mom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
	     const Real x2Mom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
	     const Real x3Mom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
	     const Real x1B = state(i, j, k, HydroSystem<problem_t>::x1Bfield_index);
	     const Real x2B = state(i, j, k, HydroSystem<problem_t>::x2Bfield_index);
	     const Real x3B = state(i, j, k, HydroSystem<problem_t>::x3Bfield_index);
	     const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, x1B, x2B, x3B, Egas);
	     c = sqrt(gamma*((gamma-1.d0)*max(Eint,SMALL_PR)/rho;
	     rho_j = pi * pow(c/(sqrt(G)*dx/jeansNo),2);
	     b2 = x1B*x1B + x2B*x2B + x3B*x3B;
	     if (Egas <= 0)
	       {
		 beta = 2.d0*rho *C_ISO*C_ISO / max(b2, 1.d-100)
	       }
	     else
	       {
		 pres = (gamma -1.d0)*Eint;
		 beta = 2.d0 * pres / max(b2, 1.d-100);
	       }

	       rho_J=rho_J * (1.d0 + (7.4d-1/beta));
	       if (rho > rho_J)
		 {
		   ParticleType p;
		   p.id()      = ParticleType::NextID();
		   p.cpu()   = ParallelDescriptor::MyProc();
		   p.pos(0) = ;
		   p.pos(1) = ;
		   p.pos(2) = ;

		   // AoS real data
		   
		   if (Egas > 0) especific = (Egas - 0.5d0*b2) / rho;
		   
		   // set sink particle mass
                   p.rdata(0) = (rho - rho_J) * vol
		   state(i, j, k, HydroSystem<problem_t>::density_index) = rho_J;

		   // set sink velocioty equal to the velocity of the gas in the cell,
		   // and remove an equal amount of momentum from the cell.
		   p.rdata(1) = x1Mom * (1.0d0 - rho_J / rho ) * vol;
		   p.rdata(2) = x2Mom * (1.0d0 - rho_J / rho ) * vol;
		   p.rdata(3) = x3Mom * (1.0d0 - rho_J / rho ) * vol;
		   state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = x1Mom * rho_J / rho;
		   state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = x2Mom * rho_J / rho;
		   state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = x3Mom * rho_J / rho;

		   // Remove the appropriate amount of thermal energy from the cell.
		   if (Egas > 0)
		     {
		       state(i, j, k, HydroSystem<problem_t>::energy_index) = especific * rho_J +
			 0.5d0 * b2;
		     }
		 }
	     }
	}
    }
} // namespace qukka:: sink_particles

#endif // SINKPARTICLES_HPP_


