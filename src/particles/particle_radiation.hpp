#ifndef PARTICLE_RADIATION_HPP_
#define PARTICLE_RADIATION_HPP_

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_ParticleInterpolators.H"
#include "hydro/hydro_system.hpp"

namespace quokka
{
//-------------------- Radiation depositions --------------------

// // Template trait for luminosity functions - can be specialized for different problems
// template <typename problem_t, typename ContainerType> struct LuminosityTraits {
// 	AMREX_GPU_DEVICE static auto stellarLuminosity(const ContainerType &/*p*/, const amrex::Real /*current_time*/, const int /*massIndex*/, const int
// /*birthTimeIndex*/, const int /*lumIndex*/) -> amrex::GpuArray<Real, Physics_Traits<problem_t>::nGroups>
// 	{
// 		// Default implementation returns array of zeros
// 		amrex::GpuArray<Real, Physics_Traits<problem_t>::nGroups> result{};
// 		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
// 			result[g] = 0.0;
// 		}
// 		return result;
// 	}
// };

// Functor for depositing radiation energy from particles onto the grid
struct RadDeposition {
	double current_time{}; // Current simulation time
	int start_part_comp{}; // Starting component in particle data
	int start_mesh_comp{}; // Starting component in mesh data
	int num_comp{};	       // Number of components to deposit
	int birthTimeIndex{};  // Index for particle birth time

	// Operator to perform radiation deposition using linear interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<Real> const &radEnergySource,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		// Deposit radiation energy only if particle is active
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
					      if (current_time < part.rdata(birthTimeIndex) || current_time >= part.rdata(birthTimeIndex + 1)) {
						      return 0.0;
					      }
					      return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
				      });
	}
};
} // namespace quokka

#endif