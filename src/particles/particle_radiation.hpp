#ifndef PARTICLE_RADIATION_HPP_
#define PARTICLE_RADIATION_HPP_

#include <algorithm>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"

//-------------------- Radiation depositions --------------------

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

// A simple luminosity function for testing purpose. Keep it linear function of mass for easy answer validation.
// L/(M / M_sun) = L_sun = 4e33 erg/s
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto simple_luminosity(const Real mass, const Real age, const int group) -> Real
{
	const double is_on = age < 1.0e14 ? 1.0 : 0.0;		   // 3 Myr
	return 4.0e33 * (mass / C::M_solar) * (group + 1) * is_on; // erg / s
}

// Functor for depositing radiation energy from particles onto the grid using mass-based luminosity
struct MassBasedRadDeposition {
	double current_time{}; // Current simulation time
	int massIndex{};       // Index for particle mass
	int start_mesh_comp{}; // Starting component in mesh data
	int num_comp{};	       // Number of components to deposit
	int birthTimeIndex{};  // Index for particle birth time

	// Operator to perform mass-based radiation deposition using linear interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<Real> const &radEnergySource,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, radEnergySource, massIndex, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			const Real age = current_time - part.rdata(birthTimeIndex);
			return simple_luminosity(part.rdata(massIndex), age, comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		});
	}
};

#endif