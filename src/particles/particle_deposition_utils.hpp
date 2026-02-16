#ifndef PARTICLE_DEPOSITION_UTILS_HPP_
#define PARTICLE_DEPOSITION_UTILS_HPP_

#include "AMReX_MultiFab.H"
#include "AMReX_ParticleMesh.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"

namespace quokka
{

//==============================================================================
// Particle Property Deposition Utilities
//==============================================================================

/// Functor for depositing particle mass density
struct ParticleMassDensityDeposition {
	int mass_comp{};
	int start_mesh_comp{};
	int num_comp{};

	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &deposition_array,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, deposition_array, mass_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		});
	}
};

//==============================================================================
// Utility Functions for Particle Deposition
//==============================================================================

/// Deposit particle mass density for a given particle type
template <typename ContainerType>
void depositParticleMassDensity(ContainerType *container, amrex::MultiFab &deposition_field, int lev, int mass_comp, int start_mesh_comp = 0)
{
	const BL_PROFILE("depositParticleMassDensity");

	ParticleMassDensityDeposition deposition_functor;
	deposition_functor.mass_comp = mass_comp;
	deposition_functor.start_mesh_comp = start_mesh_comp;
	deposition_functor.num_comp = 1;

	amrex::ParticleToMesh(*container, deposition_field, lev, deposition_functor, false);
}

} // namespace quokka

#endif // PARTICLE_DEPOSITION_UTILS_HPP_
