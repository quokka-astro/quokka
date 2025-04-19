#ifndef PARTICLE_DEPOSITION_HPP_
#define PARTICLE_DEPOSITION_HPP_

#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_ParticleMesh.H"
#include "AMReX_REAL.H"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"

namespace quokka
{

//-------------------- Radiation depositions --------------------

// Functor for depositing radiation energy from particles onto the grid
struct RadDeposition {
	Real current_time{};   // Current simulation time
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

#if AMREX_SPACEDIM == 3

//-------------------- Mass depositions --------------------

// Functor for depositing particle mass onto the grid
struct MassDeposition {
	Real Gconst{};	       // Gravitational constant
	int start_part_comp{}; // Starting component in particle data
	int start_mesh_comp{}; // Starting component in mesh data
	int num_comp{};	       // Number of components to deposit

	// Operator to perform mass deposition using linear interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<Real> const &rho,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		// Deposit mass weighted by 4 pi G
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			return 4.0 * M_PI * Gconst * part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		});
	}
};

//-------------------- Supernova depositions --------------------

// Functor for depositing supernova energy and momentum from particles onto the grid
// This is a simplified version of the SNDeposition functor that deposits mass and energy uniformly
// to 5³ cells centered on the particle's cell. It is used for testing purposes.
// Note: the deposition radius must be <= nghost_cc_
struct SNDeposition {
	Real step_end_time{};	   // Current simulation time
	int start_part_comp{};	   // Starting component in particle data
	int start_mesh_comp{};	   // Starting component in mesh data
	int birthTimeIndex{};	   // Index for particle birth time
	int evolutionStageIndex{}; // Index for particle evolution stage
	Real SN_time = particle_param2;

	// For some unknown reason, stencil_width < 3 results in larger error in SNR mass when a particle is at the refinement boundary.
	static constexpr int stencil_width = 3;

	// Abort if stencil_width > nghost_cc_ - 1.
	// The particle can drift one cell out of the valid zone during kickParticlesAllLevels().
	// A stencil_width > nghost_cc_ - 1 would result in particles depositing energy/momentum outside the ghost zones.
	// We can't use AMRSimulation<problem_t>::nghost_cc_ and have to hard-code 3 here because we don't have a problem_t template parameter.
	static_assert(stencil_width <= 3, "stencil_width must be <= nghost_cc_");

	// Operator to perform supernova deposition using cloud-in-cell approach
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<Real> const &state,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		// Check if the particle has an integer component for evolution stage
		if constexpr (ContainerType::NInt > 0) {
			// Check if this is a supernova progenitor
			bool is_sn_progenitor = false;
			if (evolutionStageIndex >= 0) {
				is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));
			}

			if (is_sn_progenitor && step_end_time > p.rdata(birthTimeIndex) + SN_time) {
				// Find the cell containing the particle
				int base_i = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
				int base_j = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
				int base_k = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

				// Calculate the volume factor for normalization (5³ cells)
				const int num_cells = (2 * stencil_width + 1) * (2 * stencil_width + 1) * (2 * stencil_width + 1);
				const Real vol_factor = (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2])) / num_cells;

				// Deposit evenly to 5³ cells centered on the particle's cell
				const Real pdensity = p.rdata(start_part_comp) * vol_factor;
				const Real penergy = pdensity; // for testing: energy = mass
				const Real pmomentum = 0.0;    // for testing: momentum = 0

				for (int kk = -stencil_width; kk <= stencil_width; ++kk) {
					for (int jj = -stencil_width; jj <= stencil_width; ++jj) {
						for (int ii = -stencil_width; ii <= stencil_width; ++ii) {
							// Add the contribution to each cell
							// We assume start_mesh_comp is the density index followed by the momentum indices and then the energy
							// index
							amrex::Gpu::Atomic::AddNoRet(&state(base_i + ii, base_j + jj, base_k + kk, start_mesh_comp), pdensity);
							amrex::Gpu::Atomic::AddNoRet(&state(base_i + ii, base_j + jj, base_k + kk, start_mesh_comp + 1),
										     pmomentum);
							amrex::Gpu::Atomic::AddNoRet(&state(base_i + ii, base_j + jj, base_k + kk, start_mesh_comp + 2),
										     pmomentum);
							amrex::Gpu::Atomic::AddNoRet(&state(base_i + ii, base_j + jj, base_k + kk, start_mesh_comp + 3),
										     pmomentum);
							amrex::Gpu::Atomic::AddNoRet(&state(base_i + ii, base_j + jj, base_k + kk, start_mesh_comp + 4),
										     penergy);
						}
					}
				}

				// Note: We cannot modify the particle here because it's passed as const reference
				// The evolution stage update is now handled by the updateEvolutionStage function
			}
		}
	}
};

// Function to update particle evolution stages from SNProgenitor to SNRemnant
template <typename ContainerType>
void updateEvolutionStage(ContainerType *container, int lev_min, Real step_end_time, int birthTimeIndex, int evolutionStageIndex)
{
	if (container == nullptr || evolutionStageIndex < 0 || birthTimeIndex < 0) {
		return;
	}

	const Real SN_time = particle_param2;

	for (int lev = lev_min; lev <= container->finestLevel(); ++lev) {
		for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
			auto &particles = pti.GetArrayOfStructs();
			auto *pData = particles().data();
			const amrex::Long np = pti.numParticles();

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
				auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

				// Check if this is a supernova progenitor
				bool is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

				// Update the particle's evolution stage if it's time
				if (is_sn_progenitor && step_end_time > p.rdata(birthTimeIndex) + SN_time) {
					p.idata(evolutionStageIndex) = static_cast<int>(StellarEvolutionStage::SNRemnant);
				}
			});
		}
	}
}

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_DEPOSITION_HPP_