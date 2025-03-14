#ifndef PARTICLE_DEPOSITION_HPP_
#define PARTICLE_DEPOSITION_HPP_

#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_ParticleMesh.H"
#include "AMReX_REAL.H"
#include "particles/particle_types.hpp"

namespace quokka
{

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
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
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
	amrex::Real Gconst{};  // Gravitational constant
	int start_part_comp{}; // Starting component in particle data
	int start_mesh_comp{}; // Starting component in mesh data
	int num_comp{};	       // Number of components to deposit

	// Operator to perform mass deposition using linear interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &rho,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		// Deposit mass weighted by 4 pi G
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			return 4.0 * M_PI * Gconst * part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		});
	}
};

//-------------------- Supernova depositions --------------------

constexpr double SN_time = 0.0015; // for testing: SN onset time = 1

// Functor for depositing supernova energy and momentum from particles onto the grid
// This is a simplified version of the SNDeposition functor that deposits mass and energy uniformly
// to 5³ cells centered on the particle's cell. It is used for testing purposes.
struct SNDeposition {
	double current_time{};	   // Current simulation time
	int start_part_comp{};	   // Starting component in particle data
	int start_mesh_comp{};	   // Starting component in mesh data
	int birthTimeIndex{};	   // Index for particle birth time
	int evolutionStageIndex{}; // Index for particle evolution stage

	// Operator to perform supernova deposition using cloud-in-cell approach
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &state,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		// Check if the particle has an integer component for evolution stage
		if constexpr (ContainerType::NInt > 0) {
			// Check if this is a supernova progenitor
			bool is_sn_progenitor = false;
			if (evolutionStageIndex >= 0) {
				is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));
			}

			if (is_sn_progenitor && current_time >= p.rdata(birthTimeIndex) + SN_time) {
				// Find the cell containing the particle
				int base_i = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
				int base_j = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
				int base_k = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

				// Calculate the volume factor for normalization (5³ cells)
				const int num_cells = 125; // 5³ cells
				const amrex::Real vol_factor = (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2])) / num_cells;

				static constexpr int stencil_width = 2;

				// Deposit evenly to 5³ cells centered on the particle's cell
				const amrex::Real pmass = p.rdata(start_part_comp) * vol_factor;
				const amrex::Real penergy = pmass; // for testing: energy = mass
				const amrex::Real pmomentum = 0.0; // for testing: momentum = 0

				for (int kk = -stencil_width; kk <= stencil_width; ++kk) {
					for (int jj = -stencil_width; jj <= stencil_width; ++jj) {
						for (int ii = -stencil_width; ii <= stencil_width; ++ii) {
							// Add the contribution to each cell
							// We assume start_mesh_comp is the density index followed by the momentum indices and then the energy
							// index
							amrex::Gpu::Atomic::AddNoRet(&state(base_i + ii, base_j + jj, base_k + kk, start_mesh_comp), pmass);
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
void updateEvolutionStage(ContainerType *container, int lev, amrex::Real current_time, int birthTimeIndex, int evolutionStageIndex)
{
	if (container == nullptr || evolutionStageIndex < 0 || birthTimeIndex < 0) {
		return;
	}

	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		const amrex::Long np = pti.numParticles();

		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
			auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

			// Check if this is a supernova progenitor
			bool is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

			// Update the particle's evolution stage if it's time
			if (is_sn_progenitor && current_time >= p.rdata(birthTimeIndex) + SN_time) {
				p.idata(evolutionStageIndex) = static_cast<int>(StellarEvolutionStage::SNRemnant);
			}
		});
	}
}

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_DEPOSITION_HPP_