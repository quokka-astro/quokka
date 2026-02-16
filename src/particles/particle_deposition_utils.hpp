#ifndef PARTICLE_DEPOSITION_UTILS_HPP_
#define PARTICLE_DEPOSITION_UTILS_HPP_

#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "AMReX_MultiFab.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "particles/particle_types.hpp"

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

/// Functor for depositing particle momentum density
struct ParticleMomentumDensityDeposition {
	int mass_comp{};
	int vel_start_comp{};
	int start_mesh_comp{};

	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &deposition_array,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		const amrex::Real cell_volume_inv = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
		const int local_mass_comp = mass_comp;

		// Deposit momentum components (mass * velocity)
		for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
			interp.ParticleToMesh(p, deposition_array, vel_start_comp + dim, start_mesh_comp + dim, 1,
					      [=] AMREX_GPU_DEVICE(const ContainerType &part, int vel_comp) {
						      const amrex::Real mass = part.rdata(local_mass_comp);
						      const amrex::Real velocity = part.rdata(vel_comp);
						      return mass * velocity * cell_volume_inv;
					      });
		}
	}
};

/// Functor for depositing particle kinetic energy density
struct ParticleKineticEnergyDensityDeposition {
	int mass_comp{};
	int vel_start_comp{};
	int start_mesh_comp{};
	int num_comp{};

	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &deposition_array,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		const int local_mass_comp = mass_comp;
		const int local_vel_start_comp = vel_start_comp;
		const amrex::Real cell_volume_inv = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
		interp.ParticleToMesh(p, deposition_array, mass_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			const amrex::Real mass = part.rdata(local_mass_comp);
			const amrex::Real vx = part.rdata(local_vel_start_comp);
			const amrex::Real vy = part.rdata(local_vel_start_comp + 1);
			const amrex::Real vz = part.rdata(local_vel_start_comp + 2);
			const amrex::Real kinetic_energy = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
			return kinetic_energy * cell_volume_inv;
		});
	}
};

/// Functor for depositing particle number density
struct ParticleNumberDensityDeposition {
	int start_mesh_comp{};
	int num_comp{};

	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &deposition_array,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, deposition_array, 0, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			// Deposit unit weight for number density
			return 1.0 * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
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

	container->template ParticleToMesh<amrex::ParticleInterpolator::Linear>(*container, deposition_field, lev, deposition_functor, false, false);
}

/// Deposit particle momentum density for a given particle type
template <typename ContainerType>
void depositParticleMomentumDensity(ContainerType *container, amrex::MultiFab &deposition_field, int lev, int mass_comp, int vel_start_comp,
				    int start_mesh_comp = 0)
{
	const BL_PROFILE("depositParticleMomentumDensity");

	ParticleMomentumDensityDeposition deposition_functor;
	deposition_functor.mass_comp = mass_comp;
	deposition_functor.vel_start_comp = vel_start_comp;
	deposition_functor.start_mesh_comp = start_mesh_comp;

	container->template ParticleToMesh<amrex::ParticleInterpolator::Linear>(*container, deposition_field, lev, deposition_functor, false, false);
}

/// Deposit particle kinetic energy density for a given particle type
template <typename ContainerType>
void depositParticleKineticEnergyDensity(ContainerType *container, amrex::MultiFab &deposition_field, int lev, int mass_comp, int vel_start_comp,
					 int start_mesh_comp = 0)
{
	const BL_PROFILE("depositParticleKineticEnergyDensity");

	ParticleKineticEnergyDensityDeposition deposition_functor;
	deposition_functor.mass_comp = mass_comp;
	deposition_functor.vel_start_comp = vel_start_comp;
	deposition_functor.start_mesh_comp = start_mesh_comp;
	deposition_functor.num_comp = 1;

	container->template ParticleToMesh<amrex::ParticleInterpolator::Linear>(*container, deposition_field, lev, deposition_functor, false, false);
}

/// Deposit particle number density for a given particle type
template <typename ContainerType>
void depositParticleNumberDensity(ContainerType *container, amrex::MultiFab &deposition_field, int lev, int start_mesh_comp = 0)
{
	const BL_PROFILE("depositParticleNumberDensity");

	ParticleNumberDensityDeposition deposition_functor;
	deposition_functor.start_mesh_comp = start_mesh_comp;
	deposition_functor.num_comp = 1;

	container->template ParticleToMesh<amrex::ParticleInterpolator::Linear>(*container, deposition_field, lev, deposition_functor, false, false);
}

//==============================================================================
// Particle Type Specific Deposition Functions
//==============================================================================

/// Deposit properties for CIC particles
template <typename ContainerType>
void depositCICParticleProperties(ContainerType *container, amrex::MultiFab &mass_field, amrex::MultiFab &momentum_field, amrex::MultiFab &energy_field,
				  amrex::MultiFab &number_field, int lev)
{
	static_assert(std::is_same_v<ContainerType, CICParticleContainer>, "Container type must be CICParticleContainer");

	// Deposit mass density
	depositParticleMassDensity(container, mass_field, lev, CICParticleMassIdx, 0);

	// Deposit momentum density
	depositParticleMomentumDensity(container, momentum_field, lev, CICParticleMassIdx, CICParticleVxIdx, 0);

	// Deposit kinetic energy density
	depositParticleKineticEnergyDensity(container, energy_field, lev, CICParticleMassIdx, CICParticleVxIdx, 0);

	// Deposit number density
	depositParticleNumberDensity(container, number_field, lev, 0);
}

/// Deposit properties for StochasticStellarPop particles
template <typename problem_t, typename ContainerType>
void depositStochasticStellarPopParticleProperties(ContainerType *container, amrex::MultiFab &mass_field, amrex::MultiFab &momentum_field,
						   amrex::MultiFab &energy_field, amrex::MultiFab &number_field, int lev)
{
	static_assert(std::is_same_v<ContainerType, StochasticStellarPopParticleContainer<problem_t>>,
		      "Container type must be StochasticStellarPopParticleContainer");

	// Deposit mass density
	depositParticleMassDensity(container, mass_field, lev, StochasticStellarPopParticleMassIdx, 0);

	// Deposit momentum density
	depositParticleMomentumDensity(container, momentum_field, lev, StochasticStellarPopParticleMassIdx, StochasticStellarPopParticleVxIdx, 0);

	// Deposit kinetic energy density
	depositParticleKineticEnergyDensity(container, energy_field, lev, StochasticStellarPopParticleMassIdx, StochasticStellarPopParticleVxIdx, 0);

	// Deposit number density
	depositParticleNumberDensity(container, number_field, lev, 0);
}

/// Deposit properties for Sink particles
template <typename ContainerType>
void depositSinkParticleProperties(ContainerType *container, amrex::MultiFab &mass_field, amrex::MultiFab &momentum_field, amrex::MultiFab &energy_field,
				   amrex::MultiFab &number_field, int lev)
{
	static_assert(std::is_same_v<ContainerType, SinkParticleContainer>, "Container type must be SinkParticleContainer");

	// Deposit mass density
	depositParticleMassDensity(container, mass_field, lev, SinkParticleMassIdx, 0);

	// Deposit momentum density
	depositParticleMomentumDensity(container, momentum_field, lev, SinkParticleMassIdx, SinkParticleVxIdx, 0);

	// Deposit kinetic energy density
	depositParticleKineticEnergyDensity(container, energy_field, lev, SinkParticleMassIdx, SinkParticleVxIdx, 0);

	// Deposit number density
	depositParticleNumberDensity(container, number_field, lev, 0);
}

/// Deposit properties for Test particles
template <typename problem_t, typename ContainerType>
void depositTestParticleProperties(ContainerType *container, amrex::MultiFab &mass_field, amrex::MultiFab &momentum_field, amrex::MultiFab &energy_field,
				   amrex::MultiFab &number_field, int lev)
{
	static_assert(std::is_same_v<ContainerType, TestParticleContainer<problem_t>>, "Container type must be TestParticleContainer");

	// Deposit mass density
	depositParticleMassDensity(container, mass_field, lev, TestParticleMassIdx, 0);

	// Deposit momentum density
	depositParticleMomentumDensity(container, momentum_field, lev, TestParticleMassIdx, TestParticleVxIdx, 0);

	// Deposit kinetic energy density
	depositParticleKineticEnergyDensity(container, energy_field, lev, TestParticleMassIdx, TestParticleVxIdx, 0);

	// Deposit number density
	depositParticleNumberDensity(container, number_field, lev, 0);
}

//==============================================================================
// Generic Particle Deposition Interface
//==============================================================================

/// Generic function to deposit particle properties based on particle type
template <typename problem_t>
void depositParticlePropertiesByType(const std::string &particleType, void *container, amrex::MultiFab &mass_field, amrex::MultiFab &momentum_field,
				     amrex::MultiFab &energy_field, amrex::MultiFab &number_field, int lev)
{
	if (particleType == "CIC") {
		auto *cicContainer = static_cast<CICParticleContainer *>(container);
		depositCICParticleProperties(cicContainer, mass_field, momentum_field, energy_field, number_field, lev);
	} else if (particleType == "StochasticStellarPop") {
		auto *stellarContainer = static_cast<StochasticStellarPopParticleContainer<problem_t> *>(container);
		depositStochasticStellarPopParticleProperties<problem_t>(stellarContainer, mass_field, momentum_field, energy_field, number_field, lev);
	} else if (particleType == "Sink") {
		auto *sinkContainer = static_cast<SinkParticleContainer *>(container);
		depositSinkParticleProperties(sinkContainer, mass_field, momentum_field, energy_field, number_field, lev);
	} else if (particleType == "Test") {
		auto *testContainer = static_cast<TestParticleContainer<problem_t> *>(container);
		depositTestParticleProperties<problem_t>(testContainer, mass_field, momentum_field, energy_field, number_field, lev);
	} else {
		amrex::Abort("Unsupported particle type for deposition: " + particleType);
	}
}

} // namespace quokka

#endif // PARTICLE_DEPOSITION_UTILS_HPP_
