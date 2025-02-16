#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "AMReX_AmrParticles.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_ParIter.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "physics_info.hpp"

// Assumptions for any particle type:
// 1. For massive particles, velocity components start after mass
// 2. Birth time, if existing, is always followed by death time

namespace quokka
{

// Enum class to identify different particle types
enum class ParticleType {
	CIC,   // Gravitating particles
	Rad,   // Radiation particles
	CICRad // Gravitating radiation particles
};

// Indices for gravitating particles (CIC_particles), mass + 3 velocity components
enum CICParticleDataIdx {
	CICParticleMassIdx = 0, // Mass of the particle
	CICParticleVxIdx,	// Velocity in x direction
	CICParticleVyIdx,	// Velocity in y direction
	CICParticleVzIdx	// Velocity in z direction
};

// Number of real components for CIC_particles, mass + 3 velocity components
constexpr int CICParticleRealComps = 4;

// Type definitions for CIC_particles container and iterator
using CICParticleContainer = amrex::AmrParticleContainer<CICParticleRealComps>;
using CICParticleIterator = amrex::ParIter<CICParticleRealComps>;

// Indices for radiation particles (Rad_particles), birth time + death time + radiation groups
enum RadParticleDataIdx {
	RadParticleBirthTimeIdx = 0, // Time when particle becomes active
	RadParticleDeathTimeIdx,     // Time when particle becomes inactive
	RadParticleLumIdx	     // Base index for luminosity components
};

// Number of real components for Rad_particles, birth time + death time + radiation groups
template <typename problem_t>
constexpr int RadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 2 + Physics_Traits<problem_t>::nGroups; // birth_time death_time lum1 ... lumN
	} else {
		return 2; // birth_time death_time
	}
}();

// Type definitions for Rad_particles container and iterator
template <typename problem_t> using RadParticleContainer = amrex::AmrParticleContainer<RadParticleRealComps<problem_t>>;
template <typename problem_t> using RadParticleIterator = amrex::ParIter<RadParticleRealComps<problem_t>>;

// Indices for gravitating radiation particles (CICRad_particles), mass + 3 velocity components + birth time + death time + radiation groups
enum CICRadParticleDataIdx {
	CICRadParticleMassIdx = 0,  // Mass of the particle
	CICRadParticleVxIdx,	    // Velocity in x direction
	CICRadParticleVyIdx,	    // Velocity in y direction
	CICRadParticleVzIdx,	    // Velocity in z direction
	CICRadParticleBirthTimeIdx, // Time when particle becomes active
	CICRadParticleDeathTimeIdx, // Time when particle becomes inactive
	CICRadParticleLumIdx	    // Base index for luminosity components
};

// Number of real components for CICRad_particles, mass + 3 velocity components + birth time + death time + radiation groups
template <typename problem_t>
constexpr int CICRadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 6 + Physics_Traits<problem_t>::nGroups; // mass, vx, vy, vz, birth_time, death_time, lum[nGroups]
	} else {
		return 6; // mass, vx, vy, vz, birth_time, death_time
	}
}();

// Type definitions for CICRad_particles container and iterator
template <typename problem_t> using CICRadParticleContainer = amrex::AmrParticleContainer<CICRadParticleRealComps<problem_t>>;
template <typename problem_t> using CICRadParticleIterator = amrex::ParIter<CICRadParticleRealComps<problem_t>>;

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
		// Deposit mass weighted by 4πG
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) { return 4.0 * M_PI * Gconst * part.rdata(comp); });
	}
};

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

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

// Base class for particle descriptors using type erasure pattern
class PhysicsParticleDescriptorBase
{
      protected:
	int massIndex_{-1};		 // Index for particle mass (-1 if not used)
	int lumIndex_{-1};		 // Index for radiation luminosity (-1 if not used)
	int birthTimeIndex_{-1};	 // Index for birth time (-1 if not used)
	bool interactsWithHydro_{false}; // Whether particles interact with hydrodynamics

      public:
	PhysicsParticleDescriptorBase(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), interactsWithHydro_(hydro_interact)
	{
	}

	virtual ~PhysicsParticleDescriptorBase() = default;

	// Copy and move constructors and assignment operators with trailing return types
	PhysicsParticleDescriptorBase(const PhysicsParticleDescriptorBase &) = default;
	auto operator=(const PhysicsParticleDescriptorBase &) -> PhysicsParticleDescriptorBase & = default;
	PhysicsParticleDescriptorBase(PhysicsParticleDescriptorBase &&) = default;
	auto operator=(PhysicsParticleDescriptorBase &&) -> PhysicsParticleDescriptorBase & = default;

	// Getter methods for particle properties
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto getMassIndex() const -> int { return massIndex_; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto getLumIndex() const -> int { return lumIndex_; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto getBirthTimeIndex() const -> int { return birthTimeIndex_; }
	[[nodiscard]] auto getInteractsWithHydro() const -> bool { return interactsWithHydro_; }

	// Virtual interface for particle operations
	[[nodiscard]] virtual auto getParticlePositions(int lev) const -> std::vector<std::array<double, AMREX_SPACEDIM>> = 0;

	// Pure virtual methods that must be implemented by derived classes
	virtual void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) = 0;
	virtual void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst) = 0;
	virtual void redistribute(int lev) = 0;
	virtual void redistribute(int lev, int ngrow) = 0;
	virtual void writePlotFile(const std::string &plotfilename, const std::string &name) = 0;
	virtual void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) = 0;
	virtual void driftParticles(int lev, amrex::Real dt) = 0;
	virtual void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const &acceleration) = 0;
};

// Concrete implementation of particle descriptor for specific container types
template <typename ContainerType, ParticleType particleType> class PhysicsParticleDescriptor : public PhysicsParticleDescriptorBase
{
      private:
	ContainerType *container_{}; // Pointer to the actual particle container
	static constexpr ParticleType particleType_ = particleType;

      public:
	// Get the particle type
	[[nodiscard]] static constexpr auto getParticleType() -> ParticleType { return particleType_; }

	// Constructor initializing descriptor with container and particle properties
	PhysicsParticleDescriptor(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact, ContainerType *container)
	    : PhysicsParticleDescriptorBase(mass_idx, lum_idx, birth_time_idx, hydro_interact), container_(container)
	{
	}

	// Implementation of particle position retrieval
	[[nodiscard]] auto getParticlePositions(int lev) const -> std::vector<std::array<double, AMREX_SPACEDIM>> override
	{
		std::vector<std::array<double, AMREX_SPACEDIM>> positions;
		if (container_ != nullptr) {
			const auto &particles = container_->GetParticles(lev);
			for (const auto &kv : particles) {
				const auto &pbox = kv.second;
				const auto &aos = pbox.GetArrayOfStructs();
				for (int i = 0; i < pbox.numParticles(); ++i) {
					const auto &p = aos[i];
					positions.push_back({AMREX_D_DECL(p.pos(0), p.pos(1), p.pos(2))});
				}
			}
		}
		return positions;
	}

	// Implementation of particle drift (position update based on velocity)
	void driftParticles(int lev, amrex::Real dt) override
	{
		if (container_ != nullptr) {
			const int mass_idx = this->getMassIndex(); // capture value instead of this pointer

			if (mass_idx >= 0) {
				for (typename ContainerType::ParIterType pIter(*container_, lev); pIter.isValid(); ++pIter) {
					auto &particles = pIter.GetArrayOfStructs();
					auto *pData = particles().data();
					const amrex::Long np = pIter.numParticles();

					amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
						auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
						// update particle position based on velocity components
						for (int i = 0; i < AMREX_SPACEDIM; ++i) {
							if (mass_idx >= 0) { // use captured value instead of this->getMassIndex()
								// For massive particles, velocity components start after mass
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
								p.pos(i) += dt * p.rdata(mass_idx + 1 + i);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
							}
						}
					});
				}
			}
		}
	}

	// Implementation of particle kick (velocity update based on acceleration)
	void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const &accel) override
	{
		if (container_ != nullptr) {
			const int mass_idx = this->getMassIndex(); // capture value instead of this pointer

			if (mass_idx >= 0) {
				for (typename ContainerType::ParIterType pIter(*container_, lev); pIter.isValid(); ++pIter) {
					auto &particles = pIter.GetArrayOfStructs();
					auto *pData = particles().data();
					const amrex::Long np = pIter.numParticles();

					const auto &accel_arr = accel.array(pIter);
					const auto &geom = container_->Geom(lev);
					const auto plo = geom.ProbLoArray();
					const auto dx_inv = geom.InvCellSizeArray();

					amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
						auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
						amrex::ParticleInterpolator::Linear interp(p, plo, dx_inv);

						// Interpolate acceleration from grid to particle and update velocity
						interp.MeshToParticle(
						    p, accel_arr, 0, mass_idx + 1, AMREX_SPACEDIM,
						    [=] AMREX_GPU_DEVICE(amrex::Array4<const amrex::Real> const &acc, int i, int j, int k, int comp) {
							    return acc(i, j, k, comp); // no weighting
						    },
						    [=] AMREX_GPU_DEVICE(typename ContainerType::ParticleType & p, int comp, amrex::Real acc_comp) {
							    // kick particle by updating its velocity
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
							    p.rdata(comp) += 0.5 * dt * static_cast<amrex::ParticleReal>(acc_comp);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
						    });
					});
				}
			}
		}
	}

	// Implementation of radiation deposition from particles to grid
	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) override
	{
		if (container_ != nullptr && this->getLumIndex() >= 0) {
			amrex::ParticleToMesh(*container_, radEnergySource, lev,
					      RadDeposition{current_time, this->getLumIndex(), 0, nGroups, this->getBirthTimeIndex()}, false);
		}
	}

	// Implementation of mass deposition from particles to grid
	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst) override
	{
		if (container_ != nullptr && this->getMassIndex() >= 0) {
			amrex::ParticleToMesh(*container_, amrex::GetVecOfPtrs(rhs), 0, finest_lev, MassDeposition{Gconst, this->getMassIndex(), 0, 1}, true);
		}
	}

	// Implementation of particle redistribution within a level
	void redistribute(int lev) override
	{
		if (container_ != nullptr) {
			container_->Redistribute(lev);
		}
	}

	// Implementation of particle redistribution with ghost cells
	void redistribute(int lev, int ngrow) override
	{
		if (container_ != nullptr) {
			container_->Redistribute(lev, container_->finestLevel(), ngrow);
		}
	}

	// Implementation of particle data output to plot file
	void writePlotFile(const std::string &plotfilename, const std::string &name) override
	{
		if (container_ != nullptr) {
			container_->WritePlotFile(plotfilename, name);
		}
	}

	// Implementation of particle data output to checkpoint file
	void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) override
	{
		if (container_ != nullptr) {
			container_->Checkpoint(checkpointname, name, include_header);
		}
	}
};

// Registry managing different types of physics particles
template <typename problem_t> class PhysicsParticleRegister
{
      private:
	// Map storing particle descriptors, indexed by particle type name
	std::map<std::string, std::unique_ptr<PhysicsParticleDescriptorBase>> particleRegistry_;

      public:
	// Constructor
	PhysicsParticleRegister() = default;
	// Destructor
	~PhysicsParticleRegister() = default;

	// Check if registry contains any massive particles
	[[nodiscard]] auto HasMassiveParticles() const -> bool
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				return true;
			}
		}
		return false;
	}

	// TODO(cch): replace name with particleType to get rid of the string comparison
	// Register a new particle type with specified properties
	template <typename ContainerType>
	void registerParticleType(const std::string &name, int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact, ContainerType *container)
	{
		std::unique_ptr<PhysicsParticleDescriptorBase> descriptor;
		if (name == "CIC_particles") {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, ParticleType::CIC>>(mass_idx, lum_idx, birth_time_idx,
														   hydro_interact, container);
		} else if (name == "Rad_particles") {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, ParticleType::Rad>>(mass_idx, lum_idx, birth_time_idx,
														   hydro_interact, container);
		} else if (name == "CICRad_particles") {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, ParticleType::CICRad>>(mass_idx, lum_idx, birth_time_idx,
														      hydro_interact, container);
		} else {
			amrex::Abort("Unknown particle type: " + name);
		}
		particleRegistry_[name] = std::move(descriptor);
	}

	// Retrieve a particle descriptor by name
	[[nodiscard]] auto getParticleDescriptor(const std::string &name) const -> const PhysicsParticleDescriptorBase *
	{
		auto it = particleRegistry_.find(name);
		if (it != particleRegistry_.end()) {
			return it->second.get();
		}
		return nullptr;
	}

	// Deposit radiation from all luminous particles
	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getLumIndex() >= 0) {
				descriptor->depositRadiation(radEnergySource, lev, current_time, Physics_Traits<problem_t>::nGroups);
			}
		}
	}

	// Deposit mass from all massive particles
	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->depositMass(rhs, finest_lev, Gconst);
			}
		}
	}

	// Redistribute all particles within a level
	void redistribute(int lev)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev);
		}
	}

	// Redistribute all particles with ghost cells
	void redistribute(int lev, int ngrow)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev, ngrow);
		}
	}

	// Write all particle data to plot file
	void writePlotFile(const std::string &plotfilename)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->writePlotFile(plotfilename, name);
		}
	}

	// Write all particle data to checkpoint file
	void writeCheckpoint(const std::string &checkpointname, bool include_header) const
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->writeCheckpoint(checkpointname, name, include_header);
		}
	}

	// Update positions of all massive particles
	void driftParticlesAllLevels(amrex::Real dt, int finest_level)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				for (int lev = 0; lev <= finest_level; ++lev) {
					descriptor->driftParticles(lev, dt);
				}
			}
		}
	}

	// Update velocities of all massive particles
	void kickParticlesAllLevels(amrex::Real dt, amrex::Vector<amrex::MultiFab> &acceleration)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				for (int lev = 0; lev <= acceleration.size() - 1; ++lev) {
					descriptor->kickParticles(lev, dt, acceleration[lev]);
				}
			}
		}
	}

	// Prevent copying or moving of the registry to ensure single ownership
	PhysicsParticleRegister(const PhysicsParticleRegister &) = delete;
	auto operator=(const PhysicsParticleRegister &) -> PhysicsParticleRegister & = delete;
	PhysicsParticleRegister(PhysicsParticleRegister &&) = delete;
	auto operator=(PhysicsParticleRegister &&) -> PhysicsParticleRegister & = delete;
};

} // namespace quokka

#endif // PHYSICS_PARTICLES_HPP_
