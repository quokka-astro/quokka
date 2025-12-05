#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include <fmt/format.h>

#include "AMReX_Array4.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"

#include "particle_IO.hpp"
#include "particle_accretion.hpp"
#include "particle_creation.hpp"
#include "particle_deposition.hpp"
#include "particle_destruction.hpp"
#include "particle_types.hpp"
#include "particle_update.hpp"
#include "physics_info.hpp"

namespace quokka
{

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

// Base class for particle descriptors using type erasure pattern
class PhysicsParticleDescriptorBase
{
      protected:
	int massIndex_{-1};		// Index for particle mass (-1 if not used)
	int lumIndex_{-1};		// Index for radiation luminosity (-1 if not used)
	int birthTimeIndex_{-1};	// Index for birth time (-1 if not used)
	bool allowsCreation_{false};	// Whether particles can be created during simulation
	bool allowsDestruction_{false}; // Whether particles can be destroyed during simulation
	int evolutionStageIndex_{-1};	// Index for evolution stage (-1 if not used)
	bool allowsAccretion_{false};	// Whether particles can accrete gas

	bool forceFinestLevel_{false}; // Whether particles are forced to live in the finest level

      public:
	PhysicsParticleDescriptorBase(int mass_idx, int lum_idx, int birth_time_idx, bool allows_creation, bool allows_destruction = false,
				      int evolution_stage_idx = -1, bool allows_accretion = false)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), allowsCreation_(allows_creation),
	      allowsDestruction_(allows_destruction), evolutionStageIndex_(evolution_stage_idx), allowsAccretion_(allows_accretion)
	{
	}

	virtual ~PhysicsParticleDescriptorBase() = default;

	// Copy and move constructors and assignment operators with trailing return types
	PhysicsParticleDescriptorBase(const PhysicsParticleDescriptorBase &) = default;
	auto operator=(const PhysicsParticleDescriptorBase &) -> PhysicsParticleDescriptorBase & = default;
	PhysicsParticleDescriptorBase(PhysicsParticleDescriptorBase &&) = default;
	auto operator=(PhysicsParticleDescriptorBase &&) -> PhysicsParticleDescriptorBase & = default;

	// Getter methods for particle properties
	[[nodiscard]] AMREX_FORCE_INLINE auto getMassIndex() const -> int { return massIndex_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getLumIndex() const -> int { return lumIndex_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getBirthTimeIndex() const -> int { return birthTimeIndex_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getAllowsCreation() const -> bool { return allowsCreation_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getAllowsDestruction() const -> bool { return allowsDestruction_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getEvolutionStageIndex() const -> int { return evolutionStageIndex_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getAllowsAccretion() const -> bool { return allowsAccretion_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getForceFinestLevel() const -> bool { return forceFinestLevel_; }

	// setter methods for particle properties
	AMREX_FORCE_INLINE void setEvolutionStageIndex(int evolution_stage_idx) { evolutionStageIndex_ = evolution_stage_idx; }
	AMREX_FORCE_INLINE void setAllowsAccretion(bool allows_accretion) { allowsAccretion_ = allows_accretion; }
	AMREX_FORCE_INLINE void setForceFinestLevel(bool force) { forceFinestLevel_ = force; }

	// New method to get particle positions and data
	[[nodiscard]] virtual auto getParticleDataAtAllLevels() const
	    -> std::tuple<std::vector<int64_t>, std::vector<std::vector<double>>, std::vector<std::vector<int>>> = 0;

	// Get particle data at level lev
	[[nodiscard]] virtual auto getParticleDataAtLevel(int lev) const -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> = 0;

	// Pure virtual methods that must be implemented by derived classes
	virtual void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) = 0;

	// Redistribute particles at level lev and above
	virtual void redistribute(int lev) = 0;

	// Redistribute particles at level lev and above with ngrow ghost cells
	virtual void redistribute(int lev, int ngrow) = 0;

	// Write particle data to plot file
	virtual void writePlotFile(const std::string &plotfilename, const std::string &name) = 0;

	// Write particle data to checkpoint file
	virtual void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) = 0;

	// Write units info of particles to checkpoint/plotfile
	virtual void writeUnitsFile(const std::string &snapshot_name, const std::string &name) = 0;

	// Print statistics of particles
	virtual void printParticleStatistics() const = 0;

	// Save particle data to file
	virtual void saveParticleDataToFile(const std::string &plotfilename, const std::string &name) = 0;

	// Get the number of particles
	[[nodiscard]] virtual auto getNumParticles() const -> int = 0;

	// Compute total stellar mass
	[[nodiscard]] virtual auto computeStellarMass() const -> amrex::Real = 0;

#if AMREX_SPACEDIM == 3
	virtual void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) = 0;

	// Drift particle at level lev_min and above for time dt. Note that subcycling is not supported.
	virtual void driftParticles(int lev_min, int lev_max, amrex::Real dt) const = 0;

	// Kick particles at level lev_min and above for time dt. Note that subcycling is not supported.
	virtual void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const &accel) = 0;

	// Destroy particles at level lev_min and above
	virtual void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt) = 0;

	virtual void splitParticles(int lev, int splitFactor) = 0;
	[[nodiscard]] virtual auto computeMaxParticleSpeed(int lev) const -> amrex::ValLocPair<amrex::Real, amrex::RealVect> = 0;

	// Tag cells around particles for refinement
	virtual void tagCellsAroundParticles(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) const = 0;

	//----- Methods that are implemented for some but not all particle types, so they cannot be pure virtual -----

	virtual auto depositSN(amrex::MultiFab & /*state*/, int /*lev*/, amrex::Real /*time*/, amrex::Real /*dt*/) -> amrex::Real { return 0.0_rt; }

	virtual void computeSinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, int lev, amrex::Real time, amrex::Real dt)
	{ /* Default empty implementation */ }

	// Create particles from hydro state at the finest level
	// Note: particles are not allowed to spawn outside of real cells. If they do, we will need a redistribution immediately after this call in order to
	// make particle-mesh interaction work.
	virtual void createParticlesFromState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt)
	{ /* Default empty implementation */ }

	virtual void applySinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, const amrex::Geometry &geom, int lev, amrex::Real time,
					amrex::Real dt)
	{ /* Default empty implementation */
	}

	// Update particle properties (e.g., luminosity) based on current state
	virtual void updateParticleProperties(amrex::Real current_time) { /* Default empty implementation */ }
#endif // AMREX_SPACEDIM == 3
};

// Concrete implementation of particle descriptor for specific container types
template <typename ContainerType, typename problem_t, ParticleType particleType> class PhysicsParticleDescriptor : public PhysicsParticleDescriptorBase
{
      private:
	static constexpr ParticleType particleType_ = particleType;

      protected:
	ContainerType *container_{}; // Pointer to the actual particle container - moved to protected

      public:
	// Get the particle type
	[[nodiscard]] static constexpr auto getParticleType() -> ParticleType { return particleType_; }

	// Constructor initializing descriptor with container and particle properties
	PhysicsParticleDescriptor(ContainerType *container, int mass_idx, int lum_idx, int birth_time_idx, bool allows_creation,
				  bool allows_destruction = false, int evolution_stage_idx = -1, bool allows_accretion = false)
	    : PhysicsParticleDescriptorBase(mass_idx, lum_idx, birth_time_idx, allows_creation, allows_destruction, evolution_stage_idx, allows_accretion),
	      container_(container)
	{
	}

	// Get positions and fields data from all particles at level 0 from all ranks and gather them on rank 0.
	// This method creates a temporary particle container on rank 0 and copies all particles to it.
	// The returned data for each particle contains:
	// - first:
	//   - First AMREX_SPACEDIM elements are positions [x,y,z]
	//   - Remaining elements are particle data (e.g., mass, velocities, etc.)
	// - second:
	//   - Integer data (e.g., id, type, etc.)
	// Only rank 0 will return the actual particle data, other ranks return an empty vector.
	// @return: tuple of vectors of particle data on rank 0, empty vectors on other ranks
	[[nodiscard]] auto getParticleDataAtAllLevels() const
	    -> std::tuple<std::vector<int64_t>, std::vector<std::vector<double>>, std::vector<std::vector<int>>> override
	{
		return particle_io::getParticleDataAtAllLevels(container_);
	}

	[[nodiscard]] auto getParticleDataAtLevel(int lev) const -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> override
	{
		return particle_io::getParticleDataAtLevel(container_, lev);
	}

	// Get the number of particles in the container
	[[nodiscard]] auto getNumParticles() const -> int override
	{
		if (container_ != nullptr) {
			return static_cast<int>(container_->TotalNumberOfParticles(true, false));
		}
		return 0;
	}

	// Compute total stellar mass
	[[nodiscard]] auto computeStellarMass() const -> amrex::Real override
	{
		amrex::Real total_mass = 0.0;
		if (container_ != nullptr && this->getMassIndex() >= 0) {
			const int mass_idx = this->getMassIndex();
			amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
			using ReduceDataType = amrex::ReduceData<amrex::Real>;
			using PTDType = typename ContainerType::ParticleTileType::ConstParticleTileDataType;

			// Sum mass over all particles at all levels
			for (int lev = 0; lev <= container_->finestLevel(); ++lev) {
				auto result_tuple = amrex::ParticleReduce<ReduceDataType>(*container_, lev, 
					[=] AMREX_GPU_DEVICE(const PTDType &p_type, const int i) noexcept -> amrex::Real
					{
						return p_type.m_aos[i].rdata(mass_idx);
					},
					reduce_ops);
				total_mass += amrex::get<0>(result_tuple);
			}
		}
		amrex::ParallelAllReduce::Sum(total_mass, amrex::ParallelContext::CommunicatorSub());
		return total_mass;
	}

#if AMREX_SPACEDIM == 3

	// Implementation of mass deposition from particles to grid
	void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) override
	{
		if (container_ != nullptr && this->getMassIndex() >= 0) {
			// zero_out_input is false because we want to accumulate mass
			// vol_weight is false because MassDeposition does the volume weighting
			amrex::ParticleToMesh(*container_, rhs, 0, finest_lev, MassDeposition{Gconst, this->getMassIndex(), 0, 1}, false, false);

			// Deposit count into the last component of rhs
			const int count_comp = 1; // Second component is the count
			amrex::ParticleToMesh(*container_, rhs, 0, finest_lev, DepositionCount{this->getMassIndex(), count_comp, 1}, false, false);
		}
	}

	void driftParticles(int lev_min, int lev_max, amrex::Real dt) const override
	{
		if (container_ != nullptr) {
			const int mass_idx = this->getMassIndex(); // capture value instead of this pointer

			if (mass_idx >= 0) {
				for (int lev = lev_min; lev <= lev_max; ++lev) {
					for (typename ContainerType::ParIterType pIter(*container_, lev); pIter.isValid(); ++pIter) {
						auto &particles = pIter.GetArrayOfStructs();
						auto *pData = particles().data();
						const amrex::Long np = pIter.numParticles();

						amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
							auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
							// update particle position based on velocity components
							for (int i = 0; i < AMREX_SPACEDIM; ++i) {
								if (mass_idx + 1 + i < ContainerType::ParticleType::NReal) {
									p.pos(i) += dt * p.rdata(mass_idx + 1 + i);
								}
							}
						});
					}
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
							    if (comp < ContainerType::ParticleType::NReal) {
								    p.rdata(comp) += 0.5 * dt * static_cast<amrex::ParticleReal>(acc_comp);
							    }
						    });
					});
				}
			}
		}
	}

	void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt) override
	{
		if (container_ != nullptr) {
			ParticleDestructionTraits<particleType_>::template destroyParticles<problem_t, ContainerType>(
			    container_, this->getMassIndex(), lev_min, current_time, dt, this->getBirthTimeIndex(), this->getEvolutionStageIndex());
		}
	}

	void splitParticles(int const lev, int const splitFactor) override
	{
		if (container_ != nullptr && this->getMassIndex() >= 0) {
			for (typename ContainerType::ParIterType pIter(*container_, lev); pIter.isValid(); ++pIter) {
				// Update NextID to include particles that will be created
				const amrex::Long npart_old = pIter.numParticles();
				const unsigned int max_new_particles = splitFactor * npart_old;
				const amrex::Long pid = ContainerType::ParticleType::NextID();
				ContainerType::ParticleType::NextID(pid + max_new_particles);

				// Resize particle tile
				auto &particles = pIter.GetArrayOfStructs();
				particles.resize(npart_old + max_new_particles);

				// Create the particles
				const auto &geom = container_->Geom(lev);
				const auto dxinv = geom.InvCellSizeArray();
				const auto dx = geom.CellSizeArray();
				const auto plo = geom.ProbLoArray();
				const int cpu_id = amrex::ParallelDescriptor::MyProc();
				const int mass_idx = this->getMassIndex();
				auto *pdata_old = particles().data();
				auto *pdata_new = particles().data() + npart_old;

				amrex::ParallelForRNG(npart_old, [=] AMREX_GPU_DEVICE(int64_t idx, amrex::RandomEngine const &rng) {
					// compute cell corner position (x0, y0, z0) of the old particle
					const int i = static_cast<int>((pdata_old[idx].pos(0) - plo[0]) * dxinv[0]); // NOLINT
					const int j = static_cast<int>((pdata_old[idx].pos(1) - plo[1]) * dxinv[1]); // NOLINT
					const int k = static_cast<int>((pdata_old[idx].pos(2) - plo[2]) * dxinv[2]); // NOLINT
					amrex::Real const x0 = plo[0] + (i * dx[0]);
					amrex::Real const y0 = plo[1] + (j * dx[1]);
					amrex::Real const z0 = plo[2] + (k * dx[2]);

					// mark old particle for deletion
					auto &p_old = pdata_old[idx]; // NOLINT
					p_old.id() = -1;
					amrex::Real const old_mass = p_old.rdata(mass_idx);

					// create new particles
					auto *new_particles = &pdata_new[idx * splitFactor]; // NOLINT
					for (int idx_new = 0; idx_new < splitFactor; ++idx_new) {
						auto &p_new = new_particles[idx_new]; // NOLINT
						// copy old particle properties
						for (int rc = 0; rc < ContainerType::ParticleType::NReal; ++rc) {
							p_new.rdata(rc) = p_old.rdata(rc);
						}
						for (int ic = 0; ic < ContainerType::ParticleType::NInt; ++ic) {
							p_new.idata(ic) = p_old.idata(ic);
						}
						// set new particle properties
						p_new.cpu() = cpu_id;
						p_new.id() = pid + idx * splitFactor + idx_new;
						p_new.pos(0) = x0 + dx[0] * amrex::Random(rng);
						p_new.pos(1) = y0 + dx[1] * amrex::Random(rng);
						p_new.pos(2) = z0 + dx[2] * amrex::Random(rng);
						p_new.rdata(mass_idx) = old_mass / static_cast<amrex::Real>(splitFactor);
					}
				});
			}
			// Remove the old particles
			container_->Redistribute(lev);
		}
	}

	// Compute maximum particle speed at a given level
	[[nodiscard]] auto computeMaxParticleSpeed(int lev) const -> amrex::ValLocPair<amrex::Real, amrex::RealVect> override
	{
		amrex::ValLocPair<amrex::Real, amrex::RealVect> max_speed{.value = 0, .index = amrex::RealVect{AMREX_D_DECL(NAN, NAN, NAN)}};

		if (container_ != nullptr && this->getMassIndex() >= 0) {
			// Only compute for particles that have velocity components
			const int mass_idx = this->getMassIndex();

			// Check if we have enough components for velocities
			if (mass_idx + 3 < ContainerType::ParticleType::NReal) {
				// Use ParticleReduce with ReduceOpMax for efficient parallel reduction
				amrex::ReduceOps<amrex::ReduceOpMax> reduce_ops;
				using ResultType = amrex::ValLocPair<amrex::Real, amrex::RealVect>;
				using ReduceDataType = amrex::ReduceData<ResultType>;

				// Perform the reduction over all particles at this level
				using PTDType = typename ContainerType::ParticleTileType::ConstParticleTileDataType;
				auto result_tuple = amrex::ParticleReduce<ReduceDataType>(
				    *container_, lev,
				    [=] AMREX_GPU_DEVICE(const PTDType &p_type, const int i) noexcept -> ResultType {
					    // Compute velocity magnitude
					    const amrex::Real vx = p_type.m_aos[i].rdata(mass_idx + 1);
					    const amrex::Real vy = p_type.m_aos[i].rdata(mass_idx + 2);
					    const amrex::Real vz = p_type.m_aos[i].rdata(mass_idx + 3);
					    const amrex::Real v2 = (vx * vx) + (vy * vy) + (vz * vz);
					    const amrex::RealVect pos{p_type[i].pos(0), p_type[i].pos(1), p_type[i].pos(2)};
					    return amrex::ValLocPair<amrex::Real, amrex::RealVect>{std::sqrt(v2), pos};
				    },
				    reduce_ops);

				// Extract the value from the tuple
				max_speed = amrex::get<0>(result_tuple);

				AMREX_ASSERT(!std::isnan(max_speed.value));
				AMREX_ASSERT(!std::isinf(max_speed.value));
			}
		}

		// Reduce across all MPI ranks to get global maximum. Use ParallelContext::CommunicatorSub() for current level and avoid using the default
		// communicator.
		amrex::ParallelAllReduce::Max(max_speed, amrex::ParallelContext::CommunicatorSub());
		return max_speed;
	}

#endif // AMREX_SPACEDIM == 3

	// Implementation of radiation deposition from particles to grid
	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) override
	{
		if (container_ != nullptr && this->getLumIndex() >= 0) {
			amrex::ParticleToMesh(*container_, radEnergySource, lev,
					      RadDeposition{current_time, this->getLumIndex(), 0, nGroups, this->getBirthTimeIndex()}, false);
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

	// Implementation of particle data output to units file
	void writeUnitsFile(const std::string &snapshot_name, const std::string &name) override
	{
		if (container_ != nullptr) {
			particle_io::writeUnitsFile<ContainerType, problem_t, particleType_>(container_, snapshot_name, name);
		}
	}

	void printParticleStatistics() const override
	{
		if (container_ != nullptr) {
			particle_io::printParticleStatistics<ContainerType, problem_t, particleType_>(container_, getMassIndex(), getEvolutionStageIndex());
		}
	}

	void saveParticleDataToFile(const std::string &filename, const std::string &name) override
	{
		if (container_ != nullptr) {
			particle_io::saveParticleDataToFile<ContainerType>(container_, filename, name);
		}
	}

#if AMREX_SPACEDIM == 3
	// Implement cell tagging around particles
	void tagCellsAroundParticles(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/) const override
	{
		if (container_ == nullptr) {
			return;
		}

		for (typename ContainerType::ParIterType pti(*container_, lev); pti.isValid(); ++pti) {
			auto &particles = pti.GetArrayOfStructs();
			auto *pData = particles().data();
			const amrex::Long np = pti.numParticles();

			// Get geometry information for this level
			const auto &geom = container_->Geom(lev);
			const auto plo = geom.ProbLoArray();
			const auto dxi = geom.InvCellSizeArray();

			const auto tag = tags.array(pti);

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
				auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				// Find the cell containing the particle
				const int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
				const int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
				const int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

				tag(ix, iy, iz) = amrex::TagBox::SET;
			});
		}
	}

	// Override updateParticleProperties for star particles
	void updateParticleProperties(amrex::Real current_time) override
	{
		// Use the traits system to update particle properties directly
		if (this->container_ != nullptr) {
			// Get the GPU tables by value (host-side access)
			constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
			auto *host_tables_ptr = quokka::g_luminosity_tables_ptr<nGroups>;

			// Only proceed if tables are initialized
			if (host_tables_ptr != nullptr && host_tables_ptr->is_initialized()) {
				// Create GPU const tables by value to pass to device
				auto const gpu_tables = host_tables_ptr->const_tables();

				// Apply the updater to all particles across all levels
				for (int lev = 0; lev <= this->container_->finestLevel(); ++lev) {
					for (typename ContainerType::ParIterType pIter(*this->container_, lev); pIter.isValid(); ++pIter) {
						auto &particles = pIter.GetArrayOfStructs();
						auto *pData = particles().data();
						const amrex::Long np = pIter.numParticles();

						amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
							auto &p = pData[idx]; // NOLINT
							ParticlePropertyUpdateTraits<particleType>::template updateProperties<
							    problem_t, typename ContainerType::ParticleType, nGroups>(p, current_time, gpu_tables);
						});
					}
				}
			}
		}
	}

	// Implementation of supernova energy and momentum deposition from particles to grid
	auto depositSN(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt) -> amrex::Real override
	{
		amrex::Real max_velocity = 0.0;

		if (this->container_ != nullptr && this->getEvolutionStageIndex() >= 0) {
			if (!quokka::disable_SN_feedback) {
				// Requires CGS units
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Physics_Traits<problem_t>::unit_system == UnitSystem::CGS,
								 "UnitSystem must be CGS for particleMeshInteraction");

				// Deposit supernova energy and momentum from all particles. This also updates the evolution stage of the particles.
				max_velocity = SNDeposition<ContainerType, problem_t>(this->container_, state, lev, time, dt, this->getMassIndex(),
										      this->getEvolutionStageIndex(), this->getBirthTimeIndex());
			} else {
				// Only update evolution stage but not deposit energy/momentum
				SNFeedbackUtils::updateEvolutionStage(this->container_, lev, time + dt, this->getBirthTimeIndex(),
								      this->getEvolutionStageIndex());
			}
		}

		return max_velocity;
	}

	// compute accretion rate
	void computeSinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, int lev, amrex::Real time, amrex::Real dt) override
	{
		SinkAccretionUtils::computeAccretion<ContainerType, problem_t>(this->container_, state, state_accretion_rate, lev, time, dt,
									       this->getMassIndex());
	}

	// apply accretion
	void applySinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, const amrex::Geometry &geom, int lev, amrex::Real time,
				amrex::Real dt) override
	{
		SinkAccretionUtils::applyAccretion<ContainerType, problem_t>(this->container_, state, state_accretion_rate, geom, lev, time, dt,
									     this->getMassIndex());
	}

	void createParticlesFromState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt) override
	{
		// Use the traits class to implement the specialized behavior
		ParticleCreationTraits<particleType>::template createParticles<problem_t, ContainerType>(
		    this->container_, this->getMassIndex(), state, accretion_rate, lev, current_time, dt, this->getEvolutionStageIndex(),
		    this->getBirthTimeIndex());
	}
#endif // AMREX_SPACEDIM == 3
};

// Registry managing different types of physics particles
template <typename problem_t> class PhysicsParticleRegister
{
      private:
	// Map storing particle descriptors, indexed by particle type enum
	std::map<ParticleType, std::unique_ptr<PhysicsParticleDescriptorBase>> particleRegistry_;

	// SFH data: nstep, time, total_mass
	std::map<std::string, std::vector<std::tuple<int, amrex::Real, amrex::Real>>> sfh_data_;

      public:
	// Constructor
	PhysicsParticleRegister() = default;
	// Destructor
	~PhysicsParticleRegister() = default;

	// Check if registry contains any massive particles
	[[nodiscard]] auto HasMassiveParticles() const -> bool
	{
		for (const auto &[name, descriptor] : particleRegistry_) { // NOSONAR
			if (descriptor->getMassIndex() >= 0) {
				return true;
			}
		}
		return false;
	}

	// Check if registry contains any radiating particles
	[[nodiscard]] auto HasRadiatingParticles() const -> bool
	{
		if (Physics_Traits<problem_t>::is_radiation_enabled) {
			for (const auto &[name, descriptor] : particleRegistry_) { // NOSONAR
				if (descriptor->getLumIndex() >= 0) {
					return true;
				}
			}
		}
		return false;
	}

	// Utility method to convert particle type to string name (for writing plotfiles/checkpoints)
	[[nodiscard]] static auto getParticleTypeName(ParticleType type) -> std::string
	{
		switch (type) {
			case ParticleType::Rad:
				return "Rad_particles";
			case ParticleType::CIC:
				return "CIC_particles";
			case ParticleType::CICRad:
				return "CICRad_particles";
			case ParticleType::Test:
				return "Test_particles";
			case ParticleType::StochasticStellarPop:
				return "StochasticStellarPop_particles";
			case ParticleType::Sink:
				return "Sink_particles";
			default:
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, "Unknown particle type");
				return "Unknown_particles";
		}
	}

	// Register a new particle type with specified properties
	template <typename ContainerType> void registerParticleType(ContainerType *container, ParticleType type)
	{
		std::unique_ptr<PhysicsParticleDescriptorBase> descriptor;

		// Create the appropriate descriptor based on the particle type
		// The parameters for the descriptor are: mass_idx, lum_idx, birth_time_idx, allows_creation, allows_destruction, evolution_stage_idx,
		// allows_accretion
		if (type == ParticleType::Rad) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::Rad>>(
			    container, -1, RadParticleLumIdx, RadParticleBirthTimeIdx, false, false);
		}
#if AMREX_SPACEDIM == 3
		else if (type == ParticleType::CIC) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::CIC>>(container, CICParticleMassIdx, -1,
															      -1, false, false);
		} else if (type == ParticleType::CICRad) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::CICRad>>(
			    container, CICRadParticleMassIdx, CICRadParticleLumIdx, CICRadParticleBirthTimeIdx, false, false);
		} else if (type == ParticleType::StochasticStellarPop) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::StochasticStellarPop>>(
			    container, StochasticStellarPopParticleMassIdx, StochasticStellarPopParticleLumIdx, StochasticStellarPopParticleBirthTimeIdx, true,
			    false, StochasticStellarPopParticleStageIdx, false);
		} else if (type == ParticleType::Sink) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::Sink>>(container, SinkParticleMassIdx,
															       -1, -1, true, false, -1, true);
		} else if (type == ParticleType::Test) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::Test>>(
			    container, TestParticleMassIdx, TestParticleLumIdx, TestParticleBirthTimeIdx, true, true, TestParticleStageIdx, false);
		}
#endif // AMREX_SPACEDIM == 3
		else {
			amrex::Abort("Unknown particle type for physics particles");
		}

		particleRegistry_[type] = std::move(descriptor);
	}

	// Retrieve a particle descriptor by type
	[[nodiscard]] auto getParticleDescriptor(ParticleType type) -> PhysicsParticleDescriptorBase *
	{
		auto it = particleRegistry_.find(type);
		if (it != particleRegistry_.end()) {
			return it->second.get();
		}
		amrex::Abort("Particle type not found");
		return nullptr;
	}

	// Deposit radiation from all luminous particles
	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time)
	{
		const BL_PROFILE("PhysicsParticleRegister::depositRadiation()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getLumIndex() >= 0) {
				descriptor->depositRadiation(radEnergySource, lev, current_time, Physics_Traits<problem_t>::nGroups);
			}
		}
	}

#if AMREX_SPACEDIM == 3
	// Deposit mass from all massive particles
	void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst)
	{
		const BL_PROFILE("PhysicsParticleRegister::depositMass()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->depositMass(rhs, finest_lev, Gconst);
			}
		}
	}

	// Deposit supernova energy and momentum from all particles
	auto depositSN(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt) -> amrex::Real
	{
		const BL_PROFILE("PhysicsParticleRegister::depositSN()");
		amrex::Real max_velocity = 0.0;
		// Each particle type handles its own buffer creation and roundoff independently
		for (const auto &[type, descriptor] : particleRegistry_) {
			const amrex::Real max_velocity_ = descriptor->depositSN(state, lev, time, dt);
			max_velocity = std::max(max_velocity, max_velocity_);
		}
		return max_velocity;
	}

	// Implementation of computeSinkAccretion
	void computeSinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, int lev, amrex::Real time, amrex::Real dt)
	{
		const BL_PROFILE("PhysicsParticleRegister::computeSinkAccretion()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getAllowsAccretion()) {
				descriptor->computeSinkAccretion(state, state_accretion_rate, lev, time, dt);
			}
		}
	}

	// Implementation of applySinkAccretion
	void applySinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, const amrex::Geometry &geom, int lev, amrex::Real time,
				amrex::Real dt)
	{
		const BL_PROFILE("PhysicsParticleRegister::applySinkAccretion()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getAllowsAccretion()) {
				descriptor->applySinkAccretion(state, state_accretion_rate, geom, lev, time, dt);
			}
		}
	}
#endif // AMREX_SPACEDIM == 3

	// Redistribute all particles within a level
	void redistribute(int lev)
	{
		const BL_PROFILE("PhysicsParticleRegister::redistribute(lev)");
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev);
		}
	}

	// Redistribute all particles with ghost cells
	void redistribute(int lev, int ngrow)
	{
		const BL_PROFILE("PhysicsParticleRegister::redistribute(lev,ngrow)");
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev, ngrow);
		}
	}

	// Write all particle data to plot file
	void writePlotFile(const std::string &plotfilename)
	{
		const BL_PROFILE("PhysicsParticleRegister::writePlotFile()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->writePlotFile(plotfilename, getParticleTypeName(type));
			descriptor->writeUnitsFile(plotfilename, getParticleTypeName(type));
			saveSFH(plotfilename, getParticleTypeName(type));
		}
	}

	// Write only specified particle types to plot file
	void writePlotFileFiltered(const std::string &plotfilename, const std::vector<std::string> &particleTypeNames)
	{
		const BL_PROFILE("PhysicsParticleRegister::writePlotFileFiltered()");

		// Iterate through registered particles and write those whose names match the requested list
		for (const auto &[type, descriptor] : particleRegistry_) {
			const std::string typeName = getParticleTypeName(type);

			// Check if this particle type is in the requested list
			if (std::find(particleTypeNames.begin(), particleTypeNames.end(), typeName) != particleTypeNames.end()) {
				descriptor->writePlotFile(plotfilename, typeName);
				descriptor->writeUnitsFile(plotfilename, typeName);
			}
		}

		// Optionally warn about requested particle types that weren't found
		for (const auto &requestedName : particleTypeNames) {
			bool found = false;
			for (const auto &[type, descriptor] : particleRegistry_) {
				if (getParticleTypeName(type) == requestedName) {
					found = true;
					break;
				}
			}
			if (!found) {
				amrex::Print() << "Warning: Requested particle type '" << requestedName << "' is not registered.\n";
			}
		}
	}

	// Write all particle data to checkpoint file
	void writeCheckpoint(const std::string &checkpointname, bool include_header) const
	{
		const BL_PROFILE("PhysicsParticleRegister::writeCheckpoint()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->writeCheckpoint(checkpointname, getParticleTypeName(type), include_header);
			descriptor->writeUnitsFile(checkpointname, getParticleTypeName(type));
			saveSFH(checkpointname, getParticleTypeName(type));
		}
	}

#if AMREX_SPACEDIM == 3
	// Update positions of all massive particles
	void driftParticlesAllLevels(amrex::Real dt, int lev_max)
	{
		const BL_PROFILE("PhysicsParticleRegister::driftParticlesAllLevels()");
		if (!quokka::disable_particle_drift) {
			for (const auto &[type, descriptor] : particleRegistry_) {
				if (descriptor->getMassIndex() >= 0) {
					descriptor->driftParticles(0, lev_max, dt);
				}
			}
		}
	}

	// Update velocities of all massive particles
	void kickParticlesAtLevel(int lev, amrex::Real dt, amrex::MultiFab &accel)
	{
		const BL_PROFILE("PhysicsParticleRegister::kickParticlesAtLevel()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->kickParticles(lev, dt, accel);
			}
		}
	}

	// Create particles based on particle type
	void createParticlesFromState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt)
	{
		const BL_PROFILE("PhysicsParticleRegister::createParticlesFromState()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			// Only create particles if the descriptor allows creation
			if (descriptor->getAllowsCreation()) {
				// Call the appropriate particle creation method based on the particle type
				descriptor->createParticlesFromState(state, accretion_rate, lev, current_time, dt);

				// redistribute particles
				// descriptor->redistribute(lev);
			}
		}
	}

	// Destroy particles based on particle type
	void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt)
	{
		const BL_PROFILE("PhysicsParticleRegister::destroyParticles()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			// Only destroy particles if the descriptor allows destruction
			if (descriptor->getAllowsDestruction()) {
				// Call the appropriate particle destruction method based on the particle type
				descriptor->destroyParticles(lev_min, current_time, dt);
			}
		}
	}

	// Compute maximum particle speed across all particle types
	[[nodiscard]] auto computeMaxParticleSpeed(int lev) const -> amrex::ValLocPair<amrex::Real, amrex::RealVect>
	{
		amrex::ValLocPair<amrex::Real, amrex::RealVect> max_speed{.value = 0, .index = amrex::RealVect{AMREX_D_DECL(NAN, NAN, NAN)}};
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				const amrex::ValLocPair<amrex::Real, amrex::RealVect> speed = descriptor->computeMaxParticleSpeed(lev);
				AMREX_ASSERT(!std::isnan(speed.value));
				max_speed = std::max(max_speed, speed);
			}
		}
		return max_speed;
	}

	// Refine grids around particles that require finest level
	void refineGridsAroundParticles(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow, const amrex::IntVect &n_error_buf)
	{
		const BL_PROFILE("PhysicsParticleRegister::refineGridsAroundParticles()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getForceFinestLevel()) {
				AMREX_ALWAYS_ASSERT(n_error_buf.min() >= 2);
				descriptor->tagCellsAroundParticles(lev, tags, time, ngrow);
			}
		}
	}

	// Update particle properties for all registered particles
	void updateParticleProperties(amrex::Real current_time)
	{
		const BL_PROFILE("PhysicsParticleRegister::updateParticleProperties()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->updateParticleProperties(current_time);
		}
	}
#endif // AMREX_SPACEDIM == 3

	// Print particle statistics
	void printParticleStatistics() const
	{
		const BL_PROFILE("PhysicsParticleRegister::printParticleStatistics()");
		amrex::Print() << ">>> Particle statistics:\n";
		amrex::Print() << fmt::format("{:<20}{:>15}\n", "Particle type", "Number of particles");

		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->printParticleStatistics();
		}
	}

	// Save particle data to file only if particle count <= max_particles
	void saveParticleDataToFileConditional(const std::string &plotfilename, int max_particles)
	{
		const BL_PROFILE("PhysicsParticleRegister::saveParticleDataToFileConditional()");
		for (const auto &[type, descriptor] : particleRegistry_) {
			const int num_particles = descriptor->getNumParticles();
			const std::string particle_type_name = getParticleTypeName(type);

			if (num_particles <= max_particles) {
				amrex::Print() << "Saving " << num_particles << " " << particle_type_name << " to CSV file\n";
				descriptor->saveParticleDataToFile(plotfilename, particle_type_name);
			} else {
				amrex::Print() << "Skipping " << particle_type_name << " CSV output: " << num_particles << " particles exceeds limit of "
					       << max_particles << "\n";
			}
		}
	}

	// Prevent copying or moving of the registry to ensure single ownership
	PhysicsParticleRegister(const PhysicsParticleRegister &) = delete;
	auto operator=(const PhysicsParticleRegister &) -> PhysicsParticleRegister & = delete;
	PhysicsParticleRegister(PhysicsParticleRegister &&) = delete;
	auto operator=(PhysicsParticleRegister &&) -> PhysicsParticleRegister & = delete;

	// Update SFH data
	void updateSFH(int nstep, amrex::Real time)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			// Only compute SFH for particles that can be created (stars)
			if (descriptor->getAllowsCreation()) {
				amrex::Real total_mass = descriptor->computeStellarMass();
				std::string typeName = getParticleTypeName(type);
				sfh_data_[typeName].emplace_back(nstep, time, total_mass);
			}
		}
	}

	// Save SFH data to file
	void saveSFH(const std::string &base_dir, const std::string &particle_type_name) const
	{
		auto it = sfh_data_.find(particle_type_name);
		if (it != sfh_data_.end()) {
			std::string filename = base_dir + "/" + particle_type_name + "/SFH.txt";
			// Continue only if directory exists (it should be created by writePlotFile/writeCheckpoint)
			const std::string dir = base_dir + "/" + particle_type_name;
			if (!amrex::FileSystem::Exists(dir)) {
				return;
			}

			// Open file for writing (overwrite)
			std::ofstream ofs(filename);
			if (ofs.is_open()) {
				ofs << "# nstep time total_mass\n";
				for (const auto &entry : it->second) {
					ofs << std::get<0>(entry) << " " << std::scientific << std::setprecision(12) << std::get<1>(entry) << " "
					    << std::get<2>(entry) << "\n";
				}
				ofs.close();
			} else {
				amrex::Print() << "Warning: Could not open SFH file " << filename << " for writing.\n";
			}
		}
	}

	// Read SFH data from file
	void readSFH(const std::string &base_dir, const std::string &particle_type_name)
	{
		std::string filename = base_dir + "/" + particle_type_name + "/SFH.txt";
		std::ifstream ifs(filename);
		if (ifs.is_open()) {
			sfh_data_[particle_type_name].clear();
			std::string line;
			while (std::getline(ifs, line)) {
				if (line.empty() || line[0] == '#') {
					continue;
				}
				std::istringstream iss(line);
				int nstep = 0;
				amrex::Real time = 0.0;
				amrex::Real mass = 0.0;
				if (iss >> nstep >> time >> mass) {
					sfh_data_[particle_type_name].emplace_back(nstep, time, mass);
				}
			}
			ifs.close();
			amrex::Print() << "Read SFH data for " << particle_type_name << " from " << filename << "\n";
		}

		// It's okay if file doesn't exist, e.g. first run, in which case you should not call this function
	}
};

} // namespace quokka

#endif // PHYSICS_PARTICLES_HPP_
