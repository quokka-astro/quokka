#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "AMReX_Array4.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "particle_creation.hpp"
#include "particle_deposition.hpp"
#include "particle_destruction.hpp"
#include "particle_types.hpp"
#include "physics_info.hpp"

namespace quokka
{

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
	bool allowsCreation_{false};	 // Whether particles can be created during simulation
	bool allowsDestruction_{false};	 // Whether particles can be destroyed during simulation

      public:
	PhysicsParticleDescriptorBase(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact, bool allows_creation, bool allows_destruction = false)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), interactsWithHydro_(hydro_interact), allowsCreation_(allows_creation),
	      allowsDestruction_(allows_destruction)
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
	[[nodiscard]] AMREX_FORCE_INLINE auto getInteractsWithHydro() const -> bool { return interactsWithHydro_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getAllowsCreation() const -> bool { return allowsCreation_; }
	[[nodiscard]] AMREX_FORCE_INLINE auto getAllowsDestruction() const -> bool { return allowsDestruction_; }

	// Virtual interface for particle operations
	[[nodiscard]] virtual auto getParticlePositions(int lev) const -> std::vector<std::array<double, AMREX_SPACEDIM>> = 0;

	// New method to get particle positions and data
	[[nodiscard]] virtual auto getParticleData(int lev) const -> std::vector<std::vector<double>> = 0;

	// Pure virtual methods that must be implemented by derived classes
	virtual void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) = 0;
	virtual void redistribute(int lev) = 0;
	virtual void redistribute(int lev, int ngrow) = 0;
	virtual void writePlotFile(const std::string &plotfilename, const std::string &name) = 0;
	virtual void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) = 0;
	virtual void writeUnitsFile(const std::string &snapshot_name, const std::string &name) = 0;
	[[nodiscard]] virtual auto getNumParticles() const -> int = 0;
#if AMREX_SPACEDIM == 3
	virtual void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) = 0;
	virtual void driftParticles(int lev, amrex::Real dt) const = 0;
	virtual void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const &acceleration) = 0;
	virtual void createParticlesFromState(amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt) const = 0;
	virtual void destroyParticles(int lev, amrex::Real current_time, amrex::Real dt) = 0;
	[[nodiscard]] virtual auto computeMaxParticleSpeed(int lev) const -> amrex::Real = 0;
#endif // AMREX_SPACEDIM == 3
};

// Concrete implementation of particle descriptor for specific container types
template <typename ContainerType, typename problem_t, ParticleType particleType> class PhysicsParticleDescriptor : public PhysicsParticleDescriptorBase
{
      private:
	ContainerType *container_{}; // Pointer to the actual particle container
	static constexpr ParticleType particleType_ = particleType;

      public:
	// Get the particle type
	[[nodiscard]] static constexpr auto getParticleType() -> ParticleType { return particleType_; }

	// Constructor initializing descriptor with container and particle properties
	PhysicsParticleDescriptor(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact, bool allows_creation, ContainerType *container,
				  bool allows_destruction = false)
	    : PhysicsParticleDescriptorBase(mass_idx, lum_idx, birth_time_idx, hydro_interact, allows_creation, allows_destruction), container_(container)
	{
	}

	// Get particle positions from all ranks and gather them on rank 0.
	// This method creates a temporary particle container on rank 0 and copies all particles to it.
	// Only rank 0 will return the actual particle positions, other ranks return an empty vector.
	// @param lev: level from which to get particles
	// @return: vector of particle positions [x,y,z] on rank 0, empty vector on other ranks
	[[nodiscard]] auto getParticlePositions(int lev) const -> std::vector<std::array<double, AMREX_SPACEDIM>> override
	{
		std::vector<std::array<double, AMREX_SPACEDIM>> positions;

		// All ranks must participate in copyParticles
		if (container_ != nullptr) {
			// Create single-box particle container for analysis on all ranks
			ContainerType analysisPC{};
			// Define a single box [0,1]^3 that will hold all particles on rank 0
			amrex::Box const box(amrex::IntVect{AMREX_D_DECL(0, 0, 0)}, amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
			amrex::Geometry const geom(box);
			amrex::BoxArray const boxArray(box);
			// Force all particles to rank 0 by using a single-rank distribution
			amrex::Vector<int> const ranks({0});
			amrex::DistributionMapping const dmap(ranks);

			// Initialize the analysis container and gather all particles to rank 0
			analysisPC.Define(geom, dmap, boxArray);
			analysisPC.copyParticles(*container_); // MPI communication happens here

			// Only rank 0 processes the particles since they're all gathered there
			if (amrex::ParallelDescriptor::IOProcessor()) {
				// Get iterator for the single box on rank 0
				typename ContainerType::ParIterType const pIter(analysisPC, lev);
				if (pIter.isValid()) {
					const amrex::Long np = pIter.numParticles();
					auto &particles = pIter.GetArrayOfStructs();

					// Transfer particle data from GPU to CPU for analysis
					typename ContainerType::ParticleType *pData = particles().data();
					amrex::Vector<typename ContainerType::ParticleType> pData_h(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, pData + np, pData_h.begin()); // NOLINT

					// Extract just the positions into the return vector
					for (int i = 0; i < np; ++i) {
						const auto &p = pData_h[i];
						positions.push_back({AMREX_D_DECL(p.pos(0), p.pos(1), p.pos(2))});
					}
				}
			}
		}

		return positions; // Empty vector on non-root ranks
	}

	// Get particle positions and data from all ranks and gather them on rank 0.
	// This method creates a temporary particle container on rank 0 and copies all particles to it.
	// The returned data for each particle contains:
	// - First AMREX_SPACEDIM elements are positions [x,y,z]
	// - Remaining elements are particle data (e.g., mass, velocities, etc.)
	// Only rank 0 will return the actual particle data, other ranks return an empty vector.
	// @param lev: level from which to get particles
	// @return: vector of particle data on rank 0, empty vector on other ranks
	[[nodiscard]] auto getParticleData(int lev) const -> std::vector<std::vector<double>> override
	{
		std::vector<std::vector<double>> particle_data;

		// All ranks must participate in copyParticles
		if (container_ != nullptr) {
			// Create single-box particle container for analysis on all ranks
			ContainerType analysisPC{};
			// Define a single box [0,1]^3 that will hold all particles on rank 0
			amrex::Box const box(amrex::IntVect{AMREX_D_DECL(0, 0, 0)}, amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
			amrex::Geometry const geom(box);
			amrex::BoxArray const boxArray(box);
			// Force all particles to rank 0 by using a single-rank distribution
			amrex::Vector<int> const ranks({0});
			amrex::DistributionMapping const dmap(ranks);

			// Initialize the analysis container and gather all particles to rank 0
			analysisPC.Define(geom, dmap, boxArray);
			analysisPC.copyParticles(*container_); // MPI communication happens here

			// Only rank 0 processes the particles since they're all gathered there
			if (amrex::ParallelDescriptor::IOProcessor()) {
				// Get iterator for the single box on rank 0
				typename ContainerType::ParIterType const pIter(analysisPC, lev);
				if (pIter.isValid()) {
					const amrex::Long np = pIter.numParticles();
					auto &particles = pIter.GetArrayOfStructs();

					// Transfer particle data from GPU to CPU for analysis
					typename ContainerType::ParticleType *pData = particles().data();
					amrex::Vector<typename ContainerType::ParticleType> pData_h(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, pData + np, pData_h.begin()); // NOLINT

					// Extract positions and real components from host data
					for (int i = 0; i < np; ++i) {
						const auto &p = pData_h[i];
						std::vector<double> data;
						// Pre-allocate to avoid reallocations
						data.reserve(AMREX_SPACEDIM + ContainerType::ParticleType::NReal);

						// First add position components
						for (int d = 0; d < AMREX_SPACEDIM; ++d) {
							data.push_back(p.pos(d));
						}

						// Then add all real components (mass, velocities, etc)
						for (int d = 0; d < ContainerType::ParticleType::NReal; ++d) {
							data.push_back(p.rdata(d));
						}

						particle_data.push_back(std::move(data));
					}
				}
			}
		}

		return particle_data; // Empty vector on non-root ranks
	}

	// Get the number of particles in the container
	[[nodiscard]] auto getNumParticles() const -> int override
	{
		if (container_ != nullptr) {
			return static_cast<int>(container_->TotalNumberOfParticles(true, false));
		}
		return 0;
	}

#if AMREX_SPACEDIM == 3

	// Implementation of mass deposition from particles to grid
	void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) override
	{
		if (container_ != nullptr && this->getMassIndex() >= 0) {
			// zero_out_input is false because we want to accumulate mass
			// vol_weight is false because MassDeposition does the volume weighting
			amrex::ParticleToMesh(*container_, rhs, 0, finest_lev, MassDeposition{Gconst, this->getMassIndex(), 0, 1}, false, false);
		}
	}

	// Implementation of particle drift (position update based on velocity)
	void driftParticles(int lev, amrex::Real dt) const override
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
							if (mass_idx + 1 + i < ContainerType::ParticleType::NReal) {
								p.pos(i) += dt * p.rdata(mass_idx + 1 + i);
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
							    if (comp < ContainerType::ParticleType::NReal) {
								    p.rdata(comp) += 0.5 * dt * static_cast<amrex::ParticleReal>(acc_comp);
							    }
						    });
					});
				}
			}
		}
	}

	void createParticlesFromState(amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt) const override
	{
		// Use the traits class to implement the specialized behavior
		ParticleCreationTraits<particleType_>::template createParticles<problem_t, ContainerType>(container_, this->getMassIndex(), state, lev,
													  current_time, dt);
	}

	void destroyParticles(int lev, amrex::Real current_time, amrex::Real dt) override
	{
		if (container_ != nullptr) {
			ParticleDestructionTraits<particleType_>::template destroyParticles<problem_t, ContainerType>(container_, this->getMassIndex(), lev,
														      current_time, dt);
		}
	}

	// Compute maximum particle speed at a given level
	[[nodiscard]] auto computeMaxParticleSpeed(int lev) const -> amrex::Real override
	{
		amrex::Real max_speed = 0.0;

		if (container_ != nullptr && this->getMassIndex() >= 0) {
			// Only compute for particles that have velocity components
			const int mass_idx = this->getMassIndex();

			// Check if we have enough components for velocities
			if (mass_idx + 3 < ContainerType::ParticleType::NReal) {
				// Use ParticleReduce with ReduceOpMax for efficient parallel reduction
				amrex::ReduceOps<amrex::ReduceOpMax> reduce_ops;
				using ReduceDataType = amrex::ReduceData<amrex::Real>;

				// Perform the reduction over all particles at this level
				using PTDType = typename ContainerType::ParticleTileType::ConstParticleTileDataType;
				auto result_tuple = amrex::ParticleReduce<ReduceDataType>(
				    *container_, lev,
				    [=] AMREX_GPU_DEVICE(const PTDType &p_type, const int i) noexcept -> amrex::Real {
					    // Compute velocity magnitude
					    const amrex::Real vx = p_type.m_aos[i].rdata(mass_idx + 1);
					    const amrex::Real vy = p_type.m_aos[i].rdata(mass_idx + 2);
					    const amrex::Real vz = p_type.m_aos[i].rdata(mass_idx + 3);
					    const amrex::Real v2 = (vx * vx) + (vy * vy) + (vz * vz);
					    return std::sqrt(v2);
				    },
				    reduce_ops);

				// Extract the value from the tuple
				max_speed = std::max(0.0, amrex::get<0>(result_tuple));

				AMREX_ASSERT(!std::isnan(max_speed));
				AMREX_ASSERT(!std::isinf(max_speed));
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
			// Only write on rank 0
			if (amrex::ParallelDescriptor::IOProcessor()) {
				// Create the full path for the Fields.yaml file
				const std::string filename = snapshot_name + "/" + name + "/Fields.yaml";

				// Open the file for writing
				std::ofstream outFile(filename);
				if (!outFile) {
					amrex::Abort("Error opening file for writing: " + filename);
				}

				// Get the units data for this particle type
				const auto &typeData = quokka::get_units_data().at(particleType_);
				if (!typeData.empty()) {
					outFile << "# field: [M, L, T, Θ]\n";
					// Write each field's units to the YAML file
					for (const auto &[fieldName, units] : typeData[0]) {
						outFile << fieldName << ": [" << units[0] << ", " << units[1] << ", " << units[2] << ", " << units[3] << "]\n";
					}
				}

				outFile.close();
			}
		}
	}
};

// Registry managing different types of physics particles
template <typename problem_t> class PhysicsParticleRegister
{
      private:
	// Map storing particle descriptors, indexed by particle type enum
	std::map<ParticleType, std::unique_ptr<PhysicsParticleDescriptorBase>> particleRegistry_;

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
			default:
				return "Unknown_particles";
		}
	}

	// Register a new particle type with specified properties
	template <typename ContainerType>
	void registerParticleType(ParticleType type, int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact, bool allows_creation,
				  ContainerType *container, bool allows_destruction = false)
	{
		std::unique_ptr<PhysicsParticleDescriptorBase> descriptor;

		// Create the appropriate descriptor based on the particle type
		if (type == ParticleType::Rad) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::Rad>>(
			    mass_idx, lum_idx, birth_time_idx, hydro_interact, allows_creation, container, allows_destruction);
		}
#if AMREX_SPACEDIM == 3
		else if (type == ParticleType::CIC) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::CIC>>(
			    mass_idx, lum_idx, birth_time_idx, hydro_interact, allows_creation, container, allows_destruction);
		} else if (type == ParticleType::CICRad) {
			descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::CICRad>>(
			    mass_idx, lum_idx, birth_time_idx, hydro_interact, allows_creation, container, allows_destruction);
		}
#endif // AMREX_SPACEDIM == 3
		else {
			amrex::Abort("Unknown particle type");
		}

		particleRegistry_[type] = std::move(descriptor);
	}

	// Retrieve a particle descriptor by type
	[[nodiscard]] auto getParticleDescriptor(ParticleType type) const -> const PhysicsParticleDescriptorBase *
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
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->depositMass(rhs, finest_lev, Gconst);
			}
		}
	}
#endif // AMREX_SPACEDIM == 3

	// Redistribute all particles within a level
	void redistribute(int lev)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev);
		}
	}

	// Redistribute all particles with ghost cells
	void redistribute(int lev, int ngrow)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev, ngrow);
		}
	}

	// Write all particle data to plot file
	void writePlotFile(const std::string &plotfilename)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->writePlotFile(plotfilename, getParticleTypeName(type));
			descriptor->writeUnitsFile(plotfilename, getParticleTypeName(type));
		}
	}

	// Write all particle data to checkpoint file
	void writeCheckpoint(const std::string &checkpointname, bool include_header) const
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->writeCheckpoint(checkpointname, getParticleTypeName(type), include_header);
			descriptor->writeUnitsFile(checkpointname, getParticleTypeName(type));
		}
	}

#if AMREX_SPACEDIM == 3
	// Update positions of all massive particles
	void driftParticlesAllLevels(amrex::Real dt, int finest_level)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				for (int lev = 0; lev <= finest_level; ++lev) {
					descriptor->driftParticles(lev, dt);
				}
			}
		}
	}

	// Update velocities of all massive particles
	void kickParticlesAtLevel(amrex::Real dt, amrex::MultiFab &acceleration, int lev)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->kickParticles(lev, dt, acceleration);
			}
		}
	}

	// Create particles based on particle type
	void createParticlesFromState(amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			// Only create particles if the descriptor allows creation
			if (descriptor->getAllowsCreation()) {
				// Call the appropriate particle creation method based on the particle type
				descriptor->createParticlesFromState(state, lev, current_time, dt);
			}
		}
	}

	// Destroy particles based on particle type
	void destroyParticles(int lev, amrex::Real current_time, amrex::Real dt)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			// Only destroy particles if the descriptor allows destruction
			if (descriptor->getAllowsDestruction()) {
				// Call the appropriate particle destruction method based on the particle type
				descriptor->destroyParticles(lev, current_time, dt);
			}
		}
	}

	// Compute maximum particle speed across all particle types
	[[nodiscard]] auto computeMaxParticleSpeed(int lev) const -> amrex::Real
	{
		amrex::Real max_speed = 0.0;
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				const amrex::Real speed = descriptor->computeMaxParticleSpeed(lev);
				AMREX_ASSERT(!std::isnan(speed));
				max_speed = std::max(max_speed, speed);
			}
		}
		return max_speed;
	}
#endif // AMREX_SPACEDIM == 3

	// Print particle statistics
	void printParticleStatistics() const
	{
		amrex::Print() << "Particle statistics:\n";
		amrex::Print() << "Particle type, Number of particles\n";
		for (const auto &[type, descriptor] : particleRegistry_) {
			amrex::Print() << getParticleTypeName(type) << ", " << descriptor->getNumParticles() << "\n";
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
