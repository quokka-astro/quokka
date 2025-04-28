#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>

#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
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
#include <fmt/format.h>

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
	PhysicsParticleDescriptorBase(int mass_idx, int lum_idx, int birth_time_idx, bool allows_creation, bool allows_destruction = false)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), allowsCreation_(allows_creation),
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
	[[nodiscard]] virtual auto getParticleDataAtLevelZero() const -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> = 0;

	// Get particle data at level lev
	[[nodiscard]] virtual auto getParticleDataAtLevel(int lev) const -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> = 0;

	// Pure virtual methods that must be implemented by derived classes
	[[nodiscard]] virtual auto isStarParticle() -> bool = 0;
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

	// Get the number of particles
	[[nodiscard]] virtual auto getNumParticles() const -> int = 0;

#if AMREX_SPACEDIM == 3
	virtual void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) = 0;

	// Drift particle at level lev_min and above for time dt. Note that subcycling is not supported.
	virtual void driftParticles(int lev_min, int lev_max, amrex::Real dt) const = 0;

	// Kick particles at level lev_min and above for time dt. Note that subcycling is not supported.
	virtual void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const &accel) = 0;

	// Create particles from hydro state at the finest level
	// Note: particles are not allowed to spawn outside of real cells. If they do, we will need a redistribution immediately after this call.
	virtual void createParticlesFromState(amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt) const = 0;

	// Destroy particles at level lev_min and above
	virtual void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt) = 0;

	[[nodiscard]] virtual auto computeMaxParticleSpeed(int lev) const -> amrex::ValLocPair<amrex::Real, amrex::RealVect> = 0;

	// Methods that are implemented for some but not all particle types, so they cannot be pure virtual

	virtual void depositSN(amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt)
	{ /* Default empty implementation */ }

	// Tag cells around particles for refinement
	virtual void tagCellsAroundParticles(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) const = 0;
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
	[[nodiscard]] auto isStarParticle() -> bool override { return false; }

	// Get the particle type
	[[nodiscard]] static constexpr auto getParticleType() -> ParticleType { return particleType_; }

	// Constructor initializing descriptor with container and particle properties
	PhysicsParticleDescriptor(ContainerType *container, int mass_idx, int lum_idx, int birth_time_idx, bool allows_creation,
				  bool allows_destruction = false)
	    : PhysicsParticleDescriptorBase(mass_idx, lum_idx, birth_time_idx, allows_creation, allows_destruction), container_(container)
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
	[[nodiscard]] auto getParticleDataAtLevelZero() const -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> override
	{
		std::vector<std::vector<double>> real_data;
		std::vector<std::vector<int>> int_data;

		// // If max level > 0, return empty vectors. This function does not support multi-level particles.
		// if (container_->finestLevel() > 0) {
		// 	return {real_data, int_data};
		// }

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
				typename ContainerType::ParIterType const pIter(analysisPC, 0);
				if (pIter.isValid()) {
					const amrex::Long np = pIter.numParticles();
					auto &particles = pIter.GetArrayOfStructs();

					// Transfer particle data from GPU to CPU for analysis
					typename ContainerType::ParticleType *pData = particles().data();
					amrex::Vector<typename ContainerType::ParticleType> pData_h(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, pData + np, pData_h.begin()); // NOLINT

					// Check if particles have integer components
					constexpr bool has_int_components = (ContainerType::ParticleType::NInt > 0);

					// Pre-size vectors to avoid reallocations
					real_data.reserve(np);
					if constexpr (has_int_components) {
						int_data.reserve(np);
					}

					// Extract positions, real components, and integer components from host data
					for (int i = 0; i < np; ++i) {
						const auto &p = pData_h[i];

						// Process real data (positions and rdata)
						std::vector<double> r_data;
						// Pre-allocate to avoid reallocations
						r_data.reserve(AMREX_SPACEDIM + ContainerType::ParticleType::NReal);

						// First add position components
						for (int d = 0; d < AMREX_SPACEDIM; ++d) {
							r_data.push_back(p.pos(d));
						}

						// Then add all real components (mass, velocities, etc)
						for (int d = 0; d < ContainerType::ParticleType::NReal; ++d) {
							r_data.push_back(p.rdata(d));
						}

						real_data.push_back(std::move(r_data));

						// Process integer data (idata) only if particles have integer components
						if constexpr (has_int_components) {
							std::vector<int> i_data;
							// Pre-allocate to avoid reallocations
							i_data.reserve(ContainerType::ParticleType::NInt);

							// Add all integer components
							for (int d = 0; d < ContainerType::ParticleType::NInt; ++d) {
								i_data.push_back(p.idata(d));
							}

							int_data.push_back(std::move(i_data));
						}
					}
				}
			}
		}

		return {real_data, int_data}; // Empty vectors on non-root ranks
	}

	[[nodiscard]] auto getParticleDataAtLevel(int lev) const -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> override
	{
		std::vector<std::vector<double>> real_data;
		std::vector<std::vector<int>> int_data;

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

			// Initialize the analysis container
			analysisPC.Define(geom, dmap, boxArray);

			// Create a single destination tile on rank 0
			auto &dst_tile = analysisPC.DefineAndReturnParticleTile(0, 0, 0);

			// Get particles only from the specified level
			const auto &particles = container_->GetParticles(lev);

			// First count total particles at this level
			int total_np = 0;
			for (const auto &kv : particles) {
				total_np += kv.second.numParticles();
			}

			// Pre-size the destination tile
			dst_tile.resize(total_np);

			// Copy particles from each tile
			int particle_offset = 0;
			for (const auto &kv : particles) {
				const auto &src_tile = kv.second;
				const int np = src_tile.numParticles();
				if (np > 0) {
					const auto &src_aos = src_tile.GetArrayOfStructs();
					auto &dst_aos = dst_tile.GetArrayOfStructs();
					amrex::Gpu::copy(amrex::Gpu::deviceToDevice, src_aos.data(), src_aos.data() + np, dst_aos.data() + particle_offset);
					particle_offset += np;
				}
			}

			// Now use MPI to gather all particles to rank 0
			analysisPC.Redistribute(); // This handles the MPI communication

			// Only rank 0 processes the particles since they're all gathered there
			if (amrex::ParallelDescriptor::IOProcessor()) {
				// Get iterator for the single box on rank 0
				typename ContainerType::ParIterType const pIter(analysisPC, 0);
				if (pIter.isValid()) {
					const amrex::Long np = pIter.numParticles();
					auto &particles = pIter.GetArrayOfStructs();

					// Transfer particle data from GPU to CPU for analysis
					typename ContainerType::ParticleType *pData = particles().data();
					amrex::Vector<typename ContainerType::ParticleType> pData_h(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, pData + np, pData_h.begin()); // NOLINT

					// Check if particles have integer components
					constexpr bool has_int_components = (ContainerType::ParticleType::NInt > 0);

					// Pre-size vectors to avoid reallocations
					real_data.reserve(np);
					if constexpr (has_int_components) {
						int_data.reserve(np);
					}

					// Process each particle
					for (int i = 0; i < np; ++i) {
						const auto &p = pData_h[i];

						// Process real data (positions and rdata)
						std::vector<double> r_data;
						// Pre-allocate to avoid reallocations
						r_data.reserve(AMREX_SPACEDIM + ContainerType::ParticleType::NReal);

						// Add position components
						for (int d = 0; d < AMREX_SPACEDIM; ++d) {
							r_data.push_back(p.pos(d));
						}

						// Add all real components
						for (int d = 0; d < ContainerType::ParticleType::NReal; ++d) {
							r_data.push_back(p.rdata(d));
						}

						real_data.push_back(std::move(r_data));

						// Process integer data if particles have integer components
						if constexpr (has_int_components) {
							std::vector<int> i_data;
							// Pre-allocate to avoid reallocations
							i_data.reserve(ContainerType::ParticleType::NInt);

							for (int d = 0; d < ContainerType::ParticleType::NInt; ++d) {
								i_data.push_back(p.idata(d));
							}

							int_data.push_back(std::move(i_data));
						}
					}
				}
			}
		}

		return {real_data, int_data}; // Empty vectors on non-root ranks
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

	void createParticlesFromState(amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt) const override
	{
		// Use the traits class to implement the specialized behavior
		ParticleCreationTraits<particleType_>::template createParticles<problem_t, ContainerType>(
		    container_, this->getMassIndex(), state, lev, current_time, dt, this->getEvolutionStageIndex(), this->getBirthTimeIndex());
	}

	void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt) override
	{
		if (container_ != nullptr) {
			ParticleDestructionTraits<particleType_>::template destroyParticles<problem_t, ContainerType>(
			    container_, this->getMassIndex(), lev_min, current_time, dt, this->getBirthTimeIndex(), this->getEvolutionStageIndex());
		}
	}

	// Compute maximum particle speed at a given level
	[[nodiscard]] auto computeMaxParticleSpeed(int lev) const -> amrex::ValLocPair<amrex::Real, amrex::RealVect> override
	{
		amrex::ValLocPair<amrex::Real, amrex::RealVect> max_speed{.value = 0, .index = amrex::RealVect { AMREX_D_DECL(NAN, NAN, NAN) }};

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
			// Only write on rank 0
			if (amrex::ParallelDescriptor::IOProcessor()) {
				// Create the full path for the Fields.yaml file
				std::string filename;
#ifdef QUOKKA_USE_OPENPMD
				// For OpenPMD, write the YAML file alongside the OpenPMD file
				filename = snapshot_name + "_" + name + ".yaml";
#else
				// For standard output, write the YAML file in the particle directory
				filename = snapshot_name + "/" + name + "/Fields.yaml";
#endif

				// Open the file for writing
				std::ofstream outFile(filename);
				if (!outFile) {
					amrex::Abort("Error opening file for writing: " + filename);
				}

				// Get the units data for this particle type
				const auto &unitsData = get_units_data();
				if (unitsData.find(particleType_) == unitsData.end()) {
					amrex::Abort(
					    "Error: Particle type not defined in units data map. Please add units for this particle type in get_units_data().");
				}

				const auto &typeData = unitsData.at(particleType_);
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

	void printParticleStatistics() const override
	{
		if (container_ != nullptr) {
			// TODO(cch): add a getParticleTypeName() method to PhysicsParticleDescriptor and call it here
			const std::string particle_type_name = PhysicsParticleRegister<problem_t>::getParticleTypeName(particleType_);
			amrex::Print() << fmt::format("{:<20}{:<15}\n", particle_type_name, getNumParticles());

			for (int lev = 0; lev <= container_->finestLevel(); ++lev) {
				// if max_level = 0 and has stellar evolution stage, print the mass and particle stage for all particles
				if (getEvolutionStageIndex() >= 0) {
					const auto [real_data, int_data] = getParticleDataAtLevel(lev);

					if (!real_data.empty()) {
						amrex::Print() << "Level " << lev << "\n";
						// Print header for detailed particle data
						amrex::Print() << fmt::format("  {:<15} | {:>20}\n", "Mass", "Stellar evolution stage");
						// amrex::Print() << fmt::format("  {}\n", std::string(15 + 3 + 20, '-'));

						// Print each particle's data with aligned columns
						for (int i = 0; i < static_cast<int>(real_data.size()); ++i) {
							amrex::Print() << fmt::format("  {:<15} | {:>20}\n", real_data[i][AMREX_SPACEDIM + getMassIndex()],
										      int_data[i][getEvolutionStageIndex()]);
						}
						amrex::Print() << "\n"; // Add extra line for readability between particle types
					}
				}
			}
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
#endif
};

// New class for star particles that adds stellar evolution capabilities
template <typename ContainerType, typename problem_t, ParticleType particleType>
class StarParticleDescriptor : public PhysicsParticleDescriptor<ContainerType, problem_t, particleType>
{
      public:
	[[nodiscard]] auto isStarParticle() -> bool override { return true; }

	// Constructor - forwards all arguments to the base class
	StarParticleDescriptor(ContainerType *container, int mass_idx, int lum_idx, int birth_time_idx, bool allows_creation, bool allows_destruction = false,
			       int evolution_stage_idx = -1, bool allows_accretion = false)
	    : PhysicsParticleDescriptor<ContainerType, problem_t, particleType>(container, mass_idx, lum_idx, birth_time_idx, allows_creation,
										allows_destruction)
	{
		this->setEvolutionStageIndex(evolution_stage_idx);
		this->setAllowsAccretion(allows_accretion);
	}

#if AMREX_SPACEDIM == 3
	// Implementation of supernova energy and momentum deposition from particles to grid
	void depositSN(amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt) override
	{
		if (this->container_ != nullptr && this->getEvolutionStageIndex() >= 0) {
			if (!quokka::disable_SN_feedback) {
				// Requires CGS units
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Physics_Traits<problem_t>::unit_system == UnitSystem::CGS,
								 "UnitSystem must be CGS for particleMeshInteraction");

				// Deposit supernova energy and momentum from all particles. This also updates the evolution stage of the particles.
				SNDeposition<ContainerType, problem_t>(this->container_, state, state_buffer, lev, time, dt, this->getMassIndex(),
								       this->getEvolutionStageIndex(), this->getBirthTimeIndex());
			} else {
				// Only update evolution stage but not deposit energy/momentum
				SNFeedbackUtils::updateEvolutionStage(this->container_, lev, time + dt, this->getBirthTimeIndex(),
								      this->getEvolutionStageIndex());
			}
		}
	}
#endif // AMREX_SPACEDIM == 3
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

	// Check if registry contains any star particles
	[[nodiscard]] auto HasStarParticles() const -> bool
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->isStarParticle()) {
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
			case ParticleType::Test:
				return "Test_particles";
			case ParticleType::StochasticStellarPop:
				return "StochasticStellarPop_particles";
			default:
				return "Unknown_particles";
		}
	}

	// Register a new particle type with specified properties
	template <typename ContainerType> void registerParticleType(ContainerType *container, ParticleType type)
	{
		std::unique_ptr<PhysicsParticleDescriptorBase> descriptor;

		// Create the appropriate descriptor based on the particle type
		// The parameters for the descriptor are: mass_idx, lum_idx, birth_time_idx, allows_creation, allows_destruction
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
		}
#endif // AMREX_SPACEDIM == 3
		else {
			amrex::Abort("Unknown particle type for physics particles");
		}

		particleRegistry_[type] = std::move(descriptor);
	}

#if AMREX_SPACEDIM == 3
	// Register a new star particle type with specified properties
	// Star particles have additional stellar evolution capabilities including supernova feedback
	template <typename ContainerType> void registerStarParticleType(ContainerType *container, ParticleType type)
	{
		std::unique_ptr<PhysicsParticleDescriptorBase> descriptor;

		// Create the appropriate star particle descriptor based on the particle type
		// The parameters for the descriptor are: mass_idx, lum_idx, birth_time_idx, allows_creation, allows_destruction, evolution_stage_idx,
		// allows_accretion
		if (type == ParticleType::StochasticStellarPop) {
			descriptor = std::make_unique<StarParticleDescriptor<ContainerType, problem_t, ParticleType::StochasticStellarPop>>(
			    container, StochasticStellarPopParticleMassIdx, StochasticStellarPopParticleLumIdx, StochasticStellarPopParticleBirthTimeIdx, true,
			    false, StochasticStellarPopParticleStageIdx, true);
		} else if (type == ParticleType::Test) {
			descriptor = std::make_unique<StarParticleDescriptor<ContainerType, problem_t, ParticleType::Test>>(
			    container, TestParticleMassIdx, TestParticleLumIdx, TestParticleBirthTimeIdx, true, true, TestParticleStageIdx, true);
		} else {
			amrex::Abort("Unknown particle type for star particles");
		}

		particleRegistry_[type] = std::move(descriptor);
	}
#endif // AMREX_SPACEDIM == 3

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

	// Deposit supernova energy and momentum from all particles
	void depositSN(amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt)
	{
		// this function is only implemented for some particle types, so we specify the particle type manually here
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->isStarParticle()) {
				descriptor->depositSN(state, state_buffer, lev, time, dt);
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
	void driftParticlesAllLevels(amrex::Real dt, int lev_max)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->driftParticles(0, lev_max, dt);
			}
		}
	}

	// Update velocities of all massive particles
	void kickParticlesAtLevel(int lev, amrex::Real dt, amrex::MultiFab &accel)
	{
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->kickParticles(lev, dt, accel);
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

				// redistribute particles
				// descriptor->redistribute(lev);
			}
		}
	}

	// Destroy particles based on particle type
	void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt)
	{
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
		for (const auto &[type, descriptor] : particleRegistry_) {
			if (descriptor->getForceFinestLevel()) {
				AMREX_ALWAYS_ASSERT(n_error_buf.min() >= 2);
				descriptor->tagCellsAroundParticles(lev, tags, time, ngrow);
			}
		}
	}
#endif // AMREX_SPACEDIM == 3

	// Print particle statistics
	void printParticleStatistics() const
	{
		amrex::Print() << ">>> Particle statistics:\n";
		amrex::Print() << fmt::format("{:<20}{:>15}\n", "Particle type", "Number of particles");

		for (const auto &[type, descriptor] : particleRegistry_) {
			descriptor->printParticleStatistics();
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
