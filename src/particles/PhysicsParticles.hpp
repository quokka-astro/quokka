#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <AMReX_AmrParticles.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParticleInterpolators.H>

#include "physics_info.hpp"

namespace quokka
{

enum CICParticleDataIdx { CICParticleMassIdx = 0, CICParticleVxIdx, CICParticleVyIdx, CICParticleVzIdx };
constexpr int CICParticleRealComps = 4; // mass vx vy vz
using CICParticleContainer = amrex::AmrParticleContainer<CICParticleRealComps>;
using CICParticleIterator = amrex::ParIter<CICParticleRealComps>;

enum RadParticleDataIdx { RadParticleBirthTimeIdx = 0, RadParticleDeathTimeIdx, RadParticleLumIdx };
template <typename problem_t> constexpr int RadParticleRealComps = 2 + Physics_Traits<problem_t>::nGroups;
template <typename problem_t> using RadParticleContainer = amrex::AmrParticleContainer<RadParticleRealComps<problem_t>>;
template <typename problem_t> using RadParticleIterator = amrex::ParIter<RadParticleRealComps<problem_t>>;

// CICRad particles
enum CICRadParticleDataIdx {
	CICRadParticleMassIdx = 0,
	CICRadParticleVxIdx,
	CICRadParticleVyIdx,
	CICRadParticleVzIdx,
	CICRadParticleBirthTimeIdx,
	CICRadParticleDeathTimeIdx,
	CICRadParticleLumIdx
};
template <typename problem_t>
constexpr int CICRadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 6 + Physics_Traits<problem_t>::nGroups; // mass vx vy vz birth_time
							       // death_time lum1 ... lumN
	} else {
		return 6; // mass vx vy vz birth_time death_time
	}
}();
template <typename problem_t> using CICRadParticleContainer = amrex::AmrParticleContainer<CICRadParticleRealComps<problem_t>>;
template <typename problem_t> using CICRadParticleIterator = amrex::ParIter<CICRadParticleRealComps<problem_t>>;

struct MassDeposition {
	amrex::Real Gconst{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};

	template <typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ParticleType &p, amrex::Array4<amrex::Real> const &rho,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ParticleType &part, int comp) {
			return 4.0 * M_PI * Gconst * part.rdata(comp); // weight by 4 pi G
		});
	}
};

struct RadDeposition {
	double current_time{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};
	int birthTimeIndex{};

	template <typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ParticleType &p, amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ParticleType &part, int comp) {
			if (current_time < part.rdata(birthTimeIndex) || current_time >= part.rdata(birthTimeIndex + 1)) {
				return 0.0;
			}
			return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		});
	}
};

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

// Non-templated base class for type erasure
class PhysicsParticleDescriptorBase {
      protected:
	int massIndex_{-1};		 // index for gravity mass, -1 if not used
	int lumIndex_{-1};		 // index for radiation luminosity, -1 if not used
	int birthTimeIndex_{-1}; // index for birth time, -1 if not used
	bool interactsWithHydro_{false}; // whether particles interact with hydro

      public:
	PhysicsParticleDescriptorBase(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), interactsWithHydro_(hydro_interact)
	{
	}
	virtual ~PhysicsParticleDescriptorBase() = default;

	// Getters
	[[nodiscard]] auto getMassIndex() const -> int { return massIndex_; }
	[[nodiscard]] auto getLumIndex() const -> int { return lumIndex_; }
	[[nodiscard]] auto getBirthTimeIndex() const -> int { return birthTimeIndex_; }
	[[nodiscard]] auto getInteractsWithHydro() const -> bool { return interactsWithHydro_; }

	// Add virtual method for getting particle positions
	[[nodiscard]] virtual auto getParticlePositions(int lev = 0) const -> std::vector<std::array<double, AMREX_SPACEDIM>> = 0;

	// Pure virtual interface for particle operations
	virtual void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) = 0;
	virtual void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst) = 0;
	virtual void redistribute(int lev) = 0;
	virtual void redistribute(int lev, int ngrow) = 0;
	virtual void writePlotFile(const std::string &plotfilename, const std::string &name) = 0;
	virtual void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) = 0;
	virtual void driftParticles(int lev, amrex::Real dt) = 0;
	virtual void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const& acceleration) = 0;
};

// Templated derived class that holds the actual particle container
template <typename ContainerType>
class PhysicsParticleDescriptor : public PhysicsParticleDescriptorBase
{
      private:
	ContainerType *container_{};

      public:
	PhysicsParticleDescriptor(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact, ContainerType *container)
	    : PhysicsParticleDescriptorBase(mass_idx, lum_idx, birth_time_idx, hydro_interact), container_(container)
	{
	}

	[[nodiscard]] auto getParticlePositions(int lev) const -> std::vector<std::array<double, AMREX_SPACEDIM>> override {
		std::vector<std::array<double, AMREX_SPACEDIM>> positions;
		if (container_ != nullptr) {
			const auto& particles = container_->GetParticles(lev);
			for (const auto& kv : particles) {
				const auto& pbox = kv.second;
				const auto& aos = pbox.GetArrayOfStructs();
				for (int i = 0; i < pbox.numParticles(); ++i) {
					const auto& p = aos[i];
					positions.push_back({AMREX_D_DECL(p.pos(0), p.pos(1), p.pos(2))});
				}
			}
		}
		return positions;
	}

	void driftParticles(int lev, amrex::Real dt) override {
		if (container_ != nullptr) {
			for (typename ContainerType::ParIterType pIter(*container_, lev); pIter.isValid(); ++pIter) {
				auto& particles = pIter.GetArrayOfStructs();
				auto* pData = particles().data();
				const amrex::Long np = pIter.numParticles();

				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					auto& p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					// update particle position based on velocity components
					for (int i = 0; i < AMREX_SPACEDIM; ++i) {
						if (this->getMassIndex() >= 0) {
							// For CIC particles, velocity components start after mass
							p.pos(i) += dt * p.rdata(this->getMassIndex() + 1 + i);
						}
					}
				});
			}
		}
	}

	void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const& accel) override {
		if (container_ != nullptr && this->getMassIndex() >= 0) {
			for (typename ContainerType::ParIterType pIter(*container_, lev); pIter.isValid(); ++pIter) {
				auto& particles = pIter.GetArrayOfStructs();
				auto* pData = particles().data();
				const amrex::Long np = pIter.numParticles();

				const auto& accel_arr = accel.array(pIter);
				const auto& geom = container_->Geom(lev);
				const auto plo = geom.ProbLoArray();
				const auto dx_inv = geom.InvCellSizeArray();

				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					auto& p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					amrex::ParticleInterpolator::Linear interp(p, plo, dx_inv);
					
					interp.MeshToParticle(p, accel_arr, 0, this->getMassIndex() + 1, AMREX_SPACEDIM,
						[=] AMREX_GPU_DEVICE(amrex::Array4<const amrex::Real> const& acc, int i, int j, int k, int comp) {
							return acc(i, j, k, comp); // no weighting
						},
						[=] AMREX_GPU_DEVICE(typename ContainerType::ParticleType& p, int comp, amrex::Real acc_comp) {
							// kick particle by updating its velocity
							p.rdata(comp) += 0.5 * dt * static_cast<amrex::ParticleReal>(acc_comp);
						});
				});
			}
		}
	}

	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) override
	{
		if (container_ != nullptr && this->getLumIndex() >= 0) {
			amrex::ParticleToMesh(*container_, radEnergySource, lev,
					      RadDeposition{current_time, this->getLumIndex(), 0, nGroups, this->getBirthTimeIndex()},
					      false);
		}
	}

	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst) override
	{
		if (container_ != nullptr && this->getMassIndex() >= 0) {
			amrex::ParticleToMesh(*container_, amrex::GetVecOfPtrs(rhs), 0, finest_lev,
					      MassDeposition{Gconst, this->getMassIndex(), 0, 1}, true);
		}
	}

	void redistribute(int lev) override
	{
		if (container_ != nullptr) {
			container_->Redistribute(lev);
		}
	}

	void redistribute(int lev, int ngrow) override
	{
		if (container_ != nullptr) {
			container_->Redistribute(lev, container_->finestLevel(), ngrow);
		}
	}

	void writePlotFile(const std::string &plotfilename, const std::string &name) override
	{
		if (container_ != nullptr) {
			container_->WritePlotFile(plotfilename, name);
		}
	}

	void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) override
	{
		if (container_ != nullptr) {
			container_->Checkpoint(checkpointname, name, include_header);
		}
	}
};

// Registry for physics particles
template <typename problem_t> class PhysicsParticleRegister
{
      private:
	std::map<std::string, std::unique_ptr<PhysicsParticleDescriptorBase>> particleRegistry_;

      public:
	PhysicsParticleRegister() = default;
	~PhysicsParticleRegister() = default;

	// Register a new particle type
	template <typename ContainerType>
	void registerParticleType(const std::string &name, int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact, ContainerType *container)
	{
		auto descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType>>(mass_idx, lum_idx, birth_time_idx, hydro_interact, container);
		particleRegistry_[name] = std::move(descriptor);
	}

	// Get a particle descriptor
	[[nodiscard]] auto getParticleDescriptor(const std::string &name) const -> const PhysicsParticleDescriptorBase *
	{
		auto it = particleRegistry_.find(name);
		if (it != particleRegistry_.end()) {
			return it->second.get();
		}
		return nullptr;
	}

	// Deposit radiation from all particles that have luminosity
	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getLumIndex() >= 0) {
				descriptor->depositRadiation(radEnergySource, lev, current_time, Physics_Traits<problem_t>::nGroups);
			}
		}
	}

	// Deposit mass from all particles that have mass for gravity calculation
	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->depositMass(rhs, finest_lev, Gconst);
			}
		}
	}

	// Run Redistribute(lev) on all particles in particleRegistry_
	void redistribute(int lev)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev);
		}
	}

	// Run Redistribute(lev, ngrow) on all particles in particleRegistry_
	void redistribute(int lev, int ngrow)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev, ngrow);
		}
	}

	// Run WritePlotFile(plotfilename, name) on all particles in particleRegistry_
	void writePlotFile(const std::string &plotfilename)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->writePlotFile(plotfilename, name);
		}
	}

	// Run Checkpoint(checkpointname, name, true) on all particles in particleRegistry_
	void writeCheckpoint(const std::string &checkpointname, bool include_header) const
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->writeCheckpoint(checkpointname, name, include_header);
		}
	}

	// Delete copy/move constructors/assignments
	PhysicsParticleRegister(const PhysicsParticleRegister &) = delete;
	PhysicsParticleRegister &operator=(const PhysicsParticleRegister &) = delete;
	PhysicsParticleRegister(PhysicsParticleRegister &&) = delete;
	PhysicsParticleRegister &operator=(PhysicsParticleRegister &&) = delete;

	// In PhysicsParticleRegister class, after depositMass
	void driftParticlesAllLevels(amrex::Real dt, int finest_level) {
		for (const auto& [name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {  // Only drift particles that have mass
				for (int lev = 0; lev <= finest_level; ++lev) {
					descriptor->driftParticles(lev, dt);
				}
			}
		}
	}

	void kickParticlesAllLevels(amrex::Real dt, amrex::Vector<amrex::MultiFab>& acceleration) {
		for (const auto& [name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {  // Only kick particles that have mass
				for (int lev = 0; lev <= acceleration.size()-1; ++lev) {
					descriptor->kickParticles(lev, dt, acceleration[lev]);
				}
			}
		}
	}
};

} // namespace quokka

#endif // PHYSICS_PARTICLES_HPP_
