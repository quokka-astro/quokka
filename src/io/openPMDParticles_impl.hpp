#ifndef OPENPMD_PARTICLES_IMPL_HPP_
#define OPENPMD_PARTICLES_IMPL_HPP_

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuPrint.H>
#include <AMReX_GpuUtility.H>
#include <AMReX_Loop.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particles.H>

#include "openPMD/openPMD.hpp"
#include "particles/PhysicsParticles.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace quokka::OpenPMDOutput
{
namespace detail
{
inline std::vector<std::string> getPositionComponentLabels()
{
#if AMREX_SPACEDIM == 1
	return {"x"};
#elif AMREX_SPACEDIM == 2
	return {"x", "y"};
#else
	return {"x", "y", "z"};
#endif
}

inline std::vector<std::string> getVelocityComponentLabels()
{
#if AMREX_SPACEDIM == 1
	return {"x"};
#elif AMREX_SPACEDIM == 2
	return {"x", "y"};
#else
	return {"x", "y", "z"};
#endif
}

inline std::vector<std::string> makeLuminosityComponentLabels(int n)
{
	std::vector<std::string> labels;
	labels.reserve(n);
	for (int i = 0; i < n; ++i) {
		labels.emplace_back("g" + std::to_string(i));
	}
	return labels;
}

// Layout information describing which particle attributes are available
struct ParticleLayout {
	bool has_mass = false;
	int mass_index = -1;
	bool has_velocity = false;
	int velocity_index = -1;
	bool has_birth = false;
	int birth_index = -1;
	bool has_death = false;
	int death_index = -1;
	bool has_luminosity = false;
	int luminosity_index = -1;
	int luminosity_components = 0;
	bool has_stage = false;
	int stage_index = -1;
};

struct ParticleCountSummary {
	std::vector<unsigned long long> local_counts;
	std::vector<unsigned long long> global_counts;
	std::vector<unsigned long long> level_offsets;
	std::vector<unsigned long long> rank_offsets;
	unsigned long long total_global = 0;
};

inline ParticleLayout buildParticleLayout(const PhysicsParticleDescriptorBase &descriptor, int num_real_comps)
{
	ParticleLayout layout{};
	layout.mass_index = descriptor.getMassIndex();
	if (layout.mass_index >= 0 && layout.mass_index < num_real_comps) {
		layout.has_mass = true;
		const int velocity_start = layout.mass_index + 1;
		if (velocity_start + AMREX_SPACEDIM - 1 < num_real_comps) {
			layout.has_velocity = true;
			layout.velocity_index = velocity_start;
		}
	}

	layout.birth_index = descriptor.getBirthTimeIndex();
	if (layout.birth_index >= 0 && layout.birth_index < num_real_comps) {
		layout.has_birth = true;
		layout.death_index = layout.birth_index + 1;
		if (layout.death_index >= 0 && layout.death_index < num_real_comps) {
			layout.has_death = true;
		}
	}

	layout.luminosity_index = descriptor.getLumIndex();
	if (layout.luminosity_index >= 0 && layout.luminosity_index < num_real_comps) {
		layout.has_luminosity = true;
		layout.luminosity_components = num_real_comps - layout.luminosity_index;
	}

	layout.stage_index = descriptor.getEvolutionStageIndex();
	if (layout.stage_index >= 0) {
		layout.has_stage = true;
	}
	return layout;
}



template <typename ContainerType>
ParticleCountSummary computeParticleCountSummary(ContainerType &container)
{
	ParticleCountSummary summary{};
	const int finest = container.finestLevel();
	if (finest < 0) {
		return summary;
	}

	const int levels = finest + 1;
	summary.local_counts.assign(levels, 0ULL);
	summary.global_counts.assign(levels, 0ULL);
	summary.level_offsets.assign(levels, 0ULL);
	summary.rank_offsets.assign(levels, 0ULL);

	for (int lev = 0; lev <= finest; ++lev) {
		for (auto const &kv : container.GetParticles(lev)) {
			summary.local_counts[lev] += static_cast<unsigned long long>(kv.second.numParticles());
		}

		unsigned long long local = summary.local_counts[lev];
		amrex::Long local_long = static_cast<amrex::Long>(local);
		amrex::Long total_long = local_long;
		amrex::ParallelDescriptor::ReduceLongSum(total_long);
		summary.global_counts[lev] = static_cast<unsigned long long>(total_long);

		int const nprocs = amrex::ParallelDescriptor::NProcs();
		amrex::Vector<amrex::Long> gathered(nprocs, 0);
		amrex::Vector<unsigned long long> offsets(nprocs, 0ULL);
		amrex::ParallelDescriptor::Gather(&local_long, 1, gathered.data(), 1, amrex::ParallelDescriptor::IOProcessorNumber());
		if (amrex::ParallelDescriptor::IOProcessor()) {
			unsigned long long running = 0;
			for (int ip = 0; ip < nprocs; ++ip) {
				offsets[ip] = running;
				running += static_cast<unsigned long long>(gathered[ip]);
			}
		}
		amrex::ParallelDescriptor::Bcast(offsets.data(), nprocs, amrex::ParallelDescriptor::IOProcessorNumber());
		summary.rank_offsets[lev] = offsets[amrex::ParallelDescriptor::MyProc()];
	}

	unsigned long long cumulative = 0;
	for (int lev = 0; lev <= finest; ++lev) {
		summary.level_offsets[lev] = cumulative;
		cumulative += summary.global_counts[lev];
	}
	summary.total_global = cumulative;

	return summary;
}

inline void initializeSpecies(openPMD::ParticleSpecies &species, const std::vector<std::string> &position_components,
                             const ParticleLayout &layout, const std::vector<std::string> &velocity_components,
                             const std::vector<std::string> &luminosity_components, unsigned long long total_particles)
{
	auto real_dataset = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {total_particles});
	auto id_dataset = openPMD::Dataset(openPMD::determineDatatype<uint64_t>(), {total_particles});

	for (auto const &comp : position_components) {
		species["position"][comp].resetDataset(real_dataset);
	}
	species["id"][openPMD::RecordComponent::SCALAR].resetDataset(id_dataset);

	if (layout.has_mass) {
		species["mass"][openPMD::RecordComponent::SCALAR].resetDataset(real_dataset);
	}

	if (layout.has_velocity) {
		for (auto const &comp : velocity_components) {
			species["velocity"][comp].resetDataset(real_dataset);
		}
	}

	if (layout.has_birth) {
		species["birthTime"][openPMD::RecordComponent::SCALAR].resetDataset(real_dataset);
	}

	if (layout.has_death) {
		species["deathTime"][openPMD::RecordComponent::SCALAR].resetDataset(real_dataset);
	}

	if (layout.has_luminosity) {
		for (auto const &comp : luminosity_components) {
			species["luminosity"][comp].resetDataset(real_dataset);
		}
	}

	if (layout.has_stage) {
		auto int_dataset = openPMD::Dataset(openPMD::determineDatatype<int>(), {total_particles});
		species["evolutionStage"][openPMD::RecordComponent::SCALAR].resetDataset(int_dataset);
	}
}

template <typename T>
inline void storeChunk(openPMD::RecordComponent component, const T *ptr, unsigned long long offset, unsigned long long extent)
{
	const std::vector<uint64_t> offset_vec{offset};
	const std::vector<uint64_t> extent_vec{extent};
	component.storeChunkRaw(ptr, offset_vec, extent_vec);
}

template <typename DescriptorType>
void writeParticleSpecies(openPMD::Series &series, openPMD::Iteration &iteration, DescriptorType &descriptor,
                         const std::string &species_name)
{
	using ContainerType = typename DescriptorType::ContainerT;
	ContainerType *container = descriptor.getContainer();
	if (container == nullptr) {
		return;
	}

	auto counts = computeParticleCountSummary(*container);
	const unsigned long long total_particles = counts.total_global;

	auto position_components = getPositionComponentLabels();
	auto velocity_components = getVelocityComponentLabels();

	const int num_real_comps = ContainerType::ParticleType::NReal;
	auto layout = buildParticleLayout(descriptor, num_real_comps);
	std::vector<std::string> luminosity_components;
	if (layout.has_luminosity) {
		luminosity_components = makeLuminosityComponentLabels(layout.luminosity_components);
	}

	auto species = iteration.particles[species_name];
	initializeSpecies(species, position_components, layout, velocity_components, luminosity_components, total_particles);

	if (total_particles == 0) {
		return;
	}

	const int finest = container->finestLevel();
	for (int lev = 0; lev <= finest; ++lev) {
		unsigned long long level_offset = counts.level_offsets[lev] + counts.rank_offsets[lev];
		unsigned long long running_offset = 0;

		for (typename ContainerType::ParConstIterType pti(*container, lev); pti.isValid(); ++pti) {
			const int num_particles = pti.numParticles();
			if (num_particles == 0) {
				continue;
			}

			const unsigned long long global_offset = level_offset + running_offset;
			running_offset += static_cast<unsigned long long>(num_particles);

			auto const &aos = pti.GetArrayOfStructs();
			auto const *aos_ptr = aos().data();

			amrex::Gpu::ManagedVector<amrex::ParticleReal> pos_x(num_particles);
#if AMREX_SPACEDIM >= 2
			amrex::Gpu::ManagedVector<amrex::ParticleReal> pos_y(num_particles);
#endif
#if AMREX_SPACEDIM >= 3
			amrex::Gpu::ManagedVector<amrex::ParticleReal> pos_z(num_particles);
#endif
			amrex::Gpu::ManagedVector<unsigned long long> ids(num_particles);

			amrex::Gpu::ManagedVector<amrex::ParticleReal> mass;
			if (layout.has_mass) {
				mass.resize(num_particles);
			}

			std::array<amrex::Gpu::ManagedVector<amrex::ParticleReal>, AMREX_SPACEDIM> velocity_dests;
			const int vel_components = layout.has_velocity ? static_cast<int>(position_components.size()) : 0;
			for (int d = 0; d < vel_components; ++d) {
				velocity_dests[d].resize(num_particles);
			}

			amrex::Gpu::ManagedVector<amrex::ParticleReal> birth;
			if (layout.has_birth) {
				birth.resize(num_particles);
			}

			amrex::Gpu::ManagedVector<amrex::ParticleReal> death;
			if (layout.has_death) {
				death.resize(num_particles);
			}

			std::vector<amrex::Gpu::ManagedVector<amrex::ParticleReal>> luminosity_buffers;
			if (layout.has_luminosity) {
				luminosity_buffers.resize(layout.luminosity_components);
				for (auto &buf : luminosity_buffers) {
					buf.resize(num_particles);
				}
			}

			amrex::Gpu::ManagedVector<int> stage;
			if (layout.has_stage) {
				stage.resize(num_particles);
			}

			auto const &soa = pti.GetStructOfArrays();
			const int soa_real_components = soa.NumRealComps();
			const int soa_int_components = soa.NumIntComps();
			constexpr int aos_real_components = ContainerType::ParticleType::NReal;
			constexpr int aos_int_components = ContainerType::ParticleType::NInt;

			auto const *mass_src_soa =
			    (layout.has_mass && layout.mass_index >= 0 && layout.mass_index < soa_real_components)
			        ? soa.GetRealData(layout.mass_index).data()
			        : nullptr;
			const int mass_rdata_index =
			    (layout.has_mass && layout.mass_index >= 0 && layout.mass_index < aos_real_components) ? layout.mass_index : -1;

			std::array<const amrex::ParticleReal *, AMREX_SPACEDIM> velocity_src_soa{};
			velocity_src_soa.fill(nullptr);
			std::array<int, AMREX_SPACEDIM> velocity_rdata_indices{};
			velocity_rdata_indices.fill(-1);
			for (int d = 0; d < vel_components; ++d) {
				const int comp_index = layout.velocity_index + d;
				if (comp_index >= 0 && comp_index < soa_real_components) {
					velocity_src_soa[d] = soa.GetRealData(comp_index).data();
				} else if (comp_index >= 0 && comp_index < aos_real_components) {
					velocity_rdata_indices[d] = comp_index;
				}
			}
			auto const *vel_src_soa_x = (vel_components > 0) ? velocity_src_soa[0] : nullptr;
			const int vel_rdata_index_x = (vel_components > 0) ? velocity_rdata_indices[0] : -1;
#if AMREX_SPACEDIM >= 2
			auto const *vel_src_soa_y = (vel_components > 1) ? velocity_src_soa[1] : nullptr;
			const int vel_rdata_index_y = (vel_components > 1) ? velocity_rdata_indices[1] : -1;
#endif
#if AMREX_SPACEDIM >= 3
			auto const *vel_src_soa_z = (vel_components > 2) ? velocity_src_soa[2] : nullptr;
			const int vel_rdata_index_z = (vel_components > 2) ? velocity_rdata_indices[2] : -1;
#endif

			auto const *birth_src_soa =
			    (layout.has_birth && layout.birth_index >= 0 && layout.birth_index < soa_real_components)
			        ? soa.GetRealData(layout.birth_index).data()
			        : nullptr;
			const int birth_rdata_index =
			    (layout.has_birth && layout.birth_index >= 0 && layout.birth_index < aos_real_components) ? layout.birth_index : -1;

			auto const *death_src_soa =
			    (layout.has_death && layout.death_index >= 0 && layout.death_index < soa_real_components)
			        ? soa.GetRealData(layout.death_index).data()
			        : nullptr;
			const int death_rdata_index =
			    (layout.has_death && layout.death_index >= 0 && layout.death_index < aos_real_components) ? layout.death_index : -1;

			const int *stage_src_soa =
			    (layout.has_stage && layout.stage_index >= 0 && layout.stage_index < soa_int_components)
			        ? soa.GetIntData(layout.stage_index).data()
			        : nullptr;
			const int stage_idata_index =
			    (layout.has_stage && layout.stage_index >= 0 && layout.stage_index < aos_int_components) ? layout.stage_index : -1;

			std::vector<const amrex::ParticleReal *> luminosity_src_soa(layout.luminosity_components, nullptr);
			std::vector<int> luminosity_rdata_indices(layout.luminosity_components, -1);
			if (layout.has_luminosity) {
				for (int idx = 0; idx < layout.luminosity_components; ++idx) {
					const int lum_index = layout.luminosity_index + idx;
					if (lum_index >= 0 && lum_index < soa_real_components) {
						luminosity_src_soa[idx] = soa.GetRealData(lum_index).data();
					} else if (lum_index >= 0 && lum_index < aos_real_components) {
						luminosity_rdata_indices[idx] = lum_index;
					}
				}
			}

			if (layout.has_mass) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mass_src_soa != nullptr || mass_rdata_index >= 0,
				    "Particle layout advertises mass, but no storage location was found.");
			}
			for (int d = 0; d < vel_components; ++d) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(velocity_src_soa[d] != nullptr || velocity_rdata_indices[d] >= 0,
				    "Particle layout advertises velocity component, but no storage location was found.");
			}
			if (layout.has_birth) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(birth_src_soa != nullptr || birth_rdata_index >= 0,
				    "Particle layout advertises birth time, but no storage location was found.");
			}
			if (layout.has_death) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(death_src_soa != nullptr || death_rdata_index >= 0,
				    "Particle layout advertises death time, but no storage location was found.");
			}
			if (layout.has_stage) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(stage_src_soa != nullptr || stage_idata_index >= 0,
				    "Particle layout advertises evolution stage, but no storage location was found.");
			}
			for (int idx = 0; idx < layout.luminosity_components; ++idx) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
				    luminosity_src_soa[idx] != nullptr || luminosity_rdata_indices[idx] >= 0,
				    "Particle layout advertises luminosity component, but no storage location was found.");
			}

			auto *pos_x_ptr = pos_x.data();
#if AMREX_SPACEDIM >= 2
			auto *pos_y_ptr = pos_y.data();
#endif
#if AMREX_SPACEDIM >= 3
			auto *pos_z_ptr = pos_z.data();
#endif
			auto *id_ptr = ids.data();
			auto *mass_dst = layout.has_mass ? mass.data() : nullptr;
			auto *birth_dst = layout.has_birth ? birth.data() : nullptr;
			auto *death_dst = layout.has_death ? death.data() : nullptr;
			int *stage_dst = layout.has_stage ? stage.data() : nullptr;
			auto *vel_dst_x = (vel_components > 0) ? velocity_dests[0].data() : nullptr;
#if AMREX_SPACEDIM >= 2
			auto *vel_dst_y = (vel_components > 1) ? velocity_dests[1].data() : nullptr;
#endif
#if AMREX_SPACEDIM >= 3
			auto *vel_dst_z = (vel_components > 2) ? velocity_dests[2].data() : nullptr;
#endif

			amrex::ParallelFor(num_particles,
			    [aos_ptr, pos_x_ptr,
#if AMREX_SPACEDIM >= 2
			     pos_y_ptr,
#endif
#if AMREX_SPACEDIM >= 3
			     pos_z_ptr,
#endif
			     id_ptr, mass_src_soa, mass_rdata_index, mass_dst, vel_src_soa_x, vel_rdata_index_x, vel_dst_x,
#if AMREX_SPACEDIM >= 2
			     vel_src_soa_y, vel_rdata_index_y, vel_dst_y,
#endif
#if AMREX_SPACEDIM >= 3
			     vel_src_soa_z, vel_rdata_index_z, vel_dst_z,
#endif
			     birth_src_soa, birth_rdata_index, birth_dst, death_src_soa, death_rdata_index, death_dst, stage_src_soa,
			     stage_idata_index, stage_dst] AMREX_GPU_DEVICE(int i) noexcept {
				auto const &p = aos_ptr[i];
				pos_x_ptr[i] = p.pos(0);
#if AMREX_SPACEDIM >= 2
				pos_y_ptr[i] = p.pos(1);
#endif
#if AMREX_SPACEDIM >= 3
				pos_z_ptr[i] = p.pos(2);
#endif
				id_ptr[i] = p.id();
				if (mass_dst) {
					if (mass_src_soa != nullptr) {
						mass_dst[i] = mass_src_soa[i];
					} else if (mass_rdata_index >= 0) {
						mass_dst[i] = p.rdata(mass_rdata_index);
					}
				}
				if (vel_dst_x) {
					if (vel_src_soa_x != nullptr) {
						vel_dst_x[i] = vel_src_soa_x[i];
					} else if (vel_rdata_index_x >= 0) {
						vel_dst_x[i] = p.rdata(vel_rdata_index_x);
					}
				}
#if AMREX_SPACEDIM >= 2
				if (vel_dst_y) {
					if (vel_src_soa_y != nullptr) {
						vel_dst_y[i] = vel_src_soa_y[i];
					} else if (vel_rdata_index_y >= 0) {
						vel_dst_y[i] = p.rdata(vel_rdata_index_y);
					}
				}
#endif
#if AMREX_SPACEDIM >= 3
				if (vel_dst_z) {
					if (vel_src_soa_z != nullptr) {
						vel_dst_z[i] = vel_src_soa_z[i];
					} else if (vel_rdata_index_z >= 0) {
						vel_dst_z[i] = p.rdata(vel_rdata_index_z);
					}
				}
#endif
				if (birth_dst) {
					if (birth_src_soa != nullptr) {
						birth_dst[i] = birth_src_soa[i];
					} else if (birth_rdata_index >= 0) {
						birth_dst[i] = p.rdata(birth_rdata_index);
					}
				}
				if (death_dst) {
					if (death_src_soa != nullptr) {
						death_dst[i] = death_src_soa[i];
					} else if (death_rdata_index >= 0) {
						death_dst[i] = p.rdata(death_rdata_index);
					}
				}
				if (stage_dst) {
					if (stage_src_soa != nullptr) {
						stage_dst[i] = stage_src_soa[i];
					} else if (stage_idata_index >= 0) {
						stage_dst[i] = p.idata(stage_idata_index);
					}
				}
			});

			for (int idx = 0; idx < layout.luminosity_components; ++idx) {
				auto *lum_dst = luminosity_buffers[idx].data();
				if (auto const *lum_src = luminosity_src_soa[idx]; lum_src != nullptr) {
					amrex::ParallelFor(num_particles,
					    [lum_src, lum_dst] AMREX_GPU_DEVICE(int i) noexcept { lum_dst[i] = lum_src[i]; });
				} else {
					const int lum_rdata_index = luminosity_rdata_indices[idx];
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lum_rdata_index >= 0,
					    "Luminosity component storage index is invalid.");
					amrex::ParallelFor(num_particles, [aos_ptr, lum_dst, lum_rdata_index] AMREX_GPU_DEVICE(int i) noexcept {
						lum_dst[i] = aos_ptr[i].rdata(lum_rdata_index);
					});
				}
			}

			amrex::Gpu::Device::streamSynchronize();

			const unsigned long long extent = static_cast<unsigned long long>(num_particles);
			storeChunk(species["position"]["x"], pos_x_ptr, global_offset, extent);
#if AMREX_SPACEDIM >= 2
			storeChunk(species["position"]["y"], pos_y_ptr, global_offset, extent);
#endif
#if AMREX_SPACEDIM >= 3
			storeChunk(species["position"]["z"], pos_z_ptr, global_offset, extent);
#endif
			storeChunk(species["id"][openPMD::RecordComponent::SCALAR], id_ptr, global_offset, extent);

			if (layout.has_mass) {
				storeChunk(species["mass"][openPMD::RecordComponent::SCALAR], mass_dst, global_offset, extent);
			}

			if (vel_components > 0) {
				storeChunk(species["velocity"][velocity_components[0]], vel_dst_x, global_offset, extent);
			}
#if AMREX_SPACEDIM >= 2
			if (vel_components > 1) {
				storeChunk(species["velocity"][velocity_components[1]], vel_dst_y, global_offset, extent);
			}
#endif
#if AMREX_SPACEDIM >= 3
			if (vel_components > 2) {
				storeChunk(species["velocity"][velocity_components[2]], vel_dst_z, global_offset, extent);
			}
#endif

			if (layout.has_birth) {
				storeChunk(species["birthTime"][openPMD::RecordComponent::SCALAR], birth_dst, global_offset, extent);
			}

			if (layout.has_death) {
				storeChunk(species["deathTime"][openPMD::RecordComponent::SCALAR], death_dst, global_offset, extent);
			}

			if (layout.has_luminosity) {
				for (int idx = 0; idx < layout.luminosity_components; ++idx) {
					storeChunk(
					    species["luminosity"][luminosity_components[idx]],
					    luminosity_buffers[idx].data(), global_offset, extent);
				}
			}

			if (layout.has_stage) {
				storeChunk(
				    species["evolutionStage"][openPMD::RecordComponent::SCALAR],
				    stage_dst, global_offset, extent);
			}
		}
	}
}

} // namespace detail

template <typename problem_t>
void WriteParticles(openPMD::Series &series, openPMD::Iteration &iteration,
                    PhysicsParticleRegister<problem_t> &particle_register, amrex::Real /*time*/)
{
	particle_register.forEachDescriptor([
	    &series, &iteration](ParticleType type, PhysicsParticleDescriptorBase &descriptor_base) {
		const std::string species_name = PhysicsParticleRegister<problem_t>::getParticleTypeName(type);

		switch (type) {
		case ParticleType::Rad: {
			using Descriptor = PhysicsParticleDescriptor<RadParticleContainer<problem_t>, problem_t, ParticleType::Rad>;
			if (auto *typed = dynamic_cast<Descriptor *>(&descriptor_base)) {
				detail::writeParticleSpecies(series, iteration, *typed, species_name);
			}
			break;
		}
#if AMREX_SPACEDIM == 3
		case ParticleType::CIC: {
			using Descriptor = PhysicsParticleDescriptor<CICParticleContainer, problem_t, ParticleType::CIC>;
			if (auto *typed = dynamic_cast<Descriptor *>(&descriptor_base)) {
				detail::writeParticleSpecies(series, iteration, *typed, species_name);
			}
			break;
		}
		case ParticleType::CICRad: {
			using Descriptor = PhysicsParticleDescriptor<CICRadParticleContainer<problem_t>, problem_t, ParticleType::CICRad>;
			if (auto *typed = dynamic_cast<Descriptor *>(&descriptor_base)) {
				detail::writeParticleSpecies(series, iteration, *typed, species_name);
			}
			break;
		}
		case ParticleType::StochasticStellarPop: {
			using Descriptor = PhysicsParticleDescriptor<StochasticStellarPopParticleContainer<problem_t>, problem_t, ParticleType::StochasticStellarPop>;
			if (auto *typed = dynamic_cast<Descriptor *>(&descriptor_base)) {
				detail::writeParticleSpecies(series, iteration, *typed, species_name);
			}
			break;
		}
		case ParticleType::Sink: {
			using Descriptor = PhysicsParticleDescriptor<SinkParticleContainer, problem_t, ParticleType::Sink>;
			if (auto *typed = dynamic_cast<Descriptor *>(&descriptor_base)) {
				detail::writeParticleSpecies(series, iteration, *typed, species_name);
			}
			break;
		}
		case ParticleType::Test: {
			using Descriptor = PhysicsParticleDescriptor<TestParticleContainer<problem_t>, problem_t, ParticleType::Test>;
			if (auto *typed = dynamic_cast<Descriptor *>(&descriptor_base)) {
				detail::writeParticleSpecies(series, iteration, *typed, species_name);
			}
			break;
		}
#endif
		default:
			break;
		}
	});
}

} // namespace quokka::OpenPMDOutput

#endif // OPENPMD_PARTICLES_IMPL_HPP_
