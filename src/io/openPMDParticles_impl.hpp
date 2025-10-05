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
#include "particles/global_particle_id.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
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

struct ParticleCountSummary {
	std::vector<unsigned long long> local_counts;
	std::vector<unsigned long long> global_counts;
	std::vector<unsigned long long> level_offsets;
	std::vector<unsigned long long> rank_offsets;
	unsigned long long total_global = 0;
};

template <typename ContainerType> ParticleCountSummary computeParticleCountSummary(ContainerType &container)
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
			      const std::vector<OpenPMDRealAttribute> &real_attributes, const std::vector<OpenPMDIntAttribute> &int_attributes,
			      unsigned long long total_particles)
{
	auto real_dataset = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {total_particles});
	auto id_dataset = openPMD::Dataset(openPMD::determineDatatype<uint64_t>(), {total_particles});

	for (auto const &comp : position_components) {
		species["position"][comp].resetDataset(real_dataset);
	}
	species["id"][openPMD::RecordComponent::SCALAR].resetDataset(id_dataset);

	for (auto const &attribute : real_attributes) {
		auto record = species[attribute.record_name];
		for (auto const &component : attribute.components) {
			if (component.label.empty()) {
				record[openPMD::RecordComponent::SCALAR].resetDataset(real_dataset);
			} else {
				record[component.label].resetDataset(real_dataset);
			}
		}
	}

	if (!int_attributes.empty()) {
		auto int_dataset = openPMD::Dataset(openPMD::determineDatatype<int>(), {total_particles});
		for (auto const &attribute : int_attributes) {
			auto record = species[attribute.record_name];
			for (auto const &component : attribute.components) {
				if (component.label.empty()) {
					record[openPMD::RecordComponent::SCALAR].resetDataset(int_dataset);
				} else {
					record[component.label].resetDataset(int_dataset);
				}
			}
		}
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
void writeParticleSpecies(openPMD::Series &series, openPMD::Iteration &iteration, DescriptorType &descriptor, const std::string &species_name)
{
	using ContainerType = typename DescriptorType::ContainerT;
	ContainerType *container = descriptor.getContainer();
	if (container == nullptr) {
		return;
	}

	auto counts = computeParticleCountSummary(*container);
	const unsigned long long total_particles = counts.total_global;

	auto position_components = getPositionComponentLabels();
	const auto real_attributes = descriptor.getOpenPMDRealAttributes();
	const auto int_attributes = descriptor.getOpenPMDIntAttributes();

	auto species = iteration.particles[species_name];
	initializeSpecies(species, position_components, real_attributes, int_attributes, total_particles);

	if (total_particles == 0) {
		return;
	}

	const int finest = container->finestLevel();
	for (int lev = 0; lev <= finest; ++lev) {
		std::vector<std::shared_ptr<void>> chunk_lifetimes;
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
			auto const &soa = pti.GetStructOfArrays();
			const int soa_real_components = soa.NumRealComps();
			const int soa_int_components = soa.NumIntComps();
			constexpr int aos_real_components = ContainerType::ParticleType::NReal;
			constexpr int aos_int_components = ContainerType::ParticleType::NInt;
			const unsigned long long extent = static_cast<unsigned long long>(num_particles);

			for (std::size_t dim = 0; dim < position_components.size(); ++dim) {
				auto buffer = std::shared_ptr<amrex::ParticleReal>(
				    new amrex::ParticleReal[num_particles],
				    [](amrex::ParticleReal const *p) { delete[] p; });
				auto *dst = buffer.get();
				for (int i = 0; i < num_particles; ++i) {
					dst[i] = aos_ptr[i].pos(static_cast<int>(dim));
				}
				auto const &label = position_components[dim];
				species["position"][label].storeChunkRaw(
				    dst, {global_offset}, {extent});
				chunk_lifetimes.emplace_back(buffer, static_cast<void *>(dst));
			}

			auto ids = std::shared_ptr<uint64_t>(
			    new uint64_t[num_particles],
			    [](uint64_t const *p) { delete[] p; });
			auto *ids_dst = ids.get();
			for (int i = 0; i < num_particles; ++i) {
				ids_dst[i] = ::quokka::particle::localIdToGlobal(aos_ptr[i].id(), aos_ptr[i].cpu());
			}
			species["id"][openPMD::RecordComponent::SCALAR].storeChunkRaw(
			    ids_dst, {global_offset}, {extent});
			chunk_lifetimes.emplace_back(ids, static_cast<void *>(ids_dst));

			for (std::size_t attr_idx = 0; attr_idx < real_attributes.size(); ++attr_idx) {
				auto const &attribute = real_attributes[attr_idx];
				auto record = species[attribute.record_name];
				for (std::size_t comp_idx = 0; comp_idx < attribute.components.size(); ++comp_idx) {
					auto const &component = attribute.components[comp_idx];
					auto &target = component.label.empty() ? record[openPMD::RecordComponent::SCALAR]
					                                    : record[component.label];
					const int soa_index = component.soa_index;
					const int aos_index = component.aos_index;
					const bool has_soa = (soa_index >= 0 && soa_index < soa_real_components);
					const bool has_aos = (aos_index >= 0 && aos_index < aos_real_components);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(has_soa || has_aos,
									 "Particle descriptor reported a real attribute without accessible storage");
					if (has_soa) {
						auto const *src = soa.GetRealData(soa_index).data();
						target.storeChunkRaw(src, {global_offset}, {extent});
					} else {
						auto buffer = std::shared_ptr<amrex::ParticleReal>(
						    new amrex::ParticleReal[num_particles],
						    [](amrex::ParticleReal const *p) { delete[] p; });
						auto *dst = buffer.get();
						for (int i = 0; i < num_particles; ++i) {
							dst[i] = aos_ptr[i].rdata(aos_index);
						}
						target.storeChunkRaw(dst, {global_offset}, {extent});
						chunk_lifetimes.emplace_back(buffer, static_cast<void *>(dst));
					}
				}
			}

			for (std::size_t attr_idx = 0; attr_idx < int_attributes.size(); ++attr_idx) {
				auto const &attribute = int_attributes[attr_idx];
				auto record = species[attribute.record_name];
				for (std::size_t comp_idx = 0; comp_idx < attribute.components.size(); ++comp_idx) {
					auto const &component = attribute.components[comp_idx];
					auto &target = component.label.empty() ? record[openPMD::RecordComponent::SCALAR]
					                                    : record[component.label];
					const int soa_index = component.soa_index;
					const int aos_index = component.aos_index;
					const bool has_soa = (soa_index >= 0 && soa_index < soa_int_components);
					const bool has_aos = (aos_index >= 0 && aos_index < aos_int_components);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(has_soa || has_aos,
									 "Particle descriptor reported an integer attribute without accessible storage");
					if (has_soa) {
						auto const *src = soa.GetIntData(soa_index).data();
						target.storeChunkRaw(src, {global_offset}, {extent});
					} else {
						auto buffer = std::shared_ptr<int>(
						    new int[num_particles],
						    [](int const *p) { delete[] p; });
						auto *dst = buffer.get();
						for (int i = 0; i < num_particles; ++i) {
							dst[i] = aos_ptr[i].idata(aos_index);
						}
						target.storeChunkRaw(dst, {global_offset}, {extent});
						chunk_lifetimes.emplace_back(buffer, static_cast<void *>(dst));
					}
				}
			}
		}

		series.flush();
	}
}

} // namespace detail

template <typename problem_t>
void WriteParticles(openPMD::Series &series, openPMD::Iteration &iteration, PhysicsParticleRegister<problem_t> &particle_register, amrex::Real /*time*/)
{
	particle_register.forEachDescriptor([&series, &iteration](ParticleType type, PhysicsParticleDescriptorBase &descriptor_base) {
		const std::string species_name = PhysicsParticleRegister<problem_t>::getParticleTypeName(type);
		descriptor_base.writeOpenPMD(series, iteration, species_name);
	});
}

} // namespace quokka::OpenPMDOutput

#ifdef QUOKKA_USE_OPENPMD
namespace quokka
{
template <typename ContainerType, typename problem_t, ParticleType particleType>
auto PhysicsParticleDescriptor<ContainerType, problem_t, particleType>::getOpenPMDRealAttributes() const -> std::vector<OpenPMDRealAttribute>
{
	std::vector<OpenPMDRealAttribute> attributes;
	constexpr int aos_real_components = ContainerType::ParticleType::NReal;
	auto velocity_fallback_labels = OpenPMDOutput::detail::getVelocityComponentLabels();

	const int mass_index = this->getMassIndex();
	if (mass_index >= 0 && mass_index < aos_real_components) {
		OpenPMDRealAttribute mass_attr;
		auto mass_name = this->getAosRealComponentName(mass_index);
		if (mass_name.empty()) {
			mass_name = "mass";
		}
		mass_attr.record_name = mass_name;
		OpenPMDRealComponent component;
		component.label = "";
		component.soa_index = mass_index;
		component.aos_index = mass_index;
		mass_attr.components.push_back(component);
		attributes.push_back(std::move(mass_attr));

		const int velocity_start = mass_index + 1;
		if (velocity_start + AMREX_SPACEDIM - 1 < aos_real_components) {
			OpenPMDRealAttribute velocity_attr;
			auto velocity_record = this->getVelocityRecordName();
			if (velocity_record.empty()) {
				velocity_record = "velocity";
			}
			velocity_attr.record_name = velocity_record;
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				const int comp_index = velocity_start + d;
				OpenPMDRealComponent vel_component;
				vel_component.soa_index = comp_index;
				vel_component.aos_index = comp_index;
				vel_component.label = this->deriveComponentLabel(velocity_attr.record_name, comp_index, velocity_fallback_labels[d]);
				velocity_attr.components.push_back(std::move(vel_component));
			}
			attributes.push_back(std::move(velocity_attr));
		}
	}

	const int birth_index = this->getBirthTimeIndex();
	if (birth_index >= 0 && birth_index < aos_real_components) {
		OpenPMDRealAttribute birth_attr;
		auto birth_name = this->getAosRealComponentName(birth_index);
		if (birth_name.empty()) {
			birth_name = "birthTime";
		}
		birth_attr.record_name = birth_name;
		OpenPMDRealComponent birth_component;
		birth_component.label = "";
		birth_component.soa_index = birth_index;
		birth_component.aos_index = birth_index;
		birth_attr.components.push_back(birth_component);
		attributes.push_back(std::move(birth_attr));

		const int death_index = birth_index + 1;
		if (death_index >= 0 && death_index < aos_real_components) {
			OpenPMDRealAttribute death_attr;
			auto death_name = this->getAosRealComponentName(death_index);
			if (death_name.empty()) {
				death_name = "deathTime";
			}
			death_attr.record_name = death_name;
			OpenPMDRealComponent death_component;
			death_component.label = "";
			death_component.soa_index = death_index;
			death_component.aos_index = death_index;
			death_attr.components.push_back(death_component);
			attributes.push_back(std::move(death_attr));
		}
	}

	const int luminosity_index = this->getLumIndex();
	if (luminosity_index >= 0 && luminosity_index < aos_real_components) {
		const int luminosity_components = aos_real_components - luminosity_index;
		if (luminosity_components > 0) {
			OpenPMDRealAttribute luminosity_attr;
			auto luminosity_record = this->getLuminosityRecordName();
			if (luminosity_record.empty()) {
				luminosity_record = "luminosity";
			}
			luminosity_attr.record_name = luminosity_record;
			auto fallback_labels = OpenPMDOutput::detail::makeLuminosityComponentLabels(luminosity_components);
			for (int idx = 0; idx < luminosity_components; ++idx) {
				const int comp_index = luminosity_index + idx;
				OpenPMDRealComponent lum_component;
				lum_component.soa_index = comp_index;
				lum_component.aos_index = comp_index;
				lum_component.label = this->deriveComponentLabel(luminosity_attr.record_name, comp_index, fallback_labels[idx]);
				luminosity_attr.components.push_back(std::move(lum_component));
			}
			attributes.push_back(std::move(luminosity_attr));
		}
	}

	return attributes;
}

template <typename ContainerType, typename problem_t, ParticleType particleType>
auto PhysicsParticleDescriptor<ContainerType, problem_t, particleType>::getOpenPMDIntAttributes() const -> std::vector<OpenPMDIntAttribute>
{
	std::vector<OpenPMDIntAttribute> attributes;
	constexpr int aos_int_components = ContainerType::ParticleType::NInt;
	const int stage_index = this->getEvolutionStageIndex();
	if (stage_index >= 0) {
		OpenPMDIntAttribute stage_attr;
		auto record_name = this->getAosIntComponentName(stage_index);
		if (record_name.empty()) {
			record_name = "evolutionStage";
		}
		stage_attr.record_name = record_name;
		OpenPMDIntComponent component;
		component.label = "";
		component.soa_index = stage_index;
		component.aos_index = (stage_index < aos_int_components) ? stage_index : -1;
		stage_attr.components.push_back(component);
		attributes.push_back(std::move(stage_attr));
	}
	return attributes;
}

template <typename ContainerType, typename problem_t, ParticleType particleType>
void PhysicsParticleDescriptor<ContainerType, problem_t, particleType>::writeOpenPMD(openPMD::Series &series, openPMD::Iteration &iteration,
										     const std::string &species_name)
{
	OpenPMDOutput::detail::writeParticleSpecies(series, iteration, *this, species_name);
}
} // namespace quokka
#endif

#endif // OPENPMD_PARTICLES_IMPL_HPP_
