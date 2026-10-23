#ifndef PARTICLE_IO_HPP_
#define PARTICLE_IO_HPP_

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "particle_types.hpp"
#include "particles/particle_container.hpp"
#include <format>
#include <limits>

namespace quokka
{

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

namespace particle_io
{

// Gather flat host buffers without constructing or redistributing an analysis
// particle container. Every rank participates, including ranks with no particles.
template <typename T> auto gatherBuffer(const std::vector<T> &local) -> std::vector<T>
{
	AMREX_ALWAYS_ASSERT(local.size() <= static_cast<size_t>(std::numeric_limits<int>::max()));
	const int root = amrex::ParallelDescriptor::IOProcessorNumber();
	const auto counts = amrex::ParallelDescriptor::Gather(static_cast<int>(local.size()), root);
	std::vector<int> offsets(counts.size());
	size_t total = 0;
	for (size_t rank = 0; rank < counts.size(); ++rank) {
		AMREX_ALWAYS_ASSERT(total + counts[rank] <= static_cast<size_t>(std::numeric_limits<int>::max()));
		offsets[rank] = static_cast<int>(total);
		total += counts[rank];
	}
	std::vector<T> gathered(total);
	amrex::ParallelDescriptor::Gatherv(local.data(), static_cast<int>(local.size()), gathered.data(), counts, offsets, root);
	return gathered;
}

// Rows preserve the existing diagnostic convention: positions followed by
// logical real fields, with integer fields in a separate row. Invalid particles
// are omitted, matching the previous analysis container's redistribution.
template <typename ContainerType>
[[nodiscard]] auto gatherParticleData(ContainerType *container, int firstLevel, int lastLevel)
    -> std::tuple<std::vector<int64_t>, std::vector<std::vector<double>>, std::vector<std::vector<int>>>
{
	if (container == nullptr) {
		return {};
	}
	const auto layout = container->Layout();
	const int nr = AMREX_SPACEDIM + layout.numReal;
	const int ni = layout.numInt;
	std::vector<int64_t> ids;
	std::vector<double> reals;
	std::vector<int> integers;
	for (int lev = firstLevel; lev <= lastLevel; ++lev) {
		for (auto &[key, tile] : container->GetParticles(lev)) {
			amrex::ignore_unused(key);
			const auto np = tile.numParticles();
			const auto data = tile.getParticleTileData();
			amrex::Gpu::DeviceVector<int64_t> deviceIDs(np);
			amrex::Gpu::DeviceVector<double> deviceReals(static_cast<size_t>(np) * nr);
			amrex::Gpu::DeviceVector<int> deviceInts(static_cast<size_t>(np) * ni);
			auto *idOut = deviceIDs.data();
			auto *realOut = deviceReals.data();
			auto *intOut = deviceInts.data();
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(amrex::Long i) noexcept {
				const auto p = ParticleView(data, i, layout);
				idOut[i] = p.id();
				for (int d = 0; d < AMREX_SPACEDIM; ++d) {
					realOut[i * nr + d] = p.pos(d);
				}
				for (int c = 0; c < layout.numReal; ++c) {
					realOut[i * nr + AMREX_SPACEDIM + c] = p.rdata(c);
				}
				for (int c = 0; c < ni; ++c) {
					intOut[i * ni + c] = p.idata(c);
				}
			});
			std::vector<int64_t> hostIDs(deviceIDs.size());
			std::vector<double> hostReals(deviceReals.size());
			std::vector<int> hostInts(deviceInts.size());
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, deviceIDs.begin(), deviceIDs.end(), hostIDs.begin());
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, deviceReals.begin(), deviceReals.end(), hostReals.begin());
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, deviceInts.begin(), deviceInts.end(), hostInts.begin());
			for (amrex::Long i = 0; i < np; ++i) {
				if (hostIDs[i] <= 0) {
					continue;
				}
				ids.push_back(hostIDs[i]);
				reals.insert(reals.end(), hostReals.begin() + i * nr, hostReals.begin() + (i + 1) * nr);
				integers.insert(integers.end(), hostInts.begin() + i * ni, hostInts.begin() + (i + 1) * ni);
			}
		}
	}
	ids = gatherBuffer(ids);
	reals = gatherBuffer(reals);
	integers = gatherBuffer(integers);
	std::vector<std::vector<double>> realRows(ids.size());
	std::vector<std::vector<int>> intRows(ni > 0 ? ids.size() : 0);
	for (size_t i = 0; i < ids.size(); ++i) {
		realRows[i].assign(reals.begin() + i * nr, reals.begin() + (i + 1) * nr);
		if (ni > 0) {
			intRows[i].assign(integers.begin() + i * ni, integers.begin() + (i + 1) * ni);
		}
	}
	return {std::move(ids), std::move(realRows), std::move(intRows)};
}

template <typename ContainerType>
[[nodiscard]] auto getParticleDataAtAllLevels(ContainerType *container)
    -> std::tuple<std::vector<int64_t>, std::vector<std::vector<double>>, std::vector<std::vector<int>>>
{
	return gatherParticleData(container, 0, container != nullptr ? container->finestLevel() : -1);
}

template <typename ContainerType>
[[nodiscard]] auto getParticleDataAtLevel(ContainerType *container, int lev) -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>>
{
	auto [ids, reals, integers] = gatherParticleData(container, lev, lev);
	return {std::move(reals), std::move(integers)};
}

// Write units info of particles to checkpoint/plotfile
inline void writeUnitsFile(ParticleContainer *container, const std::string &snapshot_name, const std::string &name)
{
	if (container != nullptr) {
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

			const auto &definitions = container->Schema().UnitDefinitions();
			if (!definitions.empty()) {
				outFile << "# field: [M, L, T, Θ]\n";
				for (const auto &[fieldName, units] : definitions) {
					outFile << fieldName << ": [" << units[0] << ", " << units[1] << ", " << units[2] << ", " << units[3] << "]\n";
				}
			}

			outFile.close();
		}
	}
}

// Print statistics of particles
template <typename ContainerType, typename problem_t, ParticleType particleType>
void printParticleStatistics(ContainerType *container, int massIndex, int evolutionStageIndex)
{
	if (container != nullptr) {
		// Get particle type name
		const std::string particle_type_name = PhysicsParticleRegister<problem_t>::getParticleTypeName(particleType);
		amrex::Print() << std::format("number of {} = {}\n", particle_type_name, static_cast<int>(container->TotalNumberOfParticles(true, false)));

		const int max_number_to_print = 100;

		for (int lev = 0; lev <= container->finestLevel(); ++lev) {
			// Get particle data at this level
			const auto [real_data, int_data] = getParticleDataAtLevel(container, lev);

			if (!real_data.empty()) {
				amrex::Print() << "Level " << lev << "\n";
				// Print header for detailed particle data
				if (evolutionStageIndex >= 0) {
					amrex::Print() << std::format("\t{:>20} | {:>20}\n", "mass", "evolution stage");
				} else {
					amrex::Print() << std::format("\t{:>20}\n", "mass");
				}

				// Print each particle's data with aligned columns
				const int n_print = std::min(static_cast<int>(real_data.size()), max_number_to_print);
				int i = 0;
				for (; i < n_print; ++i) {
					if (evolutionStageIndex >= 0) {
						amrex::Print() << std::format("\t{:20.13e} | {:>20}\n", real_data[i][AMREX_SPACEDIM + massIndex],
									      int_data[i][evolutionStageIndex]);
					} else {
						amrex::Print() << std::format("\t{:20.13e}\n", real_data[i][AMREX_SPACEDIM + massIndex]);
					}
				}
				if (i == max_number_to_print) {
					amrex::Print() << std::format("\t...\n");
				}
			}
		}
	}
}

// Save particle data to a text file
// The text file will contain the following format:
// - First line: Number of particles
// - Remaining lines: Particle data (positions, real components, integer components)
//
// Note: Only rank 0 will write the file, but all ranks must participate in the data gathering.
// @param container: Particle container
// @param filename: Name of the text file to write
// @return: true if file was written successfully, false otherwise
template <typename ContainerType> auto saveParticleDataToTxtFile(ContainerType *container, const std::string &filename, const std::string &name) -> bool
{
	// Get all particle data
	const auto [particle_ids, real_data, int_data] = getParticleDataAtAllLevels(container);

	// Only rank 0 writes the file
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string const full_filename = filename + "/" + name + ".txt";
		std::ofstream outFile(full_filename);
		if (!outFile) {
			return false;
		}

		// Write number of particles
		outFile << real_data.size() << "\n";

		// Write data
		for (size_t i = 0; i < real_data.size(); ++i) {
			// Write position and real components
			for (size_t j = 0; j < real_data[i].size(); ++j) {
				outFile << std::scientific << std::setprecision(15) << real_data[i][j] << " ";
			}

			// Write integer components
			if (!int_data.empty()) {
				for (size_t j = 1; j < int_data[i].size(); ++j) {
					outFile << int_data[i][j] << " ";
				}
			}
			outFile << "\n";
		}

		outFile.close();
		return true;
	}

	return true; // Non-root ranks always succeed
}

} // namespace particle_io

} // namespace quokka

#endif // PARTICLE_IO_HPP_
