#ifndef PARTICLE_IO_HPP_
#define PARTICLE_IO_HPP_

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "global_particle_id.hpp"
#include "particle_types.hpp"
#include <fmt/format.h>

namespace quokka
{

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

namespace particle_io
{

// Get positions and fields data from all particles across all levels and gather them on rank 0.
// This method creates a temporary particle container on all ranks (though only rank 0 will contain
// particles after redistribution) and copies all particles from all levels into it.
//
// The returned data for each particle contains:
// - first: vector of particle IDs
// - second:
//   - First AMREX_SPACEDIM elements are positions [x,y,z]
//   - Remaining elements are particle data (e.g., mass, velocities, etc.)
// - third:
//   - Integer data (e.g., type, etc.)
//
// Only rank 0 will return the actual particle data, other ranks return empty vectors.
// @return: tuple of vectors containing particle data on rank 0, empty vectors on other ranks
template <typename ContainerType>
[[nodiscard]] auto getParticleDataAtAllLevels(ContainerType *container)
    -> std::tuple<std::vector<std::uint64_t>, std::vector<std::vector<double>>, std::vector<std::vector<int>>>
{
	std::vector<std::uint64_t> particle_ids;
	std::vector<std::vector<double>> real_data;
	std::vector<std::vector<int>> int_data;

	if (container != nullptr) {
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

		// Count total particles across all levels
		int total_np = 0;
		for (int lev = 0; lev <= container->finestLevel(); ++lev) {
			const auto &particles = container->GetParticles(lev);
			for (const auto &kv : particles) {
				total_np += kv.second.numParticles();
			}
		}

		// Pre-size the destination tile
		dst_tile.resize(total_np);

		// Copy particles from all levels to the destination tile
		int particle_offset = 0;
		for (int lev = 0; lev <= container->finestLevel(); ++lev) {
			const auto &particles = container->GetParticles(lev);

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
				auto const &idcpu_device = pIter.GetStructOfArrays().GetIdCPUData();

				// Transfer particle data from GPU to CPU for analysis
				typename ContainerType::ParticleType *pData = particles().data();
				amrex::Vector<typename ContainerType::ParticleType> pData_h(np);
				amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, pData + np, pData_h.begin()); // NOLINT

				// Check if particles have integer components
				constexpr bool has_int_components = (ContainerType::ParticleType::NInt > 0);

				// Pre-size vectors to avoid reallocations
				particle_ids.reserve(np);
				particle_ids.reserve(np);
				real_data.reserve(np);
				if constexpr (has_int_components) {
					int_data.reserve(np);
				}

				// Process each particle
				for (int i = 0; i < np; ++i) {
					const auto &p = pData_h[i];
					particle_ids.push_back(particle::localIdToGlobal(p.id(), p.cpu()));

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
		} else {
			particle_ids.clear();
		}
	}

	return {particle_ids, real_data, int_data}; // Empty vectors on non-root ranks
}

// Get particle data at a specific level
template <typename ContainerType>
[[nodiscard]] auto getParticleDataAtLevel(ContainerType *container, int lev) -> std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>>
{
	std::vector<std::vector<double>> real_data;
	std::vector<std::vector<int>> int_data;

	if (container != nullptr) {
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
		const auto &particles = container->GetParticles(lev);

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

// Write units info of particles to checkpoint/plotfile
template <typename ContainerType, typename problem_t, ParticleType particleType>
void writeUnitsFile(ContainerType *container, const std::string &snapshot_name, const std::string &name)
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

			// Get the units data for this particle type
			const auto &unitsData = get_units_data();
			if (unitsData.find(particleType) == unitsData.end()) {
				amrex::Abort(
				    "Error: Particle type not defined in units data map. Please add units for this particle type in get_units_data().");
			}

			const auto &typeData = unitsData.at(particleType);
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

// Print statistics of particles
template <typename ContainerType, typename problem_t, ParticleType particleType>
void printParticleStatistics(ContainerType *container, int massIndex, int evolutionStageIndex)
{
	if (container != nullptr) {
		// Get particle type name
		const std::string particle_type_name = PhysicsParticleRegister<problem_t>::getParticleTypeName(particleType);
		amrex::Print() << fmt::format("number of {} = {}\n", particle_type_name, static_cast<int>(container->TotalNumberOfParticles(true, false)));

		const int max_number_to_print = 100;

		for (int lev = 0; lev <= container->finestLevel(); ++lev) {
			// Get particle data at this level
			const auto [real_data, int_data] = getParticleDataAtLevel(container, lev);

			if (!real_data.empty()) {
				amrex::Print() << "Level " << lev << "\n";
				// Print header for detailed particle data
				if (evolutionStageIndex >= 0) {
					amrex::Print() << fmt::format("\t{:>20} | {:>20}\n", "mass", "evolution stage");
				} else {
					amrex::Print() << fmt::format("\t{:>20}\n", "mass");
				}

				// Print each particle's data with aligned columns
				const int n_print = std::min(static_cast<int>(real_data.size()), max_number_to_print);
				int i = 0;
				for (; i < n_print; ++i) {
					if (evolutionStageIndex >= 0) {
						amrex::Print() << fmt::format("\t{:20.13e} | {:>20}\n", real_data[i][AMREX_SPACEDIM + massIndex],
									      int_data[i][evolutionStageIndex]);
					} else {
						amrex::Print() << fmt::format("\t{:20.13e}\n", real_data[i][AMREX_SPACEDIM + massIndex]);
					}
				}
				if (i == max_number_to_print) {
					amrex::Print() << fmt::format("\t...\n");
				}
			}
		}
	}
}

// Save particle data to a CSV file
// This function gathers all particle data from all ranks and saves it to a CSV file on rank 0.
// The CSV file will contain the following columns:
// - ID: Particle ID
// - x, y, z: Particle positions
// - real_0, real_1, ...: Real components (e.g., mass, velocities, etc.)
// - int_0, int_1, ...: Integer components (if any)
//
// Note: Only rank 0 will write the file, but all ranks must participate in the data gathering.
// @param container: Particle container
// @param filename: Name of the CSV file to write
// @return: true if file was written successfully, false otherwise
template <typename ContainerType> auto saveParticleDataToFile(ContainerType *container, const std::string &filename, const std::string &name) -> bool
{
	// Get all particle data
	const auto [particle_ids, real_data, int_data] = getParticleDataAtAllLevels(container);

	// Only rank 0 writes the file
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream outFile(filename + "/" + name + ".csv");
		if (!outFile) {
			return false;
		}

		// Write header
		outFile << "ID";
		// Position columns
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			outFile << ",pos_" << d;
		}
		// Real component columns
		for (int d = 0; d < ContainerType::ParticleType::NReal; ++d) {
			outFile << ",real_" << d;
		}
		// Integer component columns
		if constexpr (ContainerType::ParticleType::NInt > 0) {
			for (int d = 0; d < ContainerType::ParticleType::NInt; ++d) {
				outFile << ",int_" << d;
			}
		}
		outFile << "\n";

		// Write data
		for (size_t i = 0; i < real_data.size(); ++i) {
			// Write particle ID
			outFile << particle_ids[i];

			// Write position and real components
			for (const auto &val : real_data[i]) {
				outFile << "," << std::scientific << std::setprecision(15) << val;
			}

			// Write remaining integer components (skip ID which was written first)
			if constexpr (ContainerType::ParticleType::NInt > 1) {
				for (size_t j = 1; j < int_data[i].size(); ++j) {
					outFile << "," << int_data[i][j];
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
