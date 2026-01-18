#ifndef PARTICLE_IO_HPP_
#define PARTICLE_IO_HPP_

#include <array>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "AMReX_Gpu.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParticleTransformation.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "particle_types.hpp"
#include <fmt/format.h>

namespace quokka
{

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

namespace particle_io
{

template <typename ContainerType>
void configureAnalysisContainer(ContainerType &analysisPC, const ContainerType &container)
{
	analysisPC.SetArena(container.arena());

	const std::vector<std::string> real_names = container.GetRealSoANames();
	const std::vector<std::string> int_names = container.GetIntSoANames();

	std::vector<std::string> real_ct_names(ContainerType::NArrayReal);
	for (int ic = 0; ic < ContainerType::NArrayReal; ++ic) {
		real_ct_names.at(ic) = real_names.at(ic);
	}
	std::vector<std::string> int_ct_names(ContainerType::NArrayInt);
	for (int ic = 0; ic < ContainerType::NArrayInt; ++ic) {
		int_ct_names.at(ic) = int_names.at(ic);
	}

	analysisPC.SetSoACompileTimeNames(real_ct_names, int_ct_names);

	for (int ic = 0; ic < container.NumRuntimeRealComps(); ++ic) {
		analysisPC.AddRealComp(real_names.at(ic + ContainerType::NArrayReal));
	}
	for (int ic = 0; ic < container.NumRuntimeIntComps(); ++ic) {
		analysisPC.AddIntComp(int_names.at(ic + ContainerType::NArrayInt));
	}
}

template <typename ContainerType>
void initParticlesFromAscii(ContainerType *container, const std::string &file, int nreal_extra)
{
	if (container == nullptr) {
		return;
	}

	if (nreal_extra > container->NumRuntimeRealComps()) {
		amrex::Abort("initParticlesFromAscii: nreal_extra exceeds runtime real component count");
	}

	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ifstream input(file);
		if (!input) {
			amrex::FileOpenFailed(file);
		}

		int np = 0;
		input >> np;
		if (np <= 0) {
			amrex::Abort("Particle init file has no particles: " + file);
		}

		auto &particle_tile = container->DefineAndReturnParticleTile(0, 0, 0);
		const amrex::Long old_size = particle_tile.numParticles();
		particle_tile.resize(old_size + np);

		auto ptd = particle_tile.getParticleTileData();
		const int cpu_id = amrex::ParallelDescriptor::MyProc();

		const amrex::Long pid = ContainerType::ParticleType::NextID();
		ContainerType::ParticleType::NextID(pid + np);

		std::array<amrex::Vector<amrex::ParticleReal>, AMREX_SPACEDIM> pos_host{};
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			pos_host[d].resize(np);
		}
		std::vector<amrex::Vector<amrex::ParticleReal>> runtime_host(nreal_extra);
		for (int r = 0; r < nreal_extra; ++r) {
			runtime_host[r].resize(np);
		}
		amrex::Vector<uint64_t> idcpu_host(np);

		for (int i = 0; i < np; ++i) {
			amrex::Real pos[AMREX_SPACEDIM] = {0.0};
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				input >> pos[d];
			}

			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				pos_host[d][i] = pos[d];
			}

			for (int r = 0; r < nreal_extra; ++r) {
				amrex::Real value = 0.0;
				input >> value;
				runtime_host[r][i] = value;
			}

			idcpu_host[i] = amrex::SetParticleIDandCPU(pid + i, cpu_id);
		}

		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			amrex::Gpu::copy(amrex::Gpu::hostToDevice, pos_host[d].begin(), pos_host[d].end(), ptd.m_rdata[d] + old_size);
		}
		for (int r = 0; r < nreal_extra; ++r) {
			amrex::Gpu::copy(amrex::Gpu::hostToDevice, runtime_host[r].begin(), runtime_host[r].end(), ptd.m_runtime_rdata[r] + old_size);
		}
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, idcpu_host.begin(), idcpu_host.end(), ptd.m_idcpu + old_size);
		amrex::Gpu::streamSynchronize();
	}

	container->Redistribute();
}

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
    -> std::tuple<std::vector<int64_t>, std::vector<std::vector<double>>, std::vector<std::vector<int>>>
{
	std::vector<int64_t> particle_ids;
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
		configureAnalysisContainer(analysisPC, *container);

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
					amrex::copyParticles(dst_tile, src_tile, 0, particle_offset, np);
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
				const auto ptd = pIter.GetParticleTile().getConstParticleTileData();
				const int num_real_ct = ContainerType::NArrayReal;
				const int num_real_rt = ptd.m_num_runtime_real;
				const int num_real = num_real_ct + num_real_rt;
				const int num_int_ct = ContainerType::NArrayInt;
				const int num_int_rt = ptd.m_num_runtime_int;
				const int num_int = num_int_ct + num_int_rt;

				// Pre-size vectors to avoid reallocations
				particle_ids.reserve(np);
				real_data.reserve(np);
				if (num_int > 0) {
					int_data.reserve(np);
				}

				amrex::Vector<uint64_t> idcpu_h(np);
				amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_idcpu, ptd.m_idcpu + np, idcpu_h.begin()); // NOLINT

				std::vector<amrex::Vector<amrex::ParticleReal>> real_host(num_real);
				for (int comp = 0; comp < num_real_ct; ++comp) {
					real_host[comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_rdata[comp], ptd.m_rdata[comp] + np, real_host[comp].begin()); // NOLINT
				}
				for (int comp = 0; comp < num_real_rt; ++comp) {
					real_host[num_real_ct + comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_runtime_rdata[comp], ptd.m_runtime_rdata[comp] + np,
							 real_host[num_real_ct + comp].begin()); // NOLINT
				}

				std::vector<amrex::Vector<int>> int_host(num_int);
				for (int comp = 0; comp < num_int_ct; ++comp) {
					int_host[comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_idata[comp], ptd.m_idata[comp] + np, int_host[comp].begin());
				}
				for (int comp = 0; comp < num_int_rt; ++comp) {
					int_host[num_int_ct + comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_runtime_idata[comp], ptd.m_runtime_idata[comp] + np,
							 int_host[num_int_ct + comp].begin());
				}

				// Process each particle
				for (int i = 0; i < np; ++i) {
					// Get particle ID
					particle_ids.push_back(static_cast<int64_t>(amrex::ConstParticleIDWrapper(idcpu_h[i])));

					// Process real data (positions and rdata)
					std::vector<double> r_data;
					// Pre-allocate to avoid reallocations
					r_data.reserve(num_real);
					for (int d = 0; d < num_real; ++d) {
						r_data.push_back(static_cast<double>(real_host[d][i]));
					}

					real_data.push_back(std::move(r_data));

					// Process integer data if particles have integer components
					if (num_int > 0) {
						std::vector<int> i_data;
						// Pre-allocate to avoid reallocations
						i_data.reserve(num_int);

						for (int d = 0; d < num_int; ++d) {
							i_data.push_back(int_host[d][i]);
						}

						int_data.push_back(std::move(i_data));
					}
				}
			}
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
		configureAnalysisContainer(analysisPC, *container);

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
				amrex::copyParticles(dst_tile, src_tile, 0, particle_offset, np);
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
				const auto ptd = pIter.GetParticleTile().getConstParticleTileData();
				const int num_real_ct = ContainerType::NArrayReal;
				const int num_real_rt = ptd.m_num_runtime_real;
				const int num_real = num_real_ct + num_real_rt;
				const int num_int_ct = ContainerType::NArrayInt;
				const int num_int_rt = ptd.m_num_runtime_int;
				const int num_int = num_int_ct + num_int_rt;

				// Pre-size vectors to avoid reallocations
				real_data.reserve(np);
				if (num_int > 0) {
					int_data.reserve(np);
				}

				std::vector<amrex::Vector<amrex::ParticleReal>> real_host(num_real);
				for (int comp = 0; comp < num_real_ct; ++comp) {
					real_host[comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_rdata[comp], ptd.m_rdata[comp] + np, real_host[comp].begin()); // NOLINT
				}
				for (int comp = 0; comp < num_real_rt; ++comp) {
					real_host[num_real_ct + comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_runtime_rdata[comp], ptd.m_runtime_rdata[comp] + np,
							 real_host[num_real_ct + comp].begin()); // NOLINT
				}

				std::vector<amrex::Vector<int>> int_host(num_int);
				for (int comp = 0; comp < num_int_ct; ++comp) {
					int_host[comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_idata[comp], ptd.m_idata[comp] + np, int_host[comp].begin());
				}
				for (int comp = 0; comp < num_int_rt; ++comp) {
					int_host[num_int_ct + comp].resize(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptd.m_runtime_idata[comp], ptd.m_runtime_idata[comp] + np,
							 int_host[num_int_ct + comp].begin());
				}

				// Process each particle
				for (int i = 0; i < np; ++i) {
					// Process real data (positions and rdata)
					std::vector<double> r_data;
					// Pre-allocate to avoid reallocations
					r_data.reserve(num_real);
					for (int d = 0; d < num_real; ++d) {
						r_data.push_back(static_cast<double>(real_host[d][i]));
					}

					real_data.push_back(std::move(r_data));

					// Process integer data if particles have integer components
					if (num_int > 0) {
						std::vector<int> i_data;
						// Pre-allocate to avoid reallocations
						i_data.reserve(num_int);

						for (int d = 0; d < num_int; ++d) {
							i_data.push_back(int_host[d][i]);
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
			if (!unitsData.contains(particleType)) {
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

			// Write integer components (skip the first component for backward compatibility)
			if (!int_data.empty() && int_data[i].size() > 1) {
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
