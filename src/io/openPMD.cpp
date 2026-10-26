//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file openPMD.cpp
///  \brief openPMD I/O for snapshots

#include "io/openPMD.hpp"
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>

// AMReX headers
#include "AMReX.H"
#include "AMReX_Exception.H"
#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Reduce.H"
#include "AMReX_Scan.H"

// openPMD headers
#include "openPMD/IterationEncoding.hpp"
#include "openPMD/openPMD.hpp"

namespace quokka::OpenPMDOutput
{

namespace detail
{
/** \brief
 * Convert an IntVect to a std::vector<std::uint64_t>
 * and reverse the order of the elements
 * (used for compatibility with the openPMD API)
 */
auto getReversedVec(const amrex::IntVect &v) -> std::vector<std::uint64_t>
{
	// Convert the IntVect v to and std::vector u
	std::vector<std::uint64_t> u = {AMREX_D_DECL(static_cast<std::uint64_t>(v[0]), static_cast<std::uint64_t>(v[1]), static_cast<std::uint64_t>(v[2]))};
	// Reverse the order of elements, if v corresponds to the indices of a Fortran-order array (like an AMReX FArrayBox)
	// but u is intended to be used with a C-order API (like openPMD)
	std::reverse(u.begin(), u.end());
	return u;
}

/** \brief
 * Convert Real* pointer to a std::vector<double>,
 * and reverse the order of the elements
 * (used for compatibility with the openPMD API)
 */
auto getReversedVec(const amrex::Real *v) -> std::vector<double>
{
	// Convert Real* v to and std::vector u
	std::vector<double> u = {AMREX_D_DECL(static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2]))}; // NOLINT
	// Reverse the order of elements, if v corresponds to the indices of a Fortran-order array (like an AMReX FArrayBox)
	// but u is intended to be used with a C-order API (like openPMD)
	std::reverse(u.begin(), u.end());
	return u;
}

void SetupMeshComponent(openPMD::Mesh &mesh, const amrex::Geometry &full_geom, amrex::IndexType ixType)
{
	amrex::Box const global_box = amrex::convert(full_geom.Domain(), ixType);
	auto global_size = getReversedVec(global_box.size());
	std::vector<double> const grid_spacing = getReversedVec(full_geom.CellSize());
	std::vector<double> const global_offset = getReversedVec(full_geom.ProbLo());
	std::vector<std::string> const coordinateLabels{AMREX_D_DECL("x", "y", "z")};
	std::vector<std::string> const grid_axes{coordinateLabels.rbegin(), coordinateLabels.rend()}; // reverse order

	// Prepare the type of dataset that will be written
	mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
	mesh.setGridSpacing(grid_spacing);
	mesh.setGridGlobalOffset(global_offset);
	mesh.setAxisLabels(grid_axes); // use C-ordering
	mesh.setAttribute("fieldSmoothing", "none");

	auto mesh_comp = mesh[openPMD::MeshRecordComponent::SCALAR];
	auto const dataset = openPMD::Dataset(openPMD::determineDatatype<amrex::Real>(), global_size);
	mesh_comp.resetDataset(dataset);
	// staggering in units of the cell size: 0.5 for cell-centered axes, 0 for nodal ones
	const auto stagger = [&](int d) { return ixType.cellCentered(d) ? amrex::Real(0.5) : amrex::Real(0.0); };
	std::vector<amrex::Real> relativePosition{AMREX_D_DECL(stagger(0), stagger(1), stagger(2))};
	std::reverse(relativePosition.begin(), relativePosition.end());
	mesh_comp.setPosition(relativePosition);
}

// Number of nodes (shared-memory domains) in the job.
auto NodeCount() -> int
{
#ifdef AMREX_USE_MPI
	MPI_Comm node = MPI_COMM_NULL;
	MPI_Comm_split_type(amrex::ParallelDescriptor::Communicator(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
	int nodeRank = 0;
	MPI_Comm_rank(node, &nodeRank);
	MPI_Comm_free(&node);
	int nodes = nodeRank == 0 ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &nodes, 1, MPI_INT, MPI_SUM, amrex::ParallelDescriptor::Communicator());
	return nodes;
#else
	return 1;
#endif
}

// openPMD options for every output series. openpmd.aggregation (auto, on or off)
// selects ADIOS2 BP5 aggregation: on keeps BP5's default (TwoLevelShm, one
// aggregator per node), off lets every rank write its own subfile. EveryoneWrites
// alone still shares one subfile per node, so off also sets one aggregator per
// rank; a shared subfile sits on a single OST on Lustre. Without
// aggregation, writes are faster on a single node (about 2.4x on one CUDA node),
// but the number of files grows with the rank count. auto aggregates from
// openpmd.aggregation_min_nodes nodes upward; the default of 32 nodes (256
// ranks at 8 per node) caps subfiles at AMReX's default of 256 files per
// MultiFab for native output. Evaluated once, collectively.
auto SeriesOptions() -> const std::string &
{
	static const std::string options = [] {
		const amrex::ParmParse pp("openpmd");
		std::string mode = "auto";
		pp.query("aggregation", mode);
		int minNodes = 32;
		pp.query("aggregation_min_nodes", minNodes);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mode == "auto" || mode == "on" || mode == "off", "openpmd.aggregation must be auto, on or off");
		const int nodes = NodeCount();
		const bool aggregate = mode == "on" || (mode == "auto" && nodes >= minNodes);
		amrex::Print() << "openPMD: ADIOS2 aggregation " << (aggregate ? "on" : "off") << " (openpmd.aggregation = " << mode << ", " << nodes
			       << " node(s), openpmd.aggregation_min_nodes = " << minNodes << ")\n";
		if (aggregate) {
			return std::string("{}");
		}
		return R"({"adios2": {"engine": {"parameters": {"AggregationType": "EveryoneWrites", "NumAggregators": ")" +
		       std::to_string(amrex::ParallelDescriptor::NProcs()) + R"("}}}})";
	}();
	return options;
}

auto CreateSeries(const std::string &filename) -> openPMD::Series
{
#ifdef AMREX_USE_MPI
	auto series = openPMD::Series(filename, openPMD::Access::CREATE, amrex::ParallelDescriptor::Communicator(), SeriesOptions());
#else
	auto series = openPMD::Series(filename, openPMD::Access::CREATE, SeriesOptions());
#endif
	series.setSoftware("Quokka", "1.0");
	series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
	return series;
}

auto GetMeshComponentName(int meshLevel, std::string const &field_name) -> std::string
{
	std::string new_field_name = field_name;
	// convert dashes to underscores
	std::replace(new_field_name.begin(), new_field_name.end(), '-', '_');
	if (meshLevel > 0) {
		new_field_name += std::string("_lvl").append(std::to_string(meshLevel));
	}
	return new_field_name;
}
} // namespace detail
//----------------------------------------------------------------------------------------
/// \brief Write cell-centered MultiFab using openPMD.
void WriteFile(const std::vector<std::string> &varnames, int const output_levels, const amrex::Vector<const amrex::MultiFab *> &mf,
	       const amrex::Vector<amrex::Geometry> &geom, const std::string &output_basename, amrex::Real const time, int const file_number,
	       const std::function<void(openPMD::Series &, openPMD::Iteration &)> &writeParticles)
{
	// open file
	auto series = detail::CreateSeries(output_basename + "%05T.bp");

	auto series_iteration = series.iterations[file_number];
	series_iteration.open();
	series_iteration.setTime(time);
	auto meshes = series_iteration.meshes;

	// loop over levels up to output_levels
	for (int lev = 0; lev < output_levels; lev++) {
		amrex::Geometry full_geom = geom[lev];
		amrex::Box const &global_box = full_geom.Domain();
		int const ncomp = mf[lev]->nComp();

		for (int icomp = 0; icomp < ncomp; icomp++) {
			const std::string field_name = detail::GetMeshComponentName(lev, varnames[icomp]);
			if (!meshes.contains(field_name)) {
				auto mesh = meshes[field_name];
				detail::SetupMeshComponent(mesh, full_geom);
			}
		}

		// pass data pointers for each box to ADIOS
		for (int icomp = 0; icomp < ncomp; icomp++) {
			std::string const field_name = detail::GetMeshComponentName(lev, varnames[icomp]);
			openPMD::MeshRecordComponent mesh = series_iteration.meshes[field_name][openPMD::MeshRecordComponent::SCALAR];

			// Loop through the multifab, and store each box as a chunk in the openPMD file.
			for (amrex::MFIter mfi(*mf[lev]); mfi.isValid(); ++mfi) {
				amrex::FArrayBox const &fab = (*mf[lev])[mfi];
				amrex::Box const &local_box = fab.box();

				// Determine the offset and size of this chunk
				amrex::IntVect const box_offset = local_box.smallEnd() - global_box.smallEnd();
				auto chunk_offset = detail::getReversedVec(box_offset); // this overflows if ghost zones are used (!)
				auto chunk_size = detail::getReversedVec(local_box.size());

				detail::StoreDeviceChunk(mesh, fab.dataPtr(icomp), local_box.numPts(), chunk_offset, chunk_size);
			}
		}

		// Flush this level to disk once its staged copies are complete. A rank without
		// patches on this level has nothing new to flush, and openPMD would skip its
		// part of BP5's collective write.
		amrex::Gpu::streamSynchronize();
		series_iteration.touch();
		series.flush();
	}

	if (writeParticles) {
		writeParticles(series, series_iteration);
	}

	// close file
	series.close();
}

namespace
{
template <typename T> auto HostCopy(const amrex::ArrayView<const T> &source, amrex::Long count)
{
	std::vector<T> host(count);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, source.begin(), source.begin() + count, host.begin());
	return host;
}

struct ParticleTileOutput {
	amrex::Gpu::DeviceVector<amrex::ParticleReal> deviceReals;
	amrex::Gpu::DeviceVector<int> deviceInts;
	amrex::Gpu::DeviceVector<std::uint64_t> deviceIDs;
	std::vector<amrex::ParticleReal> hostReals;
	std::vector<int> hostInts;
	std::vector<std::uint64_t> hostIDs;
	std::vector<const amrex::ParticleReal *> reals;
	std::vector<const int *> integers;
	const std::uint64_t *ids = nullptr;
	const int *levels = nullptr;
	// Whether the pointers above refer to host memory (the host fallback) rather
	// than device memory.
	bool onHost = false;

	void Prepare(const ParticleContainer::ParticleTileType &tile, int lev, std::uint64_t count, ParticleLayout layout, amrex::Long bufferLimit)
	{
		const auto &soa = tile.GetStructOfArrays();
		const auto n = tile.numParticles();
		const int nr = AMREX_SPACEDIM + layout.numReal;
		reals.resize(nr);
		integers.resize(layout.numInt);
		const bool compact = count != static_cast<std::uint64_t>(n);
		const auto bytes =
		    count * sizeof(int) + (compact ? count * (nr * sizeof(amrex::ParticleReal) + layout.numInt * sizeof(int) + sizeof(std::uint64_t)) : 0);
		try {
			// A configurable scratch limit also allows deterministic fallback tests.
			if (bufferLimit >= 0 && bytes > static_cast<std::uint64_t>(bufferLimit)) {
				throw amrex::OutOfMemoryError("openPMD particle scratch limit exceeded");
			}
			deviceInts.resize(count * (compact ? layout.numInt + 1 : 1));
			if (compact) {
				deviceReals.resize(count * nr);
				deviceIDs.resize(count);
			}
			// Finish all allocations before launching kernels, so an allocation
			// failure can release partial buffers without waiting on their use.
			auto *outInts = deviceInts.data();
			levels = outInts;
			amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(std::uint64_t i) { outInts[i] = lev; });
			if (!compact) {
				// The normal path borrows the existing arrays, including idcpu.
				ids = soa.GetIdCPUData().data();
				for (int c = 0; c < nr; ++c) {
					reals[c] = soa.GetRealData(c).data();
				}
				for (int c = 0; c < layout.numInt; ++c) {
					integers[c] = soa.GetIntData(c).data();
				}
			} else {
				const auto data = tile.getConstParticleTileData();
				auto *outReals = deviceReals.data();
				auto *outIDs = deviceIDs.data();
				const auto *inIDs = soa.GetIdCPUData().data();
				const int ni = layout.numInt;
				// An exclusive prefix sum over the valid mask keeps the tile's particle
				// order, so output is reproducible from run to run.
				amrex::Scan::PrefixSum<amrex::Long>(
				    n, [=] AMREX_GPU_DEVICE(int i) -> amrex::Long { return data.id(i) > 0 ? 1 : 0; },
				    [=] AMREX_GPU_DEVICE(int i, amrex::Long const &j) {
					    if (data.id(i) > 0) {
						    outIDs[j] = inIDs[i];
						    for (int c = 0; c < nr; ++c) {
							    outReals[c * count + j] = data.rdata(c, i);
						    }
						    for (int c = 0; c < ni; ++c) {
							    outInts[(c + 1) * count + j] = data.idata(c, i);
						    }
					    }
				    },
				    amrex::Scan::Type::exclusive, amrex::Scan::noRetSum);
				ids = outIDs;
				for (int c = 0; c < nr; ++c) {
					reals[c] = outReals + c * count;
				}
				for (int c = 0; c < layout.numInt; ++c) {
					integers[c] = outInts + (c + 1) * count;
				}
			}
			amrex::Gpu::streamSynchronize();
			return;
		} catch (const std::bad_alloc &) {
			// AMReX arenas throw OutOfMemoryError (a bad_alloc) on allocation
			// failure. Release partial device allocations before staging on host.
			amrex::Gpu::DeviceVector<amrex::ParticleReal>().swap(deviceReals);
			amrex::Gpu::DeviceVector<int>().swap(deviceInts);
			amrex::Gpu::DeviceVector<std::uint64_t>().swap(deviceIDs);
		}
		onHost = true;
		const auto identity = HostCopy(soa.GetIdCPUData(), n);
		std::vector<int> valid;
		valid.reserve(count);
		for (int i = 0; i < n; ++i) {
			if (amrex::particle_impl::unpack_id(identity[i]) > 0) {
				valid.push_back(i);
				hostIDs.push_back(identity[i]);
			}
		}
		AMREX_ALWAYS_ASSERT(valid.size() == count);
		hostReals.resize(count * nr);
		hostInts.resize(count * (layout.numInt + 1));
		std::fill_n(hostInts.data(), count, lev);
		for (int c = 0; c < nr; ++c) {
			const auto host = HostCopy(soa.GetRealData(c), n);
			for (std::size_t j = 0; j < count; ++j) {
				hostReals[c * count + j] = host[valid[j]];
			}
			reals[c] = hostReals.data() + c * count;
		}
		for (int c = 0; c < layout.numInt; ++c) {
			const auto host = HostCopy(soa.GetIntData(c), n);
			for (std::size_t j = 0; j < count; ++j) {
				hostInts[(c + 1) * count + j] = host[valid[j]];
			}
			integers[c] = hostInts.data() + (c + 1) * count;
		}
		ids = hostIDs.data();
		levels = hostInts.data();
	}
};

void SetParticleUnits(openPMD::Record record, const std::array<int, 4> &powers, const std::array<double, 4> &units)
{
	record.setUnitDimension({{openPMD::UnitDimension::M, powers[0]},
				 {openPMD::UnitDimension::L, powers[1]},
				 {openPMD::UnitDimension::T, powers[2]},
				 {openPMD::UnitDimension::theta, powers[3]}});
	double scale = 1.0;
	for (int d = 0; d < 4; ++d) {
		if (powers[d] != 0) {
			scale *= std::pow(units[d], powers[d]);
		}
	}
	record.setAttribute("quokka_unitSI_known", std::isfinite(scale));
	for (auto &[name, component] : record) {
		amrex::ignore_unused(name);
		component.setUnitSI(std::isfinite(scale) ? scale : 1.0);
	}
}
} // namespace

void WriteParticles(openPMD::Series &series, openPMD::Iteration &iteration, const ParticleContainer &particles, const std::string &name,
		    const std::array<double, 4> &units, bool restart)
{
	const auto &schema = particles.Schema();
	const auto layout = particles.Layout();
	// Count valid particles without staging a rank's entire population. Retain
	// only tile references; particle buffers below live for one flush round.
	struct Tile {
		int level;
		const ParticleContainer::ParticleTileType *data;
		std::uint64_t count;
	};
	std::vector<Tile> tiles;
	std::uint64_t local = 0;
	for (int lev = 0; lev <= particles.finestLevel(); ++lev) {
		for (const auto &[key, tile] : particles.GetParticles(lev)) {
			amrex::ignore_unused(key);
			if (tile.numParticles() == 0) {
				continue;
			}
			std::uint64_t count = 0;
			try {
				const auto data = tile.getConstParticleTileData();
				count = amrex::Reduce::Sum<std::uint64_t>(tile.numParticles(),
									  [=] AMREX_GPU_DEVICE(int i) -> std::uint64_t { return data.id(i) > 0 ? 1 : 0; });
			} catch (const std::bad_alloc &) {
				const auto identity = HostCopy(tile.GetStructOfArrays().GetIdCPUData(), tile.numParticles());
				count = std::count_if(identity.begin(), identity.end(), [](auto id) { return amrex::particle_impl::unpack_id(id) > 0; });
			}
			local += count;
			if (count > 0) {
				tiles.push_back({lev, &tile, count});
			}
		}
	}
	std::vector<std::uint64_t> counts(amrex::ParallelDescriptor::NProcs(), 0);
#ifdef AMREX_USE_MPI
	MPI_Allgather(&local, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, amrex::ParallelDescriptor::Communicator());
#else
	counts[0] = local;
#endif
	const auto total = std::accumulate(counts.begin(), counts.end(), std::uint64_t{0});
	auto offset = std::accumulate(counts.begin(), counts.begin() + amrex::ParallelDescriptor::MyProc(), std::uint64_t{0});
	if (restart) {
		detail::WriteCheckpointParticleAttributes(iteration, particles, name, total);
	}
	// An absent species represents a globally empty population.
	if (total == 0) {
		return;
	}
	auto species = iteration.particles[name];
	species.setAttribute("quokka_id_encoding", "AMReX packed idcpu (ID and creation rank)");
	const auto define = [&]<typename T>(openPMD::RecordComponent component) {
		component.resetDataset(openPMD::Dataset(openPMD::determineDatatype<T>(), {total}));
	};
	const std::array<std::string, 3> axes{"x", "y", "z"};
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		define.operator()<amrex::ParticleReal>(species["position"][axes[d]]);
		auto component = species["positionOffset"][axes[d]];
		component.resetDataset(openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {total}));
		component.makeConstant(amrex::ParticleReal{0});
	}
	SetParticleUnits(species["position"], {0, 1, 0, 0}, units);
	SetParticleUnits(species["positionOffset"], {0, 1, 0, 0}, units);
	define.operator()<std::uint64_t>(species["id"][openPMD::RecordComponent::SCALAR]);
	define.operator()<int>(species["amr_level"][openPMD::RecordComponent::SCALAR]);
	for (int c = 0; c < layout.numReal; ++c) {
		auto record = species[schema.RealNames()[c]];
		define.operator()<amrex::ParticleReal>(record[openPMD::RecordComponent::SCALAR]);
		const auto &unitKey = schema.RealUnitsKeys()[c];
		if (!unitKey.empty()) {
			SetParticleUnits(record, schema.UnitDefinitions().at(unitKey), units);
		}
	}
	for (int c = 0; c < layout.numInt; ++c) {
		define.operator()<int>(species[schema.IntNames()[c]][openPMD::RecordComponent::SCALAR]);
	}
	// Define the same datasets on every rank before submitting any chunks.
	series.flush();
	const auto rankEnd = offset + local;
	amrex::Long bufferLimit = -1;
	amrex::ParmParse("openpmd").query("particle_device_buffer_bytes", bufferLimit);
	// Group tiles into flush rounds of up to flushBytes of particle data per rank,
	// so the number of collective flushes follows data volume rather than the
	// number of tiles. A tile larger than the budget gets a round of its own.
	amrex::Long flushBytes = amrex::Long{256} << 20;
	amrex::ParmParse("openpmd").query("particle_flush_bytes", flushBytes);
	const auto recordBytes = (AMREX_SPACEDIM + layout.numReal) * sizeof(amrex::ParticleReal) + (layout.numInt + 1) * sizeof(int) + sizeof(std::uint64_t);
	std::vector<std::size_t> roundStart;
	std::uint64_t roundBytes = 0;
	for (std::size_t t = 0; t < tiles.size(); ++t) {
		const auto bytes = tiles[t].count * recordBytes;
		if (roundStart.empty() || roundBytes + bytes > static_cast<std::uint64_t>(flushBytes)) {
			roundStart.push_back(t);
			roundBytes = 0;
		}
		roundBytes += bytes;
	}
	const auto localRounds = static_cast<amrex::Long>(roundStart.size());
	roundStart.push_back(tiles.size());
	auto rounds = localRounds;
	amrex::ParallelDescriptor::ReduceLongMax(rounds);
	for (amrex::Long round = 0; round < rounds; ++round) {
		const auto first = round < localRounds ? roundStart[round] : tiles.size();
		const auto last = round < localRounds ? roundStart[round + 1] : tiles.size();
		std::vector<ParticleTileOutput> roundBuffers(last - first);
		for (auto t = first; t < last; ++t) {
			auto &buffers = roundBuffers[t - first];
			const auto &[lev, tile, count] = tiles[t];
			buffers.Prepare(*tile, lev, count, layout, bufferLimit);
			const auto store = [&](openPMD::RecordComponent component, const auto *values) {
#ifdef AMREX_USE_GPU
				if (!buffers.onHost) {
					detail::StoreDeviceChunk(component, values, count, {offset}, {count});
					return;
				}
#endif
				component.storeChunkRaw(values, {offset}, {count});
			};
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				store(species["position"][axes[d]], buffers.reals[d]);
			}
			store(species["id"][openPMD::RecordComponent::SCALAR], buffers.ids);
			store(species["amr_level"][openPMD::RecordComponent::SCALAR], buffers.levels);
			for (int c = 0; c < layout.numReal; ++c) {
				store(species[schema.RealNames()[c]][openPMD::RecordComponent::SCALAR], buffers.reals[layout.RealStorage(c)]);
			}
			for (int c = 0; c < layout.numInt; ++c) {
				store(species[schema.IntNames()[c]][openPMD::RecordComponent::SCALAR], buffers.integers[c]);
			}
			offset += count;
		}
		// Staged copies must be complete, and all borrowed pointers and scratch buffers
		// remain alive through this collective flush, including on ranks that have
		// already run out of tiles.
		// Otherwise openPMD can skip a clean iteration on an idle rank,
		// leaving other ranks inside BP5's collective PerformDataWrite.
		amrex::Gpu::streamSynchronize();
		iteration.touch();
		series.flush(R"({"adios2": {"engine": {"preferred_flush_target": "disk"}}})");
	}
	AMREX_ALWAYS_ASSERT(offset == rankEnd);
}

} // namespace quokka::OpenPMDOutput
