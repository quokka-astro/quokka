#include "io/openPMD.hpp"
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>

namespace quokka::OpenPMDOutput
{
namespace
{
auto EncodeNames(const amrex::Vector<std::string> &names) -> std::string
{
	std::ostringstream encoded;
	for (const auto &name : names) {
		encoded << std::quoted(name) << ' ';
	}
	return encoded.str();
}
auto LevelKey(const std::string &key, int lev) -> std::string { return "quokka_" + key + "_lvl" + std::to_string(lev); }
auto ParticleKey(const std::string &name, const std::string &key) -> std::string { return "quokka_particles_" + name + "_" + key; }
// Offset and extent of a box within the level's dataset, in openPMD (C) order.
auto ChunkOffset(const amrex::Box &box, const amrex::Box &domain) -> openPMD::Offset { return detail::getReversedVec(box.smallEnd() - domain.smallEnd()); }
auto ChunkExtent(const amrex::Box &box) -> openPMD::Extent { return detail::getReversedVec(box.size()); }
// Loads floating-point data stored as float or double into host memory at dest,
// converting when the stored precision differs from T. A conversion is returned
// as a callback to run after the next flush; it is empty when the data loads directly.
template <typename T>
auto LoadReal(openPMD::RecordComponent record, T *dest, const openPMD::Offset &offset, const openPMD::Extent &extent) -> std::function<void()>
{
	const auto type = record.getDatatype();
	if (type == openPMD::determineDatatype<T>()) {
		record.loadChunkRaw(dest, offset, extent);
		return {};
	}
	const auto convert = [&]<typename Stored>() -> std::function<void()> {
		const auto n = std::accumulate(extent.begin(), extent.end(), std::uint64_t{1}, std::multiplies<>());
		auto stored = std::make_shared<std::vector<Stored>>(n);
		record.loadChunkRaw(stored->data(), offset, extent);
		return [stored, dest] { std::copy(stored->begin(), stored->end(), dest); };
	};
	if (type == openPMD::Datatype::FLOAT) {
		return convert.template operator()<float>();
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(type == openPMD::Datatype::DOUBLE, "openPMD checkpoint floating-point data must be stored as float or double");
	return convert.template operator()<double>();
}
} // namespace

auto CreateCheckpointSeries(const std::string &basename) -> openPMD::Series
{
	// Pad like native checkpoint names, e.g. chk0000010.bp.
	const std::string filename = basename + "%07T.bp";
#ifdef AMREX_USE_MPI
	auto series = openPMD::Series(filename, openPMD::Access::CREATE, amrex::ParallelDescriptor::Communicator());
#else
	auto series = openPMD::Series(filename, openPMD::Access::CREATE);
#endif
	series.setSoftware("Quokka", "1.0");
	series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
	return series;
}

auto OpenCheckpointSeries(const std::string &path) -> openPMD::Series
{
	// An explicit file name opens only this checkpoint of the file-based series.
#ifdef AMREX_USE_MPI
	return {path, openPMD::Access::READ_ONLY, amrex::ParallelDescriptor::Communicator()};
#else
	return {path, openPMD::Access::READ_ONLY};
#endif
}

auto CheckpointIteration(openPMD::Series &series) -> openPMD::Iteration
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(series.iterations.size() == 1, "openPMD checkpoint must contain exactly one iteration");
	auto iteration = series.iterations.begin()->second;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(iteration.containsAttribute("quokka_checkpoint_version") &&
					     iteration.getAttribute("quokka_checkpoint_version").get<int>() == 1,
					 "Not a supported Quokka openPMD checkpoint (plotfiles cannot be used as checkpoints)");
	AMREX_ALWAYS_ASSERT(iteration.getAttribute("quokka_spacedim").get<int>() == AMREX_SPACEDIM);
	return iteration;
}

void SetCheckpointVersion(openPMD::Iteration &iteration)
{
	iteration.setAttribute("quokka_checkpoint_version", 1);
	iteration.setAttribute("quokka_spacedim", AMREX_SPACEDIM);
}

void WriteCheckpointNames(openPMD::Iteration &iteration, const std::string &key, const amrex::Vector<std::string> &names)
{
	iteration.setAttribute("quokka_" + key, EncodeNames(names));
}

auto CheckpointNamesMatch(openPMD::Iteration &iteration, const std::string &key, const amrex::Vector<std::string> &names) -> bool
{
	return iteration.getAttribute("quokka_" + key).get<std::string>() == EncodeNames(names);
}

auto ReadCheckpointBoxArray(openPMD::Iteration &iteration, int lev) -> amrex::BoxArray
{
	std::istringstream input(iteration.getAttribute(LevelKey("boxes", lev)).get<std::string>());
	amrex::BoxArray boxes;
	boxes.readFrom(input);
	AMREX_ALWAYS_ASSERT(boxes.size() > 0);
	return boxes;
}

void WriteCheckpointMesh(openPMD::Iteration &iteration, const amrex::MultiFab &mf, const amrex::Geometry &geom, int lev,
			 const amrex::Vector<std::string> &names)
{
	AMREX_ALWAYS_ASSERT(mf.nComp() == static_cast<int>(names.size()));
	const auto ixType = mf.ixType();
	const auto domain = amrex::convert(geom.Domain(), ixType);
	// Every centering on a level shares the cell-centered hierarchy recorded here.
	std::ostringstream boxes;
	amrex::convert(mf.boxArray(), amrex::IndexType::TheCellType()).writeOn(boxes);
	iteration.setAttribute(LevelKey("boxes", lev), boxes.str());
	// Stage one level and centering in pinned memory. Device pointers cannot be
	// passed instead: GPU-enabled ADIOS2 treats GPU buffers as column-major and
	// reverses the dimensions of multidimensional chunks. Holes in refined levels
	// stay unwritten; shared faces are written by every patch that holds them.
	amrex::MultiFab host(mf.boxArray(), mf.DistributionMap(), mf.nComp(), 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	amrex::MultiFab::Copy(host, mf, 0, 0, mf.nComp(), 0);
	amrex::Gpu::streamSynchronize();
	for (int c = 0; c < mf.nComp(); ++c) {
		const auto meshName = detail::GetMeshComponentName(lev, names[c]);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!iteration.meshes.contains(meshName), "Duplicate checkpoint mesh name " + meshName);
		auto mesh = iteration.meshes[meshName];
		detail::SetupMeshComponent(mesh, geom, ixType);
		auto record = mesh[openPMD::MeshRecordComponent::SCALAR];
		for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
			const auto &box = mfi.validbox();
			record.storeChunkRaw(host[mfi].dataPtr(c), ChunkOffset(box, domain), ChunkExtent(box));
		}
	}
	iteration.seriesFlush();
}

void ReadCheckpointMesh(openPMD::Iteration &iteration, amrex::MultiFab &mf, const amrex::Box &cellDomain, int lev, amrex::IndexType ixType,
			const amrex::Vector<std::string> &names)
{
	const auto boxes = amrex::convert(ReadCheckpointBoxArray(iteration, lev), ixType);
	const auto domain = amrex::convert(cellDomain, ixType);
	const auto ncomp = static_cast<int>(names.size());
	const amrex::DistributionMapping dm(boxes);
	// Stage in pinned memory, for the same reason as WriteCheckpointMesh.
	amrex::MultiFab host(boxes, dm, ncomp, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	std::vector<std::function<void()>> conversions;
	for (int c = 0; c < ncomp; ++c) {
		const auto meshName = detail::GetMeshComponentName(lev, names[c]);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(iteration.meshes.contains(meshName), "Checkpoint is missing mesh " + meshName);
		auto record = iteration.meshes.at(meshName)[openPMD::MeshRecordComponent::SCALAR];
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(record.getExtent() == ChunkExtent(domain),
						 "Checkpoint mesh " + meshName + " does not match the restart domain");
		for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
			const auto &box = mfi.validbox();
			if (auto convert = LoadReal(record, host[mfi].dataPtr(c), ChunkOffset(box, domain), ChunkExtent(box))) {
				conversions.push_back(std::move(convert));
			}
		}
	}
	iteration.seriesFlush();
	for (const auto &convert : conversions) {
		convert();
	}
	mf.define(boxes, dm, ncomp, 0);
	amrex::MultiFab::Copy(mf, host, 0, 0, ncomp, 0);
	amrex::Gpu::streamSynchronize();
}

namespace detail
{
void WriteCheckpointParticleAttributes(openPMD::Iteration &iteration, const ParticleContainer &particles, const std::string &name, std::uint64_t total)
{
	// Record even empty populations, which have no species in the iteration.
	const auto &schema = particles.Schema();
	iteration.setAttribute(ParticleKey(name, "real_names"), EncodeNames(schema.RealNames()));
	iteration.setAttribute(ParticleKey(name, "int_names"), EncodeNames(schema.IntNames()));
	auto next = ParticleContainer::ParticleType::NextID();
	ParticleContainer::ParticleType::NextID(next);
	amrex::ParallelDescriptor::ReduceLongMax(next);
	iteration.setAttribute(ParticleKey(name, "next_id"), static_cast<std::int64_t>(next));
	iteration.setAttribute(ParticleKey(name, "count"), total);
}
} // namespace detail

auto HasCheckpointParticles(openPMD::Iteration &iteration, const std::string &name) -> bool { return iteration.containsAttribute(ParticleKey(name, "count")); }

void ReadCheckpointParticles(openPMD::Iteration &iteration, ParticleContainer &particles, const std::string &name)
{
	AMREX_ALWAYS_ASSERT(particles.TotalNumberOfParticles(false) == 0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(HasCheckpointParticles(iteration, name), "Missing openPMD checkpoint particle population " + name);
	const auto &schema = particles.Schema();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(iteration.getAttribute(ParticleKey(name, "real_names")).get<std::string>() == EncodeNames(schema.RealNames()) &&
					     iteration.getAttribute(ParticleKey(name, "int_names")).get<std::string>() == EncodeNames(schema.IntNames()),
					 "Particle restart field names/order do not match the population schema");
	const auto &reals = schema.RealNames();
	const auto &ints = schema.IntNames();
	const auto total = iteration.getAttribute(ParticleKey(name, "count")).get<std::uint64_t>();
	const auto next = iteration.getAttribute(ParticleKey(name, "next_id")).get<std::int64_t>();
	const auto rank = static_cast<std::uint64_t>(amrex::ParallelDescriptor::MyProc());
	const auto ranks = static_cast<std::uint64_t>(amrex::ParallelDescriptor::NProcs());
	const auto begin = total / ranks * rank + std::min(total % ranks, rank);
	const auto count = total / ranks + (rank < total % ranks ? 1 : 0);
	// Distribute contiguous records across the new ranks, then locate particles
	// on the current hierarchy. Creation ranks in packed IDs remain unchanged.
	std::vector<std::uint64_t> ids(count);
	std::vector<std::vector<amrex::ParticleReal>> realData(particles.NumRealComps(), std::vector<amrex::ParticleReal>(count));
	std::vector<std::vector<int>> intData(particles.NumIntComps(), std::vector<int>(count));
	std::vector<std::function<void()>> conversions;
	if (total > 0) {
		auto species = iteration.particles.at(name);
		const auto load = [&](openPMD::RecordComponent record, auto &data) {
			using Value = typename std::decay_t<decltype(data)>::value_type;
			AMREX_ALWAYS_ASSERT(record.getExtent() == openPMD::Extent{total});
			if constexpr (std::is_floating_point_v<Value>) {
				if (count > 0) {
					if (auto convert = LoadReal(record, data.data(), {begin}, {count})) {
						conversions.push_back(std::move(convert));
					}
				}
			} else {
				AMREX_ALWAYS_ASSERT(record.getDatatype() == openPMD::determineDatatype<Value>());
				if (count > 0) {
					record.loadChunkRaw(data.data(), {begin}, {count});
				}
			}
		};
		load(species["id"][openPMD::RecordComponent::SCALAR], ids);
		const std::array<std::string, 3> axes{"x", "y", "z"};
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			load(species["position"][axes[d]], realData[d]);
		}
		for (int c = 0; c < particles.Layout().numReal; ++c) {
			load(species[reals[c]][openPMD::RecordComponent::SCALAR], realData[AMREX_SPACEDIM + c]);
		}
		for (int c = 0; c < particles.NumIntComps(); ++c) {
			load(species[ints[c]][openPMD::RecordComponent::SCALAR], intData[c]);
		}
	}
	iteration.seriesFlush();
	for (const auto &convert : conversions) {
		convert();
	}
	amrex::Long maximum = 0;
	for (const auto id : ids) {
		const auto unpacked = amrex::particle_impl::unpack_id(id);
		AMREX_ALWAYS_ASSERT(unpacked > 0);
		maximum = std::max(maximum, static_cast<amrex::Long>(unpacked));
	}
	AMREX_ALWAYS_ASSERT(next > maximum);
	particles.resizeData();
	if (count > 0) {
		// Redistribute accepts nonlocal input tiles and transfers them to owners.
		auto &tile = particles.DefineAndReturnParticleTile(0, 0, 0);
		tile.resize(count);
		auto &soa = tile.GetStructOfArrays();
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(), soa.GetIdCPUData().begin());
		for (int c = 0; c < particles.NumRealComps(); ++c) {
			amrex::Gpu::copy(amrex::Gpu::hostToDevice, realData[c].begin(), realData[c].end(), soa.GetRealData(c).begin());
		}
		for (int c = 0; c < particles.NumIntComps(); ++c) {
			amrex::Gpu::copy(amrex::Gpu::hostToDevice, intData[c].begin(), intData[c].end(), soa.GetIntData(c).begin());
		}
	}
	particles.Redistribute();
	particles.RegisterRestartNextID(next);
	AMREX_ALWAYS_ASSERT(static_cast<std::uint64_t>(particles.TotalNumberOfParticles()) == total);
}
} // namespace quokka::OpenPMDOutput
