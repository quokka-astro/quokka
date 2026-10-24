#include "AMReX_FileSystem.H"
#include "io/openPMD.hpp"
#include <iomanip>
#include <sstream>

namespace quokka::OpenPMDOutput
{
namespace
{
auto OpenSeries(const std::string &path, openPMD::Access access) -> openPMD::Series
{
#ifdef AMREX_USE_MPI
	return {path, access, amrex::ParallelDescriptor::Communicator()};
#else
	return {path, access};
#endif
}
auto PatchOffsets(const amrex::BoxArray &boxes) -> std::vector<std::uint64_t>
{
	std::vector<std::uint64_t> offsets(boxes.size() + 1, 0);
	for (int i = 0; i < boxes.size(); ++i) {
		offsets[i + 1] = offsets[i] + boxes[i].numPts();
	}
	return offsets;
}
auto EncodeNames(const amrex::Vector<std::string> &names) -> std::string
{
	std::ostringstream encoded;
	for (const auto &name : names) {
		encoded << std::quoted(name) << ' ';
	}
	return encoded.str();
}
void CheckVersion(openPMD::Iteration &iteration)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(iteration.getAttribute("quokka_checkpoint_version").get<int>() == 1, "Unsupported openPMD checkpoint version");
	AMREX_ALWAYS_ASSERT(iteration.getAttribute("quokka_spacedim").get<int>() == AMREX_SPACEDIM);
}
void SetVersion(openPMD::Iteration &iteration)
{
	iteration.setAttribute("quokka_checkpoint_version", 1);
	iteration.setAttribute("quokka_spacedim", AMREX_SPACEDIM);
	iteration.setTime(0.0);
}
} // namespace

void WriteCheckpointMultiFab(const amrex::MultiFab &mf, const std::string &path)
{
	auto series = OpenSeries(path, openPMD::Access::CREATE);
	auto iteration = series.iterations[0];
	SetVersion(iteration);
	std::ostringstream boxes;
	mf.boxArray().writeOn(boxes);
	iteration.setAttribute("quokka_boxes", boxes.str());
	iteration.setAttribute("quokka_ncomp", mf.nComp());
	const auto offsets = PatchOffsets(mf.boxArray());
	// A packed patch array avoids overlapping writes of shared faces and holes
	// in refined levels. BoxArray metadata defines the spatial interpretation.
	amrex::MultiFab host(mf.boxArray(), mf.DistributionMap(), mf.nComp(), 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	amrex::MultiFab::Copy(host, mf, 0, 0, mf.nComp(), 0);
	amrex::Gpu::streamSynchronize();
	for (int c = 0; c < mf.nComp(); ++c) {
		auto mesh = iteration.meshes["component_" + std::to_string(c)];
		mesh.setAxisLabels({"patch_cell"});
		mesh.setGridSpacing(std::vector<double>{1.0});
		mesh.setGridGlobalOffset({0.0});
		auto record = mesh[openPMD::RecordComponent::SCALAR];
		record.setPosition(std::vector<double>{0.0});
		record.resetDataset(openPMD::Dataset(openPMD::determineDatatype<amrex::Real>(), {offsets.back()}));
		for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
			const auto i = mfi.index();
			record.storeChunkRaw(host[mfi].dataPtr(c), {offsets[i]}, {offsets[i + 1] - offsets[i]});
		}
	}
	series.close();
}

void ReadCheckpointMultiFab(amrex::MultiFab &mf, const std::string &path, int expectedComponents)
{
	auto series = OpenSeries(path, openPMD::Access::READ_ONLY);
	auto iteration = series.iterations.at(0);
	CheckVersion(iteration);
	const auto ncomp = iteration.getAttribute("quokka_ncomp").get<int>();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ncomp == expectedComponents, "Checkpoint component count does not match this executable");
	std::istringstream input(iteration.getAttribute("quokka_boxes").get<std::string>());
	amrex::BoxArray boxes;
	boxes.readFrom(input);
	AMREX_ALWAYS_ASSERT(boxes.size() > 0);
	const auto offsets = PatchOffsets(boxes);
	const amrex::DistributionMapping dm(boxes);
	amrex::MultiFab host(boxes, dm, ncomp, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	for (int c = 0; c < ncomp; ++c) {
		auto record = iteration.meshes.at("component_" + std::to_string(c))[openPMD::RecordComponent::SCALAR];
		AMREX_ALWAYS_ASSERT(record.getExtent() == openPMD::Extent{offsets.back()});
		AMREX_ALWAYS_ASSERT(record.getDatatype() == openPMD::determineDatatype<amrex::Real>());
		for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
			const auto i = mfi.index();
			record.loadChunkRaw(host[mfi].dataPtr(c), {offsets[i]}, {offsets[i + 1] - offsets[i]});
		}
	}
	series.close();
	mf.define(boxes, dm, ncomp, 0);
	amrex::MultiFab::Copy(mf, host, 0, 0, ncomp, 0);
	amrex::Gpu::streamSynchronize();
}

void WriteCheckpointParticles(const ParticleContainer &particles, const std::string &directory, const std::string &name)
{
	const auto path = directory + "/" + name;
	amrex::UtilCreateCleanDirectory(path, true);
	auto series = OpenSeries(path + "/particles.bp", openPMD::Access::CREATE);
	auto iteration = series.iterations[0];
	SetVersion(iteration);
	const auto &schema = particles.Schema();
	// Store even empty schemas and populations; plotfiles omit empty species.
	iteration.setAttribute("quokka_real_names", EncodeNames(schema.RealNames()));
	iteration.setAttribute("quokka_int_names", EncodeNames(schema.IntNames()));
	auto next = ParticleContainer::ParticleType::NextID();
	ParticleContainer::ParticleType::NextID(next);
	amrex::ParallelDescriptor::ReduceLongMax(next);
	iteration.setAttribute("quokka_next_id", static_cast<std::int64_t>(next));
	iteration.setAttribute("quokka_count", static_cast<std::uint64_t>(particles.TotalNumberOfParticles()));
	const auto unknown = std::numeric_limits<double>::quiet_NaN();
	WriteParticles(series, iteration, particles, name, {unknown, unknown, unknown, unknown});
	series.close();
}

void ReadCheckpointParticles(ParticleContainer &particles, const std::string &directory, const std::string &name)
{
	AMREX_ALWAYS_ASSERT(particles.TotalNumberOfParticles(false) == 0);
	auto series = OpenSeries(directory + "/" + name + "/particles.bp", openPMD::Access::READ_ONLY);
	auto iteration = series.iterations.at(0);
	CheckVersion(iteration);
	const auto &schema = particles.Schema();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(iteration.getAttribute("quokka_real_names").get<std::string>() == EncodeNames(schema.RealNames()) &&
					     iteration.getAttribute("quokka_int_names").get<std::string>() == EncodeNames(schema.IntNames()),
					 "Particle restart field names/order do not match the population schema");
	const auto &reals = schema.RealNames();
	const auto &ints = schema.IntNames();
	const auto total = iteration.getAttribute("quokka_count").get<std::uint64_t>();
	const auto next = iteration.getAttribute("quokka_next_id").get<std::int64_t>();
	const auto rank = static_cast<std::uint64_t>(amrex::ParallelDescriptor::MyProc());
	const auto ranks = static_cast<std::uint64_t>(amrex::ParallelDescriptor::NProcs());
	const auto begin = total / ranks * rank + std::min(total % ranks, rank);
	const auto count = total / ranks + (rank < total % ranks ? 1 : 0);
	// Distribute contiguous records across the new ranks, then locate particles
	// on the current hierarchy. Creation ranks in packed IDs remain unchanged.
	std::vector<std::uint64_t> ids(count);
	std::vector<std::vector<amrex::ParticleReal>> realData(particles.NumRealComps(), std::vector<amrex::ParticleReal>(count));
	std::vector<std::vector<int>> intData(particles.NumIntComps(), std::vector<int>(count));
	if (total > 0) {
		auto species = iteration.particles.at(name);
		const auto load = [&](openPMD::RecordComponent record, auto &data) {
			using Value = typename std::decay_t<decltype(data)>::value_type;
			AMREX_ALWAYS_ASSERT(record.getExtent() == openPMD::Extent{total});
			AMREX_ALWAYS_ASSERT(record.getDatatype() == openPMD::determineDatatype<Value>());
			if (count > 0) {
				record.loadChunkRaw(data.data(), {begin}, {count});
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
	series.close();
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
