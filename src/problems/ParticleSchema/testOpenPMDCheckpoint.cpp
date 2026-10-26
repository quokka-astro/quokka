#include "AMReX_ParmParse.H"
#include "AMReX_VisMF.H"
#include "io/openPMD.hpp"

namespace
{
auto ParticleValues(const quokka::ParticleContainer &particles) -> std::map<std::uint64_t, std::vector<amrex::ParticleReal>>
{
	std::map<std::uint64_t, std::vector<amrex::ParticleReal>> result;
	for (int lev = 0; lev <= particles.finestLevel(); ++lev) {
		for (const auto &[key, tile] : particles.GetParticles(lev)) {
			amrex::ignore_unused(key);
			const auto &soa = tile.GetStructOfArrays();
			std::vector<std::uint64_t> ids(tile.numParticles());
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(), soa.GetIdCPUData().begin() + tile.numParticles(), ids.begin());
			std::vector<std::vector<amrex::ParticleReal>> values(particles.NumRealComps(), std::vector<amrex::ParticleReal>(ids.size()));
			for (int c = 0; c < particles.NumRealComps(); ++c) {
				amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(c).begin(), soa.GetRealData(c).begin() + tile.numParticles(),
						 values[c].begin());
			}
			for (std::size_t i = 0; i < ids.size(); ++i) {
				std::vector<amrex::ParticleReal> fields;
				for (const auto &component : values) {
					fields.push_back(component[i]);
				}
				AMREX_ALWAYS_ASSERT(result.emplace(ids[i], fields).second);
			}
		}
	}
	return result;
}

void CompareTracers(const std::string &reference, const std::string &candidate, int levels)
{
	amrex::Vector<amrex::Geometry> geoms;
	amrex::Vector<amrex::BoxArray> grids;
	amrex::Vector<amrex::DistributionMapping> maps;
	const amrex::RealBox bounds({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1.2, 1.2, 1.2)});
	const std::array<int, AMREX_SPACEDIM> periodic{};
	for (int lev = 0; lev < levels; ++lev) {
		amrex::MultiFab mf;
		amrex::VisMF::Read(mf, reference + "/Level_" + std::to_string(lev) + "/Cell");
		grids.push_back(mf.boxArray());
		maps.push_back(mf.DistributionMap());
		const auto domain = lev == 0 ? grids[0].minimalBox() : amrex::refine(geoms[lev - 1].Domain(), 2);
		geoms.emplace_back(domain, &bounds, 0, periodic.data());
	}
	quokka::ParticleLayout layout;
	layout.numReal = AMREX_SPACEDIM;
	layout.velocity = 0;
	amrex::Vector<std::string> names;
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		names.push_back("real_comp" + std::to_string(d));
	}
	const quokka::ParticleSchema schema(quokka::ParticleType::Tracer, "tracer_particles", names, {}, amrex::Vector<amrex::ParticleReal>(AMREX_SPACEDIM, 0.),
					    {}, amrex::Vector<std::string>(AMREX_SPACEDIM), layout, {});
	const amrex::Vector<amrex::IntVect> ratios(levels - 1, amrex::IntVect(2));
	quokka::ParticleContainer expected(schema, geoms, maps, grids, ratios);
	quokka::ParticleContainer actual(schema, geoms, maps, grids, ratios);
	expected.Restart(reference, "tracer_particles");
	actual.Restart(candidate, "tracer_particles");
	quokka::ParticleContainer::FinalizeRestarts();
	AMREX_ALWAYS_ASSERT(expected.RestartNextID() == actual.RestartNextID());
	// Both populations share grids and ownership, so each rank compares its own particles.
	bool same = ParticleValues(expected) == ParticleValues(actual);
	amrex::ParallelDescriptor::ReduceBoolAnd(same);
	AMREX_ALWAYS_ASSERT(same);
}

// Values depend only on position, so shared faces hold identical copies.
void FillMesh(amrex::MultiFab &mf)
{
	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const auto a = mf.array(mfi);
		amrex::ParallelFor(mfi.validbox(), mf.nComp(),
				   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { a(i, j, k, n) = 100000 * n + 1000 * k + 100 * j + 10 * i; });
	}
}

auto MeshNames(int direction) -> amrex::Vector<std::string>
{
	// The dash exercises the plotfile name conversion.
	const auto suffix = "_" + std::to_string(direction + 1);
	return {"a" + suffix, "b-c" + suffix, "d" + suffix};
}

// Plotfile meshes take the same device-to-backend path as checkpoints. A
// non-cubic domain split into non-cubic patches catches reversed chunk dimensions.
void CheckPlotfileMesh()
{
	const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(7, 3, 5)));
	const amrex::RealBox bounds({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
	const std::array<int, AMREX_SPACEDIM> periodic{};
	const amrex::Geometry geom(domain, &bounds, 0, periodic.data());
	amrex::BoxArray ba(domain);
	ba.maxSize(amrex::IntVect(AMREX_D_DECL(4, 2, 3)));
	amrex::MultiFab mf(ba, amrex::DistributionMapping(ba), 3, 0);
	FillMesh(mf);
	const std::vector<std::string> names{"a", "b", "c"};
	quokka::OpenPMDOutput::WriteFile(names, 1, {&mf}, {geom}, "mesh_plt", 0., 5);
	amrex::ParallelDescriptor::Barrier();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		openPMD::Series series("mesh_plt00005.bp", openPMD::Access::READ_ONLY);
		auto iteration = series.iterations[5];
		const auto n = domain.length();
		for (int c = 0; c < 3; ++c) {
			auto record = iteration.meshes[names[c]][openPMD::MeshRecordComponent::SCALAR];
			AMREX_ALWAYS_ASSERT(record.getExtent() == quokka::OpenPMDOutput::detail::getReversedVec(domain.size()));
			const auto data = record.loadChunk<amrex::Real>();
			series.flush();
			// C order: x varies fastest.
			for (amrex::Long idx = 0; idx < domain.numPts(); ++idx) {
				const auto i = idx % n[0];
				const auto j = AMREX_SPACEDIM > 1 ? (idx / n[0]) % n[1] : 0;
				const auto k = AMREX_SPACEDIM > 2 ? idx / (n[0] * n[1]) : 0;
				AMREX_ALWAYS_ASSERT(data.get()[idx] == 100000 * c + 1000 * k + 100 * j + 10 * i);
			}
		}
	}
}

// With writeCheckpoint == 0, read an existing checkpoint, e.g. one written by a
// build of the other floating-point precision.
void CheckMeshes(int writeCheckpoint)
{
	const amrex::Box domain(amrex::IntVect(-2), amrex::IntVect(5));
	const amrex::RealBox bounds({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
	const std::array<int, AMREX_SPACEDIM> periodic{};
	// Level 1 covers only part of its domain, leaving holes in its datasets.
	const std::array<amrex::Box, 2> domains{domain, amrex::refine(domain, 2)};
	std::array<amrex::BoxArray, 2> boxes{amrex::BoxArray(domain), amrex::BoxArray(amrex::Box(amrex::IntVect(0), amrex::IntVect(7)))};
	for (auto &ba : boxes) {
		ba.maxSize(4);
	}
	const auto ixType = [](int direction) {
		return direction < 0 ? amrex::IndexType::TheCellType() : amrex::IndexType(amrex::IntVect::TheDimensionVector(direction));
	};
	if (writeCheckpoint != 0) {
		auto series = quokka::OpenPMDOutput::CreateCheckpointSeries("mesh");
		auto iteration = series.iterations[7];
		quokka::OpenPMDOutput::SetCheckpointVersion(iteration);
		for (int lev = 0; lev < 2; ++lev) {
			const amrex::Geometry geom(domains[lev], &bounds, 0, periodic.data());
			// Cover cells and all face directions.
			for (int direction = -1; direction < AMREX_SPACEDIM; ++direction) {
				const auto ba = amrex::convert(boxes[lev], ixType(direction));
				// Deliberately put every write patch on rank zero; the reader balances.
				const amrex::DistributionMapping dm(amrex::Vector<int>(ba.size(), 0));
				amrex::MultiFab source(ba, dm, 3, 2);
				source.setVal(-9999.);
				FillMesh(source);
				quokka::OpenPMDOutput::WriteCheckpointMesh(iteration, source, geom, lev, MeshNames(direction));
			}
		}
		series.close();
	}
	auto series = quokka::OpenPMDOutput::OpenCheckpointSeries("mesh0000007.bp");
	AMREX_ALWAYS_ASSERT(series.iterationEncoding() == openPMD::IterationEncoding::fileBased);
	auto iteration = quokka::OpenPMDOutput::CheckpointIteration(series);
	AMREX_ALWAYS_ASSERT(iteration.meshes.contains("b_c_0") && iteration.meshes.contains("b_c_0_lvl1"));
	{
		const bool previous = amrex::system::throw_exception;
		amrex::system::throw_exception = true;
		bool rejected = false;
		try {
			amrex::MultiFab missing;
			quokka::OpenPMDOutput::ReadCheckpointMesh(iteration, missing, domain, 0, ixType(-1), {"missing"});
		} catch (const std::exception &) {
			rejected = true;
		}
		amrex::system::throw_exception = previous;
		AMREX_ALWAYS_ASSERT(rejected);
	}
	for (int lev = 0; lev < 2; ++lev) {
		AMREX_ALWAYS_ASSERT(quokka::OpenPMDOutput::ReadCheckpointBoxArray(iteration, lev) == boxes[lev]);
		for (int direction = -1; direction < AMREX_SPACEDIM; ++direction) {
			amrex::MultiFab restored;
			quokka::OpenPMDOutput::ReadCheckpointMesh(iteration, restored, domains[lev], lev, ixType(direction), MeshNames(direction));
			AMREX_ALWAYS_ASSERT(restored.boxArray() == amrex::convert(boxes[lev], ixType(direction)) && restored.nGrow() == 0);
			amrex::MultiFab expected(restored.boxArray(), restored.DistributionMap(), 3, 0);
			FillMesh(expected);
			amrex::MultiFab::Subtract(restored, expected, 0, 0, 3, 0);
			for (int n = 0; n < 3; ++n) {
				AMREX_ALWAYS_ASSERT(restored.norm0(n) == 0.);
			}
		}
	}
	series.close();
}
} // namespace
auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	{
		std::string reference;
		amrex::ParmParse pp;
		pp.query("reference", reference);
		if (reference.empty()) {
			int writeCheckpoint = 1;
			pp.query("write_checkpoint", writeCheckpoint);
			CheckMeshes(writeCheckpoint);
			CheckPlotfileMesh();
		} else {
			std::string candidate;
			pp.get("candidate", candidate);
			int levels = 1;
			pp.query("levels", levels);
			for (int lev = 0; lev < levels; ++lev) {
				const auto suffix = "/Level_" + std::to_string(lev) + "/Cell";
				amrex::MultiFab expected, actual;
				amrex::VisMF::Read(expected, reference + suffix);
				amrex::VisMF::Read(actual, candidate + suffix);
				AMREX_ALWAYS_ASSERT(expected.nComp() == actual.nComp());
				amrex::MultiFab aligned(expected.boxArray(), expected.DistributionMap(), expected.nComp(), 0);
				aligned.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
				aligned.ParallelCopy(actual);
				AMREX_ALWAYS_ASSERT(!aligned.contains_nan());
				amrex::MultiFab::Subtract(aligned, expected, 0, 0, expected.nComp(), 0);
				for (int c = 0; c < expected.nComp(); ++c) {
					AMREX_ALWAYS_ASSERT(aligned.norm0(c) == 0.);
				}
			}
			CompareTracers(reference, candidate, levels);
		}
	}
	amrex::Finalize();
}
