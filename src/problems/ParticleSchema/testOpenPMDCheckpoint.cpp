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
	AMREX_ALWAYS_ASSERT(amrex::ParallelDescriptor::NProcs() == 1);
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
	AMREX_ALWAYS_ASSERT(ParticleValues(expected) == ParticleValues(actual));
}

void CheckMeshes()
{
	amrex::BoxArray boxes(amrex::Box(amrex::IntVect(-2), amrex::IntVect(5)));
	boxes.maxSize(4);
	// Cover cells and all face directions, including distinct duplicate faces.
	for (int direction = -1; direction < AMREX_SPACEDIM; ++direction) {
		auto ba = boxes;
		if (direction >= 0) {
			ba.convert(amrex::IntVect::TheDimensionVector(direction));
		}
		// Deliberately put every write patch on rank zero; the reader balances.
		const amrex::DistributionMapping dm(amrex::Vector<int>(ba.size(), 0));
		amrex::MultiFab source(ba, dm, 3, 2);
		source.setVal(-9999.);
		for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
			const auto a = source.array(mfi);
			const auto patch = mfi.index();
			amrex::ParallelFor(mfi.validbox(), 3, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
				a(i, j, k, n) = 100000 * n + 10000 * patch + 100 * k + 10 * j + i;
			});
		}
		const auto path = "mesh_" + std::to_string(direction + 1) + ".bp";
		quokka::OpenPMDOutput::WriteCheckpointMultiFab(source, path);
		amrex::MultiFab restored;
		quokka::OpenPMDOutput::ReadCheckpointMultiFab(restored, path, 3);
		if (direction == -1) {
			const bool previous = amrex::system::throw_exception;
			amrex::system::throw_exception = true;
			bool rejected = false;
			try {
				amrex::MultiFab incompatible;
				quokka::OpenPMDOutput::ReadCheckpointMultiFab(incompatible, path, 4);
			} catch (const std::exception &) {
				rejected = true;
			}
			amrex::system::throw_exception = previous;
			AMREX_ALWAYS_ASSERT(rejected);
		}
		AMREX_ALWAYS_ASSERT(restored.boxArray() == ba && restored.nGrow() == 0);
		amrex::MultiFab expected(ba, restored.DistributionMap(), 3, 0);
		for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
			const auto a = expected.array(mfi);
			const auto patch = mfi.index();
			amrex::ParallelFor(mfi.validbox(), 3, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
				a(i, j, k, n) = 100000 * n + 10000 * patch + 100 * k + 10 * j + i;
			});
		}
		amrex::MultiFab::Subtract(restored, expected, 0, 0, 3, 0);
		for (int n = 0; n < 3; ++n) {
			AMREX_ALWAYS_ASSERT(restored.norm0(n) == 0.);
		}
	}
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
			CheckMeshes();
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
