#include "AMReX_ParmParse.H"
#include "io/openPMD.hpp"
#include <limits>
#include <set>

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	{
		const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(1));
		const amrex::RealBox bounds({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
		const std::array<int, AMREX_SPACEDIM> periodic{};
		const amrex::Geometry geom(domain, &bounds, 0, periodic.data());
		amrex::BoxArray boxes(domain);
		boxes.maxSize(1);
		int emptyRank = 0;
		amrex::ParmParse pp;
		pp.query("empty_rank", emptyRank);
		int hostFallback = 0;
		pp.query("host_fallback", hostFallback);
		if (hostFallback == 1 || (hostFallback == 2 && amrex::ParallelDescriptor::MyProc() == 0)) {
			amrex::ParmParse("openpmd").add("particle_device_buffer_bytes", amrex::Long{0});
		}
		const amrex::DistributionMapping dm =
		    emptyRank != 0 ? amrex::DistributionMapping(amrex::Vector<int>(boxes.size(), 0)) : amrex::DistributionMapping(boxes);
		quokka::ParticleLayout layout;
		layout.numReal = 1;
		layout.numInt = 1;
		const quokka::ParticleSchema schema(quokka::ParticleType::Test, "test", {"mass"}, {"stage"}, {7.}, {3}, {"mass"}, layout,
						    {{"mass", {1, 0, 0, 0}}});
		amrex::IntVect fineLo(0);
		fineLo[0] = 2;
		const amrex::BoxArray fineBoxes(amrex::Box(fineLo, amrex::IntVect(3)));
		const amrex::Vector<amrex::Geometry> geoms{geom, amrex::refine(geom, 2)};
		const amrex::Vector<amrex::DistributionMapping> maps{dm, amrex::DistributionMapping(amrex::Vector<int>(1, 0))};
		const amrex::Vector<amrex::BoxArray> grids{boxes, fineBoxes};
		quokka::ParticleContainer particles(schema, geoms, maps, grids, amrex::Vector<amrex::IntVect>{amrex::IntVect(2)});
		particles.InitOnePerCell(0.5, 0.5, 0.5);
		// Invalidate the first cell without redistributing: all fields must use the same mask.
		for (int lev = 0; lev <= particles.finestLevel(); ++lev) {
			for (auto &[key, tile] : particles.GetParticles(lev)) {
				amrex::ignore_unused(key);
				const auto data = tile.getParticleTileData();
				amrex::ParallelFor(tile.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
					bool first = true;
					int cell = 0;
					for (int d = 0; d < AMREX_SPACEDIM; ++d) {
						first = first && data.pos(d, i) < 0.5;
						if (data.pos(d, i) > 0.5) {
							cell |= 1 << d;
						}
					}
					data.rdata(AMREX_SPACEDIM, i) = 7 + cell;
					data.idata(0, i) = 3 + cell;
					if (first || (AMREX_SPACEDIM > 1 && cell == 1)) {
						data.id(i) = -1;
					}
				});
			}
		}
		amrex::Gpu::streamSynchronize();
		// Valid IDs in tile order: a single-rank writer must preserve this order exactly.
		std::vector<std::uint64_t> expectedOrder;
		for (int lev = 0; lev <= particles.finestLevel(); ++lev) {
			for (const auto &[key, tile] : particles.GetParticles(lev)) {
				amrex::ignore_unused(key);
				const auto idcpu = tile.GetStructOfArrays().GetIdCPUData();
				std::vector<std::remove_cv_t<std::remove_reference_t<decltype(*idcpu.begin())>>> host(tile.numParticles());
				amrex::Gpu::copy(amrex::Gpu::deviceToHost, idcpu.begin(), idcpu.begin() + tile.numParticles(), host.begin());
				std::copy_if(host.begin(), host.end(), std::back_inserter(expectedOrder),
					     [](auto id) { return amrex::particle_impl::unpack_id(id) > 0; });
			}
		}
		quokka::ParticleContainer empty(schema, geom, dm, boxes);
		int checkpoint = 0;
		pp.query("checkpoint", checkpoint);
		quokka::ParticleContainer restored(schema, geoms, maps, grids, amrex::Vector<amrex::IntVect>{amrex::IntVect(2)});
		quokka::ParticleContainer restoredEmpty(schema, geom, dm, boxes);
		if (checkpoint != 0) {
			// write_checkpoint=0 reads an existing checkpoint, e.g. from the other precision.
			int writeCheckpoint = 1;
			pp.query("write_checkpoint", writeCheckpoint);
			if (writeCheckpoint != 0) {
				auto series = quokka::OpenPMDOutput::CreateCheckpointSeries("particle_checkpoint");
				auto iteration = series.iterations[3];
				quokka::OpenPMDOutput::SetCheckpointVersion(iteration);
				const auto nan = std::numeric_limits<double>::quiet_NaN();
				quokka::OpenPMDOutput::WriteParticles(series, iteration, particles, "test", {nan, nan, nan, nan}, true);
				quokka::OpenPMDOutput::WriteParticles(series, iteration, empty, "empty", {nan, nan, nan, nan}, true);
				series.close();
			}
			auto series = quokka::OpenPMDOutput::OpenCheckpointSeries("particle_checkpoint0000003.bp");
			auto iteration = quokka::OpenPMDOutput::CheckpointIteration(series);
			quokka::OpenPMDOutput::ReadCheckpointParticles(iteration, restored, "test");
			quokka::OpenPMDOutput::ReadCheckpointParticles(iteration, restoredEmpty, "empty");
			series.close();
			AMREX_ALWAYS_ASSERT(restoredEmpty.TotalNumberOfParticles() == 0);
			quokka::ParticleContainer::FinalizeRestarts();
			const auto next = quokka::ParticleContainer::ReserveIDs(1);
			AMREX_ALWAYS_ASSERT(next >= restored.RestartNextID() && next >= restoredEmpty.RestartNextID());
		}
#if openPMD_HAVE_ADIOS2
		const std::string path = "particle_openpmd_%T.bp";
#else
		const std::string path = "particle_openpmd_%T.json";
#endif
		{
#if defined(AMREX_USE_MPI) && openPMD_HAVE_ADIOS2
			openPMD::Series output(path, openPMD::Access::CREATE, amrex::ParallelDescriptor::Communicator());
#else
			openPMD::Series output(path, openPMD::Access::CREATE);
#endif
			auto iteration = output.iterations[0];
			quokka::OpenPMDOutput::WriteParticles(output, iteration, checkpoint != 0 ? restored : particles, "test", {0.001, 0.01, 1., 1.});
			quokka::OpenPMDOutput::WriteParticles(output, iteration, empty, "empty", {1., 1., 1., 1.});
			output.close();
		}
		amrex::ParallelDescriptor::Barrier();
		if (amrex::ParallelDescriptor::IOProcessor()) {
			openPMD::Series input(path, openPMD::Access::READ_ONLY);
			auto iteration = input.iterations[0];
			AMREX_ALWAYS_ASSERT(!iteration.particles.contains("empty"));
			auto species = iteration.particles["test"];
			const auto scalar = openPMD::RecordComponent::SCALAR;
			const auto count = (1U << AMREX_SPACEDIM) - (AMREX_SPACEDIM > 1 ? 2 : 1);
			AMREX_ALWAYS_ASSERT(species["id"][scalar].getExtent()[0] == count);
#if openPMD_HAVE_ADIOS2
			// The fine tile has half the particles; coarse tiles have one each.
			// A rank-sized write would collapse these into a larger chunk.
			const auto chunks = species["id"][scalar].availableChunks();
			AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM == 1 ? chunks.size() == 1 : chunks.size() > 1);
			std::vector<int> coverage(count, 0);
			for (const auto &chunk : chunks) {
				AMREX_ALWAYS_ASSERT(chunk.extent[0] > 0 && chunk.extent[0] <= (1U << (AMREX_SPACEDIM - 1)));
				AMREX_ALWAYS_ASSERT(chunk.offset[0] + chunk.extent[0] <= count);
				for (auto i = chunk.offset[0]; i < chunk.offset[0] + chunk.extent[0]; ++i) {
					++coverage[i];
				}
			}
			AMREX_ALWAYS_ASSERT(std::all_of(coverage.begin(), coverage.end(), [](int n) { return n == 1; }));
#endif
			const auto ids = species["id"][scalar].loadChunk<std::uint64_t>();
			const auto mass = species["mass"][scalar].loadChunk<amrex::ParticleReal>();
			const auto stage = species["stage"][scalar].loadChunk<int>();
			const auto levels = species["amr_level"][scalar].loadChunk<int>();
			std::array<std::shared_ptr<amrex::ParticleReal>, AMREX_SPACEDIM> positions;
			const std::array<std::string, 3> axes{"x", "y", "z"};
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				positions[d] = species["position"][axes[d]].loadChunk<amrex::ParticleReal>();
				AMREX_ALWAYS_ASSERT(species["positionOffset"][axes[d]].constant());
				AMREX_ALWAYS_ASSERT(species["position"][axes[d]].unitSI() == 0.01);
			}
			AMREX_ALWAYS_ASSERT(species["mass"][scalar].unitSI() == 0.001);
			input.flush();
			if (checkpoint == 0 && amrex::ParallelDescriptor::NProcs() == 1) {
				AMREX_ALWAYS_ASSERT(std::equal(expectedOrder.begin(), expectedOrder.end(), ids.get(), ids.get() + count));
			}
			std::set<std::uint64_t> unique;
			std::set<unsigned> cells;
			for (unsigned i = 0; i < count; ++i) {
				AMREX_ALWAYS_ASSERT(amrex::particle_impl::unpack_id(ids.get()[i]) > 0);
				unique.insert(ids.get()[i]);
				AMREX_ALWAYS_ASSERT(levels.get()[i] == (positions[0].get()[i] > 0.5 ? 1 : 0));
				unsigned cell = 0;
				for (int d = 0; d < AMREX_SPACEDIM; ++d) {
					const auto x = positions[d].get()[i];
					AMREX_ALWAYS_ASSERT(x == 0.25 || x == 0.75);
					if (x == 0.75) {
						cell |= 1U << d;
					}
				}
				AMREX_ALWAYS_ASSERT(cell != 0 && (AMREX_SPACEDIM == 1 || cell != 1));
				AMREX_ALWAYS_ASSERT(mass.get()[i] == 7. + cell && stage.get()[i] == 3 + static_cast<int>(cell));
				cells.insert(cell);
			}
			AMREX_ALWAYS_ASSERT(unique.size() == count && cells.size() == count);
			if (checkpoint != 0) {
				// Checkpoint species match plotfile species; empty populations have none.
				openPMD::Series saved("particle_checkpoint0000003.bp", openPMD::Access::READ_ONLY);
				AMREX_ALWAYS_ASSERT(!saved.iterations.at(3).particles.contains("empty"));
				auto savedSpecies = saved.iterations.at(3).particles.at("test");
				// openPMD-api cannot convert while loading, so compare raw records only when
				// the checkpoint has this build's precision. Restored values were checked above.
				if (savedSpecies["mass"][scalar].getDatatype() == openPMD::determineDatatype<amrex::ParticleReal>()) {
					const auto savedIDs = savedSpecies["id"][scalar].loadChunk<std::uint64_t>();
					const auto savedMass = savedSpecies["mass"][scalar].loadChunk<amrex::ParticleReal>();
					saved.flush();
					std::map<std::uint64_t, amrex::ParticleReal> identity;
					for (unsigned i = 0; i < count; ++i) {
						identity.emplace(savedIDs.get()[i], savedMass.get()[i]);
					}
					for (unsigned i = 0; i < count; ++i) {
						AMREX_ALWAYS_ASSERT(identity.at(ids.get()[i]) == mass.get()[i]);
					}
				}
			}
		}
	}
	amrex::Finalize();
}
