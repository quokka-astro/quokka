#ifndef QUOKKA_PARTICLE_TRACER_TEST_HPP_
#define QUOKKA_PARTICLE_TRACER_TEST_HPP_
#include "particle_test_records.hpp"
#include "particles/particle_schema_traits.hpp"
#include "particles/particle_tracer.hpp"
namespace particle_test
{
template <typename Problem> void CheckTracers(const amrex::Geometry &geom, const amrex::DistributionMapping &dm, const amrex::BoxArray &ba)
{
	const auto schema = quokka::ParticleSchemaTraits<quokka::ParticleType::Tracer, Problem>::Create();
	AMREX_ALWAYS_ASSERT(schema.Layout().numReal == AMREX_SPACEDIM && schema.Layout().numInt == 0);
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		AMREX_ALWAYS_ASSERT(schema.RealNames()[d] == "real_comp" + std::to_string(d));
	}
	quokka::ParticleContainer tracers(schema, geom, dm, ba);
	tracers.InitOnePerCell(0.5, 0.5, 0.5);
	// A different mesh decomposition exercises the velocity remapping path.
	const amrex::BoxArray velocityBoxes(geom.Domain());
	const amrex::DistributionMapping velocityMap(velocityBoxes);
	amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> velocity;
	const auto dx = geom.CellSizeArray();
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		velocity[d].define(amrex::convert(velocityBoxes, amrex::IntVect::TheDimensionVector(d)), velocityMap, 1, 2);
		for (amrex::MFIter mfi(velocity[d]); mfi.isValid(); ++mfi) {
			const auto array = velocity[d].array(mfi);
			amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const amrex::GpuArray<int, 3> cell{i, j, k};
				array(i, j, k) = 0.1 * (d + 1) + 0.2 * cell[d] * dx[d];
			});
		}
	}
	const auto before = quokka::particle_io::getParticleDataAtAllLevels(&tracers);
	constexpr amrex::Real dt = 0.01;
	quokka::AdvectTracers(tracers, velocity.data(), 0, dt);
	const auto after = quokka::particle_io::getParticleDataAtAllLevels(&tracers);
	AMREX_ALWAYS_ASSERT(std::get<0>(before) == std::get<0>(after));
	for (size_t i = 0; i < std::get<1>(before).size(); ++i) {
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			const auto x = std::get<1>(before)[i][d];
			const auto midpoint = x + 0.5 * dt * (0.1 * (d + 1) + 0.2 * x);
			const auto expectedVelocity = 0.1 * (d + 1) + 0.2 * midpoint;
			const auto tolerance = 64 * std::numeric_limits<amrex::ParticleReal>::epsilon();
			AMREX_ALWAYS_ASSERT(std::abs(std::get<1>(after)[i][d] - (x + dt * expectedVelocity)) < tolerance);
			AMREX_ALWAYS_ASSERT(std::abs(std::get<1>(after)[i][AMREX_SPACEDIM + d] - expectedVelocity) < tolerance);
		}
	}
	tracers.Checkpoint("schema_tracer_checkpoint", schema.Name());
	quokka::ParticleContainer restored(schema, geom, dm, ba);
	restored.Restart("schema_tracer_checkpoint", schema.Name());
	quokka::ParticleContainer::FinalizeRestarts();
	AMREX_ALWAYS_ASSERT(CaptureRecords(tracers) == CaptureRecords(restored));

	// Write the pre-migration tracer format with nonzero velocities, then read it
	// through the common wrapper without renaming or dropping components.
	amrex::TracerParticleContainer legacy(geom, dm, ba);
	amrex::TracerParticleContainer::ParticleInitData initial;
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		initial.real_struct_data[d] = 10 + d;
	}
	legacy.InitOnePerCell(0.5, 0.5, 0.5, initial);
	legacy.Checkpoint("schema_legacy_tracer_checkpoint", schema.Name());
	const auto legacyRecords = CaptureLegacyRecords(legacy);
	quokka::ParticleContainer migrated(schema, geom, dm, ba);
	migrated.Restart("schema_legacy_tracer_checkpoint", schema.Name());
	quokka::ParticleContainer::FinalizeRestarts();
	AMREX_ALWAYS_ASSERT(legacyRecords == CaptureRecords(migrated));
}
} // namespace particle_test
#endif
