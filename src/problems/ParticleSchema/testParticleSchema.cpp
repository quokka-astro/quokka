#include "AMReX.H"
#include "AMReX_Print.H"
#include "particle_tracer_test.hpp"
#include "particles/particle_IO.hpp"
#include "particles/particle_container.hpp"
#include "particles/particle_schema_traits.hpp"

#include <fstream>
#include <functional>
#include <stdexcept>
#include <type_traits>

struct SchemaProblem {};
template <> struct Physics_Traits<SchemaProblem> : DefaultPhysicsTraits {
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2;
};

struct SchemaExtraProblem {};
struct SchemaExtraModel : quokka::ToyStellarModel {
	static constexpr int nExtraReal = 3;
	static constexpr int nExtraInt = 2;
};
template <> struct Physics_Traits<SchemaExtraProblem> : DefaultPhysicsTraits {
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 4;
};
template <> struct Particle_Traits<SchemaExtraProblem> : DefaultParticleTraits {
	using stellar_model = SchemaExtraModel;
};

namespace
{
template <typename Container> consteval auto CheckRestrictedInterface() -> bool
{
	static_assert(!requires(Container & pc) { pc.AddRealComp("extra"); });
	static_assert(!requires(Container & pc) { pc.AddIntComp("extra"); });
	static_assert(!requires(Container & pc) { pc.ResizeRuntimeRealComp(1, true); });
	static_assert(!requires(Container & pc) { pc.ResizeRuntimeIntComp(1, true); });
	static_assert(!requires(Container & pc) { pc.SetArena(amrex::The_Arena()); });
	static_assert(!requires(Container & pc) { pc.addParticles(pc); });
	static_assert(!requires(Container & pc) { pc.copyParticles(pc); });
	static_assert(!requires(Container & pc) { pc.make_alike(); });
	static_assert(!requires(Container & pc) { pc.WriteAsciiFile("particles"); });
	static_assert(!requires(Container & pc) { pc.Restart("checkpoint", "population", true); });
	return true;
}
static_assert(CheckRestrictedInterface<quokka::ParticleContainer>());

// Exercise the production always-on host diagnostics without terminating MPI.
// AMReX's exception mode changes error delivery, not the assertion conditions.
void RequireFatalRejected(const std::function<void()> &operation, const std::string &message)
{
	struct ExceptionMode {
		bool previous = amrex::system::throw_exception;
		ExceptionMode() { amrex::system::throw_exception = true; }
		~ExceptionMode() { amrex::system::throw_exception = previous; }
	};
	bool rejected = false;
	{
		const ExceptionMode mode;
		try {
			operation();
		} catch (const amrex::RuntimeError &error) {
			rejected = std::string(error.what()).find(message) != std::string::npos;
		}
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rejected, "Expected particle lifecycle diagnostic was not raised");
}

void RequireRejected(const std::function<void()> &function)
{
	bool rejected = false;
	try {
		function();
	} catch (const std::invalid_argument &) {
		rejected = true;
	}
	AMREX_ALWAYS_ASSERT(rejected);
}

auto SchemaWithDefaults() -> quokka::ParticleSchema
{
	auto base = quokka::ParticleSchemaTraits<quokka::ParticleType::Test, SchemaProblem>::Create();
	auto real = base.RealDefaults();
	auto integer = base.IntDefaults();
	real[base.Layout().mass] = 17;
	real[base.Layout().deathTime] = 91;
	integer[base.Layout().evolutionStage] = 3;
	return {base.Population(), base.Name(), base.RealNames(), base.IntNames(), real, integer, base.RealUnitsKeys(), base.Layout(), base.UnitDefinitions()};
}

// No problem_t or population-specific C++ type is needed for introspection.
void CheckIntrospection(const quokka::ParticleContainer &first, const quokka::ParticleContainer &second)
{
	AMREX_ALWAYS_ASSERT(first.Schema().Name() == "Test_particles");
	AMREX_ALWAYS_ASSERT(second.Schema().Name() == "CIC_particles");
	AMREX_ALWAYS_ASSERT(first.Schema().RealNames().size() == 8);
	AMREX_ALWAYS_ASSERT(second.Schema().RealNames().size() == 4);
	AMREX_ALWAYS_ASSERT(first.Schema().IntNames().size() == 1);
	AMREX_ALWAYS_ASSERT(second.Schema().IntNames().empty());
	AMREX_ALWAYS_ASSERT(first.Schema().RealUnitsKeys()[first.Schema().RealIndex("mass")] == "mass");
	AMREX_ALWAYS_ASSERT(first.Schema().RealUnitsKeys()[first.Schema().RealIndex("luminosity_1")] == "luminosity");
	AMREX_ALWAYS_ASSERT(first.Layout().RealStorage(0) == AMREX_SPACEDIM);
	AMREX_ALWAYS_ASSERT(first.Layout().IntStorage(0) == 0);
}

void CheckValues(const quokka::ParticleContainer &pc, amrex::Long count, amrex::ParticleReal mass)
{
	AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == count);
	const auto layout = pc.Layout();
	amrex::ReduceOps<amrex::ReduceOpSum> op;
	amrex::ReduceData<int> result(op);
	using Tuple = decltype(result)::Type;
	for (int lev = 0; lev <= pc.finestLevel(); ++lev) {
		for (const auto &[key, tile] : pc.GetParticles(lev)) {
			amrex::ignore_unused(key);
			AMREX_ALWAYS_ASSERT(tile.arena() == pc.arena());
			AMREX_ALWAYS_ASSERT(tile.GetRealData("mass").data() == tile.GetRealData(AMREX_SPACEDIM).data());
			const auto data = tile.getConstParticleTileData();
			op.eval(tile.numParticles(), result, [=] AMREX_GPU_DEVICE(int i) -> Tuple {
				const quokka::ParticleView p(data, i, layout);
				int bad = p.rdata(layout.mass) != mass || p.rdata(layout.deathTime) != 91 || p.idata(layout.evolutionStage) != 3;
				for (int d = 0; d < AMREX_SPACEDIM; ++d) {
					bad += p.pos(d) < 0 || p.pos(d) >= 1;
				}
				bad += !p.id().is_valid();
				return {bad};
			});
		}
	}
	AMREX_ALWAYS_ASSERT(amrex::get<0>(result.value(op)) == 0);
}
// Exercise full nonzero payloads for each runtime schema, including zero int fields.
template <quokka::ParticleType population, typename Problem = SchemaProblem>
void CheckPopulationIO(const amrex::Geometry &geom, const amrex::DistributionMapping &dm, const amrex::BoxArray &ba, bool multilevel = false)
{
	amrex::Vector<amrex::Geometry> geoms{geom};
	amrex::Vector<amrex::DistributionMapping> maps{dm};
	amrex::Vector<amrex::BoxArray> boxes{ba};
	const amrex::Vector<amrex::IntVect> ratios{amrex::IntVect(2)};
	if (multilevel) {
		geoms.push_back(amrex::refine(geom, 2));
		boxes.emplace_back(amrex::Box(amrex::IntVect(0), amrex::IntVect(3)));
		maps.emplace_back(boxes.back());
	}
	const amrex::Long count = multilevel ? 128 : 3;
	const auto schema = quokka::ParticleSchemaTraits<population, Problem>::Create();
	quokka::ParticleContainer original(schema, geoms, maps, boxes, ratios);
	quokka::ParticleContainer::ParticleInitData initial;
	for (int c = 0; c < schema.Layout().numReal; ++c) {
		initial.real_array_data.push_back(1.25 + c);
	}
	for (int c = 0; c < schema.Layout().numInt; ++c) {
		initial.int_array_data.push_back(17 + c);
	}
	original.InitRandom(count, 123, initial);
	if (multilevel) {
		AMREX_ALWAYS_ASSERT(original.NumberOfParticlesAtLevel(0) > 0 && original.NumberOfParticlesAtLevel(1) > 0);
		// Reflect positions across the domain to force transfer between AMR levels.
		for (int lev = 0; lev <= original.finestLevel(); ++lev) {
			for (auto &[key, tile] : original.GetParticles(lev)) {
				amrex::ignore_unused(key);
				const auto data = tile.getParticleTileData();
				amrex::ParallelFor(tile.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
					for (int d = 0; d < AMREX_SPACEDIM; ++d) {
						data.pos(d, i) = 1 - data.pos(d, i);
					}
				});
			}
		}
		const auto beforeRedistribution = particle_test::CaptureRecords(original);
		amrex::Long expectedFine = 0;
		for (const auto &row : beforeRedistribution) {
			bool inFine = true;
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				inFine = inFine && std::get<2>(row)[d] < 0.5;
			}
			expectedFine += inFine;
		}
		amrex::ParallelDescriptor::Bcast(&expectedFine, 1, amrex::ParallelDescriptor::IOProcessorNumber());
		original.Redistribute();
		AMREX_ALWAYS_ASSERT(original.NumberOfParticlesAtLevel(1) == expectedFine);
		AMREX_ALWAYS_ASSERT(original.NumberOfParticlesAtLevel(0) == count - expectedFine);
		AMREX_ALWAYS_ASSERT(beforeRedistribution == particle_test::CaptureRecords(original));
	}

	const auto checkpoint = "schema_all_" + schema.Name() + "_" + std::to_string(schema.Layout().numReal) + "_" + std::to_string(schema.Layout().numInt) +
				(multilevel ? "_amr" : "");
	original.Checkpoint(checkpoint, schema.Name());
	original.WritePlotFile(checkpoint + "_plot", schema.Name());
	const auto header = amrex::ParticleHeader::read(checkpoint + "_plot", schema.Name());
	AMREX_ALWAYS_ASSERT(header.real_comp_names == schema.RealNames() && header.int_comp_names == schema.IntNames());
	quokka::ParticleContainer restored(schema, geoms, maps, boxes, ratios);
	restored.Restart(checkpoint, schema.Name());
	quokka::ParticleContainer::FinalizeRestarts();
	AMREX_ALWAYS_ASSERT(restored.TotalNumberOfParticles() == count);
	AMREX_ALWAYS_ASSERT(particle_test::CaptureRecords(original) == particle_test::CaptureRecords(restored));
	quokka::ParticleContainer empty(schema, geoms, maps, boxes, ratios);
	empty.Checkpoint(checkpoint + "_empty", schema.Name());
	quokka::ParticleContainer emptyRestored(schema, geoms, maps, boxes, ratios);
	emptyRestored.Restart(checkpoint + "_empty", schema.Name());
	quokka::ParticleContainer::FinalizeRestarts();
	AMREX_ALWAYS_ASSERT(emptyRestored.TotalNumberOfParticles() == 0);

	constexpr int legacyReals = [] {
		using namespace quokka;
		if constexpr (population == ParticleType::Rad) {
			return RadParticleRealComps<Problem>;
		} else if constexpr (population == ParticleType::CIC) {
			return CICParticleRealComps;
		} else if constexpr (population == ParticleType::CICRad) {
			return CICRadParticleRealComps<Problem>;
		} else if constexpr (population == ParticleType::Test) {
			return TestParticleRealComps<Problem>;
		} else if constexpr (population == ParticleType::Sink) {
			return SinkParticleRealComps;
		} else if constexpr (population == ParticleType::Star) {
			return StarParticleRealComps<Problem>;
		} else if constexpr (population == ParticleType::StochasticStellarPop) {
			return StochasticStellarPopParticleRealComps<Problem>;
		} else {
			return AMREX_SPACEDIM;
		}
	}();
	constexpr int legacyInts = [] {
		using namespace quokka;
		if constexpr (population == ParticleType::Test || population == ParticleType::StochasticStellarPop) {
			return 1;
		} else if constexpr (population == ParticleType::Star) {
			return StarParticleIntComps<Problem>;
		} else {
			return 0;
		}
	}();
	AMREX_ALWAYS_ASSERT(legacyReals == schema.Layout().numReal && legacyInts == schema.Layout().numInt);
	amrex::ParticleContainer<legacyReals, legacyInts> legacy(geoms, maps, boxes, ratios);
	typename decltype(legacy)::ParticleInitData legacyInitial;
	std::copy(initial.real_array_data.begin(), initial.real_array_data.end(), legacyInitial.real_struct_data.begin());
	std::copy(initial.int_array_data.begin(), initial.int_array_data.end(), legacyInitial.int_struct_data.begin());
	legacy.InitRandom(count, 456, legacyInitial);
	legacy.Checkpoint(checkpoint + "_legacy", schema.Name(), true, schema.RealNames(), schema.IntNames());
	quokka::ParticleContainer migrated(schema, geoms, maps, boxes, ratios);
	migrated.Restart(checkpoint + "_legacy", schema.Name());
	quokka::ParticleContainer::FinalizeRestarts();
	AMREX_ALWAYS_ASSERT(particle_test::CaptureLegacyRecords(legacy) == particle_test::CaptureRecords(migrated));

	const auto layout = restored.Layout();
	amrex::ReduceOps<amrex::ReduceOpSum> op;
	amrex::ReduceData<int> result(op);
	using Tuple = decltype(result)::Type;
	for (int lev = 0; lev <= restored.finestLevel(); ++lev) {
		for (const auto &[key, tile] : restored.GetParticles(lev)) {
			amrex::ignore_unused(key);
			const auto data = tile.getConstParticleTileData();
			op.eval(tile.numParticles(), result, [=] AMREX_GPU_DEVICE(int i) -> Tuple {
				const quokka::ParticleView p(data, i, layout);
				int bad = 0;
				for (int c = 0; c < layout.numReal; ++c) {
					bad += p.rdata(c) != 1.25 + c;
				}
				for (int c = 0; c < layout.numInt; ++c) {
					bad += p.idata(c) != 17 + c;
				}
				return {bad};
			});
		}
	}
	AMREX_ALWAYS_ASSERT(amrex::get<0>(result.value(op)) == 0);
}
} // namespace

int main(int argc, char **argv)
{
	amrex::Initialize(argc, argv);
	{
		const auto schema = SchemaWithDefaults();
		// Construct every population's schema, including retained tracer velocities.
		const auto checkSchema = []<quokka::ParticleType population>() {
			const auto candidate = quokka::ParticleSchemaTraits<population, SchemaProblem>::Create();
			for (int i = 0; i < candidate.Layout().numReal; ++i) {
				AMREX_ALWAYS_ASSERT(candidate.RealIndex(candidate.RealNames()[i]) == i);
				AMREX_ALWAYS_ASSERT(candidate.Layout().RealStorage(i) == AMREX_SPACEDIM + i);
			}
			for (int i = 0; i < candidate.Layout().numInt; ++i) {
				AMREX_ALWAYS_ASSERT(candidate.IntIndex(candidate.IntNames()[i]) == i);
			}
		};
		checkSchema.template operator()<quokka::ParticleType::Rad>();
		checkSchema.template operator()<quokka::ParticleType::CIC>();
		checkSchema.template operator()<quokka::ParticleType::CICRad>();
		checkSchema.template operator()<quokka::ParticleType::StochasticStellarPop>();
		checkSchema.template operator()<quokka::ParticleType::Sink>();
		checkSchema.template operator()<quokka::ParticleType::Test>();
		checkSchema.template operator()<quokka::ParticleType::Star>();
		checkSchema.template operator()<quokka::ParticleType::Tracer>();
		const auto invalid = [&](amrex::Vector<std::string> names, quokka::ParticleLayout layout, amrex::Vector<amrex::ParticleReal> defaults) {
			const quokka::ParticleSchema bad(schema.Population(), schema.Name(), std::move(names), schema.IntNames(), std::move(defaults),
							 schema.IntDefaults(), schema.RealUnitsKeys(), layout, schema.UnitDefinitions());
		};
		auto names = schema.RealNames();
		names[1] = names[0];
		RequireRejected([&]() { invalid(names, schema.Layout(), schema.RealDefaults()); });
		names[1] = "";
		RequireRejected([&]() { invalid(names, schema.Layout(), schema.RealDefaults()); });
		auto layout = schema.Layout();
		layout.mass = layout.numReal;
		RequireRejected([&]() { invalid(schema.RealNames(), layout, schema.RealDefaults()); });
		layout.mass = -2;
		RequireRejected([&]() { invalid(schema.RealNames(), layout, schema.RealDefaults()); });
		RequireRejected([&]() { invalid(schema.RealNames(), schema.Layout(), {}); });

		const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(3));
		amrex::RealBox bounds;
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			bounds.setLo(d, 0);
			bounds.setHi(d, 1);
		}
		const int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
		const amrex::Geometry geom(domain, &bounds, 0, periodic);
		amrex::BoxArray ba(domain);
		ba.maxSize(2);
		const amrex::DistributionMapping dm(ba);
		CheckPopulationIO<quokka::ParticleType::Rad>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::CIC>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::CICRad>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Test>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Sink>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Star>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::StochasticStellarPop>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Tracer>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Star, SchemaExtraProblem>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Rad, SchemaExtraProblem>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::CICRad, SchemaExtraProblem>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Test, SchemaExtraProblem>(geom, dm, ba);
		CheckPopulationIO<quokka::ParticleType::Rad>(geom, dm, ba, true);
		CheckPopulationIO<quokka::ParticleType::CIC>(geom, dm, ba, true);
		CheckPopulationIO<quokka::ParticleType::CICRad>(geom, dm, ba, true);
		CheckPopulationIO<quokka::ParticleType::Test>(geom, dm, ba, true);
		CheckPopulationIO<quokka::ParticleType::Sink>(geom, dm, ba, true);
		CheckPopulationIO<quokka::ParticleType::Star>(geom, dm, ba, true);
		CheckPopulationIO<quokka::ParticleType::StochasticStellarPop>(geom, dm, ba, true);
		CheckPopulationIO<quokka::ParticleType::Tracer>(geom, dm, ba, true);

		particle_test::CheckTracers<SchemaProblem>(geom, dm, ba);
		quokka::ParticleContainer pc(schema, geom, dm, ba);
		quokka::ParticleContainer cic(quokka::ParticleSchemaTraits<quokka::ParticleType::CIC, SchemaProblem>::Create(), geom, dm, ba);
		CheckIntrospection(pc, cic);
		// A custom instance unit definition must govern output, not population globals.
		auto customUnits = schema.UnitDefinitions();
		customUnits["mass"] = {9, 8, 7, 6};
		const quokka::ParticleSchema customSchema(schema.Population(), schema.Name(), schema.RealNames(), schema.IntNames(), schema.RealDefaults(),
							  schema.IntDefaults(), schema.RealUnitsKeys(), schema.Layout(), customUnits);
		quokka::ParticleContainer custom(customSchema, geom, dm, ba);
		custom.Checkpoint("schema_custom_units", customSchema.Name());
		quokka::particle_io::writeUnitsFile(&custom, "schema_custom_units", customSchema.Name());
		if (amrex::ParallelDescriptor::IOProcessor()) {
			std::ifstream unitsFile("schema_custom_units/" + customSchema.Name() + "/Fields.yaml");
			const std::string content((std::istreambuf_iterator<char>(unitsFile)), std::istreambuf_iterator<char>());
			AMREX_ALWAYS_ASSERT(content.find("mass: [9, 8, 7, 6]\n") != std::string::npos);
		}
		customUnits.erase("mass");
		RequireRejected([&] {
			const quokka::ParticleSchema bad(schema.Population(), schema.Name(), schema.RealNames(), schema.IntNames(), schema.RealDefaults(),
							 schema.IntDefaults(), schema.RealUnitsKeys(), schema.Layout(), customUnits);
		});

		pc.InitRandom(13, 42);
		CheckValues(pc, 13, 17);
		const auto [gatheredIDs, gatheredReals, gatheredInts] = quokka::particle_io::getParticleDataAtAllLevels(&pc);
		if (amrex::ParallelDescriptor::IOProcessor()) {
			AMREX_ALWAYS_ASSERT(gatheredIDs.size() == 13 && gatheredReals.size() == 13 && gatheredInts.size() == 13);
			for (size_t i = 0; i < gatheredIDs.size(); ++i) {
				AMREX_ALWAYS_ASSERT(gatheredIDs[i] > 0);
				AMREX_ALWAYS_ASSERT(gatheredReals[i][AMREX_SPACEDIM + schema.Layout().mass] == 17);
				AMREX_ALWAYS_ASSERT(gatheredInts[i][schema.Layout().evolutionStage] == 3);
			}
		} else {
			AMREX_ALWAYS_ASSERT(gatheredIDs.empty() && gatheredReals.empty() && gatheredInts.empty());
		}
		// One contributing rank, invalid records, and positions outside the mesh:
		// gathering must preserve field values without redistributing the source.
		quokka::ParticleContainer diagnostic(schema, geom, dm, ba);
		if (amrex::ParallelDescriptor::IOProcessor()) {
			auto &tile = diagnostic.DefineAndReturnParticleTile(0, 0, 0);
			diagnostic.ResizeAndInitialize(tile, 2);
			const auto data = tile.getParticleTileData();
			const auto diagnosticLayout = diagnostic.Layout();
			amrex::ParallelFor(2, [=] AMREX_GPU_DEVICE(int i) {
				const auto p = quokka::ParticleView(data, i, diagnosticLayout);
				p.id() = i == 0 ? 37 : -1;
				for (int d = 0; d < AMREX_SPACEDIM; ++d) {
					p.pos(d) = -100 - d;
				}
				for (int c = 0; c < diagnosticLayout.numReal; ++c) {
					p.rdata(c) = 200 + c;
				}
				for (int c = 0; c < diagnosticLayout.numInt; ++c) {
					p.idata(c) = 300 + c;
				}
			});
		}
		const auto [diagnosticIDs, diagnosticReals, diagnosticInts] = quokka::particle_io::getParticleDataAtAllLevels(&diagnostic);
		const auto [levelReals, levelInts] = quokka::particle_io::getParticleDataAtLevel(&diagnostic, 0);
		AMREX_ALWAYS_ASSERT(levelReals == diagnosticReals && levelInts == diagnosticInts);
		if (amrex::ParallelDescriptor::IOProcessor()) {
			AMREX_ALWAYS_ASSERT(diagnosticIDs.size() == 1 && diagnosticIDs[0] == 37);
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				AMREX_ALWAYS_ASSERT(diagnosticReals[0][d] == -100 - d);
			}
			for (int c = 0; c < schema.Layout().numReal; ++c) {
				AMREX_ALWAYS_ASSERT(diagnosticReals[0][AMREX_SPACEDIM + c] == 200 + c);
			}
			AMREX_ALWAYS_ASSERT(diagnosticInts[0][0] == 300);
		}
		AMREX_ALWAYS_ASSERT(diagnostic.TotalNumberOfParticles(false) == 2);
		const auto [emptyIDs, emptyReals, emptyInts] = quokka::particle_io::getParticleDataAtAllLevels(&cic);
		AMREX_ALWAYS_ASSERT(emptyIDs.empty() && emptyReals.empty() && emptyInts.empty());
		pc.Checkpoint("rtsoa_checkpoint", schema.Name());
		RequireFatalRejected([&]() { pc.Restart("rtsoa_checkpoint", schema.Name()); }, "Particle restart requires an empty destination");
		RequireFatalRejected([&]() { cic.Restart("rtsoa_checkpoint", schema.Name()); }, "Particle restart field names/order");
		AMREX_ALWAYS_ASSERT(!quokka::ParticleContainer::RestartsPending());
		quokka::ParticleContainer restored(schema, geom, dm, ba);
		restored.Restart("rtsoa_checkpoint", schema.Name());
		AMREX_ALWAYS_ASSERT(quokka::ParticleContainer::RestartsPending());
		const std::string pendingMessage = "Finalize all particle restarts before creating particles";
		RequireFatalRejected([&]() { static_cast<void>(quokka::ParticleContainer::ReserveIDs(1)); }, pendingMessage);
		RequireFatalRejected([&]() { cic.InitRandom(1, 42); }, pendingMessage);
		RequireFatalRejected([&]() { cic.InitOnePerCell(0.5, 0.5, 0.5); }, pendingMessage);
		RequireFatalRejected([&]() { cic.InitFromAsciiFile("must_not_be_opened.txt", 0); }, pendingMessage);
		quokka::ParticleContainer::ParticleTileType uninitializedTile;
		RequireFatalRejected([&]() { cic.ResizeAndInitialize(uninitializedTile, 1); }, pendingMessage);
		AMREX_ALWAYS_ASSERT(cic.TotalNumberOfParticles(false) == 0);
		quokka::ParticleContainer::FinalizeRestarts();
		AMREX_ALWAYS_ASSERT(!quokka::ParticleContainer::RestartsPending());
		CheckValues(restored, 13, 17);
		restored.WritePlotFile("rtsoa_plot", schema.Name());
		const auto header = amrex::ParticleHeader::read("rtsoa_plot", schema.Name());
		AMREX_ALWAYS_ASSERT(header.real_comp_names == schema.RealNames());
		AMREX_ALWAYS_ASSERT(header.int_comp_names == schema.IntNames());

		quokka::ParticleContainer fromAscii(schema, geom, dm, ba);
		if (amrex::ParallelDescriptor::IOProcessor()) {
			std::ofstream file("schema_particles.txt");
			file << "1\n";
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				file << "0.25 ";
			}
			file << "23\n";
		}
		amrex::ParallelDescriptor::Barrier();
		fromAscii.InitFromAsciiFile("schema_particles.txt", 1);
		CheckValues(fromAscii, 1, 23);

		// Exercise the legacy AoS writer and the RTSoA reader with equivalent
		// field order. Archived production checkpoints remain a separate gate.
		amrex::ParticleContainer<8, 1> legacy(geom, dm, ba);
		decltype(legacy)::ParticleInitData legacyDefaults;
		std::copy(schema.RealDefaults().begin(), schema.RealDefaults().end(), legacyDefaults.real_struct_data.begin());
		std::copy(schema.IntDefaults().begin(), schema.IntDefaults().end(), legacyDefaults.int_struct_data.begin());
		legacy.InitRandom(11, 93, legacyDefaults);
		legacy.Checkpoint("legacy_checkpoint", schema.Name(), true, schema.RealNames(), schema.IntNames());
		quokka::ParticleContainer fromLegacy(schema, geom, dm, ba);
		fromLegacy.Restart("legacy_checkpoint", schema.Name());
		quokka::ParticleContainer::FinalizeRestarts();
		CheckValues(fromLegacy, 11, 17);

		// A checkpoint header alone is insufficient: validate against restored
		// records, across ranks, even when a later population has a lower header.
		cic.InitRandom(19, 73);
		amrex::Long expectedNext = 1;
		for (auto &[key, tile] : cic.GetParticles(0)) {
			amrex::ignore_unused(key);
			const auto data = tile.getParticleTileData();
			const amrex::Long baseID = 10000 + 1000 * amrex::ParallelDescriptor::MyProc() + 100 * key.first;
			amrex::ParallelFor(tile.numParticles(), [=] AMREX_GPU_DEVICE(int i) noexcept { data.id(i) = baseID + i; });
			if (tile.numParticles() > 0) {
				expectedNext = std::max(expectedNext, baseID + tile.numParticles());
			}
		}
		amrex::ParallelDescriptor::ReduceLongMax(expectedNext);
		quokka::ParticleContainer::ParticleType::NextID(700);
		cic.Checkpoint("rtsoa_ids", cic.Schema().Name());
		quokka::ParticleContainer empty(schema, geom, dm, ba);
		quokka::ParticleContainer::ParticleType::NextID(900);
		empty.Checkpoint("rtsoa_ids", empty.Schema().Name());
		for (bool reverse : {false, true}) {
			quokka::ParticleContainer first(cic.Schema(), geom, dm, ba);
			quokka::ParticleContainer second(schema, geom, dm, ba);
			quokka::ParticleContainer::ParticleType::NextID(1);
			if (reverse) {
				second.Restart("rtsoa_ids", second.Schema().Name());
				first.Restart("rtsoa_ids", first.Schema().Name());
			} else {
				first.Restart("rtsoa_ids", first.Schema().Name());
				second.Restart("rtsoa_ids", second.Schema().Name());
			}
			quokka::ParticleContainer::FinalizeRestarts();
			const auto next = quokka::ParticleContainer::ParticleType::NextID();
			AMREX_ALWAYS_ASSERT(next == expectedNext);
			second.InitRandom(3, 81);
			CheckValues(second, 3, 17);
		}
		amrex::Print() << "Particle schema, defaults, logical indexing, introspection and I/O checks passed\n";
	}
	amrex::Finalize();
}
