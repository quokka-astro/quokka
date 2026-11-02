// Optional migration benchmark. The legacy branch is used only when this file
// is copied into an isolated pre-migration source tree for paired measurements.
#include "AMReX.H"
#include "AMReX_ParmParse.H"
#include "particles/PhysicsParticles.hpp"
#include <array>
#if __has_include("particles/particle_container.hpp")
#include "particles/particle_schema_traits.hpp"
#define QUOKKA_BENCHMARK_RTSOA
#endif

struct ParticleBenchmarkProblem {};
struct SinkBenchmarkProblem {};
template <> struct Physics_Traits<SinkBenchmarkProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
};
template <> struct quokka::EOS_Traits<SinkBenchmarkProblem> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0e5;
	static constexpr double mean_molecular_weight = C::m_p;
};
template <> struct HydroSystem_Traits<SinkBenchmarkProblem> {
	static constexpr bool reconstruct_eint = false;
};
struct LuminosityBenchmarkProblem {};
template <> struct Physics_Traits<LuminosityBenchmarkProblem> : DefaultPhysicsTraits {
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2;
};

namespace
{
#ifdef QUOKKA_BENCHMARK_RTSOA
using Container = quokka::ParticleContainer;
#else
using Container = quokka::CICParticleContainer;
#endif

auto MakeContainer(const amrex::Geometry &geom, const amrex::DistributionMapping &dm, const amrex::BoxArray &ba) -> std::unique_ptr<Container>
{
#ifdef QUOKKA_BENCHMARK_RTSOA
	return std::make_unique<Container>(quokka::ParticleSchemaTraits<quokka::ParticleType::CIC, ParticleBenchmarkProblem>::Create(), geom, dm, ba);
#else
	return std::make_unique<Container>(amrex::Vector<amrex::Geometry>{geom}, amrex::Vector<amrex::DistributionMapping>{dm},
					   amrex::Vector<amrex::BoxArray>{ba}, amrex::Vector<int>{});
#endif
}

template <typename F> void Measure(const std::string &name, int iterations, const F &function)
{
	amrex::Gpu::streamSynchronize();
	amrex::ParallelDescriptor::Barrier();
	BL_PROFILE(name); // NOLINT(misc-const-correctness)
	for (int i = 0; i < iterations; ++i) {
		function(i);
	}
	amrex::Gpu::streamSynchronize();
}

// Validate mass, velocity, and the translated cell-center lattice independently
// of the production drift/kick implementation and the particle ordering.
void CheckParticles(Container &pc, amrex::Long count, int cells, amrex::Real mass, amrex::Real displacement, amrex::Real velocity)
{
	AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == count);
	amrex::ReduceOps<amrex::ReduceOpMax> op;
	amrex::ReduceData<amrex::Real> result(op);
	using Tuple = decltype(result)::Type;
	for (Container::ParConstIterType iter(pc, 0); iter.isValid(); ++iter) {
		const auto data = iter.GetParticleTile().getConstParticleTileData();
#ifdef QUOKKA_BENCHMARK_RTSOA
		const auto layout = pc.Layout();
#endif
		op.eval(iter.numParticles(), result, [=] AMREX_GPU_DEVICE(int i) -> Tuple {
#ifdef QUOKKA_BENCHMARK_RTSOA
			const auto p = quokka::ParticleView(data, i, layout);
#else
			const auto &p = data.m_aos[i];
#endif
			amrex::Real error = std::abs(p.rdata(0) - mass);
			if (!p.id().is_valid() || !std::isfinite(p.rdata(0))) {
				error = 1.0;
			}
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				if (!std::isfinite(p.rdata(1 + d)) || !std::isfinite(p.pos(d))) {
					return {1.0};
				}
				error = amrex::max(error, std::abs(p.rdata(1 + d) - velocity));
				const amrex::Real index = (p.pos(d) - displacement) * cells - 0.5;
				error = amrex::max(error, std::abs(index - std::round(index)));
			}
			return {error};
		});
	}
	auto error = amrex::get<0>(result.value(op));
	amrex::ParallelDescriptor::ReduceRealMax(error);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error < 1.0e-10, "Particle benchmark analytic validation failed");
}

// Keep sinks away from physical boundaries so the conservation check accounts
// for all accreted gas without a boundary flux. Overlapping stencils exercise
// the production atomic deposits and per-cell accretion limiting.
void BenchmarkAccretion(int cells, int iterations)
{
	using Hydro = HydroSystem<SinkBenchmarkProblem>;
	const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
	const amrex::RealBox bounds({0., 0., 0.}, {1.e20, 1.e20, 1.e20});
	const std::array<int, 3> periodic = {1, 1, 1};
	const amrex::Geometry geom(domain, &bounds, 0, periodic.data());
	const amrex::BoxArray ba(domain);
	const amrex::DistributionMapping dm(ba);
#ifdef QUOKKA_BENCHMARK_RTSOA
	using Sinks = quokka::ParticleContainer;
	Sinks sinks(quokka::ParticleSchemaTraits<quokka::ParticleType::Sink, SinkBenchmarkProblem>::Create(), geom, dm, ba);
#else
	using Sinks = quokka::SinkParticleContainer;
	Sinks sinks(amrex::Vector<amrex::Geometry>{geom}, amrex::Vector<amrex::DistributionMapping>{dm}, amrex::Vector<amrex::BoxArray>{ba},
		    amrex::Vector<int>{});
#endif
	Sinks::ParticleInitData initial;
#ifdef QUOKKA_BENCHMARK_RTSOA
	initial.real_array_data.assign(sinks.Layout().numReal, 0.0);
	initial.int_array_data.assign(sinks.Layout().numInt, 0);
	auto &reals = initial.real_array_data;
#else
	auto &reals = initial.real_struct_data;
#endif
	constexpr amrex::Real sinkMass = 10.0 * C::M_solar;
	constexpr amrex::Real rho = 1.e-20;
	constexpr amrex::Real dt = 1.e10;
	reals[quokka::SinkParticleMassIdx] = sinkMass;
	const amrex::Long count = domain.numPts() / 64;
	const amrex::RealBox interior({2.5e19, 2.5e19, 2.5e19}, {7.5e19, 7.5e19, 7.5e19});
	sinks.InitRandom(count, 42, initial, true, interior);
	// Use an identical lattice in both layouts, independent of initializer RNG
	// consumption. This benchmark runs first, so its IDs start at one.
	const int side = cells / 4;
	AMREX_ALWAYS_ASSERT(cells % 4 == 0);
	for (Sinks::ParIterType iter(sinks, 0); iter.isValid(); ++iter) {
		const auto data = iter.GetParticleTile().getParticleTileData();
		amrex::ParallelFor(iter.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
#ifdef QUOKKA_BENCHMARK_RTSOA
			auto p = data[i];
#else
			auto &p = data.m_aos[i];
#endif
			amrex::Long index = p.id();
			--index;
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				p.pos(d) = 2.5e19 + (static_cast<amrex::Real>(index % side) + 0.5) * (5.e19 / side);
				index /= side;
			}
		});
	}
	sinks.Redistribute();
	amrex::MultiFab state(ba, dm, Hydro::nvar_, 3);
	amrex::MultiFab rate(ba, dm, Physics_NumVars::numHydroVars + 1, 3);
	amrex::MultiFab floor(ba, dm, 1, 3);
	state.setVal(0.0);
	state.setVal(rho, Hydro::density_index, 1, 3);
	state.setVal(rho * 1.e10, Hydro::energy_index, 1, 3);
	state.setVal(rho * 1.e10, Hydro::internalEnergy_index, 1, 3);
	floor.setVal(1.e-30);
	for (int repeat = 0; repeat < iterations; ++repeat) {
		state.FillBoundary(geom.periodicity());
		rate.setVal(0.0);
		Measure("ParticleBenchmark::accretion", 1, [&](int) {
			quokka::SinkAccretionUtils::computeAccretion<Sinks, SinkBenchmarkProblem>(&sinks, state, rate, nullptr, 0, repeat * dt, dt,
												  quokka::SinkParticleMassIdx);
			quokka::SinkAccretionUtils::applyAccretion<Sinks, SinkBenchmarkProblem>(&sinks, state, rate, nullptr, geom, 0, repeat * dt, dt,
												quokka::SinkParticleMassIdx, floor);
		});
	}
	AMREX_ALWAYS_ASSERT(sinks.TotalNumberOfParticles() == count);
	amrex::ReduceOps<amrex::ReduceOpSum> op;
	amrex::ReduceData<amrex::Real> result(op);
	using Tuple = decltype(result)::Type;
	for (Sinks::ParConstIterType iter(sinks, 0); iter.isValid(); ++iter) {
		const auto data = iter.GetParticleTile().getConstParticleTileData();
#ifdef QUOKKA_BENCHMARK_RTSOA
		const auto layout = sinks.Layout();
#endif
		op.eval(iter.numParticles(), result, [=] AMREX_GPU_DEVICE(int i) -> Tuple {
#ifdef QUOKKA_BENCHMARK_RTSOA
			const auto p = quokka::ParticleView(data, i, layout);
#else
			const auto &p = data.m_aos[i];
#endif
			return {p.rdata(quokka::SinkParticleMassIdx) - sinkMass};
		});
	}
	auto gained = amrex::get<0>(result.value(op));
	amrex::ParallelDescriptor::ReduceRealSum(gained);
	// Subtract the uniform initial density cellwise before reducing, avoiding
	// cancellation against the much larger total gas mass.
	state.plus(-rho, Hydro::density_index, 1, 0);
	const auto dx = geom.CellSizeArray();
	const auto lost = -state.sum(Hydro::density_index) * dx[0] * dx[1] * dx[2];
	AMREX_ALWAYS_ASSERT(std::isfinite(gained) && std::isfinite(lost) && gained > 0.0 && lost > 0.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(gained - lost) < 1.e-8 * gained, "Sink benchmark violates gas-particle mass conservation");
	amrex::Print() << "Sink benchmark: particles=" << count << " mass gain=" << gained << " gas loss=" << lost << '\n';
}

void BenchmarkLuminosity(const amrex::Geometry &geom, const amrex::DistributionMapping &dm, const amrex::BoxArray &ba, int iterations)
{
#ifdef QUOKKA_BENCHMARK_RTSOA
	using Stars = quokka::ParticleContainer;
	Stars stars(quokka::ParticleSchemaTraits<quokka::ParticleType::StochasticStellarPop, LuminosityBenchmarkProblem>::Create(), geom, dm, ba);
#else
	using Stars = quokka::StochasticStellarPopParticleContainer<LuminosityBenchmarkProblem>;
	Stars stars(amrex::Vector<amrex::Geometry>{geom}, amrex::Vector<amrex::DistributionMapping>{dm}, amrex::Vector<amrex::BoxArray>{ba},
		    amrex::Vector<int>{});
#endif
	Stars::ParticleInitData initial;
#ifdef QUOKKA_BENCHMARK_RTSOA
	initial.real_array_data.assign(stars.Layout().numReal, 0.0);
	initial.int_array_data.assign(stars.Layout().numInt, 0);
	auto &reals = initial.real_array_data;
#else
	auto &reals = initial.real_struct_data;
#endif
	reals[quokka::StochasticStellarPopParticleMassIdx] = 10.0 * C::M_solar;
	reals[quokka::StochasticStellarPopParticleLumIdx] = -1.0;
	reals[quokka::StochasticStellarPopParticleLumIdx + 1] = -1.0;
	stars.InitOnePerCell(0.5, 0.5, 0.5, initial);
	AMREX_ALWAYS_ASSERT(stars.TotalNumberOfParticles() == geom.Domain().numPts());
	for (Stars::ParIterType iter(stars, 0); iter.isValid(); ++iter) {
		const auto data = iter.GetParticleTile().getParticleTileData();
#ifdef QUOKKA_BENCHMARK_RTSOA
		const auto layout = stars.Layout();
#endif
		amrex::ParallelFor(iter.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
#ifdef QUOKKA_BENCHMARK_RTSOA
			auto p = quokka::ParticleView(data, i, layout);
#else
			auto &p = data.m_aos[i];
#endif
			p.rdata(quokka::StochasticStellarPopParticleMassIdx) = (1.0 + 62.0 * p.pos(0)) * C::M_solar;
			p.rdata(quokka::StochasticStellarPopParticleBirthTimeIdx) = (63.0 - 31.0 * p.pos(1)) * quokka::seconds_per_year;
		});
	}
	quokka::LuminosityTables<2> tables;
	std::array<amrex::Vector<amrex::Real>, 2> coordinates;
	for (auto &axis : coordinates) {
		for (int i = 1; i <= 64; ++i) {
			axis.push_back(i);
		}
	}
	std::array<amrex::Vector<amrex::Vector<amrex::Real>>, 2> values;
	// Vary lookup coordinates across particles while keeping an independent
	// expected luminosity for each group, regardless of interpolation transforms.
	values[0].assign(64, amrex::Vector<amrex::Real>(64, 42.));
	values[1].assign(64, amrex::Vector<amrex::Real>(64, 84.));
	tables.luminosity.initialize(coordinates, values);
	quokka::g_luminosity_tables_ptr<2> = &tables;
	Measure("ParticleBenchmark::luminosity", iterations, [&](int i) {
		quokka::ParticlePropertyUpdateTraits<quokka::ParticleType::StochasticStellarPop>::updateParticleProperties<LuminosityBenchmarkProblem>(
		    &stars, (64 + i) * quokka::seconds_per_year, quokka::seconds_per_year);
	});
	quokka::g_luminosity_tables_ptr<2> = nullptr;
	amrex::ReduceOps<amrex::ReduceOpMax> op;
	amrex::ReduceData<amrex::Real> result(op);
	using Tuple = decltype(result)::Type;
	for (Stars::ParConstIterType iter(stars, 0); iter.isValid(); ++iter) {
		const auto data = iter.GetParticleTile().getConstParticleTileData();
#ifdef QUOKKA_BENCHMARK_RTSOA
		const auto layout = stars.Layout();
#endif
		op.eval(iter.numParticles(), result, [=] AMREX_GPU_DEVICE(int i) -> Tuple {
#ifdef QUOKKA_BENCHMARK_RTSOA
			const auto p = quokka::ParticleView(data, i, layout);
#else
			const auto &p = data.m_aos[i];
#endif
			const auto first = p.rdata(quokka::StochasticStellarPopParticleLumIdx);
			const auto second = p.rdata(quokka::StochasticStellarPopParticleLumIdx + 1);
			if (!std::isfinite(first) || !std::isfinite(second)) {
				return {1.0};
			}
			return {amrex::max(std::abs(first - 42.), std::abs(second - 84.))};
		});
	}
	auto error = amrex::get<0>(result.value(op));
	amrex::ParallelDescriptor::ReduceRealMax(error);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error < 1.0e-10, "Luminosity benchmark did not replace every poisoned field with the table value");
}
} // namespace

auto main(int argc, char **argv) -> int
try {
	amrex::Initialize(argc, argv);
	{
		static_assert(AMREX_SPACEDIM == 3);
		const amrex::ParmParse pp("benchmark");
		int cells = 64;
		int iterations = 20;
		int ioIterations = 3;
		pp.query("cells", cells);
		pp.query("iterations", iterations);
		pp.query("io_iterations", ioIterations);
		AMREX_ALWAYS_ASSERT(cells >= 4 && cells % 2 == 0 && iterations > 0 && ioIterations > 0);
		const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
		const amrex::RealBox bounds({0., 0., 0.}, {1., 1., 1.});
		const std::array<int, 3> periodic = {1, 1, 1};
		const amrex::Geometry geom(domain, &bounds, 0, periodic.data());
		amrex::BoxArray ba(domain);
		ba.maxSize(32);
		const amrex::DistributionMapping dm(ba);
		BenchmarkAccretion(amrex::max(cells, 16), iterations);
		BenchmarkLuminosity(geom, dm, ba, iterations);
		const auto count = domain.numPts();
		const amrex::Real mass = 1.0 / static_cast<amrex::Real>(count);
		constexpr amrex::Real initialVelocity = 0.01;
		constexpr amrex::Real acceleration = 0.02;
		constexpr amrex::Real dt = 0.001;
		const amrex::Real displacement = iterations * dt * initialVelocity;
		AMREX_ALWAYS_ASSERT(displacement < 0.25 / cells);
		auto pc = MakeContainer(geom, dm, ba);
		Container::ParticleInitData initial;
#ifdef QUOKKA_BENCHMARK_RTSOA
		initial.real_array_data = {mass, initialVelocity, initialVelocity, initialVelocity};
#else
		initial.real_struct_data = {mass, initialVelocity, initialVelocity, initialVelocity};
#endif
		pc->InitOnePerCell(0.5, 0.5, 0.5, initial);
#ifdef QUOKKA_BENCHMARK_RTSOA
		quokka::PhysicsParticleDescriptor<ParticleBenchmarkProblem, quokka::ParticleType::CIC> descriptor(pc.get(), false, false, false);
#else
		quokka::PhysicsParticleDescriptor<Container, ParticleBenchmarkProblem, quokka::ParticleType::CIC> descriptor(pc.get(), 0, -1, -1, -1, false);
#endif
		CheckParticles(*pc, count, cells, mass, 0., initialVelocity);
		amrex::Print() << "Particle benchmark: particles=" << count << " cells=" << cells << " iterations=" << iterations << '\n';
		Measure("ParticleBenchmark::drift", iterations, [&](int) { descriptor.driftParticles(0, 0, dt); });
		CheckParticles(*pc, count, cells, mass, displacement, initialVelocity);
		amrex::MultiFab accel(ba, dm, AMREX_SPACEDIM, 2);
		accel.setVal(acceleration);
		Measure("ParticleBenchmark::kick", iterations, [&](int) { descriptor.kickParticles(0, dt, accel); });
		const amrex::Real finalVelocity = initialVelocity + iterations * 0.5 * dt * acceleration;
		CheckParticles(*pc, count, cells, mass, displacement, finalVelocity);
		amrex::MultiFab density(ba, dm, 1, 1);
		// Accumulate repeated deposits, checking the independently known total mass.
		density.setVal(0.0);
		Measure("ParticleBenchmark::deposit", iterations, [&](int) { quokka::depositParticleMassDensity(pc.get(), density, 0, 0); });
		const amrex::Real deposited = density.sum(0) / static_cast<amrex::Real>(count);
		AMREX_ALWAYS_ASSERT(std::abs(deposited - iterations) < 1.0e-10 * iterations);
		for (int repeat = 0; repeat < iterations; ++repeat) {
			// Move every particle across the domain's midpoint before timing the
			// redistribution itself. This exchanges grids for the default 64^3 case.
			for (Container::ParIterType iter(*pc, 0); iter.isValid(); ++iter) {
				const auto data = iter.GetParticleTile().getParticleTileData();
				amrex::ParallelFor(iter.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
#ifdef QUOKKA_BENCHMARK_RTSOA
					auto p = data[i];
#else
					auto &p = data.m_aos[i];
#endif
					p.pos(0) += 0.5;
					if (p.pos(0) >= 1.0) {
						p.pos(0) -= 1.0;
					}
				});
			}
			Measure("ParticleBenchmark::redistribute", 1, [&](int) { pc->Redistribute(); });
		}
		CheckParticles(*pc, count, cells, mass, displacement, finalVelocity);
		Measure("ParticleBenchmark::checkpoint", ioIterations, [&](int i) {
			const auto directory = "particle_benchmark_checkpoint_" + std::to_string(i);
#ifdef QUOKKA_BENCHMARK_RTSOA
			pc->Checkpoint(directory, "CIC_particles");
#else
			pc->Checkpoint(directory, "CIC_particles", true,
				       quokka::getParticleRealCompNames<quokka::ParticleType::CIC, ParticleBenchmarkProblem>(),
				       quokka::getParticleIntCompNames<quokka::ParticleType::CIC, ParticleBenchmarkProblem>());
#endif
		});
		for (int i = 0; i < ioIterations; ++i) {
			auto restored = MakeContainer(geom, dm, ba);
			Measure("ParticleBenchmark::restart", 1, [&](int) {
				restored->Restart("particle_benchmark_checkpoint_" + std::to_string(i), "CIC_particles");
#ifdef QUOKKA_BENCHMARK_RTSOA
				Container::FinalizeRestarts();
#endif
			});
			CheckParticles(*restored, count, cells, mass, displacement, finalVelocity);
		}
	}
	amrex::Finalize();
} catch (const std::exception &error) {
	amrex::Abort(error.what());
	return 1;
}
