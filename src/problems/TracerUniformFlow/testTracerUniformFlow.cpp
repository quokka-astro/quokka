#include "QuokkaSimulation.hpp"
#include "particles/particle_IO.hpp"
#include <algorithm>
#include <array>
#include <vector>

struct TracerProblem {};
template <> struct Physics_Traits<TracerProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
};
template <> struct quokka::EOS_Traits<TracerProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> void QuokkaSimulation<TracerProblem>::setInitialConditionsOnGrid(quokka::grid const &grid)
{
	const auto a = grid.array_;
	amrex::ParallelFor(grid.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		a(i, j, k, 0) = 1.0;
		a(i, j, k, 1) = 1.0;
		a(i, j, k, 2) = 1.0;
		a(i, j, k, 3) = 1.0;
		a(i, j, k, 4) = 4.0;
		a(i, j, k, 5) = 2.5;
	});
}

template <> void QuokkaSimulation<TracerProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const auto dx = geom[lev].CellSizeArray();
	const auto arrays = tags.arrays();
	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) {
		if ((i + 0.5) * dx[0] < 0.25) {
			arrays[box](i, j, k) = amrex::TagBox::SET;
		}
	});
}

class TracerSimulation : public QuokkaSimulation<TracerProblem>
{
      public:
	using Position = std::array<long long, 3>;
	void moveNearFaces()
	{
		double offset = 0.49;
		amrex::ParmParse("tracer_test").query("offset", offset);
		if (max_level > 0) {
			AMREX_ALWAYS_ASSERT(finestLevel() == 1);
			AMREX_ALWAYS_ASSERT(boxArray(1).numPts() < Geom(1).Domain().numPts());
		}
		for (int lev = 0; lev <= finestLevel(); ++lev) {
			const auto dx = Geom(lev).CellSizeArray();
			for (amrex::AmrTracerParticleContainer::ParIterType it(*TracerPC, lev); it.isValid(); ++it) {
				auto *p = it.GetArrayOfStructs()().data();
				amrex::ParallelFor(it.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
					for (int d = 0; d < 3; ++d) {
						p[i].pos(d) += offset * dx[d];
					}
				});
			}
		}
		TracerPC->Redistribute();
	}
	auto positions(double displacement) -> std::vector<Position>
	{
		const auto [ids, reals, ints] = quokka::particle_io::getParticleDataAtAllLevels(static_cast<amrex::TracerParticleContainer *>(TracerPC.get()));
		std::vector<Position> result;
		for (const auto &p : reals) {
			Position position{};
			bool keep = true;
			for (int d = 0; d < 3; ++d) {
				const double x = p[d] + displacement;
				if (!Geom(0).isPeriodic(d) && (x < 0.0 || x >= 1.0)) {
					keep = false;
					break;
				}
				position[d] = std::llround((x - std::floor(x)) * 1.e10) % 10000000000LL;
			}
			if (keep) {
				result.push_back(position);
			}
		}
		std::sort(result.begin(), result.end());
		return result;
	}
};

auto problem_main() -> int
{
	TracerSimulation sim;
	sim.setInitialConditions();
	sim.moveNearFaces();
	const auto expected = sim.positions(0.001953125);
	sim.evolve();
	const auto actual = sim.positions(0.0);
	int failed = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		failed = expected.empty() || actual != expected;
		amrex::Print() << "Analytic tracer displacement: " << (failed ? "FAIL" : "PASS") << " (" << actual.size() << " particles)\n";
	}
	amrex::ParallelDescriptor::ReduceIntMax(failed);
	return failed;
}
