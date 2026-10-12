#include "QuokkaSimulation.hpp"

struct CoarseStepProblem {};
template <> struct Physics_Traits<CoarseStepProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
};
template <> struct quokka::EOS_Traits<CoarseStepProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};
template <> struct Particle_Traits<CoarseStepProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> void QuokkaSimulation<CoarseStepProblem>::setInitialConditionsOnGrid(quokka::grid const &grid)
{
	const auto a = grid.array_;
	amrex::ParallelFor(grid.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		a(i, j, k, 0) = 1.0;
		a(i, j, k, 1) = 0.0;
		a(i, j, k, 2) = 0.0;
		a(i, j, k, 3) = 0.0;
		a(i, j, k, 4) = 2.5;
		a(i, j, k, 5) = 2.5;
	});
}

template <> void QuokkaSimulation<CoarseStepProblem>::createInitialCICParticles()
{
	std::string filename;
	amrex::ParmParse("particle_test").get("file", filename);
	CICParticles->InitFromAsciiFile(filename, quokka::CICParticleRealComps, nullptr);
}

template <> void QuokkaSimulation<CoarseStepProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const auto dx = geom[lev].CellSizeArray();
	const auto arrays = tags.arrays();
	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) {
		if ((i + 0.5) * dx[0] < 0.25) {
			arrays[box](i, j, k) = amrex::TagBox::SET;
		}
	});
}

// Exercise the production destruction/redistribution helper with a deterministic
// test-only rule. Ordinary CIC particles have no stellar destruction model.
template <typename problem_t> struct RemoveThirdParticle {
	AMREX_GPU_HOST_DEVICE RemoveThirdParticle(int /*birth_index*/, int /*stage_index*/) {}
	template <typename Particle> AMREX_GPU_DEVICE auto operator()(Particle const &p, int mass_index, amrex::Real time, amrex::Real dt) const -> bool
	{
		return p.rdata(mass_index) == 3.0 && time >= dt;
	}
};

class CoarseStepSimulation : public QuokkaSimulation<CoarseStepProblem>
{
      public:
	void checkPositions(int step)
	{
		AMREX_ALWAYS_ASSERT(finestLevel() == 1);
		auto *descriptor = GetParticleRegister().getParticleDescriptor(quokka::ParticleType::CIC);
		const auto [ids, reals, ints] = descriptor->getParticleDataAtAllLevels();
		if (amrex::ParallelDescriptor::IOProcessor()) {
			AMREX_ALWAYS_ASSERT(reals.size() == (step >= 2 ? 2 : 3));
			for (const auto &p : reals) {
				const double initial_x = p[3] == 2.0 ? 0.251953125 : 0.248046875;
				const double velocity = p[3] == 2.0 ? -1.0 : 1.0;
				AMREX_ALWAYS_ASSERT(std::abs(p[0] - (initial_x + step * timestep * velocity)) < 1.e-12);
				AMREX_ALWAYS_ASSERT(p[1] == 0.5 && p[2] == 0.5);
				AMREX_ALWAYS_ASSERT(p[4] == velocity && p[5] == 0.0 && p[6] == 0.0);
			}
		}
		for (int lev = 0; lev <= 1; ++lev) {
			const auto [level_reals, level_ints] = descriptor->getParticleDataAtLevel(lev);
			if (amrex::ParallelDescriptor::IOProcessor()) {
				const std::size_t expected = step == 0 ? (lev == 0 ? 1 : 2) : (step == 1 && lev == 0 ? 2 : 1);
				AMREX_ALWAYS_ASSERT(level_reals.size() == expected);
				for (const auto &p : level_reals) {
					AMREX_ALWAYS_ASSERT((p[0] < 0.25) == (lev == 1));
				}
			}
		}
	}

	void computeAfterTimestep() override
	{
		++steps;
		AMREX_ALWAYS_ASSERT(dt_[0] == timestep);
		AMREX_ALWAYS_ASSERT(tNew_[0] == steps * timestep && tNew_[1] == tNew_[0]);
		AMREX_ALWAYS_ASSERT(istep[1] == steps * (do_subcycle == 1 ? 2 : 1));
		quokka::ParticleDestructionImpl::destroyParticlesImpl<CoarseStepProblem, quokka::CICParticleContainer, RemoveThirdParticle>(
		    CICParticles.get(), quokka::CICParticleMassIdx, 0, (steps - 1) * timestep, timestep, -1, -1);
		checkPositions(steps);
	}
	int steps = 0;
	static constexpr double timestep = 0.00390625;
};

auto problem_main() -> int
{
	CoarseStepSimulation sim;
	sim.setInitialConditions();
	sim.checkPositions(0);
	sim.evolve();
	AMREX_ALWAYS_ASSERT(sim.steps == 4);
	amrex::Print() << "Coarse-step particle drift and destruction checks passed\n";
	return 0;
}
