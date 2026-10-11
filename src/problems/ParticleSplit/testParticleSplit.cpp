#include "QuokkaSimulation.hpp"

struct SplitProblem {};

template <> struct Physics_Traits<SplitProblem> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int nGroups = 3;
};

template <> struct RadSystem_Traits<SplitProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 0.0;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, 4> radBoundaries{1.0, 2.0, 3.0, 4.0};
	static constexpr double beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <> struct Particle_Traits<SplitProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | ParticleSwitch::CICRad;
};

template <> void QuokkaSimulation<SplitProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const auto state = grid_elem.array_;
	amrex::ParallelFor(grid_elem.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < Physics_Indices<SplitProblem>::nvarTotal_cc; ++n) {
			state(i, j, k, n) = 0.0;
		}
		state(i, j, k, HydroSystem<SplitProblem>::density_index) = 1.0;
		state(i, j, k, HydroSystem<SplitProblem>::energy_index) = 1.0;
		state(i, j, k, HydroSystem<SplitProblem>::internalEnergy_index) = 1.0;
	});
}

template <> void QuokkaSimulation<SplitProblem>::createInitialCICParticles()
{
	CICParticles->InitFromAsciiFile("cic.txt", quokka::CICParticleRealComps, nullptr);
}

template <> void QuokkaSimulation<SplitProblem>::createInitialCICRadParticles()
{
	CICRadParticles->InitFromAsciiFile("cicrad.txt", quokka::CICRadParticleRealComps<SplitProblem>, nullptr);
}

auto problem_main() -> int
{
	QuokkaSimulation<SplitProblem> sim;
	sim.setInitialConditions();
	amrex::ParmParse pp("split_test");
	int split_factor = 1;
	if (pp.query("factor", split_factor)) {
		sim.GetParticleRegister().getParticleDescriptor(quokka::ParticleType::CICRad)->splitParticles(0, split_factor);
	}
	int expected_count = 1;
	pp.query("expected_count", expected_count);
	for (const auto type : {quokka::ParticleType::CIC, quokka::ParticleType::CICRad}) {
		auto *descriptor = sim.GetParticleRegister().getParticleDescriptor(type);
		const int count = type == quokka::ParticleType::CIC && pp.contains("factor") ? 1 : expected_count;
		AMREX_ALWAYS_ASSERT(descriptor->getNumParticles() == count);
		AMREX_ALWAYS_ASSERT(std::abs(descriptor->computeStellarMass() - 8.0) < 1.e-12);
		const auto [reals, ints] = descriptor->getParticleDataAtLevel(0);
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::GpuArray<double, 3> momentum{};
			for (const auto &particle : reals) {
				AMREX_ALWAYS_ASSERT(std::abs(particle[3] * count - 8.0) < 1.e-12);
				for (int d = 0; d < 3; ++d) {
					momentum[d] += particle[3] * particle[4 + d];
				}
				for (int d = 0; d < 3; ++d) {
					AMREX_ALWAYS_ASSERT(particle[d] == 0.5);
				}
				if (type == quokka::ParticleType::CICRad) {
					AMREX_ALWAYS_ASSERT(particle[3 + quokka::CICRadParticleBirthTimeIdx] == 2.0);
					AMREX_ALWAYS_ASSERT(particle[3 + quokka::CICRadParticleDeathTimeIdx] == 10.0);
					for (int group = 0; group < 3; ++group) {
						const auto luminosity = particle[3 + quokka::CICRadParticleLumIdx + group];
						AMREX_ALWAYS_ASSERT(std::abs(luminosity * count - 8.0 * (group + 1)) < 1.e-12);
					}
				}
			}
			for (const auto value : momentum) {
				AMREX_ALWAYS_ASSERT(std::abs(value) < 1.e-12);
			}
		}
		if (type == quokka::ParticleType::CICRad) {
			// Cover the full deposition stencil when particles lie on MPI box boundaries.
			constexpr int deposition_ghosts = amrex::ParticleInterpolator::WendlandC2<>::stencil_width / 2;
			amrex::MultiFab radiation(sim.boxArray(0), sim.DistributionMap(0), 3, deposition_ghosts);
			for (const double time : {1.0, 3.0, 10.0}) {
				radiation.setVal(0.0);
				descriptor->depositRadiation(radiation, 0, time, 3);
				const auto dx = sim.Geom(0).CellSizeArray();
				for (int group = 0; group < 3; ++group) {
					const double luminosity = radiation.sum(group) * dx[0] * dx[1] * dx[2];
					const double expected = (time < 2.0 || time >= 10.0) ? 0.0 : 8.0 * (group + 1);
					AMREX_ALWAYS_ASSERT(std::abs(luminosity - expected) < 1.e-12);
				}
			}
		}
	}
	sim.WriteCheckpointFile();
	amrex::Print() << "Particle split checks passed\n";
	return 0;
}
