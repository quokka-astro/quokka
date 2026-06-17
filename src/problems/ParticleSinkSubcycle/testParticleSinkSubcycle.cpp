/// \file testParticleSinkSubcycle.cpp
/// \brief Regression test for issue #1957: sink particles losing finest-level coverage
///        under AMR subcycling with max_level >= 2.
///
/// Setup (from ParticleSinkSubcycle_bug.toml):
///   domain = [-1.6e20, 1.6e20]^3, n_cell=32, dx_0=1e19, blocking_factor=4, n_error_buf=2
///   max_level=2, do_subcycle=1
///
///   refineGrid(lev==0): tags x < 0 (left half) → level-1 covers level-1 cells [0,31] in x
///   Particle at x=-5e18 → level-0 cell 15, level-1 cell 31 (right at the level-1 boundary)
///
///   With do_subcycle=1, a level-1-only regrid tries to create level-2 around the particle.
///   bf_lev[1] = blocking_factor[2]/ref_ratio[1] = 4/2 = 2.
///   Particle tag grown by n_error_buf=2: level-1 cells [29,33].
///   Proper interior of level-1 patch [0,31]: cells [2,29].
///   Cells 30-33 are outside the proper interior → tag CLEARED.
///   Particle cannot get level-2 coverage → finestLevel() < max_level (the bug).
///
/// Expected ctest pass (PASS_REGULAR_EXPRESSION): only on PR #1961 branch where the
/// guard fires with "ForceFinestLevel=true and max_level >= 2" abort message.
/// On development, the test runs without abort and finestLevel() < max_level is observed.

#include "AMReX.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

struct SubcycleProblem {
};

template <> struct Particle_Traits<SubcycleProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Sink;
};

template <> struct quokka::EOS_Traits<SubcycleProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_p;
};

template <> struct HydroSystem_Traits<SubcycleProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<SubcycleProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = false; // no gravity — avoids Poisson abort
	static constexpr bool is_mhd_enabled = false;
};

template <> void QuokkaSimulation<SubcycleProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho0 = C::m_p;
	const double T0 = 10.0;
	const double CV = 1. / (5. / 3. - 1.) / C::m_p * C::k_B;
	const double rho_e = CV * T0 * rho0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SubcycleProblem>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<SubcycleProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SubcycleProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SubcycleProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SubcycleProblem>::energy_index) = rho_e;
		state_cc(i, j, k, HydroSystem<SubcycleProblem>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SubcycleProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const & /*grid_elem*/) {}

template <> void QuokkaSimulation<SubcycleProblem>::createInitialSinkParticles()
{
	// Particle at x=-5e18 cm (level-0 cell 15, level-1 cell 31) — near the right boundary
	// of the level-1 patch that refineGrid creates for the left half (x < 0) of the domain.
	// mass, vx, vy, vz — 4 real extra attributes
	const int nreal_extra = 4;
	SinkParticles->SetVerbose(0);
	SinkParticles->InitFromAsciiFile("../inputs/ParticleSinkSubcycle.txt", nreal_extra, nullptr);
}

template <> void QuokkaSimulation<SubcycleProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	if (lev > 0) {
		return; // level-2 comes only from the particle tag (refineGridsAroundParticles)
	}
	// Tag the left half of the domain (x < 0) to create a level-1 patch.
	// This places the particle near the right edge of the level-1 patch.
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		if (x < 0.0) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<SubcycleProblem> sim;
	sim.cflNumber_ = 0.3;

	// On the PR branch this aborts here with:
	// "Particles with ForceFinestLevel=true and max_level >= 2 require AMR subcycling
	//  to be disabled. Set do_subcycle = 0."
	// The PASS_REGULAR_EXPRESSION in CMakeLists matches that message.
	sim.setInitialConditions();

	// On development: setForceFinestLevel is set AFTER init (matching real production usage,
	// e.g. ParticleAccretion). During evolve(), the subcycled level-1 regrid tries but fails
	// to create level-2 around the particle due to the proper-nesting constraint.
	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->setForceFinestLevel(true);

	sim.evolve();

	// Verify the particle reached the finest level.
	const int finest = sim.finestLevel();
	const auto [ids_all, data_all, idata_all] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtAllLevels();
	const auto &[data_finest, idata_finest] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(finest);

	amrex::Print() << "finestLevel() = " << finest << " (max_level = " << sim.max_level << ")\n";
	amrex::Print() << "particles total: " << data_all.size() << ", on finest level: " << data_finest.size() << "\n";

	if (finest < sim.max_level || data_finest.size() < data_all.size()) {
		amrex::Print() << "ISSUE #1957 REPRODUCED: particle did not reach finest level under "
			          "do_subcycle=1 + max_level=" << sim.max_level << "\n";
		return 1;
	}
	amrex::Print() << "Test passed: particle is on finest level\n";
	return 0;
}
