/// \file testParticleSinkSubcycle.cpp
/// \brief Regression test for issue #1957: sink particles losing finest-level coverage
///        under AMR subcycling with max_level >= 2.
///
/// Setup (from ParticleSinkSubcycle_bug.toml):
///   domain = [-1.6e20, 1.6e20]^3, n_cell=32, dx_0=1e19, blocking_factor=8, n_error_buf=2
///   max_level=2, do_subcycle=1, regrid_interval=1, max_timesteps=3
///
///   refineGrid(lev==0): tags x < -1e20 (left sixth, cells [0,5]) → with n_error_buf=2
///   growing [0,5]→[0,7] → level-1 cells [0,15] (covering level-0 cells [0,7]).
///   Particle at x=-5e18 → level-0 cell 15 → OUTSIDE the initial level-1 patch → on level-0.
///
///   Development (no fix): subcycled level-1 regrid calls AmrCore::regrid(1), which holds
///   level-1 FIXED at [0,15]. refineGridsAroundParticles(1) can only find particles stored
///   AT level-1; the particle is at level-0 and is not found. No level-2 is created.
///   At coarse step 3, the coarse regrid (AmrCore::regrid(0)) also fails to maintain level-2
///   because tagCellsAroundParticles(0) only searches level-0, missing the particle when it
///   has been promoted to level-1 or level-2. The particle bounces back to level-0, and
///   finestLevel() never reaches max_level=2.
///
///   Fix (PR #1961):
///     (a) timeStepWithSubcycling forces a level-0 regrid at the start of each coarse step
///         when ForceFinestLevel particles exist, ensuring all levels are properly built before
///         any advance occurs.
///     (b) tagCellsAroundParticles(lev) now also searches particles at ALL other levels,
///         projecting their positions to level-lev. This ensures ErrorEst at each level can
///         tag cells for the next finer level even before the particle has been redistributed.
///
///   Expected: development exits with code 1 ("ISSUE #1957 REPRODUCED").
///             PR branch exits with code 0 ("Test passed: particle is on finest level").

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
namespace
{
bool density_refinement_enabled = true; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace

template <> struct Particle_Traits<SubcycleProblem> : DefaultParticleTraits {
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
	// Particle at x=-5e18 cm — level-0 cell 15, level-1 cells 30/31.
	// With the density-only initial regrid (refineGrid tags [0,5] → level-1 [0,15]),
	// the particle is OUTSIDE the level-1 patch and is stored at level-0.
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
	if (!density_refinement_enabled) {
		return; // test variant: pure particle-driven refinement
	}
	// Tag the leftmost sixth of the domain (x < -1e20, cells [0,5]) to create a level-1 patch.
	// With n_error_buf=2, this grows to [0,7] → level-1 [0,15] (covering level-0 cells [0,7]).
	// The particle at x=-5e18 (level-0 cell 15) is OUTSIDE this patch, staying on level-0.
	// This is the geometry that exposes the subcycling bug: the subcycled regrid cannot
	// expand level-1 to cover the particle, so level-2 cannot be properly nested around it.
	const double threshold_x = -1.0e20;
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		if (x < threshold_x) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<SubcycleProblem> sim;
	sim.cflNumber_ = 0.3;

	const amrex::ParmParse pp;
	pp.query("density_refinement", density_refinement_enabled);

	// setInitialConditions: InitFromScratch runs the initial regrid (density tags only,
	// no particles exist yet) → level-1 at [0,15]. Then InitPhyParticles creates the particle
	// at x=-5e18; redistribute places it at level-0 (not covered by level-1 [0,7]).
	// On PR branch, ForceFinestLevel is already true at registerParticleType (registered in
	// InitPhyParticles) but the initial regrid already ran without the particle, so level-1
	// remains [0,15] and the particle starts at level-0 on both branches.
	sim.setInitialConditions();

	// Ensure ForceFinestLevel is set on both branches so that refineGridsAroundParticles
	// is active during evolve(). On the PR branch this is a no-op (already set at registration).
	// On development this enables particle tagging for the subcycled regrid to attempt.
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
				  "do_subcycle=1 + max_level="
			       << sim.max_level << "\n";
		return 1;
	}
	amrex::Print() << "Test passed: particle is on finest level\n";
	return 0;
}
