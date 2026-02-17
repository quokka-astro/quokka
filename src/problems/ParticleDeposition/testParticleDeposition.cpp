/// Test particle deposition utilities for per-species particle deposition.

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/PhysicsParticles.hpp"
#include "particles/particle_deposition.hpp"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"

using namespace quokka;

struct ParticleDepositionProblem {
};

template <> struct Particle_Traits<ParticleDepositionProblem> {
#if AMREX_SPACEDIM == 3
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
#else
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
#endif
};

template <> struct EOS_Traits<ParticleDepositionProblem> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<ParticleDepositionProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<ParticleDepositionProblem> {
	std::vector<amrex::Real> totalMass;
};

template <> void QuokkaSimulation<ParticleDepositionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state = grid_elem.array_;

	const amrex::Real rho0 = 1.0e-3;
	const amrex::Real P0 = 1.0e-6;
	const amrex::Real eint0 = P0 / (rho0 * (quokka::EOS_Traits<ParticleDepositionProblem>::gamma - 1.0));

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state(i, j, k, HydroSystem<ParticleDepositionProblem>::density_index) = rho0;
		state(i, j, k, HydroSystem<ParticleDepositionProblem>::x1Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<ParticleDepositionProblem>::x2Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<ParticleDepositionProblem>::x3Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<ParticleDepositionProblem>::internalEnergy_index) = eint0;
		state(i, j, k, HydroSystem<ParticleDepositionProblem>::energy_index) = eint0;
	});
}

// CIC particles are only supported in 3D builds.
#if AMREX_SPACEDIM == 3
template <> void QuokkaSimulation<ParticleDepositionProblem>::createInitialCICParticles()
{
	const int nParticles = 10;
	const amrex::Real particleMass = 1.0e-2;
	const amrex::Real particleVelocity = 1.0e5;

	const std::string particleFile = "particle_deposition_particles.txt";
	std::ofstream outFile(particleFile);
	outFile << nParticles << "\n";

	for (int n = 0; n < nParticles; ++n) {
		const amrex::Real x = 0.1 + 0.8 * static_cast<amrex::Real>(n) / static_cast<amrex::Real>(nParticles - 1);
		const amrex::Real y = 0.5;
		const amrex::Real z = 0.5;
		const amrex::Real vx = particleVelocity * (0.5 - static_cast<amrex::Real>(n) / static_cast<amrex::Real>(nParticles - 1));
		const amrex::Real vy = 0.0;
		const amrex::Real vz = 0.0;
		outFile << x << " " << y << " " << z << " " << particleMass << " " << vx << " " << vy << " " << vz << "\n";
	}
	outFile.close();

	CICParticles->SetVerbose(0);
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->InitFromAsciiFile(particleFile, nreal_extra, nullptr);
}
#endif

template <> void QuokkaSimulation<ParticleDepositionProblem>::computeAfterTimestep()
{
#if AMREX_SPACEDIM == 3
	const int lev = 0;
	const int nGhost = 0;
	const int nComp = 1;

	amrex::MultiFab massField(grids[lev], dmap[lev], nComp, nGhost);

	massField.setVal(0.0);
	depositParticleMassDensity(CICParticles.get(), massField, lev, CICParticleMassIdx, 0);

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::Real cellVol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	const amrex::Real totalMass = massField.sum(0) * cellVol;

	userData_.totalMass.push_back(totalMass);

	amrex::Print() << "Step " << istep[0] << ": Total mass = " << totalMass << "\n";
#endif
}

template <> void QuokkaSimulation<ParticleDepositionProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int /*ncomp*/) const
{
#if AMREX_SPACEDIM == 3
	if (dname == "particle_mass_density") {
		mf.setVal(0.0);
		depositParticleMassDensity(CICParticles.get(), mf, lev, CICParticleMassIdx, 0);
	}
#endif
}

auto problem_main() -> int
{
#if AMREX_SPACEDIM != 3
	amrex::Print() << "Skipping ParticleDeposition test: CIC particles are only enabled in 3D.\n";
	return 0;
#else
	QuokkaSimulation<ParticleDepositionProblem> sim;
	sim.setInitialConditions();
	sim.evolve();

	if (sim.userData_.totalMass.empty()) {
		amrex::Print() << "ERROR: No deposition diagnostics recorded.\n";
		return 1;
	}

	const amrex::Real finalMass = sim.userData_.totalMass.back();

	amrex::Print() << "Final results:\n";
	amrex::Print() << "  Total mass: " << finalMass << "\n";

	const amrex::Real expectedMass = 10.0 * 1.0e-2;
	const amrex::Real massTolerance = 1.0e-12;

	if (std::abs(finalMass - expectedMass) > massTolerance) {
		amrex::Print() << "ERROR: Mass conservation failed! Expected " << expectedMass << ", got " << finalMass << "\n";
		return 1;
	}

	amrex::Print() << "SUCCESS: Particle deposition test passed!\n";
	return 0;
#endif
}
