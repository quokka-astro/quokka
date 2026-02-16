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
#include "particles/particle_deposition_utils.hpp"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"

using namespace quokka;

struct ParticleDepositionProblem {
};

template <> struct Particle_Traits<ParticleDepositionProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
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
	std::vector<amrex::Real> totalMomentum;
	std::vector<amrex::Real> totalEnergy;
	std::vector<amrex::Real> totalNumber;
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

template <> void QuokkaSimulation<ParticleDepositionProblem>::createInitialCICParticles()
{
	const int nParticles = 10;
	const amrex::Real particleMass = 1.0e-2;
	const amrex::Real particleVelocity = 1.0e5;

	const std::string particleFile = "tests/particle_deposition_particles.txt";
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

template <> void QuokkaSimulation<ParticleDepositionProblem>::computeAfterTimestep()
{
	const int lev = 0;
	const int nGhost = 0;
	const int nComp = 1;

	amrex::MultiFab massField(grids[lev], dmap[lev], nComp, nGhost);
	amrex::MultiFab momentumField(grids[lev], dmap[lev], AMREX_SPACEDIM, nGhost);
	amrex::MultiFab energyField(grids[lev], dmap[lev], nComp, nGhost);
	amrex::MultiFab numberField(grids[lev], dmap[lev], nComp, nGhost);

	massField.setVal(0.0);
	momentumField.setVal(0.0);
	energyField.setVal(0.0);
	numberField.setVal(0.0);

	depositCICParticleProperties(CICParticles.get(), massField, momentumField, energyField, numberField, lev);

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::Real cellVol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	const amrex::Real totalMass = massField.sum(0) * cellVol;
	const amrex::Real totalMomentumX = momentumField.sum(0) * cellVol;
	const amrex::Real totalMomentumY = momentumField.sum(1) * cellVol;
	const amrex::Real totalMomentumZ = momentumField.sum(2) * cellVol;
	const amrex::Real totalEnergy = energyField.sum(0) * cellVol;
	const amrex::Real totalNumber = numberField.sum(0) * cellVol;

	userData_.totalMass.push_back(totalMass);
	userData_.totalMomentum.push_back(std::sqrt(totalMomentumX * totalMomentumX + totalMomentumY * totalMomentumY + totalMomentumZ * totalMomentumZ));
	userData_.totalEnergy.push_back(totalEnergy);
	userData_.totalNumber.push_back(totalNumber);

	amrex::Print() << "Step " << istep[0] << ": Total mass = " << totalMass << ", Total momentum = " << userData_.totalMomentum.back()
		       << ", Total energy = " << totalEnergy << ", Total number = " << totalNumber << "\n";
}

template <>
void QuokkaSimulation<ParticleDepositionProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int /*ncomp*/) const
{
	if (dname == "particle_mass_density") {
		mf.setVal(0.0);
		depositParticleMassDensity(CICParticles.get(), mf, lev, CICParticleMassIdx, 0);
	} else if (dname == "particle_number_density") {
		mf.setVal(0.0);
		depositParticleNumberDensity(CICParticles.get(), mf, lev, 0);
	}
}

auto problem_main() -> int
{
	QuokkaSimulation<ParticleDepositionProblem> sim;
	sim.setInitialConditions();
	sim.evolve();

	if (sim.userData_.totalMass.empty() || sim.userData_.totalNumber.empty()) {
		amrex::Print() << "ERROR: No deposition diagnostics recorded.\n";
		return 1;
	}

	const amrex::Real finalMass = sim.userData_.totalMass.back();
	const amrex::Real finalMomentum = sim.userData_.totalMomentum.back();
	const amrex::Real finalEnergy = sim.userData_.totalEnergy.back();
	const amrex::Real finalNumber = sim.userData_.totalNumber.back();

	amrex::Print() << "Final results:\n";
	amrex::Print() << "  Total mass: " << finalMass << "\n";
	amrex::Print() << "  Total momentum: " << finalMomentum << "\n";
	amrex::Print() << "  Total energy: " << finalEnergy << "\n";
	amrex::Print() << "  Total number: " << finalNumber << "\n";

	const amrex::Real expectedMass = 10.0 * 1.0e-2;
	const amrex::Real expectedNumber = 10.0;
	const amrex::Real massTolerance = 1.0e-12;
	const amrex::Real numberTolerance = 1.0e-12;

	if (std::abs(finalMass - expectedMass) > massTolerance) {
		amrex::Print() << "ERROR: Mass conservation failed! Expected " << expectedMass << ", got " << finalMass << "\n";
		return 1;
	}
	if (std::abs(finalNumber - expectedNumber) > numberTolerance) {
		amrex::Print() << "ERROR: Number conservation failed! Expected " << expectedNumber << ", got " << finalNumber << "\n";
		return 1;
	}

	amrex::Print() << "SUCCESS: Particle deposition test passed!\n";
	return 0;
}
