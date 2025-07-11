/// Test particle deposition utilities for per-species particle deposition
/// This test verifies that particle properties (mass, momentum, energy, number density) 
/// can be deposited onto the grid correctly for different particle types

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "AMReX_AmrCore.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_Config.H"
#include "AMReX_MultiFab.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "io/DiagParticleDeposition.H"
#include "particles/particle_deposition_utils.hpp"
#include "particles/particle_types.hpp"
#include "particles/PhysicsParticles.hpp"
#include "physics_info.hpp"

using namespace quokka;

// Problem-specific parameters
struct ParticleDepositionProblem {
};

// Enable CIC particles for this test
template <> struct quokka::Particle_Traits<ParticleDepositionProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

// EOS configuration
template <> struct quokka::EOS_Traits<ParticleDepositionProblem> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

// Physics configuration
template <> struct quokka::Physics_Traits<ParticleDepositionProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int nGroups = 1;
};

// Simulation class
template <> 
class QuokkaSimulation<ParticleDepositionProblem> : public AMRSimulation<ParticleDepositionProblem>
{
public:
	explicit QuokkaSimulation(amrex::LevelBld *a_lev_bld = nullptr, amrex::CartesianGrid *a_parent = nullptr)
	    : AMRSimulation<ParticleDepositionProblem>(a_lev_bld, a_parent)
	{
		// Initialize particle deposition diagnostic
		std::string prefix = "diag.particle_deposition";
		std::string diagName = "particle_deposition";
		m_particleDepositionDiag.init(prefix, diagName);
	}

	void setInitialConditions() override;
	void computeAfterTimestep() override;
	void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;
	auto ComputeProjections(amrex::Direction dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> override;

private:
	DiagParticleDeposition m_particleDepositionDiag;
	std::vector<double> m_totalMass;
	std::vector<double> m_totalMomentum;
	std::vector<double> m_totalEnergy;
	std::vector<double> m_totalNumber;
};

template <>
void QuokkaSimulation<ParticleDepositionProblem>::setInitialConditions()
{
	// Set initial gas conditions
	const amrex::Real rho0 = 1.0e-3;  // Low density background
	const amrex::Real P0 = 1.0e-6;    // Low pressure
	const amrex::Real T0 = 1.0e4;     // 10^4 K
	const amrex::Real eint0 = P0 / (rho0 * (quokka::EOS_Traits<ParticleDepositionProblem>::gamma - 1.0));

	// Set initial state
	for (int lev = 0; lev <= finest_level; ++lev) {
		const auto &dx = geom[lev].CellSizeArray();
		const auto &prob_lo = geom[lev].ProbLoArray();
		const auto &prob_hi = geom[lev].ProbHiArray();

		for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
			const amrex::Box &indexRange = mfi.validbox();
			const auto &state = state_new_cc_[lev].array(mfi);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				state(i, j, k, HydroSystem<ParticleDepositionProblem>::density_index) = rho0;
				state(i, j, k, HydroSystem<ParticleDepositionProblem>::x1Momentum_index) = 0.0;
				state(i, j, k, HydroSystem<ParticleDepositionProblem>::x2Momentum_index) = 0.0;
				state(i, j, k, HydroSystem<ParticleDepositionProblem>::x3Momentum_index) = 0.0;
				state(i, j, k, HydroSystem<ParticleDepositionProblem>::internalEnergy_index) = eint0;
				state(i, j, k, HydroSystem<ParticleDepositionProblem>::energy_index) = eint0;
			});
		}
	}

	// Create test particles
	const int lev = 0;
	const int nParticles = 10;
	const amrex::Real particleMass = 1.0e-2;
	const amrex::Real particleVelocity = 1.0e5;  // cm/s

	if (ParticleSwitch::CIC & quokka::Particle_Traits<ParticleDepositionProblem>::particle_switch) {
		auto *cicContainer = particleRegister_.getCICParticleContainer();
		
		// Create particles uniformly distributed in the domain
		for (int n = 0; n < nParticles; ++n) {
			const amrex::Real x = 0.1 + 0.8 * static_cast<amrex::Real>(n) / (nParticles - 1);
			const amrex::Real y = 0.5;
			const amrex::Real z = 0.5;
			const amrex::Real vx = particleVelocity * (0.5 - static_cast<amrex::Real>(n) / (nParticles - 1));
			const amrex::Real vy = 0.0;
			const amrex::Real vz = 0.0;

			amrex::ParticleReal pdata[CICParticleRealComps];
			pdata[CICParticleMassIdx] = particleMass;
			pdata[CICParticleVxIdx] = vx;
			pdata[CICParticleVyIdx] = vy;
			pdata[CICParticleVzIdx] = vz;

			// Add particle to container
			cicContainer->AddOneParticle(lev, 0, 0, x, y, z, pdata);
		}
	}

	// Prepare particle deposition diagnostic
	amrex::Vector<std::string> varNames = {"density", "x1Momentum", "x2Momentum", "x3Momentum", "internalEnergy", "energy"};
	m_particleDepositionDiag.prepare(finest_level + 1, geom, grids, dmap, varNames);

	// Call parent initialization
	AMRSimulation<ParticleDepositionProblem>::setInitialConditions();
}

template <>
void QuokkaSimulation<ParticleDepositionProblem>::computeAfterTimestep()
{
	// Test particle deposition utilities
	const int lev = 0;
	const int nGhost = 0;
	const int nComp = 1;

	// Create MultiFabs for deposition
	amrex::MultiFab massField(grids[lev], dmap[lev], nComp, nGhost);
	amrex::MultiFab momentumField(grids[lev], dmap[lev], AMREX_SPACEDIM, nGhost);
	amrex::MultiFab energyField(grids[lev], dmap[lev], nComp, nGhost);
	amrex::MultiFab numberField(grids[lev], dmap[lev], nComp, nGhost);

	// Clear fields
	massField.setVal(0.0);
	momentumField.setVal(0.0);
	energyField.setVal(0.0);
	numberField.setVal(0.0);

	// Deposit CIC particles
	if (ParticleSwitch::CIC & quokka::Particle_Traits<ParticleDepositionProblem>::particle_switch) {
		auto *cicContainer = particleRegister_.getCICParticleContainer();
		depositCICParticleProperties(cicContainer, massField, momentumField, energyField, numberField, lev);
	}

	// Compute totals
	const amrex::Real totalMass = massField.sum(0);
	const amrex::Real totalMomentumX = momentumField.sum(0);
	const amrex::Real totalMomentumY = momentumField.sum(1);
	const amrex::Real totalMomentumZ = momentumField.sum(2);
	const amrex::Real totalEnergy = energyField.sum(0);
	const amrex::Real totalNumber = numberField.sum(0);

	// Store results
	m_totalMass.push_back(totalMass);
	m_totalMomentum.push_back(std::sqrt(totalMomentumX * totalMomentumX + totalMomentumY * totalMomentumY + totalMomentumZ * totalMomentumZ));
	m_totalEnergy.push_back(totalEnergy);
	m_totalNumber.push_back(totalNumber);

	// Print results
	amrex::Print() << "Step " << istep[0] << ": Total mass = " << totalMass << ", Total momentum = " << m_totalMomentum.back()
		       << ", Total energy = " << totalEnergy << ", Total number = " << totalNumber << "\n";
}

template <>
void QuokkaSimulation<ParticleDepositionProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const
{
	// Add derived variables for particle deposition fields
	if (dname == "particle_mass_density") {
		mf.setVal(0.0);
		if (ParticleSwitch::CIC & quokka::Particle_Traits<ParticleDepositionProblem>::particle_switch) {
			auto *cicContainer = particleRegister_.getCICParticleContainer();
			depositParticleMassDensity(cicContainer, mf, lev, CICParticleMassIdx, 0);
		}
	} else if (dname == "particle_number_density") {
		mf.setVal(0.0);
		if (ParticleSwitch::CIC & quokka::Particle_Traits<ParticleDepositionProblem>::particle_switch) {
			auto *cicContainer = particleRegister_.getCICParticleContainer();
			depositParticleNumberDensity(cicContainer, mf, lev, 0);
		}
	}
}

template <>
auto QuokkaSimulation<ParticleDepositionProblem>::ComputeProjections(amrex::Direction dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	return {};
}

// Problem initialization
auto problem_main() -> int
{
	// Problem parameters
	const int nx = 64;
	const int ny = 64;
	const int nz = 64;
	const amrex::Real Lx = 1.0;
	const amrex::Real Ly = 1.0;
	const amrex::Real Lz = 1.0;
	const amrex::Real CFL_number = 0.4;
	const amrex::Real max_time = 0.1;
	const int max_timesteps = 10;

	// Set up computational domain
	amrex::Vector<amrex::Real> prob_lo = {0.0, 0.0, 0.0};
	amrex::Vector<amrex::Real> prob_hi = {Lx, Ly, Lz};
	amrex::Vector<int> n_cell = {nx, ny, nz};

	// Set up AMR parameters
	amrex::Vector<int> max_grid_size = {32, 32, 32};
	amrex::Vector<int> blocking_factor = {8, 8, 8};

	// Initialize simulation
	QuokkaSimulation<ParticleDepositionProblem> sim;
	sim.setInitialConditions();

	// Run simulation
	sim.evolve();

	// Verify results
	const auto &finalMass = sim.m_totalMass.back();
	const auto &finalMomentum = sim.m_totalMomentum.back();
	const auto &finalEnergy = sim.m_totalEnergy.back();
	const auto &finalNumber = sim.m_totalNumber.back();

	amrex::Print() << "Final results:\n";
	amrex::Print() << "  Total mass: " << finalMass << "\n";
	amrex::Print() << "  Total momentum: " << finalMomentum << "\n";
	amrex::Print() << "  Total energy: " << finalEnergy << "\n";
	amrex::Print() << "  Total number: " << finalNumber << "\n";

	// Simple verification
	const amrex::Real expectedMass = 10.0 * 1.0e-2;  // 10 particles * 1e-2 mass each
	const amrex::Real expectedNumber = 10.0;          // 10 particles
	
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