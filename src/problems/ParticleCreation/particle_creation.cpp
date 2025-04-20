/// \file particle_creation.cpp
/// \brief Defines a test problem for particle creation.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "particle_creation.hpp"

struct TestParticle {
};

constexpr double rho0 = 1.0e-5;
constexpr double dt_ = 0.001;
static bool refine_half_domain = false; // NOLINT

template <> struct quokka::EOS_Traits<TestParticle> {
	static constexpr double gamma = 1.0;	     // isothermal
	static constexpr double cs_isothermal = 3.0; //
	static constexpr double mean_molecular_weight = 1.0;
};

// Test enum to demonstrate type checking of particle_switch
enum class TestEnum : unsigned int {
	MISTAKE = 0b00000100U,
};

template <> struct Particle_Traits<TestParticle> {
	// The following will cause a compile error
	// static constexpr int particle_switch = 1;
	// static constexpr TestEnum particle_switch = TestEnum::MISTAKE;
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | TestEnum::MISTAKE;
	// This is the correct way to define the particle switch
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Test;
};

template <> struct HydroSystem_Traits<TestParticle> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<TestParticle> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<TestParticle>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<TestParticle>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<TestParticle>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<TestParticle>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<TestParticle>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<TestParticle>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<TestParticle>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<TestParticle>::createInitialTestParticles()
{
	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	TestParticles->SetVerbose(1);
	TestParticles->InitFromAsciiFile("TestParticles.txt", nreal_extra, nullptr);

	// Loop over all particle at all levels and set first integer component to SNProgenitor
	for (int lev = 0; lev <= maxLevel(); ++lev) {
		auto &particles = TestParticles->GetParticles(lev);

		for (auto &kv : particles) {
			auto &particle_array = kv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			auto *pdata = particle_array().data();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
						    // if (p.rdata(0) > 1.0e-10) {
						    // 	p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
						    // } else {
						    // 	p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::LowMassStar);
						    // }
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::LowMassStar);
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<TestParticle>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<TestParticle>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<TestParticle>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<TestParticle>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<TestParticle> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	sim.initDt_ = dt_;
	sim.maxDt_ = dt_;

	// Read parameters from input file
	const amrex::ParmParse pp("problem");
	pp.query("refine_half_domain", refine_half_domain);

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// ----- Check Test particles -----

	const int n_SNR_particles = 8;
	const int n_particle_test = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Test)->getNumParticles();

	int status = 0; // Initialize to success

	if (amrex::ParallelDescriptor::IOProcessor()) {

		amrex::Print() << "Expected number of test particles: " << n_SNR_particles << "\n";
		amrex::Print() << "Actual number of test particles: " << n_particle_test << "\n";

		status = 1;
		if (n_particle_test == n_SNR_particles) {
			status = 0;
			amrex::Print() << "Relative error within tolerance.\n";
		}
		if (status > 0) {
			amrex::Print() << "Test failed.\n";
		}
	}

	return status;
}
