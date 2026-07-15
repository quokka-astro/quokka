/// \file test_SNII_Yields.cpp
/// \brief Defines a compact StochasticStellarPop test problem for SNII yield validation.
///

#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "problems/chemical_yield_test_utils.hpp"

struct test_SNII_Yields {
};

constexpr Real gamma_ = 5. / 3.;
constexpr Real year = 3.15576e+07;
static Real n0 = 1.0e4;									// NOLINT
static Real Tamb = 10.0;								// NOLINT
static std::string initial_particles_file = "../inputs/test_SNII_Yields_particles.txt"; // NOLINT

template <> struct quokka::EOS_Traits<test_SNII_Yields> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = 1.0;
};

template <> struct Particle_Traits<test_SNII_Yields> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<test_SNII_Yields> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<test_SNII_Yields> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 3;
	static constexpr int nGroups = 1;
};

template <> void QuokkaSimulation<test_SNII_Yields>::createInitialStochasticStellarPopParticles()
{
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<test_SNII_Yields>;
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(initial_particles_file, nreal_extra, nullptr);

	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

			auto *pdata = particle_array().data();
			const int chem_base = quokka::StochasticStellarPopParticleChemistryBaseIdx<test_SNII_Yields>();

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				pdata[i].idata(quokka::StochasticStellarPopParticleStageIdx) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
				// Non-zero birth abundances catch accidental double-counting in table-driven yields.
				pdata[i].rdata(chem_base) = 1.0e-3;
				pdata[i].rdata(chem_base + 1) = 2.0e-3;
				pdata[i].rdata(chem_base + 2) = 3.0e-3;
			});
		}
	}

	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<test_SNII_Yields>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho = n0 * 1.0;
	const double e_int = 1.0 / (gamma_ - 1.0) * rho * C::k_B * Tamb;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<test_SNII_Yields>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<test_SNII_Yields>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<test_SNII_Yields>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<test_SNII_Yields>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<test_SNII_Yields>::energy_index) = e_int;
		state_cc(i, j, k, HydroSystem<test_SNII_Yields>::internalEnergy_index) = e_int;
		for (int n = 0; n < Physics_Traits<test_SNII_Yields>::numPassiveScalars; ++n) {
			state_cc(i, j, k, HydroSystem<test_SNII_Yields>::scalar0_index + n) = 0.0;
		}
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<test_SNII_Yields> sim;

	sim.reconstructionOrder_ = 3;
	sim.cflNumber_ = 0.5;
	sim.stopTime_ = 1.0e7 * year;

	const int seed = 42;
	amrex::InitRandom(seed, 1);
	// TODO: remove seed

	amrex::ParmParse const ppp("problem");
	ppp.query("Tamb", Tamb);
	ppp.query("n0", n0);
	ppp.query("initial_particles_file", initial_particles_file);

	sim.setInitialConditions();

	sim.evolve();
	quokka::ChemicalYieldTest::validateSNIIYields(sim, initial_particles_file, {"C12", "N14", "O16"});
	// TODO: move the content of the function here
	amrex::Print() << "test_SNII_Yields completed\n";
	return 0;
}
