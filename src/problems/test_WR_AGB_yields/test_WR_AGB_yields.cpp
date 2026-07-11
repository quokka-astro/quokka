/// Minimal test problem to validate WR continuous and AGB death-time metal feedback.
/// Creates two high-mass star particles, one WR source and one AGB source, and forces
/// their evolutionary stage so the feedback paths can be checked directly.

#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "problems/chemical_yield_test_utils.hpp"

struct test_WR_AGB_yields {
};

constexpr Real gamma_ = 5. / 3.;
static Real n0 = 1.0e4;									  // NOLINT
static Real Tamb = 10.0;								  // NOLINT
static std::string initial_particles_file = "../inputs/test_WR_AGB_yields_particles.txt"; // NOLINT

template <> struct quokka::EOS_Traits<test_WR_AGB_yields> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = 1.0;
};

template <> struct Particle_Traits<test_WR_AGB_yields> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<test_WR_AGB_yields> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<test_WR_AGB_yields> : DefaultPhysicsTraits {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 12; // total + SNII + WR + AGB for 3 isotopes
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gravitational_constant = C::Gconst;
	static constexpr double c_light = C::c_light;
	static constexpr double radiation_constant = C::a_rad;
};

template <> void QuokkaSimulation<test_WR_AGB_yields>::createInitialStochasticStellarPopParticles()
{
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<test_WR_AGB_yields>;
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(initial_particles_file, nreal_extra, nullptr);

	// Force particle metadata using mass, avoiding tile-local particle ordering.
	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			if (np == 0) {
				continue;
			}
			auto *pdata = particle_array().data();
			const int chem_base = quokka::StochasticStellarPopParticleChemistryBaseIdx<test_WR_AGB_yields>();

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				pdata[i].idata(quokka::StochasticStellarPopParticleStageIdx) =
				    static_cast<int>(quokka::StellarEvolutionStage::HighMassNonExploding);
				// Non-zero birth abundances catch accidental double-counting in table-driven yields.
				pdata[i].rdata(chem_base) = 1.0e-3;
				pdata[i].rdata(chem_base + 1) = 2.0e-3;
				pdata[i].rdata(chem_base + 2) = 3.0e-3;
				if (pdata[i].rdata(quokka::StochasticStellarPopParticleMassAtBirthIdx) <= 8.0 * C::M_solar) {
					pdata[i].rdata(quokka::StochasticStellarPopParticleDeathTimeIdx) = 5.0e13;
				}
			});
		}
	}

	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<test_WR_AGB_yields>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho = n0 * 1.0;
	const double e_int = 1.0 / (gamma_ - 1.0) * rho * C::k_B * Tamb;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<test_WR_AGB_yields>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<test_WR_AGB_yields>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<test_WR_AGB_yields>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<test_WR_AGB_yields>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<test_WR_AGB_yields>::energy_index) = e_int;
		state_cc(i, j, k, HydroSystem<test_WR_AGB_yields>::internalEnergy_index) = e_int;
		for (int n = 0; n < Physics_Traits<test_WR_AGB_yields>::numPassiveScalars; ++n) {
			state_cc(i, j, k, HydroSystem<test_WR_AGB_yields>::scalar0_index + n) = 0.0;
		}
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<test_WR_AGB_yields> sim;

	sim.reconstructionOrder_ = 3;
	sim.cflNumber_ = 0.5;
	sim.stopTime_ = 1.0e14;

	const int seed = 42;
	amrex::InitRandom(seed, 1);

	amrex::ParmParse const ppp("problem");
	ppp.query("Tamb", Tamb);
	ppp.query("n0", n0);
	ppp.query("initial_particles_file", initial_particles_file);

	sim.setInitialConditions();

	sim.evolve();
	quokka::ChemicalYieldTest::validateWRAGBYields(sim, initial_particles_file, {"C12", "O16", "Fe56"});
	amrex::Print() << "test_WR_AGB_yields completed\n";
	return 0;
}
