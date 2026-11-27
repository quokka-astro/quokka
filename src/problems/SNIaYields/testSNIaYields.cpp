/// \file SNIaYields.cpp
/// \brief Compact test problem for table-driven SNIa yield validation.

#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_snia_feedback.hpp"
#include "particles/particle_types.hpp"
#include "problems/YieldValidation.hpp"

#include <array>
#include <vector>

struct SNIaYields {};

constexpr Real gamma_ = 5. / 3.;
static Real n0 = 1.0;	 // NOLINT
static Real Tamb = 10.0; // NOLINT

template <> struct SimulationData<SNIaYields> {
	bool snia_deposited = false;
};

template <> struct quokka::EOS_Traits<SNIaYields> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Particle_Traits<SNIaYields> : DefaultParticleTraits {
	static constexpr bool enable_chemical_feedback = true;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<SNIaYields> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<SNIaYields> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = (quokka::ChemicalYieldLookup::max_tracked_channels + 1) * 3;
};

template <> void QuokkaSimulation<SNIaYields>::createInitialStochasticStellarPopParticles() { amrex::Print() << "No star particles needed for SNIaYields.\n"; }

template <> void QuokkaSimulation<SNIaYields>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho = n0 * C::m_u;
	const double e_int = 1.0 / (gamma_ - 1.0) * n0 * C::k_B * Tamb;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SNIaYields>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<SNIaYields>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNIaYields>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNIaYields>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNIaYields>::energy_index) = e_int;
		state_cc(i, j, k, HydroSystem<SNIaYields>::internalEnergy_index) = e_int;
		for (int n = 0; n < Physics_Traits<SNIaYields>::numPassiveScalars; ++n) {
			state_cc(i, j, k, HydroSystem<SNIaYields>::scalar0_index + n) = 0.0;
		}
	});
}

template <> void QuokkaSimulation<SNIaYields>::computeAfterTimestep()
{
	if (userData_.snia_deposited) {
		return;
	}
	userData_.snia_deposited = true;

	const int lev = 0;
	state_new_cc_[lev].FillBoundary(geom[lev].periodicity());
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto prob_hi = geom[lev].ProbHiArray();
	quokka::SNFeedbackUtils::SNFeedbackEvent event{};
	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		event.position[dir] = 0.5 * (prob_lo[dir] + prob_hi[dir]);
		event.velocity[dir] = 0.0;
	}
	const std::vector<quokka::SNFeedbackUtils::SNFeedbackEvent> events{event};

	std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc_ptr = nullptr;
	const amrex::Real snia_ejecta_mass = 1.4 * C::M_solar;
	const amrex::Real snia_energy = 1.0e51;
	const auto cell_size = geom[lev].CellSizeArray();
	const amrex::Real cell_volume = cell_size[0] * cell_size[1] * cell_size[2];
	const amrex::Real initial_mass = state_new_cc_[lev].sum(HydroSystem<SNIaYields>::density_index) * cell_volume;
	const auto empty_result =
	    quokka::SNFeedbackUtils::depositSNFeedbackEvents<SNIaYields>(state_new_cc_[lev], state_fc_ptr, geom[lev], {}, snia_ejecta_mass, snia_energy);
	AMREX_ALWAYS_ASSERT(empty_result.first == 0);
	quokka::ChemicalFeedbackUtils::depositSNIaChemicalFeedback<SNIaYields>(state_new_cc_[lev], geom[lev], {}, snia_ejecta_mass);
	const auto [deposited_events, max_velocity] =
	    quokka::SNFeedbackUtils::depositSNFeedbackEvents<SNIaYields>(state_new_cc_[lev], state_fc_ptr, geom[lev], events, snia_ejecta_mass, snia_energy);
	quokka::ChemicalFeedbackUtils::depositSNIaChemicalFeedback<SNIaYields>(state_new_cc_[lev], geom[lev], events, snia_ejecta_mass);
	AMREX_ALWAYS_ASSERT(deposited_events == 1);
	const amrex::Real final_mass = state_new_cc_[lev].sum(HydroSystem<SNIaYields>::density_index) * cell_volume;
	quokka::testing::assertYieldClose("SNIa ejecta mass", final_mass - initial_mass, snia_ejecta_mass);
	const auto tables = quokka::ChemicalYieldLookup::constTablesHost();
	for (int isotope = 0; isotope < 3; ++isotope) {
		const amrex::Real expected =
		    snia_ejecta_mass * quokka::ChemicalYieldLookup::queryYieldFraction(tables, quokka::ChemicalYieldLookup::snia_channel_index, isotope,
										       snia_ejecta_mass / C::M_solar, quokka::stellar_metallicity_fraction);
		const int scalar_base = HydroSystem<SNIaYields>::scalar0_index + isotope;
		quokka::testing::assertYieldClose("SNIa total isotope mass", state_new_cc_[lev].sum(scalar_base) * cell_volume, expected);
		for (int channel = 0; channel < quokka::ChemicalYieldLookup::max_tracked_channels; ++channel) {
			const amrex::Real channel_expected = channel == quokka::ChemicalYieldLookup::snia_channel_index ? expected : 0.0;
			quokka::testing::assertYieldClose("SNIa channel isotope mass", state_new_cc_[lev].sum(scalar_base + (channel + 1) * 3) * cell_volume,
							  channel_expected);
		}
	}
	amrex::Print() << "SNIaYields deposited SNIa events=" << deposited_events << ", max signal speed=" << max_velocity << " cm/s\n";
}

auto problem_main() -> int
{
	QuokkaSimulation<SNIaYields> sim;

	sim.reconstructionOrder_ = 3;
	sim.cflNumber_ = 0.5;
	sim.stopTime_ = 1.0e10;

	const int seed = 42;
	amrex::InitRandom(seed, 1);

	amrex::ParmParse const ppp("problem");
	ppp.query("Tamb", Tamb);
	ppp.query("n0", n0);

	sim.setInitialConditions();

	sim.evolve();
	amrex::Print() << "SNIaYields completed\n";
	return 0;
}
