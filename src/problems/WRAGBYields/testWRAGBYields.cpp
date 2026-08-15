/// \file testWRAGBYields.cpp
/// \brief Defines a compact StochasticStellarPop test problem for WR/AGB yield validation.
///

#include "AMReX_BLassert.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_chemical_yield.hpp"
#include "particles/particle_types.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr amrex::Real yield_validation_rtol = 1.0e-10;

struct InitialParticleRecord {
	std::vector<amrex::Real> rdata;
};

auto readInitialParticleRecords(const std::string &filename, int nreal) -> std::vector<InitialParticleRecord>
{
	std::ifstream input(filename);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input.is_open(), ("failed to open initial particle file: " + filename).c_str());

	int count = 0;
	input >> count;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(count >= 0, ("invalid particle count in file: " + filename).c_str());

	std::vector<InitialParticleRecord> records;
	records.reserve(static_cast<std::size_t>(count));
	for (int p = 0; p < count; ++p) {
		amrex::Real pos = 0.0;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			input >> pos;
		}

		InitialParticleRecord record{};
		record.rdata.resize(static_cast<std::size_t>(nreal));
		for (int n = 0; n < nreal; ++n) {
			input >> record.rdata[static_cast<std::size_t>(n)];
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input.good(), ("failed to read particle data from file: " + filename).c_str());
		records.push_back(std::move(record));
	}

	return records;
}

template <typename problem_t> [[nodiscard]] auto cellVolume(const QuokkaSimulation<problem_t> &sim) -> amrex::Real
{
	const auto dx = sim.Geom(0).CellSizeArray();
	return AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
}

template <typename problem_t> [[nodiscard]] auto scalarMass(const QuokkaSimulation<problem_t> &sim, int scalar_index) -> amrex::Real
{
	const int comp = HydroSystem<problem_t>::scalar0_index + scalar_index;
	return sim.state_new_cc_[0].sum(comp) * cellVolume(sim);
}

auto yieldFraction(const quokka::ChemicalYieldLookup::ChemicalYieldGpuConstTables &tables, int channel_index, int isotope_index, amrex::Real mass)
    -> amrex::Real
{
	return quokka::ChemicalYieldLookup::queryYieldFraction(tables, channel_index, isotope_index, mass / C::M_solar, quokka::stellar_metallicity_fraction);
}

void assertClose(const std::string &label, amrex::Real simulated, amrex::Real expected, amrex::Real tolerance = yield_validation_rtol)
{
	const amrex::Real error = (expected > 0.0) ? std::abs(simulated / expected - 1.0) : std::abs(simulated);
	const amrex::Real ratio = (expected > 0.0) ? simulated / expected : 1.0;
	amrex::Print() << label << ": simulated=" << simulated << " expected=" << expected << " sim/expected=" << ratio << "\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error <= tolerance, std::format("{} failed: error={} > {}", label, error, tolerance).c_str());
}

template <typename problem_t>
void validateWRAGBYields(const QuokkaSimulation<problem_t> &sim, const std::string &initial_particles_file, const std::vector<std::string> &isotopes)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(quokka::ChemicalYieldLookup::isLoaded(), "chemical yield tables were not loaded");
	const auto records = readInitialParticleRecords(initial_particles_file, quokka::StochasticStellarPopParticleRealComps<problem_t>);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(records.size() >= 2, "WRAGBYields requires at least two initial particles");

	const auto tables = quokka::ChemicalYieldLookup::constTablesHost();
	int wr_index = -1;
	int agb_index = -1;
	for (std::size_t i = 0; i < records.size(); ++i) {
		const amrex::Real mass_msun = records[i].rdata[static_cast<std::size_t>(quokka::StochasticStellarPopParticleMassAtBirthIdx)] / C::M_solar;
		if (mass_msun >= 9.0 && wr_index < 0) {
			wr_index = static_cast<int>(i);
		}
		if (mass_msun <= 8.0 && agb_index < 0) {
			agb_index = static_cast<int>(i);
		}
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(wr_index >= 0, "WRAGBYields did not find a WR-mass particle");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(agb_index >= 0, "WRAGBYields did not find an AGB-mass particle");

	const auto &wr_record = records[static_cast<std::size_t>(wr_index)].rdata;
	const auto &agb_record = records[static_cast<std::size_t>(agb_index)].rdata;
	const amrex::Real wr_mass = wr_record[static_cast<std::size_t>(quokka::StochasticStellarPopParticleMassAtBirthIdx)];
	const amrex::Real agb_mass = agb_record[static_cast<std::size_t>(quokka::StochasticStellarPopParticleMassAtBirthIdx)];
	const amrex::Real wr_birth_time = wr_record[static_cast<std::size_t>(quokka::StochasticStellarPopParticleBirthTimeIdx)];
	const amrex::Real wr_death_time = wr_record[static_cast<std::size_t>(quokka::StochasticStellarPopParticleDeathTimeIdx)];
	const amrex::Real wr_lifetime = std::max<amrex::Real>(wr_death_time - wr_birth_time, 0.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(wr_lifetime > 0.0, "WR particle has non-positive lifetime");

	const amrex::Real wr_elapsed = std::min(sim.tNew_[0] - wr_birth_time, wr_lifetime);
	const amrex::Real wr_distribution = quokka::ChemicalYieldLookup::queryWRMassLossCumulativeFraction(tables, wr_elapsed, wr_mass / C::M_solar);

	amrex::Print() << "WRAGBYields simulated/table:\n";
	for (std::size_t n = 0; n < isotopes.size(); ++n) {
		const int n_idx = static_cast<int>(n);
		const amrex::Real wr_expected = yieldFraction(tables, 1, n_idx, wr_mass) * wr_mass * wr_distribution;
		const amrex::Real agb_expected = yieldFraction(tables, 2, n_idx, agb_mass) * agb_mass;
		const amrex::Real total_expected = wr_expected + agb_expected;

		const amrex::Real measured_total = scalarMass(sim, n_idx);
		const amrex::Real measured_snii = scalarMass(sim, 3 + n_idx);
		const amrex::Real measured_wr = scalarMass(sim, 6 + n_idx);
		const amrex::Real measured_agb = scalarMass(sim, 9 + n_idx);

		assertClose(std::format("  {} total scalar_{}", isotopes[n], n_idx), measured_total, total_expected);
		assertClose(std::format("  {} WR scalar_{}", isotopes[n], 6 + n_idx), measured_wr, wr_expected);
		assertClose(std::format("  {} AGB scalar_{}", isotopes[n], 9 + n_idx), measured_agb, agb_expected);
		assertClose(std::format("  {} SNII scalar_{}", isotopes[n], 3 + n_idx), measured_snii, 0.0);
	}
}

} // namespace

struct WRAGBYields {
};

constexpr Real gamma_ = 5. / 3.;
static Real n0 = 1.0e4;									  // NOLINT
static Real Tamb = 10.0;								  // NOLINT
static std::string initial_particles_file = "../inputs/test_WR_AGB_yields_particles.txt"; // NOLINT

template <> struct quokka::EOS_Traits<WRAGBYields> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = 1.0;
};

template <> struct Particle_Traits<WRAGBYields> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<WRAGBYields> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<WRAGBYields> : DefaultPhysicsTraits {
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

template <> void QuokkaSimulation<WRAGBYields>::createInitialStochasticStellarPopParticles()
{
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<WRAGBYields>;
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
			const int chem_base = quokka::StochasticStellarPopParticleChemistryBaseIdx<WRAGBYields>();

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

template <> void QuokkaSimulation<WRAGBYields>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho = n0 * 1.0;
	const double e_int = 1.0 / (gamma_ - 1.0) * rho * C::k_B * Tamb;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<WRAGBYields>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<WRAGBYields>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<WRAGBYields>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<WRAGBYields>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<WRAGBYields>::energy_index) = e_int;
		state_cc(i, j, k, HydroSystem<WRAGBYields>::internalEnergy_index) = e_int;
		for (int n = 0; n < Physics_Traits<WRAGBYields>::numPassiveScalars; ++n) {
			state_cc(i, j, k, HydroSystem<WRAGBYields>::scalar0_index + n) = 0.0;
		}
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<WRAGBYields> sim;

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
	validateWRAGBYields(sim, initial_particles_file, {"C12", "O16", "Fe56"});
	amrex::Print() << "WRAGBYields completed\n";
	return 0;
}
