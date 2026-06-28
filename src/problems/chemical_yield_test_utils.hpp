#ifndef CHEMICAL_YIELD_TEST_UTILS_HPP_
#define CHEMICAL_YIELD_TEST_UTILS_HPP_

#include "AMReX_BLassert.H"
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

namespace quokka::ChemicalYieldTest
{

inline constexpr amrex::Real yield_validation_rtol = 1.0e-10;

struct InitialParticleRecord {
	std::vector<amrex::Real> rdata;
};

inline auto readInitialParticleRecords(const std::string &filename, int nreal) -> std::vector<InitialParticleRecord>
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

inline auto yieldFraction(const ChemicalYieldLookup::ChemicalYieldGpuConstTables &tables, int channel_index, int isotope_index, amrex::Real mass) -> amrex::Real
{
	return ChemicalYieldLookup::queryYieldFraction(tables, channel_index, isotope_index, mass / C::M_solar, stellar_metallicity_fraction);
}

inline void assertClose(const std::string &label, amrex::Real simulated, amrex::Real expected, amrex::Real tolerance = yield_validation_rtol)
{
	const amrex::Real error = (expected > 0.0) ? std::abs(simulated / expected - 1.0) : std::abs(simulated);
	const amrex::Real ratio = (expected > 0.0) ? simulated / expected : 1.0;
	amrex::Print() << label << ": simulated=" << simulated << " expected=" << expected << " sim/expected=" << ratio << "\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error <= tolerance, std::format("{} failed: error={} > {}", label, error, tolerance).c_str());
}

template <typename problem_t>
void validateSNIIYields(const QuokkaSimulation<problem_t> &sim, const std::string &initial_particles_file, const std::vector<std::string> &isotopes)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ChemicalYieldLookup::isLoaded(), "chemical yield tables were not loaded");
	const auto records = readInitialParticleRecords(initial_particles_file, StochasticStellarPopParticleRealComps<problem_t>);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!records.empty(), "test_SNII_Yields requires at least one initial particle");

	const auto tables = ChemicalYieldLookup::constTables();
	const amrex::Real birth_mass = records.front().rdata[static_cast<std::size_t>(StochasticStellarPopParticleMassAtBirthIdx)];

	amrex::Print() << "test_SNII_Yields simulated/table:\n";
	for (int n = 0; n < static_cast<int>(isotopes.size()); ++n) {
		const amrex::Real expected = yieldFraction(tables, 0, n, birth_mass) * birth_mass;
		const amrex::Real measured = scalarMass(sim, n);
		assertClose(std::format("  {} scalar_{}", isotopes[static_cast<std::size_t>(n)], n), measured, expected);
	}
}

template <typename problem_t>
void validateWRAGBYields(const QuokkaSimulation<problem_t> &sim, const std::string &initial_particles_file, const std::vector<std::string> &isotopes)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ChemicalYieldLookup::isLoaded(), "chemical yield tables were not loaded");
	const auto records = readInitialParticleRecords(initial_particles_file, StochasticStellarPopParticleRealComps<problem_t>);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(records.size() >= 2, "test_WR_AGB_yields requires at least two initial particles");

	const auto tables = ChemicalYieldLookup::constTables();
	int wr_index = -1;
	int agb_index = -1;
	for (int i = 0; i < static_cast<int>(records.size()); ++i) {
		const amrex::Real mass_msun =
		    records[static_cast<std::size_t>(i)].rdata[static_cast<std::size_t>(StochasticStellarPopParticleMassAtBirthIdx)] / C::M_solar;
		if (mass_msun >= 9.0 && wr_index < 0) {
			wr_index = i;
		}
		if (mass_msun <= 8.0 && agb_index < 0) {
			agb_index = i;
		}
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(wr_index >= 0, "test_WR_AGB_yields did not find a WR-mass particle");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(agb_index >= 0, "test_WR_AGB_yields did not find an AGB-mass particle");

	const auto &wr_record = records[static_cast<std::size_t>(wr_index)].rdata;
	const auto &agb_record = records[static_cast<std::size_t>(agb_index)].rdata;
	const amrex::Real wr_mass = wr_record[static_cast<std::size_t>(StochasticStellarPopParticleMassAtBirthIdx)];
	const amrex::Real agb_mass = agb_record[static_cast<std::size_t>(StochasticStellarPopParticleMassAtBirthIdx)];
	const amrex::Real wr_birth_time = wr_record[static_cast<std::size_t>(StochasticStellarPopParticleBirthTimeIdx)];
	const amrex::Real wr_death_time = wr_record[static_cast<std::size_t>(StochasticStellarPopParticleDeathTimeIdx)];
	const amrex::Real wr_lifetime = std::max<amrex::Real>(wr_death_time - wr_birth_time, 0.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(wr_lifetime > 0.0, "WR particle has non-positive lifetime");

	const amrex::Real wr_elapsed = std::min(sim.tNew_[0] - wr_birth_time, wr_lifetime);
	const amrex::Real wr_distribution = ChemicalYieldLookup::queryWRMassLossCumulativeFraction(tables, wr_elapsed, wr_mass / C::M_solar);

	amrex::Print() << "test_WR_AGB_yields simulated/table:\n";
	for (int n = 0; n < static_cast<int>(isotopes.size()); ++n) {
		const amrex::Real wr_expected = yieldFraction(tables, 1, n, wr_mass) * wr_mass * wr_distribution;
		const amrex::Real agb_expected = yieldFraction(tables, 2, n, agb_mass) * agb_mass;
		const amrex::Real total_expected = wr_expected + agb_expected;

		const amrex::Real measured_total = scalarMass(sim, n);
		const amrex::Real measured_snii = scalarMass(sim, 3 + n);
		const amrex::Real measured_wr = scalarMass(sim, 6 + n);
		const amrex::Real measured_agb = scalarMass(sim, 9 + n);

		assertClose(std::format("  {} total scalar_{}", isotopes[static_cast<std::size_t>(n)], n), measured_total, total_expected);
		assertClose(std::format("  {} WR scalar_{}", isotopes[static_cast<std::size_t>(n)], 6 + n), measured_wr, wr_expected);
		assertClose(std::format("  {} AGB scalar_{}", isotopes[static_cast<std::size_t>(n)], 9 + n), measured_agb, agb_expected);
		assertClose(std::format("  {} SNII scalar_{}", isotopes[static_cast<std::size_t>(n)], 3 + n), measured_snii, 0.0);
	}
}

template <typename problem_t> void validateSNIaYields(const QuokkaSimulation<problem_t> &sim, const std::vector<std::string> &isotopes, amrex::Real ejecta_mass)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ChemicalYieldLookup::isLoaded(), "chemical yield tables were not loaded");
	const auto tables = ChemicalYieldLookup::constTables();

	amrex::Print() << "test_SNIa_Yields simulated/table:\n";
	for (int n = 0; n < static_cast<int>(isotopes.size()); ++n) {
		const amrex::Real expected = yieldFraction(tables, 3, n, ejecta_mass) * ejecta_mass;
		const amrex::Real measured_total = scalarMass(sim, n);
		const amrex::Real measured_snia = scalarMass(sim, 12 + n);
		assertClose(std::format("  {} total scalar_{}", isotopes[static_cast<std::size_t>(n)], n), measured_total, expected);
		assertClose(std::format("  {} SNIa scalar_{}", isotopes[static_cast<std::size_t>(n)], 12 + n), measured_snia, expected);
	}
}

} // namespace quokka::ChemicalYieldTest

#endif // CHEMICAL_YIELD_TEST_UTILS_HPP_
