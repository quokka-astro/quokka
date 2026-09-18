/// \file test_SNII_Yields.cpp
/// \brief Defines a compact StochasticStellarPop test problem for SNII yield validation.
///

#include "AMReX_BLassert.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_chemical_yield.hpp"
#include "particles/particle_types.hpp"

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
	const amrex::Real error = std::abs(simulated - expected);
	const amrex::Real allowed_error = tolerance * std::abs(expected);
	amrex::Print() << label << ": simulated=" << simulated << " expected=" << expected << " absolute_error=" << error << "\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error <= allowed_error, std::format("{} failed: error={} > {}", label, error, allowed_error).c_str());
}

template <typename problem_t>
void validateSNIIYields(const QuokkaSimulation<problem_t> &sim, const std::string &initial_particles_file, const std::vector<std::string> &isotopes)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(quokka::ChemicalYieldLookup::isLoaded(), "chemical yield tables were not loaded");
	const auto records = readInitialParticleRecords(initial_particles_file, quokka::StochasticStellarPopParticleRealComps<problem_t>);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!records.empty(), "test_SNII_Yields requires at least one initial particle");

	const auto tables = quokka::ChemicalYieldLookup::constTablesHost();
	const amrex::Real birth_mass = records.front().rdata[static_cast<std::size_t>(quokka::StochasticStellarPopParticleMassAtBirthIdx)];

	amrex::Print() << "test_SNII_Yields simulated/table:\n";
	for (std::size_t n = 0; n < isotopes.size(); ++n) {
		const int n_idx = static_cast<int>(n);
		const amrex::Real expected = yieldFraction(tables, 0, n_idx, birth_mass) * birth_mass;
		const amrex::Real measured = scalarMass(sim, n_idx);
		assertClose(std::format("  {} scalar_{}", isotopes[n], n_idx), measured, expected);
	}
}

} // namespace

struct test_SNII_Yields {};

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
	static constexpr bool enable_chemical_feedback = true;
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

struct YieldStorageDisabled {};

template <> struct Physics_Traits<YieldStorageDisabled> : Physics_Traits<test_SNII_Yields> {};

static_assert(quokka::StochasticStellarPopParticleChemistryBlockSize<YieldStorageDisabled>() == 0);
static_assert(quokka::StochasticStellarPopParticleRealComps<YieldStorageDisabled> ==
	      quokka::StochasticStellarPopParticleLumIdx + Physics_Traits<YieldStorageDisabled>::nGroups);
static_assert(quokka::StochasticStellarPopParticleRealComps<test_SNII_Yields> ==
	      quokka::StochasticStellarPopParticleRealComps<YieldStorageDisabled> + 4 * Physics_Traits<test_SNII_Yields>::numPassiveScalars);

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
	const volatile amrex::Real zero_yield = 0.0;
	assertClose("zero yield", zero_yield, zero_yield);
	const auto legacy_names = quokka::getParticleRealCompNames<quokka::ParticleType::StochasticStellarPop, YieldStorageDisabled>();
	AMREX_ALWAYS_ASSERT(legacy_names.size() == quokka::StochasticStellarPopParticleRealComps<YieldStorageDisabled>);
	const auto chemistry_names = quokka::getParticleRealCompNames<quokka::ParticleType::StochasticStellarPop, test_SNII_Yields>();
	AMREX_ALWAYS_ASSERT(chemistry_names.size() == quokka::StochasticStellarPopParticleRealComps<test_SNII_Yields>);
	bool test_zero_yield_only = false;
	amrex::ParmParse("problem").query("test_zero_yield_only", test_zero_yield_only);
	if (test_zero_yield_only) {
		return 0;
	}
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
	validateSNIIYields(sim, initial_particles_file, {"C12", "N14", "O16"});
	amrex::Print() << "test_SNII_Yields completed\n";
	return 0;
}
