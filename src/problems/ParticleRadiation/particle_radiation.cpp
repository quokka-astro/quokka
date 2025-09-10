/// \file particle_radiation.cpp
/// \brief Defines a test problem for radiation from particles.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "util/valarray.hpp"

struct ParticleRadiationProblem {
};

constexpr double m_stars_over_M_solar = 100.0; // mass of stars read from file
constexpr double star_lum_per_M_solar = 4.0e33;
constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
constexpr double rho0 = 1.0e-8 * C::m_p; // g cm^-3
constexpr double T0 = 10.0;		 // K
constexpr double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
constexpr double initial_Erad = 1.0e-30 * CV * rho0 * T0;
constexpr double dt_ = 1.0e10; // s
constexpr double chat_over_c = 1.0e-5;
constexpr double formation_time = 1.5 * dt_;
static bool refine_half_domain = false; // NOLINT

constexpr double box_size_half = 3.0e18; // This should be fixed for this problem.
constexpr double particle_offset_from_center_ = 1e-3 * box_size_half;

// locations of the particles: a 2x2x2 grids of particles
// constexpr double box_left_edge_ = -2.0;
// need to be smaller than smallest possible cell size, but not too small to avoid huge gravitational force
const static double SN_mass = 8.0 * C::M_solar; // mass of SNProgenitor particles in grams
constexpr int n_test_particles_init = 4;	// 4 test particles created at the start of the simulation
constexpr int n_test_particles_created = 8;	// 8 test particles created and live to the end

template <> struct quokka::EOS_Traits<ParticleRadiationProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

// Test enum to demonstrate type checking of particle_switch
enum class TestEnum : unsigned int {
	MISTAKE = 0b00000100U,
};

template <> struct Particle_Traits<ParticleRadiationProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<ParticleRadiationProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleRadiationProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<ParticleRadiationProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleRadiationProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleRadiationProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0;
}

template <> void QuokkaSimulation<ParticleRadiationProblem>::createInitialStochasticStellarPopParticles()
{
	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile("../inputs/TestParticlesNoRad.txt", nreal_extra, nullptr);

	// Using a for loop from lev = 0 to StochasticStellarPopParticles->maxLevel() won't work because not all levels necessarily have particles, and when
	// some levels do not have particles, StochasticStellarPopParticles->GetParticles(lev) will result in a Segfault. Therefore, we loop over the actual
	// particle container.
	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

			auto *pdata = particle_array().data();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

// Specialization for star particles with stellar evolution
namespace quokka
{
template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
	static constexpr double star_lum_per_M_solar = 4.0e33;

	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time) noexcept
	{
		const int mass_idx = StochasticStellarPopParticleMassIdx;
		const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
		const int lum_idx = StochasticStellarPopParticleLumIdx;
		const amrex::Real age = current_time - p.rdata(birth_time_idx);
		const amrex::Real mass = p.rdata(mass_idx);

		// A simple luminosity function for testing purpose. Keep it linear function of mass for easy answer
		// validation. L/(M / M_sun) = L_sun = 4e33 erg/s
		const double is_on = age < 1.0e14 ? 1.0 : 0.0; // 3 Myr

		// Update luminosity components (they are stored consecutively starting at lum_idx)
		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
			const amrex::Real luminosity = star_lum_per_M_solar * (mass / C::M_solar) * (g + 1) * is_on; // erg / s
			p.rdata(lum_idx + g) = luminosity;
		}
	}
};
} // namespace quokka

template <> void QuokkaSimulation<ParticleRadiationProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double rho = rho0;
		const double rho_e = CV * T0 * rho;

		// Set radiation variables
		for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasInternalEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x1GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x2GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x3GasMomentum_index) = 0.0;
	});
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<ParticleRadiationProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ParticleRadiationProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ParticleRadiationProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		// Check radiation flux components
		for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
			if ((n == RadSystem<ParticleRadiationProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) && (dim == 0)) {
				return true;
			}
			if ((n == RadSystem<ParticleRadiationProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) && (dim == 1)) {
				return true;
			}
			if ((n == RadSystem<ParticleRadiationProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) && (dim == 2)) {
				return true;
			}
		}
		return false;
	};

	auto BCs_cc = quokka::BC<ParticleRadiationProblem>(quokka::BoundaryCondition::reflecting, quokka::BoundaryCondition::reflecting, quokka::BoundaryCondition::reflecting);

	// Problem initialization
	QuokkaSimulation<ParticleRadiationProblem> sim(BCs_cc);
	sim.initDt_ = dt_;
	sim.maxDt_ = dt_;

	// Read parameters from input file
	const amrex::ParmParse pp("problem");
	pp.query("refine_half_domain", refine_half_domain);

	// initialize
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	// Total radiation energy in the field
	amrex::Real total_Erad_init = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
		total_Erad_init +=
		    sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}

	// total gas energy
	const amrex::Real total_gas_energy_init = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::gasEnergy_index) * vol;

	// set force finest level to true for test particles
	// sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->setForceFinestLevel(true);

	// evolve
	sim.evolve();

	// ----- Check Stochastic particles -----

	// Total radiation energy in the field
	amrex::Real total_Erad = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
		total_Erad += sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}

	// total gas energy
	const amrex::Real total_gas_energy = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::gasEnergy_index) * vol;

	int status = 0; // Initialize to success

	if (amrex::ParallelDescriptor::IOProcessor()) {

		// print total gas energy
		amrex::Print() << "Total gas energy (initial): " << total_gas_energy_init << "\n";
		amrex::Print() << "Total gas energy (final): " << total_gas_energy << "\n";
		amrex::Print() << "Total radiation energy (initial): " << total_Erad_init / chat_over_c << "\n";
		amrex::Print() << "Total radiation energy (final): " << total_Erad / chat_over_c << "\n";

		const double total_energy_init = total_Erad_init / chat_over_c + total_gas_energy_init;
		const double total_energy = total_Erad / chat_over_c + total_gas_energy;
		const double change_of_total_energy = total_energy - total_energy_init;
		amrex::Print() << "Change of total energy: " << change_of_total_energy << "\n";

		// expected answer. Raidation is not deposited into cells, so we have to subtract dt_ from the total time.
		const double change_of_total_energy_expected = 4 * (star_lum_per_M_solar * m_stars_over_M_solar) * (sim.tNew_[0] - dt_);
		amrex::Print() << "Current time: " << sim.tNew_[0] << "\n";
		amrex::Print() << "Expected change of total energy: " << change_of_total_energy_expected << "\n";

		const double relative_error = std::abs(change_of_total_energy - change_of_total_energy_expected) / total_energy;
		amrex::Print() << "Relative error: " << relative_error << "\n";

		const double tolerance = 1e-12; // should be accurate to machine precision
		if (!(relative_error < tolerance)) {
			status = 1;
			amrex::Print() << "Test failed: change of total energy mismatch.\n";
		}

		if (status == 0) {
			amrex::Print() << "Test passed: change of total energy within tolerance.\n";
		}
	}

	return status;
}
