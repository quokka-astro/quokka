/// \file particle_creation_from_cell.cpp
/// \brief Defines a test problem for particle creation.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "math/interpolate.hpp"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "util/valarray.hpp"

struct TestParticle {
};

constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
constexpr double rho0 = 1.0e8 * C::m_p; // g cm^-3
constexpr double T0 = 10.0;		  // K
constexpr double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
constexpr double initial_Erad = 1.0e-30 * CV * rho0 * T0;
constexpr double dt_ = 1.0e10; // s
constexpr double formation_time = 1.5 * dt_;
static bool refine_half_domain = false; // NOLINT

constexpr double box_size_half = 3.0e18; // This should be fixed for this problem.
constexpr double particle_offset_from_center_ = 1e-3 * box_size_half;

// locations of the particles: a 2x2x2 grids of particles
// constexpr double box_left_edge_ = -2.0;
// need to be smaller than smallest possible cell size, but not too small to avoid huge gravitational force
const static double SN_mass = 8.0 * C::M_solar;  // mass of SNProgenitor particles in grams
constexpr int n_test_particles_init = 8;    // 8 test particles created at the start of the simulation
constexpr int n_test_particles_created = 8; // 8 test particles created and live to the end

template <> struct quokka::EOS_Traits<TestParticle> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
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
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<TestParticle> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<TestParticle> {
	static constexpr double c_hat_over_c = 1.0e-5;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TestParticle>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0e-30; // very low opacity for testing
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TestParticle>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0e-10; // very low opacity for testing
}

template <> void QuokkaSimulation<TestParticle>::createInitialTestParticles()
{
	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	TestParticles->SetVerbose(1);
	TestParticles->InitFromAsciiFile("../inputs/TestParticles.txt", nreal_extra, nullptr);

	// Using a for loop from lev = 0 to TestParticles->maxLevel() won't work because not all levels necessarily have particles, and when some levels
	// do not have particles, TestParticles->GetParticles(lev) will result in a Segfault. Therefore, we loop over the actual particle container.
	for (auto &kv : TestParticles->GetParticles()) {
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
				if (p.rdata(0) > SN_mass) {
					p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
				} else {
					// For testing purposes, we mark particles with mass < SN_mass as Removed. These particles will be removed in current
					// timestep.
					p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::Removed);
				}
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

namespace quokka
{
// Specialization for Test particle creation
template <> struct ParticleCreationTraits<ParticleType::Test> {
	// Specialized nested ParticleChecker for Test particles
	template <typename problem_t> struct ParticleChecker {
		amrex::Real current_time;
		amrex::Real dt;

		double x_L = -box_size_half; // This is a hack. Since the domain size is not passed to here, we have to manually set it for this test problem
		double offset = particle_offset_from_center_;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const & /*state_arr*/,
						 amrex::Array4<const amrex::Real> const & /*state_accretion_rate_arr*/, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::RandomEngine const & /*engine*/) const -> int
		{
			// A simple demonstration of particle creation
			// Could check density threshold or other state-based conditions

			// Calculate cell indices for the particle locations
			const int i_par1 = static_cast<int>(floor((-offset - x_L) / dx[0]));
			const int j_par1 = static_cast<int>(floor((-offset - x_L) / dx[1]));
			const int k_par1 = static_cast<int>(floor((-offset - x_L) / dx[2]));

			const int i_par2 = static_cast<int>(floor((offset - x_L) / dx[0]));
			const int j_par2 = static_cast<int>(floor((offset - x_L) / dx[1]));
			const int k_par2 = static_cast<int>(floor((offset - x_L) / dx[2]));

			const bool is_create_particle = current_time <= formation_time && current_time + dt > formation_time;
			if (is_create_particle && (i == i_par1 || i == i_par2) && (j == j_par1 || j == j_par2) && (k == k_par1 || k == k_par2)) {
				return 1;
			}
			return 0;
		}
	};

	// Specialized nested ParticleCreator for Test particles
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int birth_time_index;
		int evolution_stage_index;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;
		amrex::Real dt;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index,
				amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index), cpu_id(processor_id),
		      pid_start(particle_id_start), current_time(current_time), dt(dt)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void
		operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const & /*state_accretion_rate_arr*/, int i,
			   int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
			   amrex::Long base_offset, amrex::RandomEngine const & /*engine*/) const
		{
			if (mass_idx + 3 < ParticleType::NReal) {
				// Calculate common values for all particles
				const amrex::Real cell_density = state_arr(i, j, k, RadSystem<problem_t>::gasDensity_index);

				const amrex::Real vx = state_arr(i, j, k, RadSystem<problem_t>::x1GasMomentum_index) / cell_density;
				const amrex::Real vy = state_arr(i, j, k, RadSystem<problem_t>::x2GasMomentum_index) / cell_density;
				const amrex::Real vz = state_arr(i, j, k, RadSystem<problem_t>::x3GasMomentum_index) / cell_density;

				// Create all particles
				for (int p_idx = 0; p_idx < num_particles; ++p_idx) {
					auto &p = particles[p_idx]; // NOLINT

					// Set particle position at cell center
					p.pos(0) = plo[0] + (i + 0.5) * dx[0];
					p.pos(1) = plo[1] + (j + 0.5) * dx[1];
					p.pos(2) = plo[2] + (k + 0.5) * dx[2];

					// Set particle ID and CPU
					p.id() = pid_start + base_offset + p_idx;
					p.cpu() = cpu_id;

					// Initialize particle properties
					p.rdata(mass_idx) = SN_mass;
					p.rdata(mass_idx + 1) = vx;
					p.rdata(mass_idx + 2) = vy;
					p.rdata(mass_idx + 3) = vz;

					// set birth time to current time
					p.rdata(birth_time_index) = current_time;
					// set death time to infinity, so that particle will never die
					p.rdata(birth_time_index + 1) = std::numeric_limits<double>::max();

					// Set particle evolution stage
					p.idata(evolution_stage_index) = static_cast<int>(StellarEvolutionStage::SNProgenitor);
				}

				const amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
				const amrex::Real tot_SN_mass = num_particles * SN_mass;
				const amrex::Real cell_mass = cell_density * volume;
				AMREX_ASSERT(cell_mass > tot_SN_mass);
				state_arr(i, j, k, RadSystem<problem_t>::gasDensity_index) -= tot_SN_mass / volume;
			}
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, int lev,
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index, int birth_time_index)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<ParticleType::Test>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::Test>::template ParticleCreator>(
		    container, mass_idx, state, state_accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
};
} // namespace quokka

template <> void QuokkaSimulation<TestParticle>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double rho = rho0;
		const double rho_e = CV * T0 * rho;

		// Set radiation variables
		for (int g = 0; g < Physics_Traits<TestParticle>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<TestParticle>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad0;
			state_cc(i, j, k, RadSystem<TestParticle>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TestParticle>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TestParticle>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<TestParticle>::gasEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<TestParticle>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TestParticle>::gasInternalEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<TestParticle>::x1GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<TestParticle>::x2GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<TestParticle>::x3GasMomentum_index) = 0.0;
	});
}

template <> void QuokkaSimulation<TestParticle>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) {}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<TestParticle>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TestParticle>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TestParticle>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		// Check radiation flux components
		for (int g = 0; g < Physics_Traits<TestParticle>::nGroups; ++g) {
			if ((n == RadSystem<TestParticle>::x1RadFlux_index + Physics_NumVars::numRadVars * g) && (dim == 0)) {
				return true;
			}
			if ((n == RadSystem<TestParticle>::x2RadFlux_index + Physics_NumVars::numRadVars * g) && (dim == 1)) {
				return true;
			}
			if ((n == RadSystem<TestParticle>::x3RadFlux_index + Physics_NumVars::numRadVars * g) && (dim == 2)) {
				return true;
			}
		}
		return false;
	};

	const int ncomp_cc = RadSystem<TestParticle>::nvar_;
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

	// set force finest level to true for test particles
	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Test)->setForceFinestLevel(true);

	// evolve
	sim.evolve();

	// ----- Check Test particles -----

	const int n_particle_expected = n_test_particles_init / 2 + n_test_particles_created;
	const int n_particle_test = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Test)->getNumParticles();

	int status = 0; // Initialize to success

	if (amrex::ParallelDescriptor::IOProcessor()) {

		amrex::Print() << "Expected number of test particles: " << n_particle_expected << "\n";
		amrex::Print() << "Actual number of test particles: " << n_particle_test << "\n";

		status = 1;
		if (n_particle_test == n_particle_expected) {
			status = 0;
			amrex::Print() << "Relative error within tolerance.\n";
		}
		if (status > 0) {
			amrex::Print() << "Test failed.\n";
		}
	}

	return status;
}
