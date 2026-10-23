/// \file testParticleCreation.cpp
/// \brief Defines a test problem for particle creation.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "math/interpolate.hpp"
#include "util/BC.hpp"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_schema_traits.hpp"

#include <array>

struct TestParticle {};

constexpr double rho0 = 1.0e-5;
constexpr double dt_ = 0.001;
static bool refine_half_domain = false; // NOLINT
constexpr double B0 = 1.0e-7;		// uniform background field

// locations of the particles: a 2x2x2 grids of particles
constexpr double box_left_edge_ = -2.0; // This should be fixed for this problem.
// need to be smaller than smallest possible cell size, but not too small to avoid huge gravitational force
constexpr double particle_offset_from_center_ = 0.01;
const static double SN_mass = 1.0e-5;	    // mass of SNProgenitor particles
constexpr int n_test_particles_init = 8;    // 8 test particles created at the start of the simulation
constexpr int n_test_particles_created = 8; // 8 test particles created and live to the end

template <> struct quokka::EOS_Traits<TestParticle> {
	static constexpr double gamma = 1.0;	     // isothermal
	static constexpr double cs_isothermal = 3.0; //
	static constexpr double mean_molecular_weight = 1.0;
};

// Test enum to demonstrate type checking of particle_switch
enum class TestEnum : unsigned int {
	MISTAKE = 0b00000100U,
};

template <> struct Particle_Traits<TestParticle> : DefaultParticleTraits {
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

template <> struct Physics_Traits<TestParticle> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

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
			const auto np = ikv.second.numParticles();
			const auto layout = TestParticles->Layout();

			if (np == 0) {
				continue;
			}

			const auto data = ikv.second.getParticleTileData();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto p = quokka::ParticleView(data, i, layout); // NOLINT
				if (p.rdata(0) > 1.0e-10) {
					p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
				} else {
					// For testing purposes, we mark particles with mass < 1.0e-10 as Removed. These particles will be removed in current
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
		amrex::Real param1 = particle_param1;

		double x_L = box_left_edge_;
		double offset = particle_offset_from_center_;

		ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const & /*state_arr*/,
						 amrex::Array4<const amrex::Real> const & /*state_accretion_rate_arr*/, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const * /*cons_fc*/,
						 amrex::RandomEngine const & /*engine*/) const -> int
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

			const bool is_create_particle = current_time <= param1 && current_time + dt > param1;
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
		int death_time_index;
		int evolution_stage_index;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;
		amrex::Real dt;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start,
				int evolution_stage_index, int /*mass_at_birth_index*/, amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index),
		      evolution_stage_index(evolution_stage_index), cpu_id(processor_id), pid_start(particle_id_start), current_time(current_time), dt(dt)
		{
		}

		template <typename TileData, typename StateArray>
		AMREX_GPU_DEVICE void operator()(TileData data, amrex::Long first, ParticleLayout layout, int num_particles, StateArray const &state_arr,
						 StateArray const & /*state_accretion_rate_arr*/, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
						 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const * /*cons_fc*/, amrex::Long base_offset,
						 amrex::RandomEngine const & /*engine*/) const
		{
			if (mass_idx + 3 < layout.numReal) {
				// Calculate common values for all particles
				const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);

				const amrex::Real vx = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
				const amrex::Real vy = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
				const amrex::Real vz = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;

				// Create all particles
				for (int p_idx = 0; p_idx < num_particles; ++p_idx) {
					auto p = ParticleView(data, first + p_idx, layout); // NOLINT

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
					// set death time to current time + 0.0025 (2.5 time steps, so will evolve into SNRemnant at step 3)
					p.rdata(death_time_index) = current_time + 0.0025;

					// Set particle evolution stage
					p.idata(evolution_stage_index) = static_cast<int>(StellarEvolutionStage::SNProgenitor);
				}

				// Update cell density. For testing purposes, we remove a tiny amount of mass from the cell.
				state_arr(i, j, k, HydroSystem<problem_t>::density_index) -= 1.0e-20;
			}
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, int lev,
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index, int birth_time_index, int death_time_index,
				    int mass_at_birth_index, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<ParticleType::Test>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::Test>::template ParticleCreator>(
		    container, mass_idx, state, state_accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index, death_time_index,
		    mass_at_birth_index, state_fc, verbose);
	}
};
} // namespace quokka

template <> void QuokkaSimulation<TestParticle>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double Emag = 0.5 * B0 * B0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<TestParticle>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<TestParticle>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<TestParticle>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<TestParticle>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<TestParticle>::energy_index) = Emag;
		state_cc(i, j, k, HydroSystem<TestParticle>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<TestParticle>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<TestParticle>::mhdFirstIndex) = B_val; });
}

// Check production creation against independent expected fields and gas updates.
void checkRTSoACreation()
{
	const auto savedNextID = quokka::ParticleContainer::ParticleType::NextID();
	quokka::ParticleContainer::ParticleType::NextID(savedNextID);
	const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(3));
	const amrex::RealBox bounds({-2., -2., -2.}, {2., 2., 2.});
	const std::array<int, 3> periodic = {0, 0, 0};
	const amrex::Geometry geom(domain, &bounds, 0, periodic.data());
	amrex::BoxArray ba(domain);
	ba.maxSize(2);
	const amrex::DistributionMapping dm(ba);
	const auto schema = quokka::ParticleSchemaTraits<quokka::ParticleType::Test, TestParticle>::Create();
	quokka::ParticleContainer runtime(schema, geom, dm, ba);
	amrex::MultiFab gasState(ba, dm, HydroSystem<TestParticle>::nvar_, 0);
	amrex::MultiFab state_accretion_rate(ba, dm, gasState.nComp(), 0);
	gasState.setVal(0);
	gasState.setVal(rho0, HydroSystem<TestParticle>::density_index, 1);
	state_accretion_rate.setVal(0);
	const auto create = [&](auto &pc, amrex::MultiFab &state, double time) {
		using Container = std::decay_t<decltype(pc)>;
		quokka::ParticleCreationTraits<quokka::ParticleType::Test>::createParticles<TestParticle, Container>(
		    &pc, schema.Layout().mass, state, state_accretion_rate, 0, time, dt_, schema.Layout().evolutionStage, schema.Layout().birthTime,
		    schema.Layout().deathTime, -1);
	};
	quokka::ParticleContainer::ParticleType::NextID(100);
	create(runtime, gasState, quokka::particle_param1);
	AMREX_ALWAYS_ASSERT(runtime.TotalNumberOfParticles() == 8);
	create(runtime, gasState, quokka::particle_param1);
	AMREX_ALWAYS_ASSERT(runtime.TotalNumberOfParticles() == 16);
	const auto layout = runtime.Layout();
	amrex::ReduceOps<amrex::ReduceOpSum> op;
	amrex::ReduceData<int> result(op);
	using Tuple = decltype(result)::Type;
	for (auto &[key, tile] : runtime.GetParticles(0)) {
		const auto data = tile.getParticleTileData();
		amrex::ignore_unused(key);
		const int rank = amrex::ParallelDescriptor::MyProc();
		const auto birth = quokka::particle_param1;
		op.eval(tile.numParticles(), result, [=] AMREX_GPU_DEVICE(int i) -> Tuple {
			const auto p = quokka::ParticleView(data, i, layout);
			int bad = !p.id().is_valid() || p.id() < 100 || p.cpu() != rank;
			bad += p.rdata(layout.mass) != SN_mass;
			bad += p.idata(layout.evolutionStage) != static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				bad += std::abs(p.pos(d)) != 0.5;
			}
			for (int c = 0; c < layout.numReal; ++c) {
				amrex::ParticleReal expected = 0;
				if (c == layout.mass) {
					expected = SN_mass;
				} else if (c == layout.birthTime) {
					expected = birth;
				} else if (c == layout.deathTime) {
					expected = birth + 0.0025;
				}
				bad += p.rdata(c) != expected;
			}
			for (int c = 0; c < layout.numInt; ++c) {
				bad += p.idata(c) != static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			}
			return {bad};
		});
	}
	AMREX_ALWAYS_ASSERT(amrex::get<0>(result.value(op)) == 0);
	const auto next = quokka::ParticleContainer::ParticleType::NextID();
	quokka::ParticleContainer::ParticleType::NextID(next);
	create(runtime, gasState, quokka::particle_param1 + dt_);
	AMREX_ALWAYS_ASSERT(quokka::ParticleContainer::ParticleType::NextID() == next);
	quokka::ParticleContainer::ParticleType::NextID(next);
	runtime.Redistribute();
	AMREX_ALWAYS_ASSERT(runtime.TotalNumberOfParticles() == 16);
	for (amrex::MFIter mfi(gasState); mfi.isValid(); ++mfi) {
		const auto gas = gasState.const_array(mfi);
		op.eval(mfi.validbox(), result, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
			const bool created = (i == 1 || i == 2) && (j == 1 || j == 2) && (k == 1 || k == 2);
			const auto density = created ? (rho0 - 1.0e-20) - 1.0e-20 : rho0;
			int bad = gas(i, j, k, HydroSystem<TestParticle>::density_index) != density;
			for (int c = 0; c < HydroSystem<TestParticle>::nvar_; ++c) {
				if (c != HydroSystem<TestParticle>::density_index) {
					bad += gas(i, j, k, c) != 0;
				}
			}
			return {bad};
		});
	}
	AMREX_ALWAYS_ASSERT(amrex::get<0>(result.value(op)) == 0);
	for (auto &[key, tile] : runtime.GetParticles(0)) {
		amrex::ignore_unused(key);
		const auto data = tile.getParticleTileData();
		amrex::ParallelFor(tile.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
			const auto p = quokka::ParticleView(data, i, layout);
			if (p.pos(0) < 0) {
				p.idata(layout.evolutionStage) = static_cast<int>(quokka::StellarEvolutionStage::Removed);
			}
		});
	}
	quokka::ParticleDestructionTraits<quokka::ParticleType::Test>::destroyParticles<TestParticle>(&runtime, layout.mass, 0, 0., dt_, layout.birthTime,
												      layout.evolutionStage);
	AMREX_ALWAYS_ASSERT(runtime.TotalNumberOfParticles() == 8);
	quokka::ParticleContainer::ParticleType::NextID(savedNextID);
	amrex::Print() << "RTSoA creation fields, gas updates, and destruction passed\n";
}

auto problem_main() -> int
{
	checkRTSoACreation();
	// Problem initialization
	QuokkaSimulation<TestParticle> sim;
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
