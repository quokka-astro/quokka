/// \file gravity_3d.cpp
/// \brief Defines a test problem for self-gravity in 3D.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "gravity_3d.hpp"
#include "hydro/hydro_system.hpp"

struct BinaryOrbit {
};

// This is an ad-hoc test of particle creation and destruction.
// The initial condition consists of 2 CIC particles with a mass of 1.0. We keep track of their orbit and compare with the exact solution. In the second time
// step, 3^3 * 2 particles are created. A third of them are LowMassStar and the rest are SNProgenitor. In the third time step, all SNProgenitor particles are
// turned into SNRemnant. In the fourth time step, all SNRemnant particles are destroyed. In the end of the simulation, there are 2 CIC particles and 18 Test
// particles.

constexpr double rho0 = 1.0e-5;
constexpr double init_mass_total = rho0 * 4 * 4 * 4;

constexpr int particle_per_cell = 2;
constexpr double SN_mass = 0.1;				// mass of SNProgenitor particles
constexpr double init_test_particle_mass = 2. * 1.0e-5; // mass of Test particles
constexpr double particle_low_mass = 1.0e-20;		// very low mass particles marked for destruction
constexpr double dt_ = 0.001;
constexpr int n_expected_test_particles = 8; // 8 low_mass particles created and live to the end
constexpr int n_SN = 8;
constexpr double m_SN = (n_SN * SN_mass) + init_test_particle_mass;

static bool do_split_particles = false; // NOLINT
static int split_factor = 8;		// NOLINT

// locations of the particles: a 2x2x2 grids of particles
constexpr int loc_x1 = 31;
constexpr int loc_x2 = 32;
constexpr int loc_y1 = 31;
constexpr int loc_y2 = 32;
constexpr int loc_z1 = 31;
constexpr int loc_z2 = 32;

template <> struct quokka::EOS_Traits<BinaryOrbit> {
	static constexpr double gamma = 1.0;	     // isothermal
	static constexpr double cs_isothermal = 3.0; //
	static constexpr double mean_molecular_weight = 1.0;
};

// Test enum to demonstrate type checking of particle_switch
enum class TestEnum : unsigned int {
	MISTAKE = 0b00000100U,
};

template <> struct Particle_Traits<BinaryOrbit> {
	// The following will cause a compile error
	// static constexpr int particle_switch = 1;
	// static constexpr TestEnum particle_switch = TestEnum::MISTAKE;
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | TestEnum::MISTAKE;
	// This is the correct way to define the particle switch
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | ParticleSwitch::Test;
};

template <> struct HydroSystem_Traits<BinaryOrbit> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<BinaryOrbit> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0e-5; // set a small value to keep the cells/particles from moving
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

namespace quokka
{
// Specialization for CIC particle creation
template <> struct ParticleCreationTraits<ParticleType::Test> {
	// Specialized nested ParticleChecker for Test particles
	template <typename problem_t> struct ParticleChecker {
		amrex::Real current_time;
		amrex::Real dt;
		amrex::Real param1 = particle_param1;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) const -> int
		{
			// A simple demonstration of particle creation
			// Could check density threshold or other state-based conditions
			amrex::ignore_unused(state_arr, dx);
			const bool is_create_particle = current_time <= param1 && current_time + dt > param1;
			if (is_create_particle && (i == loc_x1 || i == loc_x2) && (j == loc_y1 || j == loc_y2) && (k == loc_z1 || k == loc_z2)) {
				return particle_per_cell;
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

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index,
				amrex::Real current_time)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index), cpu_id(processor_id),
		      pid_start(particle_id_start), current_time(current_time)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long base_offset) const
		{
			if (mass_idx + 3 < ParticleType::NReal) {
				// Calculate common values for all particles
				const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);

				const amrex::Real vx = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
				const amrex::Real vy = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
				const amrex::Real vz = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;

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
					p.rdata(mass_idx) = p_idx == 0 ? SN_mass : particle_low_mass;
					p.rdata(mass_idx + 1) = vx;
					p.rdata(mass_idx + 2) = vy;
					p.rdata(mass_idx + 3) = vz;

					// set birth time to current time
					p.rdata(birth_time_index) = current_time;

					// Set particle evolution stage
					p.idata(evolution_stage_index) = p_idx == 0 ? static_cast<int>(StellarEvolutionStage::SNProgenitor)
										    : static_cast<int>(StellarEvolutionStage::LowMassStar);
				}

				// Update cell density. For testing purposes, we remove a tiny amount of mass from the cell.
				state_arr(i, j, k, HydroSystem<problem_t>::density_index) -= 1.0e-20;
			}
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt,
				    int evolution_stage_index, int birth_time_index)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<ParticleType::Test>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::Test>::template ParticleCreator>(
		    container, mass_idx, state, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
};

// Specialization for Test particles destruction
template <> struct ParticleDestructionTraits<ParticleType::Test> {
	// Default nested ParticleChecker - determines if a particle should be destroyed
	template <typename problem_t> struct ParticleChecker {
		int birth_time_index;
		int evolution_stage_index;
		amrex::Real t_destroy = particle_param3;

		AMREX_GPU_HOST_DEVICE explicit ParticleChecker(int birth_time_index, int evolution_stage_index)
		    : birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index)
		{
		}

		template <typename ParticleType>
		AMREX_GPU_DEVICE auto operator()(ParticleType &p, int mass_idx, amrex::Real current_time, amrex::Real dt) const -> bool
		{
			// Default implementation: destroy particles with mass < 1.0
			amrex::ignore_unused(mass_idx, current_time, dt);

			// only SNRemnant will be destroyed; just for testing
			const bool is_sn_remnant = (p.idata(evolution_stage_index) == static_cast<int>(StellarEvolutionStage::SNRemnant));
			const bool is_time = (current_time + dt > t_destroy);
			return is_sn_remnant && is_time;
		}
	};

	// Main method to destroy particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void destroyParticles(ContainerType *container, int mass_idx, int lev, amrex::Real current_time, amrex::Real dt, int birth_time_index,
				     int evolution_stage_index)
	{
		// Use the common implementation with our checker type
		ParticleDestructionImpl::destroyParticlesImpl<problem_t, ContainerType,
							      ParticleDestructionTraits<ParticleType::Test>::template ParticleChecker>(
		    container, mass_idx, lev, current_time, dt, birth_time_index, evolution_stage_index);
	}
};

} // namespace quokka

template <> void QuokkaSimulation<BinaryOrbit>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<BinaryOrbit>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) {}

template <> void QuokkaSimulation<BinaryOrbit>::createInitialCICParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("Gravity3D.txt", nreal_extra, nullptr);

	// test particle splitting
	// (this is intended to only be used when restarting at a higher resolution)
	if (do_split_particles) {
		amrex::Print() << "Splitting CICParticles using split_factor = " << split_factor << "\n";
		for (int lev = 0; lev <= CICParticles->finestLevel(); ++lev) {
			amrex::Print() << "...splitting on level " << lev << "\n";
			particleRegister_.splitParticles(lev, split_factor);
		}
	}
}

template <> void QuokkaSimulation<BinaryOrbit>::createInitialTestParticles()
{
	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	TestParticles->SetVerbose(1);
	TestParticles->InitFromAsciiFile("TestParticles.txt", nreal_extra, nullptr);

	// Loop over all particles and set first integer component to SNProgenitor
	auto &particles = TestParticles->GetParticles(0);
	for (auto &kv : particles) {
		auto &particle_array = kv.second.GetArrayOfStructs();
		const int np = particle_array.numParticles();
		for (int i = 0; i < np; i++) {
			auto &p = particle_array[i];
			p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
		}
	}
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<BinaryOrbit>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<BinaryOrbit>::nvarTotal_cc;
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

	// read in runtime parameters for this test problem
	amrex::ParmParse const pp("particles");
	pp.query("do_split_particles", do_split_particles);
	pp.query("split_factor", split_factor);

	// Problem initialization
	QuokkaSimulation<BinaryOrbit> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	sim.initDt_ = dt_;
	sim.maxDt_ = dt_;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// exact solution
	const double theta = 0.5 * sim.tNew_[0];
	const double exact_x = 1.0 * std::cos(theta);
	const double exact_y = 1.0 * std::sin(theta);
	const double exact_z = 0.0;

	double position_error = 0.0;
	double position_norm = 0.0;

	int status = 0; // Initialize to success

	// get total mass in cells
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const total_mass = sim.state_new_cc_[0].sum(HydroSystem<BinaryOrbit>::density_index) * vol;
	amrex::Real const SN_remnant_mass = total_mass - init_mass_total;
	amrex::Print() << "Total SN remnant mass: " << SN_remnant_mass << "\n";
	amrex::Print() << "Expected total SN remnant mass in cells: " << m_SN << "\n";
	const double SN_remnant_mass_rel_err = std::abs(SN_remnant_mass - m_SN) / m_SN;
	amrex::Print() << "SN remnant mass relative error: " << SN_remnant_mass_rel_err << "\n";

	// ----- Check CIC particles -----

	// particle actions must be called on all ranks
	auto [real_data, int_data] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getParticleData(0);
	const int n_particle_CIC = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getNumParticles();
	const int n_particle_test = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Test)->getNumParticles();

	if (amrex::ParallelDescriptor::IOProcessor()) {

		// assume the first particle is in the first plane quadrant
		for (const auto &data : real_data) {
			// only consider particles with mass > 0.1. Those are the ones created at the start of the simulation.
			if (data[3] < 0.1) {
				continue;
			}
			// First 3 elements are positions (x,y,z)
			if (data[0] * exact_x > 0.0) {
				position_error += std::abs(data[0] - exact_x);
				position_error += std::abs(data[1] - exact_y);
				position_error += std::abs(data[2] - exact_z);
			} else {
				position_error += std::abs(data[0] - (-exact_x));
				position_error += std::abs(data[1] - (-exact_y));
				position_error += std::abs(data[2] - (-exact_z));
			}
			position_norm += std::abs(data[0]);
			position_norm += std::abs(data[1]);
			position_norm += std::abs(data[2]);
		}

		amrex::Print() << "Particle positions and data are: \n";
		for (const auto &data : real_data) {
			// Print positions
			amrex::Print() << "Position: " << data[0] << ", " << data[1] << ", " << data[2];
			// Print additional data (mass, velocities)
			amrex::Print() << " | Mass: " << data[3];
			amrex::Print() << " | Velocities: " << data[4] << ", " << data[5] << ", " << data[6] << "\n";
		}
		amrex::Print() << "Exact positions are: \n" << exact_x << ", " << exact_y << ", " << exact_z << "\n";

		// compute relative error
		const double relative_error = position_error / position_norm;

		amrex::Print() << "Position error: " << position_error << "\n";
		amrex::Print() << "Position norm: " << position_norm << "\n";
		amrex::Print() << "Relative error: " << relative_error << "\n";

		// ----- Check CIC particles -----
		const int n_expected_CIC_particles = do_split_particles ? 2 * split_factor : 2;
		amrex::Print() << "Expected number of CIC particles: " << n_expected_CIC_particles << "\n";
		amrex::Print() << "Actual number of CIC particles: " << n_particle_CIC << "\n";

		// ----- Check Test particles -----

		amrex::Print() << "Expected number of test particles: " << n_expected_test_particles << "\n";
		amrex::Print() << "Actual number of test particles: " << n_particle_test << "\n";

		// ----- Check SN remnant mass -----

		const double max_err_tol = ((sim.tNew_[0] < 1.0) && !do_split_particles) ? 0.001 : 0.05; // max error tol in cell widths
		const double max_err_tol_mass = 1.0e-8;							 // max error tol in mass
		status = 1;
		if ((relative_error < max_err_tol) && (n_particle_test == n_expected_test_particles) && (n_particle_CIC == n_expected_CIC_particles) &&
		    (SN_remnant_mass_rel_err < max_err_tol_mass)) {
			status = 0;
			amrex::Print() << "Relative error within tolerance.\n";
		}
	}

	return status;
}
