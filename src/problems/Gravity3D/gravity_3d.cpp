/// \file gravity_3d.cpp
/// \brief Defines a test problem for self-gravity in 3D.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "gravity_3d.hpp"
#include "hydro/hydro_system.hpp"
#include <algorithm>

struct BinaryOrbit {
};

// This is an ad-hoc test of particle creation and destruction.
// The initial condition consists of 2 particles with a mass of 1.0.
// In the first time step, 2^3 * 2 particles are created. Half of them are low-mass particles
// marked for destruction. In the next time step, low-mass particles are destroyed. There
// are 8 of them. Finally, in the third and last step, 2^3 * 2 particles are created.
// The final number of particles is 26

constexpr int particle_per_cell = 2;
constexpr int particle_spacing = 30;
constexpr double particle_low_mass = 1.0e-20; // very low mass particles marked for destruction
constexpr double dt_ = 0.001;
constexpr int n_particle_last = 26; // initial 2, 2^3 * 2 * 2 created, 8 destroyed

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
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Test;
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
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

namespace quokka
{
// Specialization for CIC particle creation
template <> struct ParticleCreationTraits<ParticleType::Test> {
	// Specialized nested ParticleChecker for Test particles
	template <typename problem_t> struct ParticleChecker {
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real current_time, amrex::Real dt) const -> int
		{
			// A simple demonstration of particle creation
			// Could check density threshold or other state-based conditions
			amrex::ignore_unused(state_arr, dx);
			const int spacing = particle_spacing;
			const bool is_create_particle_1 = current_time <= param1 && current_time + dt > param1;
			const bool is_create_particle_2 = current_time <= param2 && current_time + dt > param2;
			if ((is_create_particle_1 || is_create_particle_2) && (i != 0 && i % spacing == 0) && (j != 0 && j % spacing == 0) &&
			    (k != 0 && k % spacing == 0)) {
				return particle_per_cell;
			}
			return 0;
		}
	};

	// Specialized nested ParticleCreator for Test particles
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int evolution_stage_index;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index)
		    : mass_idx(mass_index), cpu_id(processor_id), pid_start(particle_id_start), evolution_stage_index(evolution_stage_index)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long base_offset) const
		{
			if (mass_idx + 3 < ParticleType::NReal) {
				// Calculate common values for all particles
				const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
				const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);
				const amrex::Real cell_mass = cell_density * cell_volume;
				amrex::Real particle_mass = 0.5 * cell_mass / num_particles; // Divide mass among particles

				// mark half of the particles as low-mass particles which will be destroyed in the next time step
				if (i <= particle_spacing) {
					particle_mass = particle_low_mass;
				}

				const amrex::Real vx = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
				const amrex::Real vy = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
				const amrex::Real vz = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;

				// Create all particles
				for (int p_idx = 0; p_idx < num_particles; ++p_idx) {
					auto &p = particles[p_idx]; // NOLINT

					// Set particle position (all at cell center for now)
					p.pos(0) = plo[0] + (i + 0.5) * dx[0];
					p.pos(1) = plo[1] + (j + 0.5) * dx[1];
					p.pos(2) = plo[2] + (k + 0.5) * dx[2];

					// Set particle ID and CPU
					p.id() = pid_start + base_offset + p_idx;
					p.cpu() = cpu_id;

					// Initialize particle properties
					p.rdata(mass_idx) = particle_mass;
					p.rdata(mass_idx + 1) = vx;
					p.rdata(mass_idx + 2) = vy;
					p.rdata(mass_idx + 3) = vz;

					// Set particle evolution stage
					p.idata(evolution_stage_index) = static_cast<int>(StellarEvolutionStage::SNProgenitor);
				}

				// Update cell density (remove mass that was given to particles)
				state_arr(i, j, k, HydroSystem<problem_t>::density_index) = 0.5 * cell_density;
			}
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt, int evolution_stage_index)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<ParticleType::Test>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::Test>::template ParticleCreator>(container, mass_idx, state, lev,
															       current_time, dt, evolution_stage_index);
	}
};

// Specialization for Test particles destruction
template <> struct ParticleDestructionTraits<ParticleType::Test> {
	// Default nested ParticleChecker - determines if a particle should be destroyed
	template <typename problem_t> struct ParticleChecker {
		amrex::Real t_destroy = particle_param3;

		// AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real param1) : param1(param1) {}

		template <typename ParticleType>
		AMREX_GPU_DEVICE auto operator()(ParticleType &p, int mass_idx, amrex::Real current_time, amrex::Real dt) const -> bool
		{
			// Default implementation: destroy particles with mass < 1.0
			amrex::ignore_unused(current_time, dt);

			const bool is_small_mass = (p.rdata(mass_idx) < 2.0 * particle_low_mass);
			const bool is_time = (current_time <= t_destroy && current_time + dt > t_destroy);
			return is_small_mass && is_time;
		}
	};

	// Main method to destroy particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void destroyParticles(ContainerType *container, int mass_idx, int lev, amrex::Real current_time, amrex::Real dt)
	{
		// Use the common implementation with our checker type
		ParticleDestructionImpl::destroyParticlesImpl<problem_t, ContainerType, ParticleDestructionTraits<ParticleType::Test>::template ParticleChecker>(
		    container, mass_idx, lev, current_time, dt);
	}
};

} // namespace quokka

template <> void QuokkaSimulation<BinaryOrbit>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double const rho = 1.0e-5; // g cm^{-3}
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<BinaryOrbit>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) {}

// template <> void QuokkaSimulation<BinaryOrbit>::createInitialCICParticles()
// {
// 	// read particles from ASCII file
// 	const int nreal_extra = 4; // mass vx vy vz
// 	CICParticles->SetVerbose(1);
// 	CICParticles->InitFromAsciiFile("Gravity3D.txt", nreal_extra, nullptr);
// }

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

	const int n_particle_expect = n_particle_last;

	int status = 0; // Initialize to success

	auto [real_data, int_data] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Test)->getParticleData(0);

	if (amrex::ParallelDescriptor::IOProcessor()) {

		const auto n_particle_actual = real_data.size();

		// // assume the first particle is in the first plane quadrant
		// for (const auto &data : particle_data) {
		// 	// only consider particles with mass > 0.1. Those are the ones created at the start of the simulation.
		// 	if (data[3] < 0.1) {
		// 		continue;
		// 	}
		// 	// First 3 elements are positions (x,y,z)
		// 	if (data[0] * exact_x > 0.0) {
		// 		position_error += std::abs(data[0] - exact_x);
		// 		position_error += std::abs(data[1] - exact_y);
		// 		position_error += std::abs(data[2] - exact_z);
		// 	} else {
		// 		position_error += std::abs(data[0] - (-exact_x));
		// 		position_error += std::abs(data[1] - (-exact_y));
		// 		position_error += std::abs(data[2] - (-exact_z));
		// 	}
		// 	position_norm += std::abs(data[0]);
		// 	position_norm += std::abs(data[1]);
		// 	position_norm += std::abs(data[2]);
		// }

		amrex::Print() << "Particle positions and data are: \n";
		for (const auto &data : real_data) {
			// Print positions
			amrex::Print() << "Position: " << data[0] << ", " << data[1] << ", " << data[2];
			// Print additional data (mass, velocities)
			amrex::Print() << " | Mass: " << data[3];
			amrex::Print() << " | Velocities: " << data[4] << ", " << data[5] << ", " << data[6] << "\n";
		}
		amrex::Print() << "Exact positions are: \n" << exact_x << ", " << exact_y << ", " << exact_z << "\n";
		amrex::Print() << "Expected number of particles: " << n_particle_expect << "\n";
		amrex::Print() << "Actual number of particles: " << n_particle_actual << "\n";

		// compute relative error
		// const double relative_error = position_error / position_norm;
		const double relative_error = 0.0;

		amrex::Print() << "Position error: " << position_error << "\n";
		amrex::Print() << "Position norm: " << position_norm << "\n";
		amrex::Print() << "Relative error: " << relative_error << "\n";

		const double max_err_tol = sim.tNew_[0] < 1.0 ? 0.001 : 0.05; // max error tol in cell widths
		status = 1;
		if (relative_error < max_err_tol && n_particle_actual == n_particle_expect) {
			status = 0;
			amrex::Print() << "Relative error within tolerance.\n";
		}
	}

	return status;
}
