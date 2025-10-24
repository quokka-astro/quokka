#ifndef PARTICLE_CREATION_HPP_
#define PARTICLE_CREATION_HPP_

#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "gcem.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "particle_types.hpp"
#include "particles/particle_utils.hpp"
#include "stellarpop_data.hpp"
#include <cmath>
#include <limits>
#include <numbers>

namespace quokka
{

// Helper namespace with implementation details for particle creation
namespace ParticleCreationImpl
{
// Common implementation of particle creation logic
template <typename problem_t, typename ContainerType, template <typename> class CheckerType, template <typename> class CreatorType>
static void createParticlesImpl(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev,
				amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1)
{
	const BL_PROFILE("ParticleCreationImpl::createParticlesImpl()");
	if (container != nullptr) {
		if (mass_idx >= 0) {
			// Counter for total particles created at this time step
			amrex::Long total_particles_created = 0;

			// Use the provided ParticleChecker type with global particle parameters
			CheckerType<problem_t> particle_checker(current_time, dt);

			for (amrex::MFIter mfi = container->MakeMFIter(lev); mfi.isValid(); ++mfi) {
				const auto &box = mfi.validbox();
				const auto &state_arr = state.array(mfi);
				const auto &accretion_rate_arr = accretion_rate.array(mfi);
				const auto &geom = container->Geom(lev);
				const auto dx = geom.CellSizeArray();
				const auto plo = geom.ProbLoArray();

				// Count particles to be created in this box
				amrex::Gpu::AsyncVector<unsigned int> counts(box.numPts()); // 1 if cell creates particle, 0 if not
				amrex::Gpu::AsyncVector<unsigned int> offset(box.numPts()); // Will store starting index for each cell's particle
				auto *pcounts = counts.data();

				// Count potential particles per cell
				amrex::ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &engine) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);
					// Check if we should create a particle at this location and time
					pcounts[index] = particle_checker(state_arr, accretion_rate_arr, i, j, k, dx, engine); // NOLINT
				});

				// Calculate exclusive prefix sum to get unique position for each particle
				// Example: counts  = [1, 0, 1, 0, 1]
				//         offset  = [0, 1, 1, 2, 2]
				const amrex::Long max_new_particles = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

				// Add to our counter
				total_particles_created += max_new_particles;

				// Update NextID to include particles that will be created
				const amrex::Long pid = ContainerType::ParticleType::NextID();
				ContainerType::ParticleType::NextID(pid + max_new_particles);

				// Get the particle tile and prepare for new particles
				auto &particle_tile = container->DefineAndReturnParticleTile(lev, mfi);
				auto &aos = particle_tile.GetArrayOfStructs();
				const int old_size = aos.size();
				aos.resize(old_size + max_new_particles);

				// Create the particles
				auto *poffset = offset.data();
				auto *pdata = aos.data() + old_size;
				const int cpu_id = amrex::ParallelDescriptor::MyProc();

				// Initialize particle creator functor using the provided ParticleCreator type
				CreatorType<problem_t> particle_creator(mass_idx, birth_time_index, cpu_id, pid, evolution_stage_index, current_time, dt);

				amrex::ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &engine) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);

					if (pcounts[index] > 0) {			  // NOLINT
						const int num_particles = pcounts[index]; // NOLINT
						auto *particles = &pdata[poffset[index]]; // NOLINT
						particle_creator(particles, num_particles, state_arr, accretion_rate_arr, i, j, k, dx, plo, poffset[index],
								 engine); // NOLINT
					}
				});
			}

			// Sum up total particles created across all processors
			amrex::Long global_total_particles = total_particles_created;
			amrex::ParallelDescriptor::ReduceLongSum(global_total_particles);

			// Print the total number of particles created at this time step
			if (amrex::ParallelDescriptor::IOProcessor()) {
				amrex::Print() << ">>>Particle creation:\n\tTime: " << current_time << " - Created " << global_total_particles
					       << " particles at level " << lev << "\n\n";
			}
		}
	}
}
} // namespace ParticleCreationImpl

// Traits class for specializing particle creation behavior
template <ParticleType particleType> struct ParticleCreationTraits {
	// Default nested ParticleChecker - determines if a particle should be created at a location
	template <typename problem_t> struct ParticleChecker {
		amrex::Real current_time;
		amrex::Real dt;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, amrex::Array4<const amrex::Real> const &accretion_rate_arr,
						 int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::RandomEngine const &engine) const -> int
		{
			// Default implementation creates no particles
			amrex::ignore_unused(state_arr, accretion_rate_arr, i, j, k, dx, engine);
			return 0;
		}
	};

	// Default nested ParticleCreator - initializes a particle's properties
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
		AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const &accretion_rate_arr,
						 int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long base_offset,
						 amrex::RandomEngine const &engine) const
		{
			// Default implementation does nothing
			amrex::ignore_unused(particles, num_particles, state_arr, accretion_rate_arr, i, j, k, dx, plo, base_offset, engine);
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev,
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1)
	{
		const BL_PROFILE("ParticleCreationTraits::createParticles()");
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<particleType>::template ParticleChecker,
							  ParticleCreationTraits<particleType>::template ParticleCreator>(
		    container, mass_idx, state, accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
};

#if AMREX_SPACEDIM == 3

// Specialization for Sink particles
template <> struct ParticleCreationTraits<ParticleType::Sink> {

	// determines if a particle should be created at a location
	template <typename problem_t> struct ParticleChecker {
		amrex::Real current_time;
		amrex::Real dt;

		static constexpr int stencil_size = ParticleUtils::stencil_size;

		static constexpr Real Gconst = C::Gconst;
		static constexpr Real gamma = quokka::EOS_Traits<problem_t>::gamma;
		static constexpr Real mu = quokka::EOS_Traits<problem_t>::mean_molecular_weight;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, amrex::Array4<const amrex::Real> const &accretion_rate_arr,
						 int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::RandomEngine const & /*engine*/) const -> int
		{
			const double dx_max = std::max({dx[0], dx[1], dx[2]});

			// Determine sound speed.
			Real cs = NAN;
			if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
				cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
			} else {
				cs = HydroSystem<problem_t>::ComputeSoundSpeed(state_arr, i, j, k);
			}

			// Jeans density.
			const auto rho_J = ParticleUtils::computeJeansDensity(cs, dx_max);
			const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);
			const double accretion_rate_cell = accretion_rate_arr(i, j, k);

			// Only form a star if
			// 1. Cell density is above Jeans density
			// 2. Cell accretion rate (a non-positive number) is zero
			// 3. Cell density is the local maximum density
			if (cell_density > rho_J && accretion_rate_cell >= 0.0) {
				bool is_local_maximum = true;
				for (int di = -3; di <= 3 && is_local_maximum; ++di) {
					for (int dj = -3; dj <= 3 && is_local_maximum; ++dj) {
						for (int dk = -3; dk <= 3 && is_local_maximum; ++dk) {
							// Skip the center cell
							if (di == 0 && dj == 0 && dk == 0) {
								continue;
							}
							// Only check cells within spherical radius of 3
							// A small epsilon is added to the right hand side to ensure both (i - stencil_size) and (i +
							// stencil_size) are included
							if (di * di + dj * dj + dk * dk <= static_cast<Real>(stencil_size * stencil_size) + 1.0e-10) {
								const Real rho_ijk = state_arr(i + di, j + dj, k + dk, HydroSystem<problem_t>::density_index);
								if (rho_ijk > cell_density) {
									is_local_maximum = false;
									break;
								}
							}
						}
					}
				}

				if (is_local_maximum) {
					return 1;
				}
			}
			return 0;
		}
	};

	// Default nested ParticleCreator - initializes a particle's properties
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int birth_time_index;
		int evolution_stage_index;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;
		amrex::Real dt;

		static constexpr Real Gconst = C::Gconst;
		static constexpr Real gamma = quokka::EOS_Traits<problem_t>::gamma;
		static constexpr Real mu = quokka::EOS_Traits<problem_t>::mean_molecular_weight;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index,
				amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index), cpu_id(processor_id),
		      pid_start(particle_id_start), current_time(current_time), dt(dt)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void
		operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const & /*accretion_rate_arr*/, int i, int j,
			   int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
			   amrex::Long base_offset, amrex::RandomEngine const & /*engine*/) const
		{
			const double dx_max = std::max({dx[0], dx[1], dx[2]});

			// Determine sound speed.
			Real cs = NAN;
			if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
				cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
			} else {
				cs = HydroSystem<problem_t>::ComputeSoundSpeed(state_arr, i, j, k);
			}

			// Jeans density.
			const auto rho_J = ParticleUtils::computeJeansDensity(cs, dx_max);

			// Calculate common values for all particles
			const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
			const amrex::Real particle_mass = (cell_density - rho_J) * cell_volume;

			const amrex::Real vx = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
			const amrex::Real vy = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
			const amrex::Real vz = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;

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
				p.rdata(mass_idx) = particle_mass / num_particles;
				// add a check to avoid compiler warnings about array-bounds
				if (mass_idx + 3 < ParticleType::NReal) {
					p.rdata(mass_idx + 1) = vx;
					p.rdata(mass_idx + 2) = vy;
					p.rdata(mass_idx + 3) = vz;
				}
			}

			// update cell density to be the threshold density
			//	const amrex::Real scale_factor = n_thresh / cell_density;
			const amrex::Real scale_factor = rho_J / cell_density;
			state_arr(i, j, k, HydroSystem<problem_t>::density_index) = rho_J;
			state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) *= scale_factor;
			state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) *= scale_factor;
			state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) *= scale_factor;
			state_arr(i, j, k, HydroSystem<problem_t>::energy_index) *= scale_factor;
			state_arr(i, j, k, HydroSystem<problem_t>::internalEnergy_index) *= scale_factor;
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev,
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<ParticleType::Sink>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::Sink>::template ParticleCreator>(
		    container, mass_idx, state, accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
};

// Specialization for StochasticStellarPop particles
template <> struct ParticleCreationTraits<ParticleType::StochasticStellarPop> {
	// Specialized nested ParticleChecker for StochasticStellarPop particles

	static constexpr amrex::Real eps_star = 0.5; // fraction of gas mass that goes into star particles
	static constexpr amrex::Real J = 0.5;	     // Jeans parameter

	// Constants for the Chabrier IMF
	// These are the parameters used in extern/ChabrierIMGCalculation.nb
	static constexpr amrex::Real m_star_high = 9.0 * C::M_solar; // all stars above this mass are considered high mass stars
	static constexpr amrex::Real m_imf_max = 120.0 * C::M_solar; // high mass limit of the IMF
	static constexpr amrex::Real alpha = 2.35;		     // slope of the powerlaw

	// fstar_high sets the mass of the high mass stars in a cell (=particle mass * fstar_high)
	// m_star_high_avg is the average mass of the high mass stars in a cell
	// Checkout docs/star_formation for more details on the physics and ChabrierIMGCalculation.nb for the derivation
	// of fstar_high and m_star_high_avg

	// // fstar is the fraction of number of high mass stars from the IMF
	static constexpr double fstar_high = 0.2055;
	static constexpr double m_star_high_avg = 19.39 * C::M_solar; // average mass of high mass stars

	ParticleCreationTraits() = default;

	template <typename problem_t> struct ParticleChecker {
		amrex::Real current_time;
		amrex::Real dt;
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;
		amrex::Real eps_ff_ = eps_ff;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr,
						 amrex::Array4<const amrex::Real> const & /*accretion_rate_arr*/, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::RandomEngine const &engine) const -> int
		{
			const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
			const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);

			const amrex::Real cs = HydroSystem<problem_t>::ComputeSoundSpeed(state_arr, i, j, k);
			const amrex::Real LambdaJ = cs / std::sqrt(C::Gconst * cell_density);
			const amrex::Real t_ff = std::sqrt(3.0 * std::numbers::pi / (32.0 * C::Gconst * cell_density));
			const amrex::Real prob_star_formation = (eps_ff_ / eps_star) * (dt / t_ff);
			const amrex::Real random_draw = amrex::Random(engine);
			int num_star = 0;

			// Check if the cell violates the Jeans condition but create a particle only if prob_star_formation > random draw
			// eps_star is the fraction of gas mass that goes into star particles
			// Checkout docs/star_formation for more details

			if ((LambdaJ < J * dx[0]) &&
			    random_draw < prob_star_formation) { // Create a particle only if LambdaJ < J*dx and prob_star_formation> random draw
				const amrex::Real particle_mass = cell_density * cell_volume * eps_star;
				const amrex::Real m_high_tot = particle_mass * fstar_high;
				amrex::Real const num_high_mass_stars_exp = m_high_tot / m_star_high_avg;
				num_star = static_cast<int>(1 + (amrex::RandomPoisson(num_high_mass_stars_exp, engine)));
			}
			return num_star;
		}
	};

	// Specialized nested ParticleCreator for StochasticStellarPop particles
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int birth_time_index;
		int evolution_stage_index;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;
		amrex::Real dt;
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;
		amrex::Real eps_ff_ = eps_ff;
		amrex::Real stellar_velocity_limit_ = stellar_velocity_limit;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index,
				amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index), cpu_id(processor_id),
		      pid_start(particle_id_start), current_time(current_time), dt(dt)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void
		operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const & /*accretion_rate_arr*/, int i, int j,
			   int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
			   amrex::Long base_offset, amrex::RandomEngine const &engine) const
		{

			if (mass_idx + 3 < ParticleType::NReal) {
				// Calculate common values for all particles
				const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);
				const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
				const amrex::Real cell_mass = cell_volume * cell_density;
				const amrex::Real vx = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
				const amrex::Real vy = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
				const amrex::Real vz = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;
				const int nscalars = Physics_Traits<problem_t>::numPassiveScalars;
				const amrex::Real particle_mass = cell_density * cell_volume * eps_star;
				const amrex::Real mass_low_mass_star = particle_mass * (1.0 - fstar_high);
				double total_momx = 0.0;
				double total_momy = 0.0;
				double total_momz = 0.0;

				// p_idx = 0 represents the low mass star and p_idx = 1, 2..  represent the high mass stars

				for (int p_idx = 0; p_idx < num_particles; ++p_idx) {
					auto &p = particles[p_idx]; // NOLINT

					// Set particle ID and CPU
					p.id() = pid_start + base_offset + p_idx;
					p.cpu() = cpu_id;

					// Set particle birth time
					p.rdata(birth_time_index) = current_time;

					// Set particle evolution stage to LowMassComposite if it is a low-mass stellar composite
					// This gets changed in the for loop below if this is a high mass star
					p.idata(evolution_stage_index) = static_cast<int>(StellarEvolutionStage::LowMassComposite);

					// Low Mass particle position at cell center
					p.pos(0) = plo[0] + (i + 0.5) * dx[0];
					p.pos(1) = plo[1] + (j + 0.5) * dx[1];
					p.pos(2) = plo[2] + (k + 0.5) * dx[2];

					// Low Mass particle mass and velocity
					p.rdata(mass_idx) = mass_low_mass_star;
					p.rdata(mass_idx + 1) = vx;
					p.rdata(mass_idx + 2) = vy;
					p.rdata(mass_idx + 3) = vz;

					p.rdata(birth_time_index + 1) = std::numeric_limits<amrex::Real>::max();
					if (p_idx > 0) {
						// This is the loop that sets the velocity of the high mass stars
						double const km_per_s = 1.e5; // convert km/s to cm/s
						double const v_min = 3.0;     // Minimum velocity from the distribution
						double const v_max = 385.0;   // Maximum velocity from the distribution
						double const beta = 1.8;      // Slope of the velocity distribution

						// Draw velocity from the power-law distribution
						double const xx_random = amrex::Random(engine);
						double v_mag =
						    xx_random * (std::pow(v_max, 1. - beta) - std::pow(v_min, 1. - beta)) + std::pow(v_min, 1. - beta);
						v_mag = std::pow(v_mag, 1. / (1. - beta)) * km_per_s; // Convert to km/s

						double const cos_theta_random =
						    (2 * amrex::Random(engine) - 1.0); // Sample cos theta from a uniform distribution between -1 to 1.
						double const phi_random =
						    (1. - amrex::Random(engine)) * 2. * std::numbers::pi; // Sample phi from a uniform distribution between 0 and 2*pi

						double const vx_random = v_mag * std::sqrt(1. - cos_theta_random * cos_theta_random) * std::cos(phi_random);
						double const vy_random = v_mag * std::sqrt(1. - cos_theta_random * cos_theta_random) * std::sin(phi_random);
						double const vz_random = v_mag * cos_theta_random;

						p.rdata(mass_idx + 1) = vx + vx_random;
						p.rdata(mass_idx + 2) = vy + vy_random;
						p.rdata(mass_idx + 3) = vz + vz_random;

						// Sample mass randomly from the IMF between m_star_high, which is the min mass and max mass in the Sukhbold
						// table
						double mass_of_star = NAN;
						const double xx = amrex::Random(engine);
						mass_of_star = xx * (std::pow(m_imf_max, 1.0 - alpha) - std::pow(m_star_high, 1.0 - alpha)) +
							       std::pow(m_star_high, 1.0 - alpha);
						mass_of_star = std::pow(mass_of_star, 1. / (1. - alpha));
						p.rdata(mass_idx) = mass_of_star;

						total_momx += p.rdata(mass_idx + 1) * p.rdata(mass_idx);
						total_momy += p.rdata(mass_idx + 2) * p.rdata(mass_idx);
						total_momz += p.rdata(mass_idx + 3) * p.rdata(mass_idx);

						p.idata(evolution_stage_index) =
						    interpolate_fate(p.rdata(mass_idx)) == 1 ? static_cast<int>(StellarEvolutionStage::SNProgenitor) : 0;
						p.rdata(birth_time_index + 1) = interpolate_death_time(p.rdata(mass_idx));
					}
				}

				if (num_particles > 1) { // Update momentum of the low mass star if there is(are) high mass star(s) in the cell
					const int p_idx = 0;
					auto &plow = particles[p_idx]; // NOLINT
					plow.rdata(mass_idx + 1) = -total_momx / plow.rdata(mass_idx);
					plow.rdata(mass_idx + 2) = -total_momy / plow.rdata(mass_idx);
					plow.rdata(mass_idx + 3) = -total_momz / plow.rdata(mass_idx);
				}

				const double factor = (1. - particle_mass / cell_mass);

				// Update the cell density to reflect mass conversion into stars
				state_arr(i, j, k, HydroSystem<problem_t>::density_index) *= factor;

				// Update the cell momentum to make sure velocities don't change
				state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) *= factor;
				state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) *= factor;
				state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) *= factor;

				// Update internal energy to relect mass change
				state_arr(i, j, k, HydroSystem<problem_t>::internalEnergy_index) *= factor;

				// Update total energy
				state_arr(i, j, k, HydroSystem<problem_t>::energy_index) *= factor;

				// Update mass scalars including passive scalars
				if (nscalars > 0) {
					for (int nn = 0; nn < nscalars; ++nn) {
						state_arr(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) *= factor;
					}
				}
			}
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev,
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1)
	{
		const BL_PROFILE("ParticleCreationTraits<StochasticStellarPop>::createParticles()");
		// Requires CGS units
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Physics_Traits<problem_t>::unit_system == UnitSystem::CGS,
						 "UnitSystem must be CGS for StochasticStellarPopulation");
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType,
							  ParticleCreationTraits<ParticleType::StochasticStellarPop>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::StochasticStellarPop>::template ParticleCreator>(
		    container, mass_idx, state, accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
}; // ParticleCreationTraits<ParticleType::StochasticStellarPop>

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_CREATION_HPP_
