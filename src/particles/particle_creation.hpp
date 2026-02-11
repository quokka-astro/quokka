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
#include <array>
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
				amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1, int death_time_index = -1,
				int mass_at_birth_index = -1, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0)
{
	const BL_PROFILE("ParticleCreationImpl::createParticlesImpl()");
	if (container != nullptr) {
		if (mass_idx >= 0) {
			// Counter for total particles created at this time step
			amrex::Long total_particles_created = 0;
			const bool has_face_centered_state = (state_fc != nullptr);

			// Use the provided ParticleChecker type with global particle parameters
			CheckerType<problem_t> particle_checker(current_time, dt);

			for (amrex::MFIter mfi = container->MakeMFIter(lev); mfi.isValid(); ++mfi) {
				const auto &box = mfi.validbox();
				const auto &state_arr = state.array(mfi);
				const auto &accretion_rate_arr = accretion_rate.array(mfi);
				const auto &geom = container->Geom(lev);
				const auto dx = geom.CellSizeArray();
				const auto plo = geom.ProbLoArray();

				std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc{};
				if (has_face_centered_state) {
					fab_fc[0] = (*state_fc)[0].const_array(mfi);
					fab_fc[1] = (*state_fc)[1].const_array(mfi);
					fab_fc[2] = (*state_fc)[2].const_array(mfi);
				}

				// Count particles to be created in this box
				amrex::Gpu::AsyncVector<unsigned int> counts(box.numPts()); // 1 if cell creates particle, 0 if not
				amrex::Gpu::AsyncVector<unsigned int> offset(box.numPts()); // Will store starting index for each cell's particle
				auto *pcounts = counts.data();

				// Count potential particles per cell
				amrex::ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &engine) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);
					// Check if we should create a particle at this location and time
					std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc_ptr =
					    (has_face_centered_state) ? &fab_fc : nullptr;
					pcounts[index] = particle_checker(state_arr, accretion_rate_arr, i, j, k, dx, fab_fc_ptr, engine); // NOLINT
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
				CreatorType<problem_t> particle_creator(mass_idx, birth_time_index, death_time_index, cpu_id, pid, evolution_stage_index,
									mass_at_birth_index, current_time, dt);

				amrex::ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &engine) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);
					std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc_ptr =
					    (has_face_centered_state) ? &fab_fc : nullptr;

					if (pcounts[index] > 0) {			  // NOLINT
						const int num_particles = pcounts[index]; // NOLINT
						auto *particles = &pdata[poffset[index]]; // NOLINT
						particle_creator(particles, num_particles, state_arr, accretion_rate_arr, i, j, k, dx, plo, fab_fc_ptr,
								 poffset[index],
								 engine); // NOLINT
					}
				});
			}

			// Sum up total particles created across all processors
			amrex::Long global_total_particles = total_particles_created;
			amrex::ParallelDescriptor::ReduceLongSum(global_total_particles);

			// Print the total number of particles created at this time step
			if (amrex::ParallelDescriptor::IOProcessor()) {
				if (verbose > 0 && global_total_particles > 0) {
					amrex::Print() << ">>>Particle creation:\n\tTime: " << current_time << " - Created " << global_total_particles
						       << " particles at level " << lev << "\n";
				}
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
						 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc,
						 amrex::RandomEngine const &engine) const -> int
		{
			// Default implementation creates no particles
			amrex::ignore_unused(state_arr, accretion_rate_arr, i, j, k, dx, fab_fc, engine);
			return 0;
		}
	};

	// Default nested ParticleCreator - initializes a particle's properties
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int birth_time_index;
		int death_time_index;
		int evolution_stage_index;
		int mass_at_birth_idx;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;
		amrex::Real dt;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start,
				int evolution_stage_index, int mass_at_birth_index, amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index),
		      evolution_stage_index(evolution_stage_index), mass_at_birth_idx(mass_at_birth_index), cpu_id(processor_id), pid_start(particle_id_start),
		      current_time(current_time), dt(dt)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const &accretion_rate_arr,
						 int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
						 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, amrex::Long base_offset,
						 amrex::RandomEngine const &engine) const
		{
			// Default implementation does nothing
			amrex::ignore_unused(particles, num_particles, state_arr, accretion_rate_arr, i, j, k, dx, plo, fab_fc, base_offset, engine);
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev,
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1,
				    int death_time_index = -1, int mass_at_birth_index = -1,
				    std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0)
	{
		const BL_PROFILE("ParticleCreationTraits::createParticles()");
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<particleType>::template ParticleChecker,
							  ParticleCreationTraits<particleType>::template ParticleCreator>(
		    container, mass_idx, state, accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index, death_time_index,
		    mass_at_birth_index, state_fc, verbose);
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
						 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc,
						 amrex::RandomEngine const & /*engine*/) const -> int
		{
			const double dx_max = std::max({dx[0], dx[1], dx[2]});

			// Determine sound speed.
			Real cs = NAN;
			if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
				cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
			} else {
				cs = HydroSystem<problem_t>::ComputeSoundSpeed(state_arr, i, j, k, fab_fc);
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
		int death_time_index;
		int evolution_stage_index;
		int mass_at_birth_idx;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;
		amrex::Real dt;

		static constexpr Real Gconst = C::Gconst;
		static constexpr Real gamma = quokka::EOS_Traits<problem_t>::gamma;
		static constexpr Real mu = quokka::EOS_Traits<problem_t>::mean_molecular_weight;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start,
				int evolution_stage_index, int mass_at_birth_index, amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index),
		      evolution_stage_index(evolution_stage_index), mass_at_birth_idx(mass_at_birth_index), cpu_id(processor_id), pid_start(particle_id_start),
		      current_time(current_time), dt(dt)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void
		operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const & /*accretion_rate_arr*/, int i, int j,
			   int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
			   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, amrex::Long base_offset,
			   amrex::RandomEngine const & /*engine*/) const
		{
			const double dx_max = std::max({dx[0], dx[1], dx[2]});

			// Determine sound speed.
			Real cs = NAN;
			if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
				cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
			} else {
				cs = HydroSystem<problem_t>::ComputeSoundSpeed(state_arr, i, j, k, fab_fc);
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
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1,
				    int death_time_index = -1, int mass_at_birth_index = -1,
				    std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<ParticleType::Sink>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::Sink>::template ParticleCreator>(
		    container, mass_idx, state, accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index, death_time_index,
		    mass_at_birth_index, state_fc, verbose);
	}
};

// Specialization for StochasticStellarPop particles
template <> struct ParticleCreationTraits<ParticleType::StochasticStellarPop> {
	// Specialized nested ParticleChecker for StochasticStellarPop particles

	static constexpr amrex::Real eps_star = 0.5; // fraction of gas mass that goes into star particles
	static constexpr amrex::Real J = 0.5;	     // Jeans number (Truelove et al. 1997)
	// Truncating the collapse at sufficiently low Jeans number is needed to prevent
	// runaway collapse to very high densities. This is absolutely critical to include because half
	// of the cell mass will turn into stars and produce composite star particle masses
	// that are so large they cause heating due to dynamical friction in the galaxy.
	static constexpr amrex::Real J_truncate = 0.01 * J; // Jeans number for guaranteed star formation

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
		amrex::Real low_mass_composite_max_mass_ = low_mass_composite_max_mass;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr,
						 amrex::Array4<const amrex::Real> const & /*accretion_rate_arr*/, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc,
						 amrex::RandomEngine const &engine) const -> int
		{
			const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
			const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);

			const amrex::Real cs = HydroSystem<problem_t>::ComputeSoundSpeed(state_arr, i, j, k, fab_fc);
			const amrex::Real LambdaJ = cs / std::sqrt(C::Gconst * cell_density);
			const amrex::Real t_ff = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * cell_density));
			const amrex::Real nominal_prob_star_formation = (eps_ff_ / eps_star) * (dt / t_ff);
			// force P_sf to 1 if we are very far below the Jeans length (as determined by J_truncate)
			const amrex::Real actual_prob_star_formation = (LambdaJ < (J_truncate * dx[0])) ? 1.0 : nominal_prob_star_formation;
			const amrex::Real random_draw = amrex::Random(engine);
			int num_star = 0;

			// Check if the cell violates the Jeans condition but create a particle only if prob_star_formation > random draw
			// eps_star is the fraction of gas mass that goes into star particles
			// Checkout docs/star_formation for more details

			if ((LambdaJ < J * dx[0]) &&
			    random_draw < actual_prob_star_formation) { // Create a particle only if LambdaJ < J*dx and actual_prob_star_formation > random draw
				const amrex::Real particle_mass = cell_density * cell_volume * eps_star;
				const amrex::Real m_high_tot = particle_mass * fstar_high;
				const amrex::Real mass_low_mass_star = particle_mass * (1.0 - fstar_high);
				amrex::Real const num_high_mass_stars_exp = m_high_tot / m_star_high_avg;
				const int num_high = static_cast<int>(amrex::RandomPoisson(num_high_mass_stars_exp, engine));
				int num_low = 1;
				if ((low_mass_composite_max_mass_ > 0.0) && (mass_low_mass_star > low_mass_composite_max_mass_)) {
					num_low = static_cast<int>(std::ceil(mass_low_mass_star / low_mass_composite_max_mass_));
				}
				num_star = num_low + num_high;
			}
			return num_star;
		}
	};

	// Specialized nested ParticleCreator for StochasticStellarPop particles
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int birth_time_index;
		int death_time_index;
		int evolution_stage_index;
		int mass_at_birth_idx;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;
		amrex::Real dt;
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;
		amrex::Real eps_ff_ = eps_ff;
		amrex::Real stellar_velocity_limit_ = stellar_velocity_limit;
		amrex::Real low_mass_composite_max_mass_ = low_mass_composite_max_mass;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start,
				int evolution_stage_index, int mass_at_birth_index, amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index),
		      evolution_stage_index(evolution_stage_index), mass_at_birth_idx(mass_at_birth_index), cpu_id(processor_id), pid_start(particle_id_start),
		      current_time(current_time), dt(dt)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void
		operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const & /*accretion_rate_arr*/, int i, int j,
			   int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
			   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const * /*fab_fc*/, amrex::Long base_offset,
			   amrex::RandomEngine const &engine) const
		{
			if (mass_idx + 3 < ParticleType::NReal) {
				constexpr amrex::Real unset_position = std::numeric_limits<amrex::Real>::max();
				// Calculate common values for all particles
				const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);
				const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
				const amrex::Real cell_mass = cell_volume * cell_density;
				const amrex::Real vx = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
				const amrex::Real vy = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
				const amrex::Real vz = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;
				constexpr int nscalars = Physics_Traits<problem_t>::numPassiveScalars;
				const amrex::Real particle_mass = cell_density * cell_volume * eps_star;
				const amrex::Real mass_low_mass_star = particle_mass * (1.0 - fstar_high);

				const int num_low = static_cast<int>(std::ceil(mass_low_mass_star / low_mass_composite_max_mass_));
				const int num_high = num_particles - num_low;
				const amrex::Real mass_low_each = mass_low_mass_star / static_cast<amrex::Real>(num_low);
				const bool split_low_mass_composite = (num_low > 1);

				double total_high_momx = 0.0;
				double total_high_momy = 0.0;
				double total_high_momz = 0.0;

				// p_idx = 0..(num_low-1) represent the low mass composites, p_idx = num_low.. represent the high mass stars

				for (int p_idx = 0; p_idx < num_particles; ++p_idx) {
					auto &p = particles[p_idx]; // NOLINT

					// Set particle ID and CPU
					p.id() = pid_start + base_offset + p_idx;
					p.cpu() = cpu_id;

					// Set particle birth time
					p.rdata(birth_time_index) = current_time;

					// Set particle position to cell center
					// (Note: if multiple particles are formed, their velocities will be randomized.)
					p.pos(0) = plo[0] + (i + 0.5) * dx[0];
					p.pos(1) = plo[1] + (j + 0.5) * dx[1];
					p.pos(2) = plo[2] + (k + 0.5) * dx[2];

					// Set birth and death metadata
					p.rdata(StochasticStellarPopParticleBirthPosXIdx) = p.pos(0);
					p.rdata(StochasticStellarPopParticleBirthPosYIdx) = p.pos(1);
					p.rdata(StochasticStellarPopParticleBirthPosZIdx) = p.pos(2);
					p.rdata(StochasticStellarPopParticleDeathPosXIdx) = unset_position;
					p.rdata(StochasticStellarPopParticleDeathPosYIdx) = unset_position;
					p.rdata(StochasticStellarPopParticleDeathPosZIdx) = unset_position;
					p.rdata(StochasticStellarPopParticleDeathTimeIdx) = unset_position;
					p.rdata(StochasticStellarPopParticleDeathDensityIdx) = unset_position;

					// Everything is now set EXCEPT for mass, velocity, evolutionary stage, and mass at birth.
					// (For SN progenitors, the death time will be overridden based on the interpolated lifetime.)
					// (Mass at birth is set at the end of this loop. It MUST be set for all star particles,
					// 	because it is used to calculate the star formation rate in outputs.)

					// This is a LowMassComposite star particle
					if (p_idx < num_low) {
						// Set particle evolution stage
						p.idata(evolution_stage_index) = static_cast<int>(StellarEvolutionStage::LowMassComposite);
						// Set particle mass
						p.rdata(mass_idx) = mass_low_each;
						// Set particle position and velocity
						if (split_low_mass_composite) {
							// Randomize velocity within the cell only when split
							const amrex::Real rx = amrex::Random(engine);
							const amrex::Real ry = amrex::Random(engine);
							const amrex::Real rz = amrex::Random(engine);
							// Set velocity dispersion to the cell-scale escape speed of the LowMassComposite cluster:
							// v_esc = sqrt(2 G M_cluster / dx), where M_cluster is the total low-mass composite mass in this cell.
							const amrex::Real dx_cell = std::min({dx[0], dx[1], dx[2]});
							const amrex::Real m_cluster = mass_low_mass_star;
							const amrex::Real v_esc = std::sqrt(2.0 * C::Gconst * m_cluster / dx_cell);
							// For U ~ Uniform[0,1], std(U - 0.5) = sqrt(1/12).
							// Choose component std = v_esc / sqrt(3), so 3D RMS dispersion is v_esc.
							const amrex::Real vdisp_norm = 2.0 * v_esc;
							p.rdata(StochasticStellarPopParticleVxIdx) = vx + vdisp_norm * (rx - 0.5);
							p.rdata(StochasticStellarPopParticleVyIdx) = vy + vdisp_norm * (ry - 0.5);
							p.rdata(StochasticStellarPopParticleVzIdx) = vz + vdisp_norm * (rz - 0.5);
						} else {
							p.rdata(StochasticStellarPopParticleVxIdx) = vx;
							p.rdata(StochasticStellarPopParticleVyIdx) = vy;
							p.rdata(StochasticStellarPopParticleVzIdx) = vz;
						}
					}

					// This is a HighMass star particle
					if (p_idx >= num_low) {
						// Set particle velocity
						{
							double constexpr km_per_s = 1.e5; // convert km/s to cm/s
							double constexpr v_min = 3.0;	  // Minimum velocity from the distribution
							double constexpr v_max = 385.0;	  // Maximum velocity from the distribution
							double constexpr beta = 1.8;	  // Slope of the velocity distribution
							double constexpr vmin_pow = gcem::pow(v_min, 1. - beta);
							double constexpr vmax_pow = gcem::pow(v_max, 1. - beta);

							// Draw velocity from the power-law distribution with inverse transform sampling
							double v_mag = amrex::Random(engine) * (vmax_pow - vmin_pow) + vmin_pow;
							v_mag = std::pow(v_mag, 1. / (1. - beta)) * km_per_s; // Convert to km/s

							// Sample cos theta from a uniform distribution between -1 to 1
							double const cos_theta_random = 2. * amrex::Random(engine) - 1.;
							double const sin_theta_random = std::sqrt(1. - cos_theta_random * cos_theta_random);
							// Sample phi from a uniform distribution between 0 and 2*pi
							double const phi_random = amrex::Random(engine) * 2. * std::numbers::pi;

							double const vx_random = v_mag * sin_theta_random * std::cos(phi_random);
							double const vy_random = v_mag * sin_theta_random * std::sin(phi_random);
							double const vz_random = v_mag * cos_theta_random;

							p.rdata(StochasticStellarPopParticleVxIdx) = vx + vx_random;
							p.rdata(StochasticStellarPopParticleVyIdx) = vy + vy_random;
							p.rdata(StochasticStellarPopParticleVzIdx) = vz + vz_random;
						}

						// Set particle mass
						{
							// Sample mass from the IMF between m_star_high and m_imf_max using inverse transform sampling
							constexpr double mimf_max_pow = gcem::pow(m_imf_max, 1.0 - alpha);
							constexpr double mstar_high_pow = gcem::pow(m_star_high, 1.0 - alpha);
							double mass_of_star = amrex::Random(engine) * (mimf_max_pow - mstar_high_pow) + mstar_high_pow;
							mass_of_star = std::pow(mass_of_star, 1. / (1. - alpha));
							p.rdata(mass_idx) = mass_of_star;

							total_high_momx += p.rdata(StochasticStellarPopParticleVxIdx) * p.rdata(mass_idx);
							total_high_momy += p.rdata(StochasticStellarPopParticleVyIdx) * p.rdata(mass_idx);
							total_high_momz += p.rdata(StochasticStellarPopParticleVzIdx) * p.rdata(mass_idx);
						}

						// Set the evolutionary stage based on whether this star will explode as a supernova
						// interpolate_whether_SN_explosion returns true if the star will undergo a supernova explosion
						p.idata(evolution_stage_index) = interpolate_whether_SN_explosion(p.rdata(mass_idx))
										     ? static_cast<int>(StellarEvolutionStage::SNProgenitor)
										     : static_cast<int>(StellarEvolutionStage::HighMassNonExploding);

						// Set the particle death time for all high mass stars (both SN progenitors and non-exploding)
						p.rdata(death_time_index) = current_time + interpolate_death_time(p.rdata(mass_idx));
					}

					// Set mass_at_birth for ALL star particles
					// (NOTE: this must be set for BOTH LowMassComposite and HighMass star particles
					// 	in order to properly track the star formation rate.)
					if (mass_at_birth_idx >= 0) {
						p.rdata(mass_at_birth_idx) = p.rdata(mass_idx);
					}
				}

				// Update momentum of the low mass composite star particles if there is(are) high mass star(s)
				if (num_high >= 1 || split_low_mass_composite) {
					// Calculate the actual total mass of all particles (high-mass stars sampled from IMF + low-mass composite)
					amrex::Real real_particle_total_mass = 0.;
					for (int pp = 0; pp < num_particles; ++pp) {
						real_particle_total_mass += particles[pp].rdata(mass_idx);
					}

					// Option 1 (preferred): Ensure COM velocity of all stars equals cell velocity (vx, vy, vz).
					// This may violate momentum conservation because real_particle_total_mass != particle_mass
					// due to stochastic sampling of high-mass star masses from the IMF.
					// However, since mass conservation is already violated in a single cell due to stochasticity,
					// it's more important to preserve correct velocities in a rotating disk.
					const amrex::Real target_low_momx = real_particle_total_mass * vx - total_high_momx;
					const amrex::Real target_low_momy = real_particle_total_mass * vy - total_high_momy;
					const amrex::Real target_low_momz = real_particle_total_mass * vz - total_high_momz;

					amrex::Real low_momx = 0.0;
					amrex::Real low_momy = 0.0;
					amrex::Real low_momz = 0.0;
					for (int pp = 0; pp < num_low; ++pp) {
						auto &plow = particles[pp]; // NOLINT
						low_momx += plow.rdata(mass_idx) * plow.rdata(StochasticStellarPopParticleVxIdx);
						low_momy += plow.rdata(mass_idx) * plow.rdata(StochasticStellarPopParticleVyIdx);
						low_momz += plow.rdata(mass_idx) * plow.rdata(StochasticStellarPopParticleVzIdx);
					}

					const amrex::Real delta_vx = (target_low_momx - low_momx) / mass_low_mass_star;
					const amrex::Real delta_vy = (target_low_momy - low_momy) / mass_low_mass_star;
					const amrex::Real delta_vz = (target_low_momz - low_momz) / mass_low_mass_star;

					for (int pp = 0; pp < num_low; ++pp) {
						auto &plow = particles[pp]; // NOLINT
						plow.rdata(StochasticStellarPopParticleVxIdx) += delta_vx;
						plow.rdata(StochasticStellarPopParticleVyIdx) += delta_vy;
						plow.rdata(StochasticStellarPopParticleVzIdx) += delta_vz;
					}

					// Option 2 (alternative): Guarantee momentum conservation.
					// This uses particle_mass (the mass removed from the cell) instead of real_particle_total_mass.
					// While this conserves momentum exactly, it results in incorrect COM velocity because
					// real_particle_total_mass != particle_mass due to stochastic sampling.
					// const amrex::Real target_low_momx2 = particle_mass * vx - total_momx;
					// const amrex::Real target_low_momy2 = particle_mass * vy - total_momy;
					// const amrex::Real target_low_momz2 = particle_mass * vz - total_momz;
					// const amrex::Real delta_vx2 = (target_low_momx2 - low_momx) / mass_low_mass_star;
					// const amrex::Real delta_vy2 = (target_low_momy2 - low_momy) / mass_low_mass_star;
					// const amrex::Real delta_vz2 = (target_low_momz2 - low_momz) / mass_low_mass_star;
				}

				// Update the hydro state to reflect the mass that was removed to create the star particle(s)
				{
					//  We use the *expectation value* of the mass of the created particles in this cell,
					// 	NOT the actual mass of the stochastically created particles, to update the hydro state.)
					const double factor = (1. - particle_mass / cell_mass);

					// Update the cell density to reflect mass conversion into stars
					state_arr(i, j, k, HydroSystem<problem_t>::density_index) *= factor;

					// Update the cell momentum to make sure velocities don't change
					state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) *= factor;
					state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) *= factor;
					state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) *= factor;

					// Update internal energy to reflect mass change
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
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev,
				    amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1,
				    int death_time_index = -1, int mass_at_birth_index = -1,
				    std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0)
	{
		const BL_PROFILE("ParticleCreationTraits<StochasticStellarPop>::createParticles()");
		// Requires CGS units
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Physics_Traits<problem_t>::unit_system == UnitSystem::CGS,
						 "UnitSystem must be CGS for StochasticStellarPopulation");
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType,
							  ParticleCreationTraits<ParticleType::StochasticStellarPop>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::StochasticStellarPop>::template ParticleCreator>(
		    container, mass_idx, state, accretion_rate, lev, current_time, dt, evolution_stage_index, birth_time_index, death_time_index,
		    mass_at_birth_index, state_fc, verbose);
	}
}; // ParticleCreationTraits<ParticleType::StochasticStellarPop>

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_CREATION_HPP_
