#ifndef PARTICLE_CREATION_HPP_
#define PARTICLE_CREATION_HPP_

#include "hydro/hydro_system.hpp"
#include "particle_types.hpp"
#include "stellarpop_data.hpp"
#include <cmath>

namespace quokka
{

// Helper namespace with implementation details for particle creation
namespace ParticleCreationImpl
{
// Common implementation of particle creation logic
template <typename problem_t, typename ContainerType, template <typename> class CheckerType, template <typename> class CreatorType>
static void createParticlesImpl(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt,
				int evolution_stage_index = -1, int birth_time_index = -1)
{
	if (container != nullptr) {
		if (mass_idx >= 0) {
			// Use the provided ParticleChecker type with global particle parameters
			CheckerType<problem_t> particle_checker(current_time, dt);

			for (amrex::MFIter mfi = container->MakeMFIter(lev); mfi.isValid(); ++mfi) {
				const auto &box = mfi.validbox();
				const auto &state_arr = state.array(mfi);
				const auto &geom = container->Geom(lev);
				const auto dx = geom.CellSizeArray();
				const auto plo = geom.ProbLoArray();

				// Count particles to be created in this box
				amrex::Gpu::DeviceVector<unsigned int> counts(box.numPts()); // 1 if cell creates particle, 0 if not
				amrex::Gpu::DeviceVector<unsigned int> offset(box.numPts()); // Will store starting index for each cell's particle
				auto *pcounts = counts.data();

				// Count potential particles per cell
				amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);
					// Check if we should create a particle at this location and time
					pcounts[index] = particle_checker(state_arr, i, j, k, dx); // NOLINT
				});

				// Calculate exclusive prefix sum to get unique position for each particle
				// Example: counts  = [1, 0, 1, 0, 1]
				//         offset  = [0, 1, 1, 2, 2]
				const unsigned int max_new_particles = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

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

				amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);

					if (pcounts[index] > 0) {									 // NOLINT
						const int num_particles = pcounts[index];						 // NOLINT
						auto *particles = &pdata[poffset[index]];						 // NOLINT
						particle_creator(particles, num_particles, state_arr, i, j, k, dx, plo, poffset[index]); // NOLINT
					}
				});
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

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) const -> int
		{
			// Default implementation creates no particles
			amrex::ignore_unused(state_arr, i, j, k, dx);
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
		AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long base_offset) const
		{
			// Default implementation does nothing
			amrex::ignore_unused(particles, num_particles, state_arr, i, j, k, dx, plo, base_offset);
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt,
				    int evolution_stage_index = -1, int birth_time_index = -1)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<particleType>::template ParticleChecker,
							  ParticleCreationTraits<particleType>::template ParticleCreator>(
		    container, mass_idx, state, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
};

#if AMREX_SPACEDIM == 3
// Specialization for StochasticStellarPop particles
template <> struct ParticleCreationTraits<ParticleType::StochasticStellarPop> {
	// Specialized nested ParticleChecker for StochasticStellarPop particles

	static constexpr amrex::Real eps_star = 0.5; // fraction of gas mass that goes into star particles
	static constexpr amrex::Real eps_ff = 0.5;   // efficiency per free fall time
	static constexpr amrex::Real J = 0.5;	     // Jeans parameter

	// Constants for the Chabrier IMF
	static constexpr amrex::Real m_star_high = 8.0 * C::M_solar; // all stars above this mass are considered high mass stars
	static constexpr amrex::Real m_imf_min = 0.08 * C::M_solar;  // lower limit of the IMF
	static constexpr amrex::Real m_imf_max = 120.0 * C::M_solar; // high mass limit of the IMF
	static constexpr amrex::Real m_imf_break = 1.0 * C::M_solar; // IMF is lognormal below this mass and powerlaw above
	static constexpr amrex::Real imf_disp = 0.55;		     // dispersion of the lognormal IMF
	static constexpr amrex::Real imf_mu = 32.599;		     //=log10(0.2 * C::M_solar), mean of the lognormal IMF, avoid compiler error
	static constexpr amrex::Real alpha = 2.35;		     // slope of the powerlaw

	static double fstar_high;      // fstar is the fraction of number of high mass stars from the IMF
	static double m_star_high_avg; // average mass of high mass stars

	ParticleCreationTraits()
	{
		auto arg = [](double mass) -> double { return (std::log10(mass) - imf_mu) / std::sqrt(2.0 * imf_disp * imf_disp); };
		double const norm_ratio =
		    std::pow(m_imf_break, (1 - alpha)) * imf_disp * std::sqrt(2.0 * M_PI) / std::exp(-arg(m_imf_break) * arg(m_imf_break));

		double const total_stars = ((1. - alpha) * norm_ratio * (std::erf(arg(m_imf_break)) - std::erf(arg(m_imf_min)))) +
					   std::pow(m_imf_max, 1.0 - alpha) - std::pow(m_imf_break, 1.0 - alpha);
		double const num_high_mass_stars = std::pow(m_imf_max, 1.0 - alpha) - std::pow(m_star_high, 1.0 - alpha);
		;

		fstar_high = num_high_mass_stars / total_stars;
		m_star_high_avg = m_imf_max * ((alpha - 1.0) / (alpha - 2.0)) * (1. - std::pow(m_star_high / m_imf_max, 2.0 - alpha)) /
				  (1. - std::pow(m_star_high / m_imf_max, 1.0 - alpha));
	}

	template <typename problem_t> struct ParticleChecker {
		amrex::Real current_time;
		amrex::Real dt;
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) const -> int
		{
			auto engine = amrex::RandomEngine();
			const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
			const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);

			const amrex::Real cs = HydroSystem<problem_t>::ComputeSoundSpeed(state_arr, i, j, k);
			const amrex::Real LambdaJ = cs / std::sqrt(C::Gconst * cell_density);
			const amrex::Real t_ff = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * cell_density));
			const amrex::Real prob_star_formation = eps_ff * dt / eps_star / t_ff;
			const amrex::Real random_draw = amrex::Random(engine);
			int num_star = 0;

			if (LambdaJ < J * dx[0] &&
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

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index,
				amrex::Real current_time, amrex::Real dt)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index), cpu_id(processor_id),
		      pid_start(particle_id_start), current_time(current_time), dt(dt)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long base_offset) const
		{
			// A simple demonstration of particle creation

			auto engine = amrex::RandomEngine();
			if (mass_idx + 3 < ParticleType::NReal) {
				// Calculate common values for all particles
				const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);
				const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
				const amrex::Real cell_mass = cell_volume * cell_density;
				amrex::Real vx = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
				amrex::Real vy = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
				amrex::Real vz = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;
				const amrex::Real t_ff = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * cell_density));
				const amrex::Real particle_mass = cell_density * cell_volume * eps_ff * dt / t_ff;
				const amrex::Real mass_low_mass_star = particle_mass * (1.0 - fstar_high);
				double total_momx = 0.0;
				double total_momy = 0.0;
				double total_momz = 0.0;

				for (int p_idx = 0; p_idx < num_particles; ++p_idx) {
					auto &p = particles[p_idx]; // NOLINT

					// Set particle ID and CPU
					p.id() = pid_start + base_offset + p_idx;
					p.cpu() = cpu_id;
					p.rdata(birth_time_index) = current_time;

					// Set particle evolution stage to 0 if it is a low mass star
					// This gets changed if there is a high mass star in the cell
					p.idata(evolution_stage_index) = p_idx;

					// Low Mass particle position at cell center
					p.pos(0) = plo[0] + (i + 0.5) * dx[0];
					p.pos(1) = plo[1] + (j + 0.5) * dx[1];
					p.pos(2) = plo[2] + (k + 0.5) * dx[2];

					// Low Mass particle mass and velocity
					p.rdata(mass_idx) = mass_low_mass_star;
					p.rdata(mass_idx + 1) = vx;
					p.rdata(mass_idx + 2) = vy;
					p.rdata(mass_idx + 3) = vz;

					p.rdata(birth_time_index + 1) = LONG_MAX;
					if (p_idx > 0) {
						double sigma_sq_x = NAN;
						double sigma_sq_y = NAN;
						double sigma_sq_z = NAN;
						double numx = 0.0;
						double numy = 0.0;
						double numz = 0.0;
						double denominator = 0.0;
						double vx_adj = NAN;
						double vy_adj = NAN;
						double vz_adj = NAN;
						double rho_adj = NAN;
						// Get the average velocity from the velocity dispersion of the surrounding cells
						// We use the velocity dispersion of the surrounding cells to get the velocity of the high mass star...
						//... from a log normal distribution
						for (int ii = i - 1; ii <= i + 1; ++ii) {
							for (int jj = j - 1; jj <= j + 1; ++jj) {
								for (int kk = k - 1; kk <= k + 1; ++kk) {

									vx_adj = (state_arr(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index)) /
										 state_arr(ii, jj, kk, HydroSystem<problem_t>::density_index);
									vy_adj = (state_arr(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index)) /
										 state_arr(ii, jj, kk, HydroSystem<problem_t>::density_index);
									vz_adj = (state_arr(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index)) /
										 state_arr(ii, jj, kk, HydroSystem<problem_t>::density_index);
									rho_adj = state_arr(ii, jj, kk, HydroSystem<problem_t>::density_index);
									numx += rho_adj * (vx_adj - (vx)) * (vx_adj - (vx));
									numy += rho_adj * (vy_adj - (vy)) * (vy_adj - (vy));
									numz += rho_adj * (vz_adj - (vz)) * (vz_adj - (vz));

									denominator += rho_adj;
								}
							}
						}
						sigma_sq_x = numx / denominator;
						sigma_sq_y = numy / denominator;
						sigma_sq_z = numz / denominator;

						p.rdata(mass_idx + 1) = (std::abs(vx) / vx) * amrex::RandomNormal(std::abs(vx), std::sqrt(sigma_sq_x), engine);
						p.rdata(mass_idx + 2) = (std::abs(vy) / vy) * amrex::RandomNormal(std::abs(vy), std::sqrt(sigma_sq_y), engine);
						p.rdata(mass_idx + 3) = (std::abs(vz) / vz) * amrex::RandomNormal(std::abs(vz), std::sqrt(sigma_sq_z), engine);

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

						p.idata(evolution_stage_index) = interpolate_fate(p.rdata(mass_idx));
						p.rdata(birth_time_index + 1) = interpolate_death_time(p.rdata(mass_idx));
						;
					}
				}

				if (num_particles > 1) { // Update momentum of the low mass star if there is(are) high mass star(s) in the cell
					const int p_idx = 0;
					auto &plow = particles[p_idx]; // NOLINT
					plow.rdata(mass_idx + 1) = -total_momx / plow.rdata(mass_idx);
					plow.rdata(mass_idx + 2) = -total_momy / plow.rdata(mass_idx);
					plow.rdata(mass_idx + 3) = -total_momz / plow.rdata(mass_idx);
				}

				const double factor = (cell_mass - particle_mass) / cell_volume / state_arr(i, j, k, HydroSystem<problem_t>::density_index);

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
			}
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt,
				    int evolution_stage_index = -1, int birth_time_index = -1)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType,
							  ParticleCreationTraits<ParticleType::StochasticStellarPop>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::StochasticStellarPop>::template ParticleCreator>(
		    container, mass_idx, state, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
}; // ParticleCreationTraits<ParticleType::StochasticStellarPop>

double ParticleCreationTraits<ParticleType::StochasticStellarPop>::fstar_high;	    // NOLINT
double ParticleCreationTraits<ParticleType::StochasticStellarPop>::m_star_high_avg; // NOLINT

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_CREATION_HPP_
