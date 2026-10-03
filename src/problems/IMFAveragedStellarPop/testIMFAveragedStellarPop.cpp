#include <algorithm>
#include <cmath>
#include <string>

#include "AMReX_BLassert.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/Random.hpp"
#include "particles/imf_supernova.hpp"
#include "particles/particle_types.hpp"

struct IMFAveragedStellarPop {
};

template <> struct Physics_Traits<IMFAveragedStellarPop> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
};

template <> struct HydroSystem_Traits<IMFAveragedStellarPop> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct quokka::EOS_Traits<IMFAveragedStellarPop> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_p;
};

template <> struct Particle_Traits<IMFAveragedStellarPop> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::IMFAveragedStellarPop;
};

template <> struct SimulationData<IMFAveragedStellarPop> {
	std::string particles_file = "../inputs/IMFAveragedStellarPop_particles.txt";
};

template <> void QuokkaSimulation<IMFAveragedStellarPop>::setInitialConditionsOnGrid(quokka::grid const &grid)
{
	const amrex::Box box = grid.indexRange_;
	const auto state = grid.array_;
	constexpr amrex::Real background_density = 1.0e-24;
	constexpr amrex::Real star_forming_density = 1.0e-21;
	constexpr amrex::Real temperature = 100.0;
	amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real density = (i == 16 && j == 16 && k == 16) ? star_forming_density : background_density;
		const amrex::Real internal_energy = density * C::k_B * temperature / ((quokka::EOS_Traits<IMFAveragedStellarPop>::gamma - 1.0) * C::m_p);
		state(i, j, k, HydroSystem<IMFAveragedStellarPop>::density_index) = density;
		state(i, j, k, HydroSystem<IMFAveragedStellarPop>::x1Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<IMFAveragedStellarPop>::x2Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<IMFAveragedStellarPop>::x3Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<IMFAveragedStellarPop>::energy_index) = internal_energy;
		state(i, j, k, HydroSystem<IMFAveragedStellarPop>::internalEnergy_index) = internal_energy;
	});
}

template <> void QuokkaSimulation<IMFAveragedStellarPop>::createInitialIMFAveragedStellarPopParticles()
{
	IMFAveragedStellarPopParticles->InitFromAsciiFile(userData_.particles_file, quokka::IMFAveragedStellarPopParticleRealComps, nullptr);
	constexpr std::uint64_t global_seed = 1234U;
	for (auto &level_entry : IMFAveragedStellarPopParticles->GetParticles()) {
		for (auto &tile_entry : level_entry) {
			auto &particle_array = tile_entry.second.GetArrayOfStructs();
			auto *particles = particle_array().data();
			const int count = particle_array.numParticles();
			amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(int index) noexcept {
				auto &particle = particles[index]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				const auto key = quokka::math::random::makeParticleKey(global_seed, static_cast<std::uint64_t>(particle.id()),
										       static_cast<std::uint32_t>(particle.cpu()));
				const auto schedule = quokka::particles::initializeSupernovaSchedule(key);
				particle.idata(quokka::IMFAveragedStellarPopParticleRNGKeyLoIdx) = static_cast<int>(static_cast<std::uint32_t>(key.value));
				particle.idata(quokka::IMFAveragedStellarPopParticleRNGKeyHiIdx) =
				    static_cast<int>(static_cast<std::uint32_t>(key.value >> 32U));
				particle.idata(quokka::IMFAveragedStellarPopParticleSNDrawIndexLoIdx) = 1;
				particle.idata(quokka::IMFAveragedStellarPopParticleSNDrawIndexHiIdx) = 0;
				particle.rdata(quokka::IMFAveragedStellarPopParticleNextSNIntensityIdx) = schedule.next_event_intensity;
			});
		}
	}
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	QuokkaSimulation<IMFAveragedStellarPop> simulation;
	simulation.setInitialConditions();
	const auto dx = simulation.geom[0].CellSizeArray();
	const amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	auto *stellar_descriptor = simulation.particleRegister_.getParticleDescriptor(quokka::ParticleType::IMFAveragedStellarPop);
	const amrex::Real stellar_mass_before_formation = stellar_descriptor->computeStellarMass();
	const amrex::Real gas_mass_before_formation = simulation.state_new_cc_[0].sum(HydroSystem<IMFAveragedStellarPop>::density_index) * volume;
	amrex::MultiFab accretion_rate(simulation.state_new_cc_[0].boxArray(), simulation.state_new_cc_[0].DistributionMap(), 1, 0);
	accretion_rate.setVal(0.0);
	simulation.particleRegister_.createParticlesFromState(simulation.state_new_cc_[0], accretion_rate, 0, 0.0, quokka::Myr_in_s);
	const amrex::Real stellar_mass_after_formation = stellar_descriptor->computeStellarMass();
	const amrex::Real gas_mass_after_formation = simulation.state_new_cc_[0].sum(HydroSystem<IMFAveragedStellarPop>::density_index) * volume;
	const amrex::Real formed_mass = stellar_mass_after_formation - stellar_mass_before_formation;
	const amrex::Real formation_mass_error = std::abs((gas_mass_before_formation - gas_mass_after_formation) - formed_mass) / std::max(formed_mass, 1.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(formed_mass > 0.0, "The eligible cell should form at least one IMF-averaged particle.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(formation_mass_error < 1.0e-8, "IMF-averaged particle formation did not conserve gas plus particle mass.");

	const amrex::Real gas_mass_before = simulation.state_new_cc_[0].sum(HydroSystem<IMFAveragedStellarPop>::density_index) * volume;

	const auto [event_count, max_velocity] = simulation.particleRegister_.depositSN(simulation.state_new_cc_[0], nullptr, 0, 0.0, 55.0 * quokka::Myr_in_s);
	const amrex::Real gas_mass_after = simulation.state_new_cc_[0].sum(HydroSystem<IMFAveragedStellarPop>::density_index) * volume;
	const amrex::Real expected_ejecta = static_cast<amrex::Real>(event_count) * 10.0 * C::M_solar;
	const amrex::Real relative_mass_error = std::abs((gas_mass_after - gas_mass_before) - expected_ejecta) / std::max(expected_ejecta, 1.0);
	const amrex::Real remaining_particle_mass = stellar_descriptor->computeStellarMass();

	amrex::Print() << "IMF-averaged formed mass = " << formed_mass / C::M_solar << " Msun, formation mass error = " << formation_mass_error
		       << ", SN events = " << event_count << ", ejecta mass error = " << relative_mass_error << ", maximum velocity = " << max_velocity << "\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(event_count > 0, "The deterministic IMF particle should schedule at least one SN by 55 Myr.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(relative_mass_error < 1.0e-8, "SN ejecta were not deposited through the Quokka feedback buffer.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(expected_ejecta > stellar_mass_after_formation,
					 "The test must inject more ejecta than the particles contained before feedback.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(remaining_particle_mass < stellar_mass_after_formation,
					 "Finite particle mass must not truncate or renormalize the scheduled Poisson events.");
	return 0;
}
