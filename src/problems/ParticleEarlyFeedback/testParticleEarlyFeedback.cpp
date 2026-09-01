/// \file testParticleEarlyFeedback.cpp
/// \brief Focused conservation test for empirically motivated early feedback.

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_early_feedback.hpp"
#include "particles/particle_types.hpp"

namespace
{
struct ParticleEarlyFeedback {};

constexpr amrex::Real gamma_gas = 5.0 / 3.0;
constexpr amrex::Real mean_molecular_weight = C::m_p;
constexpr amrex::Real density = 1.0e-21;
constexpr amrex::Real temperature = 100.0;
constexpr amrex::Real total_birth_mass = 100.0 * C::M_solar;
constexpr amrex::Real deposition_tolerance = 2.0e-8;
constexpr amrex::Real source_position = 5.0e19;

auto approximatelyEqual(amrex::Real actual, amrex::Real expected, amrex::Real relative_tolerance = 2.0e-12) -> bool
{
	const amrex::Real scale = std::max({std::abs(actual), std::abs(expected), static_cast<amrex::Real>(1.0e-300)});
	return std::abs(actual - expected) <= relative_tolerance * scale;
}

void testEquation10()
{
	constexpr amrex::Real mass = 2.0;
	constexpr amrex::Real p0 = 3.0;
	constexpr amrex::Real t_fb = 4.0;
	AMREX_ALWAYS_ASSERT(quokka::earlyFeedbackMomentumIncrement(0.0, 1.0, 2.0, mass, p0, t_fb, 1.0) == 0.0);
	AMREX_ALWAYS_ASSERT(approximatelyEqual(quokka::earlyFeedbackMomentumIncrement(2.0, 1.0, 2.0, mass, p0, t_fb, 1.0), p0 * mass * std::pow(0.25, 3)));
	AMREX_ALWAYS_ASSERT(
	    approximatelyEqual(quokka::earlyFeedbackMomentumIncrement(3.0, 2.0, 0.0, mass, p0, t_fb, 1.0), p0 * mass * (1.0 - std::pow(0.75, 3))));
	AMREX_ALWAYS_ASSERT(quokka::earlyFeedbackMomentumIncrement(4.0, 1.0, 0.0, mass, p0, t_fb, 1.0) == 0.0);
	AMREX_ALWAYS_ASSERT(approximatelyEqual(quokka::earlyFeedbackMomentumIncrement(0.0, t_fb, 0.0, mass, p0, t_fb, 0.5), 0.5 * p0 * mass));
	AMREX_ALWAYS_ASSERT(approximatelyEqual(quokka::earlyFeedbackMomentumIncrement(0.0, t_fb, 0.0, mass, p0, t_fb, 1.0), p0 * mass));

	amrex::Real partitioned_momentum = 0.0;
	for (int interval = 0; interval < 4; ++interval) {
		partitioned_momentum += quokka::earlyFeedbackMomentumIncrement(static_cast<amrex::Real>(interval), 1.0, 0.0, mass, p0, t_fb, 1.0);
	}
	AMREX_ALWAYS_ASSERT(approximatelyEqual(partitioned_momentum, quokka::earlyFeedbackMomentumIncrement(0.0, t_fb, 0.0, mass, p0, t_fb, 1.0)));
}
} // namespace

template <> struct Physics_Traits<ParticleEarlyFeedback> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 1;
	static constexpr int numPassiveScalars = numMassScalars;
};

template <> struct HydroSystem_Traits<ParticleEarlyFeedback> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct quokka::EOS_Traits<ParticleEarlyFeedback> {
	static constexpr double gamma = gamma_gas;
	static constexpr double mean_molecular_weight = ::mean_molecular_weight;
};

template <> struct Particle_Traits<ParticleEarlyFeedback> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct SimulationData<ParticleEarlyFeedback> {
	std::string particlesFile = "../inputs/ParticleEarlyFeedback_particles.txt";
	amrex::Real boostVelocity = 0.0;
	amrex::Real inflowSpeed = 0.0;
};

template <> void QuokkaSimulation<ParticleEarlyFeedback>::setInitialConditionsOnGrid(quokka::grid const &grid_element)
{
	const amrex::Box &box = grid_element.indexRange_;
	const auto state = grid_element.array_;
	const auto prob_lo = grid_element.prob_lo_;
	const auto cell_size = grid_element.dx_;
	const amrex::Real internal_energy = density * C::k_B * temperature / (mean_molecular_weight * (gamma_gas - 1.0));
	const amrex::Real boost_velocity = userData_.boostVelocity;
	const amrex::Real inflow_speed = userData_.inflowSpeed;
	amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real delta_x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * cell_size[0] - source_position;
		const amrex::Real delta_y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * cell_size[1] - source_position;
		const amrex::Real delta_z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * cell_size[2] - source_position;
		const amrex::Real radius = std::sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
		const amrex::Real inverse_radius = (radius > 0.0) ? 1.0 / radius : 0.0;
		const amrex::Real velocity_x = boost_velocity - inflow_speed * delta_x * inverse_radius;
		const amrex::Real velocity_y = -inflow_speed * delta_y * inverse_radius;
		const amrex::Real velocity_z = -inflow_speed * delta_z * inverse_radius;
		const amrex::Real momentum_x = density * velocity_x;
		const amrex::Real momentum_y = density * velocity_y;
		const amrex::Real momentum_z = density * velocity_z;
		const amrex::Real kinetic_energy = 0.5 * ((momentum_x * momentum_x) + (momentum_y * momentum_y) + (momentum_z * momentum_z)) / density;
		state(i, j, k, HydroSystem<ParticleEarlyFeedback>::density_index) = density;
		state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x1Momentum_index) = momentum_x;
		state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x2Momentum_index) = momentum_y;
		state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x3Momentum_index) = momentum_z;
		state(i, j, k, HydroSystem<ParticleEarlyFeedback>::energy_index) = internal_energy + kinetic_energy;
		state(i, j, k, HydroSystem<ParticleEarlyFeedback>::internalEnergy_index) = internal_energy;
		state(i, j, k, HydroSystem<ParticleEarlyFeedback>::scalar0_index) = density;
	});
}

template <> void QuokkaSimulation<ParticleEarlyFeedback>::createInitialStochasticStellarPopParticles()
{
	const int num_real_components = quokka::StochasticStellarPopParticleRealComps<ParticleEarlyFeedback>;
	StochasticStellarPopParticles->SetVerbose(0);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.particlesFile, num_real_components, nullptr);

	int particle_offset = 0;
	for (auto &level_entry : StochasticStellarPopParticles->GetParticles()) {
		for (auto &tile_entry : level_entry) {
			auto &particle_array = tile_entry.second.GetArrayOfStructs();
			const int num_particles = particle_array.numParticles();
			if (num_particles == 0) {
				continue;
			}
			auto *particles = particle_array().data();
			const int first_particle = particle_offset;
			amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE(int index) noexcept {
				auto &particle = particles[index]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				const int global_index = first_particle + index;
				if (global_index == 0) {
					particle.idata(quokka::StochasticStellarPopParticleStageIdx) =
					    static_cast<int>(quokka::StellarEvolutionStage::LowMassComposite);
				} else if (global_index == 1) {
					particle.idata(quokka::StochasticStellarPopParticleStageIdx) =
					    static_cast<int>(quokka::StellarEvolutionStage::HighMassNonExploding);
				} else {
					particle.idata(quokka::StochasticStellarPopParticleStageIdx) =
					    static_cast<int>(quokka::StellarEvolutionStage::SNRemnant);
				}
			});
			particle_offset += num_particles;
		}
	}
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	testEquation10();

	QuokkaSimulation<ParticleEarlyFeedback> simulation;
	amrex::ParmParse const problem_parameters("problem");
	problem_parameters.query("particles_file", simulation.userData_.particlesFile);
	problem_parameters.query("boost_velocity", simulation.userData_.boostVelocity);
	problem_parameters.query("inflow_speed", simulation.userData_.inflowSpeed);
	simulation.setInitialConditions();
	auto *stellar_descriptor = simulation.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop);
	const amrex::Real stellar_mass_before_split = stellar_descriptor->computeStellarMass();
	const amrex::Real stellar_birth_mass_before_split = stellar_descriptor->computeStellarMassAtBirth();
	stellar_descriptor->splitParticles(0, 2);
	const amrex::Real stellar_mass_after_split = stellar_descriptor->computeStellarMass();
	const amrex::Real stellar_birth_mass_after_split = stellar_descriptor->computeStellarMassAtBirth();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(approximatelyEqual(stellar_mass_after_split, stellar_mass_before_split),
					 "Particle splitting did not conserve current stellar mass.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(approximatelyEqual(stellar_birth_mass_after_split, stellar_birth_mass_before_split),
					 "Particle splitting did not conserve stellar mass_at_birth.");

	const auto cell_size = simulation.geom[0].CellSizeArray();
	const amrex::Real cell_volume = AMREX_D_TERM(cell_size[0], *cell_size[1], *cell_size[2]);
	const amrex::Real initial_gas_mass = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::density_index) * cell_volume;
	const amrex::Real initial_scalar_mass = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::scalar0_index) * cell_volume;
	const amrex::Real initial_internal_energy = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::internalEnergy_index) * cell_volume;
	using FaceStateArray = std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>;
	const amrex::Real initial_momentum_x = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::x1Momentum_index) * cell_volume;
	const amrex::Real initial_momentum_y = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::x2Momentum_index) * cell_volume;
	const amrex::Real initial_momentum_z = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::x3Momentum_index) * cell_volume;

	const amrex::Real step_time = 0.0;
	const amrex::Real dt = 1.0 * quokka::Myr_in_s;
	const auto stats = simulation.particleRegister_.depositEarlyFeedback(simulation.state_new_cc_[0], nullptr, 0, step_time, dt);
	const amrex::Real expected_momentum =
	    quokka::earlyFeedbackMomentumIncrement(step_time, dt, 0.0, total_birth_mass, quokka::EMF_p0, quokka::EMF_tFB, quokka::EMF_alpha);

	const amrex::Real momentum_x = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::x1Momentum_index) * cell_volume - initial_momentum_x;
	const amrex::Real momentum_y = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::x2Momentum_index) * cell_volume - initial_momentum_y;
	const amrex::Real momentum_z = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::x3Momentum_index) * cell_volume - initial_momentum_z;
	const auto prob_lo = simulation.geom[0].ProbLoArray();
	const amrex::Real boost_velocity = simulation.userData_.boostVelocity;
	const amrex::Real inflow_speed = simulation.userData_.inflowSpeed;
	const amrex::Real scalar_impulse = simulation.computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real delta_x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * cell_size[0] - source_position;
		    const amrex::Real delta_y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * cell_size[1] - source_position;
		    const amrex::Real delta_z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * cell_size[2] - source_position;
		    const amrex::Real radius = std::sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
		    const amrex::Real inverse_radius = (radius > 0.0) ? 1.0 / radius : 0.0;
		    const amrex::Real rho = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::density_index);
		    const amrex::Real px =
			state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x1Momentum_index) - rho * (boost_velocity - inflow_speed * delta_x * inverse_radius);
		    const amrex::Real py = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x2Momentum_index) + rho * inflow_speed * delta_y * inverse_radius;
		    const amrex::Real pz = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x3Momentum_index) + rho * inflow_speed * delta_z * inverse_radius;
		    return std::sqrt((px * px) + (py * py) + (pz * pz));
	    });
	const amrex::Real energy_consistency_error = simulation.computeVolumeIntegral(
	    [] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::density_index);
		    const amrex::Real px = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x1Momentum_index);
		    const amrex::Real py = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x2Momentum_index);
		    const amrex::Real pz = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::x3Momentum_index);
		    const amrex::Real kinetic_energy = 0.5 * ((px * px) + (py * py) + (pz * pz)) / rho;
		    const amrex::Real internal_energy = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::internalEnergy_index);
		    const amrex::Real total_energy = state(i, j, k, HydroSystem<ParticleEarlyFeedback>::energy_index);
		    return std::abs(total_energy - internal_energy - kinetic_energy);
	    });

	const amrex::Real final_gas_mass = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::density_index) * cell_volume;
	const amrex::Real final_scalar_mass = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::scalar0_index) * cell_volume;
	const amrex::Real final_internal_energy = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::internalEnergy_index) * cell_volume;
	const amrex::Real final_total_energy = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::energy_index) * cell_volume;

	amrex::Print() << "EMF requested scalar momentum = " << stats.scalar_momentum << " g cm/s (expected " << expected_momentum << ")\n"
		       << "EMF deposited scalar impulse = " << scalar_impulse << " g cm/s\n"
		       << "EMF net vector momentum = (" << momentum_x << ", " << momentum_y << ", " << momentum_z << ") g cm/s\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(stats.active_particles == 6, "EMF must include split composite, individual high-mass, and remnant particles.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(approximatelyEqual(stats.scalar_momentum, expected_momentum),
					 "Equation-10 requested momentum does not use the summed birth mass.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(approximatelyEqual(scalar_impulse, expected_momentum, deposition_tolerance),
					 "Deposited scalar impulse does not equal the Equation-10 request.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(momentum_x) <= deposition_tolerance * expected_momentum &&
					     std::abs(momentum_y) <= deposition_tolerance * expected_momentum &&
					     std::abs(momentum_z) <= deposition_tolerance * expected_momentum,
					 "Early-feedback stencil does not have zero net vector momentum.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(approximatelyEqual(final_gas_mass, initial_gas_mass), "EMF changed gas mass.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(approximatelyEqual(final_scalar_mass, initial_scalar_mass), "EMF changed a MassScalar.");
	if (simulation.userData_.inflowSpeed > 0.0) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(final_internal_energy > initial_internal_energy,
						 "Negative gas-motion work was not thermalized into the host cell.");
	} else {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(approximatelyEqual(final_internal_energy, initial_internal_energy),
						 "A uniform gas boost spuriously changed internal energy.");
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(energy_consistency_error <= deposition_tolerance * final_total_energy,
					 "EMF total-energy update is inconsistent with its kinetic-energy change.");

	quokka::EMF_enabled = false;
	const auto disabled_stats = simulation.particleRegister_.depositEarlyFeedback(simulation.state_new_cc_[0], nullptr, 0, step_time, dt);
	const amrex::Real disabled_total_energy = simulation.state_new_cc_[0].sum(HydroSystem<ParticleEarlyFeedback>::energy_index) * cell_volume;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(disabled_stats.active_particles == 0 && disabled_stats.scalar_momentum == 0.0,
					 "Disabled EMF reported active sources.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(disabled_total_energy == final_total_energy, "Disabled EMF changed the gas state.");
	return 0;
}
