#ifndef QUOKKA_PARTICLE_SCHEMA_TRAITS_HPP_
#define QUOKKA_PARTICLE_SCHEMA_TRAITS_HPP_

#include "particles/particle_schema.hpp"
#include "particles/particle_types.hpp"

namespace quokka
{
template <ParticleType population, typename problem_t> struct ParticleSchemaTraits {
	[[nodiscard]] static auto Create() -> ParticleSchema
	{
		amrex::Vector<std::string> reals;
		amrex::Vector<std::string> integers;
		std::string name;
		ParticleLayout layout;
		if constexpr (population != ParticleType::Tracer) {
			reals = getParticleRealCompNames<population, problem_t>();
			integers = getParticleIntCompNames<population, problem_t>();
		} else {
			for (int d = 0; d < AMREX_SPACEDIM; ++d) {
				reals.push_back("real_comp" + std::to_string(d));
			}
		}
		layout.numReal = static_cast<int>(reals.size());
		layout.numInt = static_cast<int>(integers.size());
		if constexpr (population == ParticleType::Rad) {
			name = "Rad_particles";
			layout.birthTime = RadParticleBirthTimeIdx;
			layout.deathTime = RadParticleDeathTimeIdx;
			layout.numLuminosities = layout.numReal - RadParticleLumIdx;
			layout.luminosity = layout.numLuminosities > 0 ? RadParticleLumIdx : -1;
		} else if constexpr (population == ParticleType::CIC) {
			name = "CIC_particles";
			layout.mass = CICParticleMassIdx;
			layout.velocity = CICParticleVxIdx;
		} else if constexpr (population == ParticleType::CICRad) {
			name = "CICRad_particles";
			layout.mass = CICRadParticleMassIdx;
			layout.velocity = CICRadParticleVxIdx;
			layout.birthTime = CICRadParticleBirthTimeIdx;
			layout.deathTime = CICRadParticleDeathTimeIdx;
			layout.numLuminosities = layout.numReal - CICRadParticleLumIdx;
			layout.luminosity = layout.numLuminosities > 0 ? CICRadParticleLumIdx : -1;
		} else if constexpr (population == ParticleType::Test) {
			name = "Test_particles";
			layout.mass = TestParticleMassIdx;
			layout.velocity = TestParticleVxIdx;
			layout.birthTime = TestParticleBirthTimeIdx;
			layout.deathTime = TestParticleDeathTimeIdx;
			layout.evolutionStage = TestParticleStageIdx;
			layout.numLuminosities = layout.numReal - TestParticleLumIdx;
			layout.luminosity = layout.numLuminosities > 0 ? TestParticleLumIdx : -1;
		} else if constexpr (population == ParticleType::Sink) {
			name = "Sink_particles";
			layout.mass = SinkParticleMassIdx;
			layout.velocity = SinkParticleVxIdx;
			layout.accretionRate = SinkParticleMdotIdx;
			layout.angularMomentum = SinkParticleLxIdx;
		} else if constexpr (population == ParticleType::Star) {
			name = "Star_particles";
			layout.mass = StarParticleMassIdx;
			layout.velocity = StarParticleVxIdx;
			layout.birthTime = StarParticleBirthTimeIdx;
			layout.accretionRate = StarParticleMdotIdx;
			layout.numLuminosities = Physics_Traits<problem_t>::nGroups;
			layout.luminosity = layout.numLuminosities > 0 ? StarParticleLumIdx : -1;
			// Stellar-model integer extensions do not imply an evolution-stage field.
		} else if constexpr (population == ParticleType::StochasticStellarPop) {
			name = "StochasticStellarPop_particles";
			layout.mass = StochasticStellarPopParticleMassIdx;
			layout.velocity = StochasticStellarPopParticleVxIdx;
			layout.birthTime = StochasticStellarPopParticleBirthTimeIdx;
			layout.deathTime = StochasticStellarPopParticleDeathTimeIdx;
			layout.massAtBirth = StochasticStellarPopParticleMassAtBirthIdx;
			layout.evolutionStage = StochasticStellarPopParticleStageIdx;
			if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
				layout.numLuminosities = Physics_Traits<problem_t>::nGroups;
				layout.luminosity = layout.numLuminosities > 0 ? StochasticStellarPopParticleLumIdx : -1;
			}
		} else if constexpr (population == ParticleType::Tracer) {
			name = "tracer_particles";
			layout.velocity = 0;
		}

		amrex::Vector<std::string> units(reals.size());
		ParticleSchema::Units definitions;
		if constexpr (population != ParticleType::Tracer) {
			definitions = get_units_data().at(population).front();
			const auto &legacy = definitions;
			for (int i = 0; i < layout.numReal; ++i) {
				if (layout.luminosity >= 0 && i >= layout.luminosity && i < layout.luminosity + layout.numLuminosities) {
					units[i] = "luminosity";
				} else if (legacy.contains(reals[i])) {
					units[i] = reals[i];
				}
			}
		}
		return {population,
			std::move(name),
			reals,
			integers,
			amrex::Vector<amrex::ParticleReal>(reals.size(), 0),
			amrex::Vector<int>(integers.size(), 0),
			std::move(units),
			layout,
			std::move(definitions)};
	}
};
} // namespace quokka
#endif
