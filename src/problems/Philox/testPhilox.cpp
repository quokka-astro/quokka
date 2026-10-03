#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "AMReX_Print.H"

#include "math/Philox.hpp"
#include "math/Random.hpp"
#include "particles/imf_supernova.hpp"

struct Philox {
};

auto problem_main() -> int
{
	using quokka::math::random::ParticleKey;
	using quokka::math::random::PhiloxCounter;
	using quokka::math::random::PhiloxKey;
	using quokka::math::random::Stream;

	struct KnownAnswer {
		PhiloxCounter counter;
		PhiloxKey key;
		PhiloxCounter expected;
	};
	// Random123's published Philox4x32-10 known-answer vectors.
	constexpr std::array<KnownAnswer, 3> known_answers{{
	    {.counter = {0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U},
	     .key = {0x00000000U, 0x00000000U},
	     .expected = {0x6627e8d5U, 0xe169c58dU, 0xbc57ac4cU, 0x9b00dbd8U}},
	    {.counter = {0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU},
	     .key = {0xffffffffU, 0xffffffffU},
	     .expected = {0x408f276dU, 0x41c83b0eU, 0xa20bc7c6U, 0x6d5451fdU}},
	    {.counter = {0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U},
	     .key = {0xa4093822U, 0x299f31d0U},
	     .expected = {0xd16cfe09U, 0x94fdccebU, 0x5001e420U, 0x24126ea1U}},
	}};

	bool passed = true;
	for (const auto &answer : known_answers) {
		passed = passed && (quokka::math::random::philox4x32(answer.counter, answer.key) == answer.expected);
	}

	constexpr ParticleKey key{0x0123456789abcdefULL};
	passed = passed && (std::bit_cast<std::uint64_t>(quokka::math::random::uniformOpen01(key, Stream::CoreCollapseSN, 0U)) == 0x3fe70a0445d8b197ULL);
	passed = passed && (std::bit_cast<std::uint64_t>(quokka::math::random::exponential(key, Stream::CoreCollapseSN, 0U)) == 0x3fd506d4b9092ba9ULL);
	for (std::uint64_t draw = 0; draw < 32; ++draw) {
		const double uniform = quokka::math::random::uniformOpen01(key, Stream::CoreCollapseSN, draw);
		const double deviate = quokka::math::random::exponential(key, Stream::CoreCollapseSN, draw);
		passed = passed && (uniform > 0.0) && (uniform < 1.0) && std::isfinite(deviate) && (deviate > 0.0);
		passed = passed && (uniform != quokka::math::random::uniformOpen01(key, Stream::TypeIaSN, draw));
		passed = passed && (std::abs(deviate + std::log(uniform)) < 2.0e-15);
	}

	// Advancing to the same final cumulative intensity in one step or many
	// must consume the identical event sequence and leave identical state.
	auto one_step = quokka::particles::initializeSupernovaSchedule(key);
	auto many_steps = quokka::particles::initializeSupernovaSchedule(key);
	const int one_step_count = quokka::particles::advanceSupernovaSchedule(one_step, 12.0);
	int many_step_count = 0;
	for (int intensity = 1; intensity <= 12; ++intensity) {
		many_step_count += quokka::particles::advanceSupernovaSchedule(many_steps, static_cast<double>(intensity));
	}
	passed = passed && (one_step_count == many_step_count) && (one_step.next_draw_index == many_steps.next_draw_index) &&
		 (one_step.next_event_intensity == many_steps.next_event_intensity);

	// Equal-mass, equal-age particles remain statistically independent because
	// their immutable identities produce independent Philox keys.
	const auto other_key = quokka::math::random::makeParticleKey(17U, 43U, 0U);
	const auto first_key = quokka::math::random::makeParticleKey(17U, 42U, 0U);
	passed = passed && (quokka::math::random::uniformOpen01(first_key, Stream::CoreCollapseSN, 0U) !=
			    quokka::math::random::uniformOpen01(other_key, Stream::CoreCollapseSN, 0U));

	constexpr int ensemble_size = 20000;
	constexpr double ensemble_intensity = 8.0;
	double count_sum = 0.0;
	double count_squared_sum = 0.0;
	for (int particle_id = 0; particle_id < ensemble_size; ++particle_id) {
		const auto ensemble_key = quokka::math::random::makeParticleKey(17U, static_cast<std::uint64_t>(particle_id), 0U);
		auto schedule = quokka::particles::initializeSupernovaSchedule(ensemble_key);
		const auto count = static_cast<double>(quokka::particles::advanceSupernovaSchedule(schedule, ensemble_intensity));
		count_sum += count;
		count_squared_sum += count * count;
	}
	const double sample_mean = count_sum / ensemble_size;
	const double sample_variance = count_squared_sum / ensemble_size - sample_mean * sample_mean;
	passed = passed && (std::abs(sample_mean - ensemble_intensity) < 0.1) && (std::abs(sample_variance - ensemble_intensity) < 0.2);

	if (!passed) {
		amrex::Print() << "Philox deterministic random-number tests failed.\n";
		return 1;
	}
	amrex::Print() << "Philox deterministic random-number tests passed.\n";
	return 0;
}
