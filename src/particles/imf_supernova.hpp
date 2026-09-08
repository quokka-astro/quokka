#ifndef IMF_SUPERNOVA_HPP_
#define IMF_SUPERNOVA_HPP_

#include <array>
#include <cstdint>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "math/Random.hpp"

namespace quokka::particles
{

/// Cumulative core-collapse SN yield for a fully sampled Kroupa IMF.
///
/// The table integrates a Kroupa (2001) IMF from 0.08--100 Msun and maps
/// 8--100 Msun progenitors to age using Quokka's existing stellar-lifetime
/// table. Linear interpolation makes the cumulative intensity continuous.
/// Values are expected explosions per solar mass at birth.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cumulativeCoreCollapseSNPerSolarMass(const double age_myr) -> double
{
	constexpr std::array<double, 14> ages_myr{0.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 45.0, 50.0, 55.0};
	constexpr std::array<double, 14> cumulative_yield{0.0,
							  0.0,
							  4.27189800074e-05,
							  5.25547204560e-04,
							  9.12908543667e-04,
							  1.84860333355e-03,
							  2.83970473275e-03,
							  4.69025093805e-03,
							  6.28258134825e-03,
							  8.67376217784e-03,
							  1.08389135883e-02,
							  1.09297147483e-02,
							  1.09297147483e-02,
							  1.09297147483e-02};
	if (age_myr <= ages_myr.front()) {
		return 0.0;
	}
	if (age_myr >= ages_myr.back()) {
		return cumulative_yield.back();
	}
	for (int i = 0; i < static_cast<int>(ages_myr.size()) - 1; ++i) {
		if (age_myr < ages_myr[i + 1]) {
			const double fraction = (age_myr - ages_myr[i]) / (ages_myr[i + 1] - ages_myr[i]);
			return cumulative_yield[i] + fraction * (cumulative_yield[i + 1] - cumulative_yield[i]);
		}
	}
	return cumulative_yield.back();
}

struct SupernovaScheduleState {
	quokka::math::random::ParticleKey key{};
	std::uint64_t next_draw_index{};
	double next_event_intensity{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto initializeSupernovaSchedule(const quokka::math::random::ParticleKey key) -> SupernovaScheduleState
{
	return {.key = key,
		.next_draw_index = 1U,
		.next_event_intensity = quokka::math::random::exponential(key, quokka::math::random::Stream::CoreCollapseSN, 0U)};
}

/// Consume every Poisson arrival whose cumulative-intensity threshold has
/// been crossed. Since the state stores the next arrival rather than a count
/// sampled per timestep, the event sequence is independent of timestep
/// partitioning and survives restart exactly.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto advanceSupernovaSchedule(SupernovaScheduleState &schedule, const double cumulative_intensity) -> int
{
	int event_count = 0;
	while (schedule.next_event_intensity <= cumulative_intensity) {
		++event_count;
		schedule.next_event_intensity +=
		    quokka::math::random::exponential(schedule.key, quokka::math::random::Stream::CoreCollapseSN, schedule.next_draw_index);
		++schedule.next_draw_index;
	}
	return event_count;
}

} // namespace quokka::particles

#endif // IMF_SUPERNOVA_HPP_
