#ifndef QUOKKA_MATH_RANDOM_HPP_
#define QUOKKA_MATH_RANDOM_HPP_

#include <cstdint>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "math/Philox.hpp"

namespace quokka::math::random
{

/// Independent random-number streams owned by a particle. Adding draws to one
/// physical process cannot perturb any other process.
enum class Stream : std::uint32_t {
	CoreCollapseSN = 0,
	TypeIaSN = 1,
	StarFormation = 2,
};

struct ParticleKey {
	std::uint64_t value{};
};

/// SplitMix64 is used only to turn the user seed and immutable particle
/// identity into a well-diffused Philox key. It is not used as a PRNG stream.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto splitmix64(std::uint64_t value) -> std::uint64_t
{
	value += 0x9E3779B97F4A7C15ULL;
	value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
	value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
	return value ^ (value >> 31U);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto makeParticleKey(const std::uint64_t global_seed, const std::uint64_t particle_id,
									const std::uint32_t creation_cpu) -> ParticleKey
{
	const std::uint64_t cpu_hash = splitmix64(static_cast<std::uint64_t>(creation_cpu));
	const std::uint64_t rotated_cpu_hash = (cpu_hash << 32U) | (cpu_hash >> 32U);
	return {splitmix64(global_seed ^ splitmix64(particle_id) ^ rotated_cpu_hash)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto generateBlock(const ParticleKey key, const Stream stream, const std::uint64_t block_index)
    -> PhiloxCounter
{
	const PhiloxCounter counter{static_cast<std::uint32_t>(block_index), static_cast<std::uint32_t>(block_index >> 32U), static_cast<std::uint32_t>(stream),
				    0U};
	const PhiloxKey philox_key{static_cast<std::uint32_t>(key.value), static_cast<std::uint32_t>(key.value >> 32U)};
	return philox4x32(counter, philox_key);
}

/// Return the indexed 52-bit variate in the open interval (0, 1).
/// Each Philox block supplies two variates, so draw indices are stable under
/// changes to batching and execution order.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto uniformOpen01(const ParticleKey key, const Stream stream, const std::uint64_t draw_index) -> double
{
	const PhiloxCounter words = generateBlock(key, stream, draw_index >> 1U);
	const int offset = static_cast<int>((draw_index & 1U) * 2U);
	const std::uint64_t bits = (static_cast<std::uint64_t>(words[offset]) << 20U) | (static_cast<std::uint64_t>(words[offset + 1]) >> 12U);
	constexpr double inverse_two_to_52 = 0x1.0p-52;
	return (static_cast<double>(bits) + 0.5) * inverse_two_to_52;
}

namespace detail
{
/// Deterministic natural logarithm for positive doubles in (0, 1].
/// Range reduction bounds |z| <= 0.1716 in the atanh series below. Twelve
/// terms then give better than double-precision truncation error.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto deterministicLog(const double input) -> double
{
	double mantissa = input;
	int exponent = 0;
	while (mantissa < 1.0) {
		mantissa *= 2.0;
		--exponent;
	}
	constexpr double sqrt_two = 1.41421356237309504880168872420969808; // NOLINT(modernize-use-std-numbers)
	if (mantissa > sqrt_two) {
		mantissa *= 0.5;
		++exponent;
	}

	const double z = (mantissa - 1.0) / (mantissa + 1.0);
	const double z_squared = z * z;
	double power = z;
	double sum = power;
	for (int denominator = 3; denominator <= 23; denominator += 2) {
		power *= z_squared;
		sum += power / static_cast<double>(denominator);
	}
	constexpr double ln_two_hi = 0.693147180559945286226763982995180413; // NOLINT(modernize-use-std-numbers)
	constexpr double ln_two_lo = 2.319046813846299558417771099e-17;
	const double reduced_log = 2.0 * sum;
	return static_cast<double>(exponent) * ln_two_hi + (reduced_log + static_cast<double>(exponent) * ln_two_lo);
}
} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto exponential(const ParticleKey key, const Stream stream, const std::uint64_t draw_index) -> double
{
	return -detail::deterministicLog(uniformOpen01(key, stream, draw_index));
}

} // namespace quokka::math::random

#endif // QUOKKA_MATH_RANDOM_HPP_
