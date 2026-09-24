#ifndef QUOKKA_MATH_PHILOX_HPP_
#define QUOKKA_MATH_PHILOX_HPP_

#include <array>
#include <cstdint>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

namespace quokka::math::random
{

using PhiloxCounter = std::array<std::uint32_t, 4>;
using PhiloxKey = std::array<std::uint32_t, 2>;

namespace detail
{
constexpr std::uint32_t philox4x32_m0 = 0xD2511F53U;
constexpr std::uint32_t philox4x32_m1 = 0xCD9E8D57U;
constexpr std::uint32_t philox4x32_w0 = 0x9E3779B9U;
constexpr std::uint32_t philox4x32_w1 = 0xBB67AE85U;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto multiplyHigh(const std::uint32_t lhs, const std::uint32_t rhs) -> std::uint32_t
{
	return static_cast<std::uint32_t>((static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs)) >> 32U);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto philoxRound(const PhiloxCounter &counter, const PhiloxKey &key) -> PhiloxCounter
{
	const std::uint64_t product0 = static_cast<std::uint64_t>(philox4x32_m0) * static_cast<std::uint64_t>(counter[0]);
	const std::uint64_t product1 = static_cast<std::uint64_t>(philox4x32_m1) * static_cast<std::uint64_t>(counter[2]);
	return {multiplyHigh(philox4x32_m1, counter[2]) ^ counter[1] ^ key[0], static_cast<std::uint32_t>(product1),
		multiplyHigh(philox4x32_m0, counter[0]) ^ counter[3] ^ key[1], static_cast<std::uint32_t>(product0)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto bumpKey(const PhiloxKey &key) -> PhiloxKey { return {key[0] + philox4x32_w0, key[1] + philox4x32_w1}; }
} // namespace detail

/// The Philox4x32-10 counter-based generator from Random123.
///
/// This implementation deliberately uses only fixed-width unsigned integer
/// arithmetic so a (counter, key) pair maps to the same four words on every
/// supported CPU and GPU backend.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto philox4x32(const PhiloxCounter &input_counter, const PhiloxKey &input_key) -> PhiloxCounter
{
	PhiloxCounter counter = input_counter;
	PhiloxKey key = input_key;
	for (int round = 0; round < 10; ++round) {
		counter = detail::philoxRound(counter, key);
		if (round != 9) {
			key = detail::bumpKey(key);
		}
	}
	return counter;
}

} // namespace quokka::math::random

#endif // QUOKKA_MATH_PHILOX_HPP_
