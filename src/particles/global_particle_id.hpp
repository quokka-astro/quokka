#ifndef GLOBAL_PARTICLE_ID_HPP_
#define GLOBAL_PARTICLE_ID_HPP_

#include <cstdint>

namespace quokka::particle
{

inline constexpr std::uint64_t localIdToGlobal(int id, int cpu) noexcept
{
	static_assert(sizeof(int) * 2u <= sizeof(std::uint64_t),
	              "int size might cause collisions in global IDs");
	return static_cast<std::uint64_t>(id) |
	       (static_cast<std::uint64_t>(cpu) << 32u);
}

} // namespace quokka::particle

#endif // GLOBAL_PARTICLE_ID_HPP_
