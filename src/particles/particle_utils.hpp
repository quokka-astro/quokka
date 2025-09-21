#ifndef PARTICLE_UTILS_HPP_
#define PARTICLE_UTILS_HPP_

#include "AMReX_MultiFab.H"
#include "fundamental_constants.H"
#include "math/FastMath.hpp"
#include "particles/particle_types.hpp"

namespace quokka::ParticleUtils
{

constexpr int stencil_size = 3;
constexpr int SN_stencil_array_size = stencil_size + 1;
constexpr double jeansNo = 0.25; // Jeans number

static_assert(stencil_size <= 3, "stencil_size must be <= 3");

constexpr amrex::Real stencil_volume = 4.0 / 3.0 * M_PI * stencil_size * stencil_size * stencil_size;

using kernel_weights_array_t =
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, SN_stencil_array_size>, SN_stencil_array_size>, SN_stencil_array_size>;

// spherical kernel, normalized to sum 1
constexpr kernel_weights_array_t kernel_spherical_3_weights_normalized = {{{{{0.00884198143074, 0.00884198143074, 0.00884198143074, 0.00416240696843},
									     {0.00884198143074, 0.00884198143074, 0.00884198143074, 0.00262865918549},
									     {0.00884198143074, 0.00884198143074, 0.00596795726055, 0.00005052308190},
									     {0.00416240696843, 0.00262865918549, 0.00005052308190, 0.00000000000000}}},
									   {{{0.00884198143074, 0.00884198143074, 0.00884198143074, 0.00262865918549},
									     {0.00884198143074, 0.00884198143074, 0.00861063982859, 0.00119306623841},
									     {0.00884198143074, 0.00861063982859, 0.00400459528385, 0.00000136166514},
									     {0.00262865918549, 0.00119306623841, 0.00000136166514, 0.00000000000000}}},
									   {{{0.00884198143074, 0.00884198143074, 0.00596795726055, 0.00005052308190},
									     {0.00884198143074, 0.00861063982859, 0.00400459528385, 0.00000136166514},
									     {0.00596795726055, 0.00400459528385, 0.00045652034325, 0.00000000000000},
									     {0.00005052308190, 0.00000136166514, 0.00000000000000, 0.00000000000000}}},
									   {{{0.00416240696843, 0.00262865918549, 0.00005052308190, 0.00000000000000},
									     {0.00262865918549, 0.00119306623841, 0.00000136166514, 0.00000000000000},
									     {0.00005052308190, 0.00000136166514, 0.00000000000000, 0.00000000000000},
									     {0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000}}}}};

// spherical kernel
constexpr kernel_weights_array_t kernel_spherical_3_weights = {{{{{1.00000000000000, 1.00000000000000, 1.00000000000000, 0.47075500000000},
								  {1.00000000000000, 1.00000000000000, 1.00000000000000, 0.29729300000000},
								  {1.00000000000000, 1.00000000000000, 0.67495700000000, 0.00571400000000},
								  {0.47075500000000, 0.29729300000000, 0.00571400000000, 0.00000000000000}}},
								{{{1.00000000000000, 1.00000000000000, 1.00000000000000, 0.29729300000000},
								  {1.00000000000000, 1.00000000000000, 0.97383600000000, 0.13493200000000},
								  {1.00000000000000, 0.97383600000000, 0.45290700000000, 0.00015400000000},
								  {0.29729300000000, 0.13493200000000, 0.00015400000000, 0.00000000000000}}},
								{{{1.00000000000000, 1.00000000000000, 0.67495700000000, 0.00571400000000},
								  {1.00000000000000, 0.97383600000000, 0.45290700000000, 0.00015400000000},
								  {0.67495700000000, 0.45290700000000, 0.05163100000000, 0.00000000000000},
								  {0.00571400000000, 0.00015400000000, 0.00000000000000, 0.00000000000000}}},
								{{{0.47075500000000, 0.29729300000000, 0.00571400000000, 0.00000000000000},
								  {0.29729300000000, 0.13493200000000, 0.00015400000000, 0.00000000000000},
								  {0.00571400000000, 0.00015400000000, 0.00000000000000, 0.00000000000000},
								  {0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000}}}}};

// uniform kernel
constexpr kernel_weights_array_t kernel_spherical_uniform_3_weights = {{{{{1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000}}},
									{{{1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000}}},
									{{{1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000}}},
									{{{1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000},
									  {1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000}}}}};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto computeJeansDensity(double cs_cell, double dx) -> double
{
	return jeansNo * jeansNo * M_PI * cs_cell * cs_cell / (C::Gconst * (dx * dx));
}

inline void roundoffMultiFab(amrex::MultiFab &mf, amrex::MultiFab &mf_count)
{
	// Apply roundoff algorithm to reduce floating-point precision errors by removing
	// the least significant bits from IEEE 754 double precision numbers.
	//
	// IEEE 754 double precision format:
	// - 1 sign bit + 11 exponent bits + 52 mantissa bits = 64 total bits
	// - The mantissa has an implicit leading 1, giving 53 bits of precision
	//
	// This version uses mf_count to compute digit_to_remove based on the relative error
	// formula: relative_error = (N - 1) * epsilon, where N is the count and epsilon
	// is machine epsilon. We convert this to binary digits and add redundancy.

	constexpr amrex::Real tiny = 1.0e10 * std::numeric_limits<amrex::Real>::min();
	constexpr amrex::Real machine_epsilon = std::numeric_limits<amrex::Real>::epsilon();
	const auto redundancy = static_cast<unsigned int>(reproducibility_roundoff_redundancy);

	// Get array accessor for all patches at once
	auto const &arr = mf.arrays();
	auto const &count_arr = mf_count.const_arrays();
	const int ncomp = mf.nComp();

	// Apply roundoff algorithm to every grid point and component in parallel
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// Process all components at this grid point
		for (int n = 0; n < ncomp; ++n) {
			const auto val = arr[bx](i, j, k, n);
			const auto count = count_arr[bx](i, j, k, n);

			if (std::abs(val) < tiny) {
				arr[bx](i, j, k, n) = 0.0;
				continue;
			}

			// Compute digit_to_remove based on count
			auto digit_to_remove = redundancy;

			if (count > 1.0) {
				// Relative error estimate: (N - 1) * epsilon
				const amrex::Real relative_error = (count - 1.0) * machine_epsilon;

				// Convert to binary digits: log2(1/relative_error)
				if (relative_error > 0.0) {
					const amrex::Real binary_digits = -FastMath::fastlg(relative_error);

					if (binary_digits < 0.0) {
						// count > 1/machine_epsilon; unlikely
						digit_to_remove = 52u;
					} else {
						// Add reproducibility_roundoff_redundancy
						digit_to_remove += static_cast<unsigned int>(binary_digits);

						// Clamp to reasonable bounds (1 to 52 bits)
						digit_to_remove = amrex::max(1u, amrex::min(52u, digit_to_remove));
					}
				}
			}

			const auto factor = static_cast<amrex::Real>((1ULL << digit_to_remove) + 1);

			volatile amrex::Real c = factor * val;
			volatile amrex::Real a = c - val;
			arr[bx](i, j, k, n) = c - a;
		}
	});
}

// Overload for cases where mf_count is not available
inline void roundoffMultiFab(amrex::MultiFab &mf)
{
	// Apply roundoff algorithm with fixed digit_to_remove when count is not available
	const unsigned int digit_to_remove = reproducibility_roundoff_redundancy + 3;
	const auto factor = static_cast<amrex::Real>((1ULL << digit_to_remove) + 1);

	constexpr amrex::Real tiny = 1.0e10 * std::numeric_limits<amrex::Real>::min();

	// Get array accessor for all patches at once
	auto const &arr = mf.arrays();
	const int ncomp = mf.nComp();

	// Apply roundoff algorithm to every grid point and component in parallel
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// Process all components at this grid point
		for (int n = 0; n < ncomp; ++n) {
			const auto val = arr[bx](i, j, k, n);

			if (std::abs(val) < tiny) {
				arr[bx](i, j, k, n) = 0.0;
				continue;
			}

			volatile amrex::Real c = factor * val;
			volatile amrex::Real a = c - val;
			arr[bx](i, j, k, n) = c - a;
		}
	});
}

} // namespace quokka::ParticleUtils

#endif // PARTICLE_UTILS_HPP_
