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

inline void roundoffMultiFab(amrex::MultiFab &mf)
{
	// Apply roundoff algorithm to reduce floating-point precision errors by removing
	// the least significant bits from IEEE 754 double precision numbers.
	//
	// IEEE 754 double precision format:
	// - 1 sign bit + 11 exponent bits + 52 mantissa bits = 64 total bits
	// - The mantissa has an implicit leading 1, giving 53 bits of precision
	//
	// This version uses the last component of mf as the count to compute digit_to_remove
	// based on the relative error formula: relative_error = (N - 1) * epsilon, where N
	// is the count and epsilon is machine epsilon. We convert this to binary digits and add redundancy.

	constexpr amrex::Real tiny = 1.0e10 * std::numeric_limits<amrex::Real>::min();
	const auto redundancy = static_cast<unsigned int>(reproducibility_roundoff_redundancy);
	const unsigned int base_digit_to_remove = redundancy + 2; // +2 to account for machine epsilon

	// Get array accessor for all patches at once
	auto const &arr = mf.arrays();
	const int ncomp = mf.nComp();
	const int count_comp = ncomp - 1; // Last component is the count

	// Apply roundoff algorithm to every grid point and component in parallel
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const auto count = arr[bx](i, j, k, count_comp); // integer represented as Real

		if (count > 1.5) {
			// Process all components at this grid point (except the count component)
			for (int n = 0; n < count_comp; ++n) {
				const auto val = arr[bx](i, j, k, n);

				if (std::abs(val) < tiny) {
					arr[bx](i, j, k, n) = 0.0;
					continue;
				}

				// Compute digit_to_remove based on count
				auto digit_to_remove = base_digit_to_remove;

				// Relative error estimate: (N - 1) * epsilon
				const amrex::Real scale_up = count - 1.0;

				// Convert to binary digits: log2(scale_up)
				if (scale_up > 0.0) {
					const amrex::Real extra_digits = std::max(0.0, FastMath::fastlg(scale_up));
					// TODO(cch): we can be less conservative and divide extra_digits by 2 to improve accuracy when scale_up is large.
					// Assuming roundoff errors are random walk, the scatter of the accumulative error is sqrt(N) * epsilon.

					digit_to_remove += static_cast<unsigned int>(extra_digits);

					// Clamp to reasonable bounds (1 to 52 bits)
					digit_to_remove = amrex::max(1U, amrex::min(52U, digit_to_remove));
				}

				const auto factor = static_cast<amrex::Real>((1ULL << digit_to_remove) + 1);

				// Use volatile to prevent compiler optimizations that would defeat the roundoff algorithm.
				// The volatile keyword ensures that:
				// 1. The compiler doesn't optimize away the intermediate calculations
				// 2. The operations are performed in the exact order specified
				// 3. The roundoff effect from the large factor multiplication is preserved
				//
				// Without volatile, an optimizing compiler might simplify (c - a) to just (factor * val - (factor * val - val)),
				// which could eliminate the intended precision truncation that occurs when adding the large factor.
				// The algorithm relies on the fact that adding a large number and then subtracting it back
				// will truncate the least significant bits due to IEEE 754 floating-point representation limits.
				volatile amrex::Real const c = factor * val;
				volatile amrex::Real const a = c - val;
				arr[bx](i, j, k, n) = c - a;
			}
		}
	});
}

} // namespace quokka::ParticleUtils

#endif // PARTICLE_UTILS_HPP_
