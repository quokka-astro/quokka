#ifndef PARTICLE_UTILS_HPP_
#define PARTICLE_UTILS_HPP_

#include "AMReX_MultiFab.H"
#include "AMReX_FabArray.H"
#include "fundamental_constants.H"
#include "math/FastMath.hpp"
#include <cmath>
#include <cstdint>

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
	// Deterministic roundoff to eliminate floating point non-associativity effects
	// We round off only the least significant bits that are affected by summation order
	// This preserves the main precision while ensuring reproducibility
	
	// Get array accessor for all patches at once
	auto arr = mf.arrays();
	const int ncomp = mf.nComp();

	constexpr Real tiny = 1e10 * std::numeric_limits<amrex::Real>::min();

	// Apply roundoff algorithm to every grid point and component in parallel
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// Process all components at this grid point
		for (int n = 0; n < ncomp; ++n) {
			amrex::Real &value = arr[bx](i, j, k, n);
			
			// Only apply roundoff to non-zero values
			if (std::abs(value) > tiny) {
				// Round to 8 significant digits to eliminate floating-point non-associativity
				// Use FastMath for optimal GPU performance
				
				const amrex::Real abs_val = std::abs(value);
				const int exponent = static_cast<int>(std::floor(FastMath::log10(abs_val)));
				
				// Calculate precision for 8 significant digits using FastMath
				// For value 1.23456789e-44, we want precision = 1e-51 (8 digits from the leading digit)
				const amrex::Real precision = FastMath::pow10(static_cast<amrex::Real>(exponent - 7));  // 8 significant digits
				
				// Round to the calculated precision
				value = std::round(value / precision) * precision;
			} else {
				value = 0.0;
			}
		}
	});
}

} // namespace quokka::ParticleUtils

#endif // PARTICLE_UTILS_HPP_
