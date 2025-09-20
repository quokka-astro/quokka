#ifndef PARTICLE_UTILS_HPP_
#define PARTICLE_UTILS_HPP_

#include "AMReX_MultiFab.H"
#include "AMReX_FabArray.H"
#include "fundamental_constants.H"
#include <cmath>

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
	// Deterministic roundoff precision: round to ~1e-12 relative precision
	// This eliminates floating point non-associativity effects from parallel particle deposition
	constexpr amrex::Real roundoff_precision = 1e-12;
	constexpr amrex::Real inv_roundoff_precision = 1.0 / roundoff_precision;

	// Get array accessor for all patches at once
	auto arr = mf.arrays();
	const int ncomp = mf.nComp();

	// Apply roundoff algorithm to every grid point and component in parallel
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// Process all components at this grid point
		for (int n = 0; n < ncomp; ++n) {
			amrex::Real &value = arr[bx](i, j, k, n);
			
			// Apply deterministic rounding to eliminate small differences
			// from non-associative floating point operations
			if (std::abs(value) > roundoff_precision) {
				// Round to specified precision by scaling, rounding, and scaling back
				value = std::round(value * inv_roundoff_precision) * roundoff_precision;
			} else {
				// Set very small values to exactly zero to ensure reproducibility
				value = 0.0;
			}
		}
	});
}

} // namespace quokka::ParticleUtils

#endif // PARTICLE_UTILS_HPP_
