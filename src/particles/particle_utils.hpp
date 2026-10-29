#ifndef PARTICLE_UTILS_HPP_
#define PARTICLE_UTILS_HPP_

#include "AMReX_Box.H"
#include "AMReX_Gpu.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
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

/// @brief Compute the Jeans density with optional MHD correction.
///
/// Standard Jeans density (pure hydro):
///   rho_J = J^2 * pi * cs^2 / (G * dx^2)
///
/// MHD-aware Jeans density:
///   rho_J = J^2 * pi * cs_eff^2 / (G * dx^2)
///   where cs_eff^2 = cs^2 * (1 + 0.74 / beta)
///
/// Plasma beta:
///   beta = P_thermal / P_magnetic
///   P_thermal = rho * cs^2 / gamma (ideal gas)
///   P_magnetic = B^2 / (8*pi) (CGS units)
///
/// The factor 0.74 accounts for magnetic pressure support against gravitational
/// collapse in the Jeans instability criterion (Mouschovias & Spitzer, 1976, ApJ,
/// 210, 326; see also Myers et al, 2013, ApJ, 766, 97)

///
/// @param cs_cell Sound speed in the cell (cm/s)
/// @param dx Cell size (cm)
/// @param plasma_beta Ratio of thermal pressure to magnetic pressure (P_thermal / P_magnetic).
///                    Set to infinity (or a very large value) for non-MHD cases.
/// @return Jeans density (g/cm^3)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto computeJeansDensity(double cs_cell, double dx, double plasma_beta = std::numeric_limits<double>::max())
    -> double
{
	// cs_eff^2 = cs^2 * (1 + 0.74/beta)
	// For beta -> infinity (non-MHD case), cs_eff^2 -> cs^2
	const double cs_eff_sq = cs_cell * cs_cell * (1.0 + 0.74 / plasma_beta);
	// rho_J = J^2 * pi * cs_eff^2 / (G * dx^2)
	return jeansNo * jeansNo * M_PI * cs_eff_sq / (C::Gconst * (dx * dx));
}

/// @brief Compute plasma beta from thermal pressure and magnetic energy.
///
/// Plasma beta:
///   beta = P_thermal / P_magnetic
///
/// In QUOKKA, we are carrying our magnetic field as B' = B / sqrt(4 pi), so
///   magnetic_energy = B^2 / 2
///   P_magnetic = B^2 / 2
///   Therefore: P_magnetic = magnetic_energy
///
/// @param pressure_thermal Thermal pressure (dyn/cm^2)
/// @param magnetic_energy Magnetic energy density = B^2 / 2 (erg/cm^3)
/// @return Plasma beta. Returns infinity if magnetic energy is zero.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto computePlasmaBeta(double pressure_thermal, double magnetic_energy) -> double
{
	if (magnetic_energy <= 0.0) {
		return std::numeric_limits<double>::max();
	}
	return pressure_thermal / magnetic_energy;
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

/// Result of surveying the particles of one feedback channel on one level.
struct FeedbackSurvey {
	int active = 0; ///< number of particles (summed over all MPI ranks) that may deposit in this channel
	int misfit = 0; ///< number of active particles (summed over all ranks) whose stencil does not fit in the shrunk buffer
};

/// Count (over all MPI ranks) the particles of `container` on level `lev` for which `pred(p)` is true, and how many of those have a deposition stencil
/// (cell containing the particle +- `reach`) that would not fit inside the grid's valid box grown by `ghost`. This is a collective call (one reduction), so
/// all ranks get the same answer and may use it for collective decisions. `pred` must be a device-callable functor (not a lambda).
template <typename ContainerType, typename ActivePred>
auto surveyFeedbackSources(ContainerType *container, amrex::MultiFab const &state, int lev, int reach, int ghost, ActivePred pred) -> FeedbackSurvey
{
	amrex::Gpu::Buffer<int> counts({0, 0});
	int *p_counts = counts.data();
	const auto &geom = container->Geom(lev);
	const auto plo = geom.ProbLoArray();
	const auto dxi = geom.InvCellSizeArray();

	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		const amrex::Long np = pti.numParticles();
		const amrex::Box fab_box = amrex::grow(state.boxArray()[pti.index()], ghost);
		const amrex::Dim3 lo = amrex::lbound(fab_box);
		const amrex::Dim3 hi = amrex::ubound(fab_box);

		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
			auto const &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			if (!pred(p)) {
				return;
			}
			amrex::Gpu::Atomic::AddNoRet(&p_counts[0], 1);
			const int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
			const int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
			const int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));
			if (ix - reach < lo.x || ix + reach > hi.x || iy - reach < lo.y || iy + reach > hi.y || iz - reach < lo.z || iz + reach > hi.z) {
				amrex::Gpu::Atomic::AddNoRet(&p_counts[1], 1);
			}
		});
	}

	int *h_counts = counts.copyToHost();
	int totals[2] = {h_counts[0], h_counts[1]}; // NOLINT
	amrex::ParallelDescriptor::ReduceIntSum(totals, 2);
	return FeedbackSurvey{totals[0], totals[1]};
}

// Zero every real component from first_comp onwards for all particles in the container.
//
// InitFromAsciiFile() only writes the nreal_extra components present in the file and leaves the
// rest of each particle's real data indeterminate. Accumulator components such as the sink-particle
// angular momentum are updated with += during accretion, so reading them before they are written
// yields whatever the underlying (host or device) allocation last held. Call this immediately after
// InitFromAsciiFile() with first_comp = nreal_extra.
template <typename ContainerType> void zeroRealComponentsFrom(ContainerType *container, const int first_comp)
{
	using PType = typename ContainerType::ParticleType;
	if (first_comp >= PType::NReal) {
		return;
	}

	for (int lev = 0; lev <= container->finestLevel(); ++lev) {
		for (auto &kv : container->GetParticles(lev)) {
			auto &particle_array = kv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			auto *pdata = particle_array().data();
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				for (int n = first_comp; n < PType::NReal; ++n) {
					p.rdata(n) = 0.0;
				}
			});
		}
	}
	amrex::Gpu::streamSynchronize();
}

} // namespace quokka::ParticleUtils

#endif // PARTICLE_UTILS_HPP_
