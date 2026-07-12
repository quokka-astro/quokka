// ABOUTME: Provides a free function template for computing volume-weighted integrals over AMR levels.
// ABOUTME: Passes both cell-centered and face-centered state arrays to the user-supplied integrand lambda.
#ifndef QUOKKA_VOLUME_INTEGRAL_HPP
#define QUOKKA_VOLUME_INTEGRAL_HPP

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include <array>

#include "physics_info.hpp"

namespace quokka
{

/// Compute a volume-weighted integral of a user-supplied function over all AMR levels.
///
/// The integrand lambda `user_f` is called as:
///   user_f(i, j, k, state_cc, state_fc)
/// where `state_cc` is the cell-centered Array4 and `state_fc` is a std::array of
/// AMREX_SPACEDIM face-centered Array4s. For non-MHD problems the face-centered
/// arrays are null (default-constructed) and should not be accessed.
///
/// @tparam problem_t  The problem type (used to determine whether face-centered data exists)
/// @tparam F          The callable type of the integrand lambda
/// @param finest_level  The finest AMR level index
/// @param state_cc      Cell-centered state MultiFabs (one per level)
/// @param state_fc      Face-centered state MultiFabs (one Array<AMREX_SPACEDIM> per level)
/// @param geom          Geometry objects (one per level)
/// @param ref_ratio     Refinement ratios between levels
/// @param user_f        The integrand lambda
/// @return The volume-weighted sum of user_f over all cells and levels
template <typename problem_t, typename F>
auto computeVolumeIntegral(int finest_level, amrex::Vector<amrex::MultiFab> const &state_cc,
			   amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &state_fc, amrex::Vector<amrex::Geometry> const &geom,
			   amrex::Vector<amrex::IntVect> const &ref_ratio, F const &user_f) -> amrex::Real
{
	const BL_PROFILE("quokka::computeVolumeIntegral()"); // NOLINT(misc-const-correctness)

	// allocate temporary multifabs
	amrex::Vector<amrex::MultiFab> q;
	q.resize(finest_level + 1);
	for (int lev = 0; lev <= finest_level; ++lev) {
		q[lev].define(state_cc[lev].boxArray(), state_cc[lev].DistributionMap(), 1, 0);
	}

	// evaluate user_f on all levels
	// (note: it is not necessary to average down)
	for (int lev = 0; lev <= finest_level; ++lev) {
		auto const &state = state_cc[lev].const_arrays();
		auto const &result = q[lev].arrays();
		if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
			auto const &fc = state_fc[lev];
			auto const &fc_x = fc[0].const_arrays();
			auto const &fc_y = fc[1].const_arrays();
			auto const &fc_z = fc[2].const_arrays();
			amrex::ParallelFor(q[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
				std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const state_fc_arr{AMREX_D_DECL(fc_x[bx], fc_y[bx], fc_z[bx])};
				result[bx](i, j, k) = user_f(i, j, k, state[bx], state_fc_arr);
			});
		} else {
			amrex::ParallelFor(q[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
				std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const state_fc_arr{};
				result[bx](i, j, k) = user_f(i, j, k, state[bx], state_fc_arr);
			});
		}
	}
	amrex::Gpu::streamSynchronize();

	// call amrex::volumeWeightedSum
	const amrex::Real result = amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(q), 0, geom, ref_ratio);
	return result;
}

} // namespace quokka

#endif // QUOKKA_VOLUME_INTEGRAL_HPP
