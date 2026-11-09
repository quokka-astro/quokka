#ifndef PARALLELFOR_JCHUNKED_HPP_
#define PARALLELFOR_JCHUNKED_HPP_
//==============================================================================
// Copyright 2026 Quokka authors
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ParallelForJChunked.hpp
/// \brief ParallelFor over a MultiFab with a cache-blocked (j-chunked) thread-to-cell order on GPUs.
///
/// AMReX maps consecutive threads to (i, j, k) with i fastest, then j, then k. A kernel that reads a (k+1) or
/// (j+1) neighbour then re-reads data that was first touched one k-plane (or one j-row) earlier. For a 256^2 plane
/// and ~30 component arrays, a k-plane is ~15 MB, larger than the 8 MB L2 of an MI250X GCD, so every k-neighbour
/// read goes back to HBM. Here consecutive threads cover i fastest, then j within a chunk of `jchunk` rows, then k,
/// then the next chunk, so a k-plane of one chunk stays in L2 until the next plane reuses it. Each cell is still
/// visited by exactly one thread, so results are identical to amrex::ParallelFor.

#include <climits>

#include "AMReX_Box.H"
#include "AMReX_FabArray.H"
#include "AMReX_GpuLaunch.H"
#include "AMReX_IntVect.H"

namespace quokka
{

/// Calls f(box_no, i, j, k) for every cell of every local box of `mf` grown by `ng`, where box_no indexes
/// mf.arrays() / mf.const_arrays(). All boxes are launched on the current stream (no extra synchronisation).
/// With jchunk <= 0, or on CPU builds, or in 1D/2D, this is amrex::ParallelFor(mf, ng, f).
template <typename MF, typename F> void ParallelForJChunked(MF const &mf, amrex::IntVect const &ng, int jchunk, F const &f)
{
#if defined(AMREX_USE_GPU) && (AMREX_SPACEDIM == 3)
	if (jchunk > 0 && amrex::Gpu::inLaunchRegion()) {
		for (int li = 0; li < mf.local_size(); ++li) {
			const int box_no = li;
			const amrex::Box bx = amrex::grow(mf.box(mf.IndexArray()[li]), ng);
			const auto lo = amrex::lbound(bx);
			const auto len = amrex::length(bx);
			const int jc = std::min(jchunk, len.y);
			const int nchunk = (len.y + jc - 1) / jc;
			const long long ntot = static_cast<long long>(len.x) * jc * len.z * nchunk;
			AMREX_ALWAYS_ASSERT(ntot < static_cast<long long>(INT_MAX));
			const auto nx = static_cast<unsigned int>(len.x);
			const auto njc = static_cast<unsigned int>(jc);
			const auto nz = static_cast<unsigned int>(len.z);
			amrex::ParallelFor(static_cast<int>(ntot), [=] AMREX_GPU_DEVICE(int t) noexcept {
				auto r = static_cast<unsigned int>(t);
				const unsigned int i = r % nx;
				r /= nx;
				const unsigned int jl = r % njc;
				r /= njc;
				const unsigned int k = r % nz;
				const unsigned int c = r / nz;
				const int j = static_cast<int>(c * njc + jl);
				if (j < len.y) {
					f(box_no, lo.x + static_cast<int>(i), lo.y + j, lo.z + static_cast<int>(k));
				}
			});
		}
		return;
	}
#endif
	amrex::ignore_unused(jchunk);
	amrex::ParallelFor(mf, ng, f);
}

/// Thread-order settings for the hydro kernels (0 = AMReX default order), set from the hydro.jchunk_* parameters.
/// Defaults from the Sedov 256^3 MI250X study (PLM, one 256^3 box): the update, chi and z-face kernels read k+1 (or
/// k+-3) neighbours and gain 2.1, 0.7 and 3.5 ms/step with 64-row chunks; the x- and y-face kernels are slower when
/// chunked, so they keep the default order. Boxes with at most 64 cells in j are unaffected.
struct HydroLaunchOrder {
	static inline int jchunkUpdate = 64;	      // UpdateStageFused
	static inline int jchunkFace[3] = {0, 0, 64}; // face (reconstruction + Riemann) kernels, per direction
	static inline int jchunkChi = 64;	      // ComputeFlatteningChiCell
};

} // namespace quokka

#endif // PARALLELFOR_JCHUNKED_HPP_
