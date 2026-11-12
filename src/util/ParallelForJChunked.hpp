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

// NVIDIA-only occupancy control. The fused hydro kernels need ~130-200 registers per thread; with AMReX's
// 256-thread blocks and -maxrregcount=255, nvcc gives the face kernels ~196 registers, so only one block (8 warps)
// fits on an SM. A positive QUOKKA_CUDA_*_MINB launches that kernel through quokka::detail::launchBounded with
// __launch_bounds__(QUOKKA_CUDA_LB_THREADS, MINB), letting nvcc cap registers at 65536 / (threads * MINB). The
// thread-to-cell mapping is unchanged, so results are bitwise identical. 0 = launch through amrex::ParallelFor.
// These have no effect on HIP or CPU builds.
#ifndef QUOKKA_CUDA_LB_THREADS
#define QUOKKA_CUDA_LB_THREADS 256
#endif
#ifndef QUOKKA_CUDA_FACE_MINB
#define QUOKKA_CUDA_FACE_MINB 0
#endif
#ifndef QUOKKA_CUDA_UPDATE_MINB
#define QUOKKA_CUDA_UPDATE_MINB 0
#endif
#ifndef QUOKKA_CUDA_CHI_MINB
#define QUOKKA_CUDA_CHI_MINB 0
#endif

namespace quokka
{
#if defined(AMREX_USE_CUDA)
namespace detail
{
template <int MT, int MINB, typename L> __global__ void __launch_bounds__(MT, MINB) launchBounded(const int n, L f)
{
	const int t = static_cast<int>(blockIdx.x) * MT + static_cast<int>(threadIdx.x);
	if (t < n) {
		f(t);
	}
}
} // namespace detail
#endif

/// Launches f(box_no, i, j, k) for every cell of the box `bx` on the current stream, with the j-chunked thread order
/// (jchunk <= 0: the AMReX default order). MINB > 0 (CUDA only) launches through detail::launchBounded.
template <int MINB = 0, typename F> void launchBoxJChunked(const int box_no, amrex::Box const &bx, int jchunk, F const &f)
{
#if defined(AMREX_USE_CUDA)
	constexpr bool bounded = (MINB > 0);
#else
	constexpr bool bounded = false;
#endif
	const auto lo = amrex::lbound(bx);
	const auto len = amrex::length(bx);
	const int jc = (jchunk > 0) ? std::min(jchunk, len.y) : len.y;
	const int nchunk = (len.y + jc - 1) / jc;
	const long long ntot = static_cast<long long>(len.x) * jc * len.z * nchunk;
	AMREX_ALWAYS_ASSERT(ntot < static_cast<long long>(INT_MAX));
	const auto nx = static_cast<unsigned int>(len.x);
	const auto njc = static_cast<unsigned int>(jc);
	const auto nz = static_cast<unsigned int>(len.z);
	auto body = [=] AMREX_GPU_DEVICE(int t) noexcept {
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
	};
#if defined(AMREX_USE_CUDA)
	if constexpr (bounded) {
		constexpr int MT = QUOKKA_CUDA_LB_THREADS;
		const int n = static_cast<int>(ntot);
		const int nblocks = (n + MT - 1) / MT;
		detail::launchBounded<MT, (MINB > 0 ? MINB : 1)><<<nblocks, MT, 0, amrex::Gpu::gpuStream()>>>(n, body);
		AMREX_GPU_ERROR_CHECK();
		return;
	}
#endif
	amrex::ParallelFor(static_cast<int>(ntot), body);
}

/// Calls f(box_no, i, j, k) for every cell of every local box of `mf` grown by `ng`, where box_no indexes
/// mf.arrays() / mf.const_arrays(). All boxes are launched on the current stream (no extra synchronisation).
/// With jchunk <= 0, or on CPU builds, or in 1D/2D, this is amrex::ParallelFor(mf, ng, f).
/// MINB > 0 (CUDA builds only) launches through detail::launchBounded with __launch_bounds__(QUOKKA_CUDA_LB_THREADS, MINB);
/// with jchunk <= 0 the thread order is then the AMReX default order (i, then j, then k), launched per box.
template <int MINB = 0, typename MF, typename F> void ParallelForJChunked(MF const &mf, amrex::IntVect const &ng, int jchunk, F const &f)
{
#if defined(AMREX_USE_GPU) && (AMREX_SPACEDIM == 3)
#if defined(AMREX_USE_CUDA)
	constexpr bool bounded = (MINB > 0);
#else
	constexpr bool bounded = false;
#endif
	if ((jchunk > 0 || bounded) && amrex::Gpu::inLaunchRegion()) {
		for (int li = 0; li < mf.local_size(); ++li) {
			launchBoxJChunked<MINB>(li, amrex::grow(mf.box(mf.IndexArray()[li]), ng), jchunk, f);
		}
		return;
	}
#endif
	amrex::ignore_unused(jchunk);
	amrex::ParallelFor(mf, ng, f);
}

#if AMREX_SPACEDIM == 3
/// Column launch: calls f(box_no, i, j, k, nrun) once per run of up to `ncol` consecutive cells along direction `dir`
/// of every local box of `mf` grown by `ng`; (i, j, k) is the first cell of the run and the run covers
/// (i, j, k) + m e_dir, m = 0 .. nrun-1. The runs tile the box exactly once. Thread order over the runs is the
/// j-chunked order above. Lets a kernel carry per-cell work (e.g. reconstructed edge states) from one cell of the run
/// to the next instead of recomputing it in a neighbouring thread. ncol <= 1 gives nrun = 1 for every cell.
template <int MINB = 0, typename MF, typename F> void ParallelForColumns(MF const &mf, amrex::IntVect const &ng, int dir, int ncol, int jchunk, F const &f)
{
	ncol = std::max(ncol, 1);
	for (int li = 0; li < mf.local_size(); ++li) {
		const amrex::Box bx = amrex::grow(mf.box(mf.IndexArray()[li]), ng);
		const int lo_d = bx.smallEnd(dir);
		const int hi_d = bx.bigEnd(dir);
		amrex::Box rbx = bx; // one point per run along dir
		rbx.setBig(dir, lo_d + (bx.length(dir) + ncol - 1) / ncol - 1);
		auto runBody = [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
			amrex::IntVect iv(AMREX_D_DECL(i, j, k));
			iv[dir] = lo_d + (iv[dir] - lo_d) * ncol;
			const int nrun = std::min(ncol, hi_d - iv[dir] + 1);
			f(box_no, iv[0], iv[1], iv[2], nrun);
		};
#if defined(AMREX_USE_GPU) && (AMREX_SPACEDIM == 3)
		if (amrex::Gpu::inLaunchRegion()) {
			launchBoxJChunked<MINB>(li, rbx, jchunk, runBody);
			continue;
		}
#endif
		amrex::ParallelFor(rbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { runBody(li, i, j, k); });
	}
}
#endif

/// Thread-order settings for the hydro kernels (0 = AMReX default order), set from the hydro.jchunk_* parameters.
/// Defaults from the Sedov 256^3 MI250X study (PLM, one 256^3 box): the update, chi and z-face kernels read k+1 (or
/// k+-3) neighbours and gain 2.1, 0.7 and 3.5 ms/step with 64-row chunks; the x- and y-face kernels are slower when
/// chunked, so they keep the default order. Boxes with at most 64 cells in j are unaffected.
struct HydroLaunchOrder {
	static inline int jchunkUpdate = 64;	      // UpdateStageFused
	static inline int jchunkFace[3] = {0, 0, 64}; // face (reconstruction + Riemann) kernels, per direction
	static inline int jchunkChi = 64;	      // ComputeFlatteningChiCell
	// ComputeFlatteningChiCell: 1 = two passes (directional coefficients stored once per cell, then the 7-point min),
	// 0 = one pass that re-evaluates 9 directional coefficients per cell. Bitwise identical results.
	static inline int chiTwoPass = 1;
	// fused face kernels: faces per thread along the face normal, per direction (1 = one face per thread). With > 1 a
	// thread reconstructs each cell once and carries its edge state to the next face (bitwise identical results).
	static inline int faceColumn[3] = {1, 1, 1};
};

} // namespace quokka

#endif // PARALLELFOR_JCHUNKED_HPP_
