#ifndef PARALLEL_FOR_HPP_
#define PARALLEL_FOR_HPP_

#include <type_traits>

#include "AMReX.H"
#include "AMReX_Dim3.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Geometry.H"

namespace quokka {

using namespace amrex;

#ifdef AMREX_USE_GPU

template <typename L>
auto ParallelFor(amrex::Gpu::KernelInfo const & /*unused*/, const int nwidth,
                 amrex::Box const &box, L &&f) noexcept
    -> std::enable_if_t<amrex::MaybeDeviceRunnable<L>::value> {
  if (amrex::isEmpty(box)) {
    return;
  }
  if (nwidth > 0) {
    const amrex::Dim3 lo = amrex::lbound(box);
    const amrex::Dim3 hi = amrex::ubound(box);
    const amrex::Dim3 ilo{lo.x + nwidth, lo.y + nwidth, lo.z + nwidth};
    const amrex::Dim3 ihi{hi.x - nwidth, hi.y - nwidth, hi.z - nwidth};

    int ncells = box.numPts();
    const auto len = amrex::length(box);
    const auto lenxy = len.x * len.y;
    const auto lenx = len.x;
    const auto ec = amrex::Gpu::ExecutionConfig(ncells);

    AMREX_LAUNCH_KERNEL(
        ec.numBlocks, ec.numThreads, 0, amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE() noexcept {
          for (int icell = blockDim.x * blockIdx.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;
               icell < ncells; icell += stride) {
            int k = icell / lenxy;
            int j = (icell - k * lenxy) / lenx;
            int i = (icell - k * lenxy) - j * lenx;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            // if (i,j,k) are in the outer region, call the function
            if (((i < ilo.x) || (i > ihi.x)) || ((j < ilo.y) || (j > ihi.y)) ||
                ((k < ilo.z) || (k > ihi.z))) {
              detail::call_f(f, i, j, k, (ncells - icell + (int)threadIdx.x));
            }
          }
        });
    AMREX_GPU_ERROR_CHECK();
  } else {
    amrex::ParallelFor(box, std::forward<L>(f));
  }
}

template <typename T, typename L,
          typename M = std::enable_if_t<std::is_integral<T>::value>>
auto ParallelFor(amrex::Gpu::KernelInfo const & /*unused*/, const int nwidth,
                 amrex::Box const &box, T ncomp, L &&f) noexcept
    -> std::enable_if_t<amrex::MaybeDeviceRunnable<L>::value> {
  if (amrex::isEmpty(box)) {
    return;
  }
  if (nwidth > 0) {
    const amrex::Dim3 lo = amrex::lbound(box);
    const amrex::Dim3 hi = amrex::ubound(box);
    const amrex::Dim3 ilo{lo.x + nwidth, lo.y + nwidth, lo.z + nwidth};
    const amrex::Dim3 ihi{hi.x - nwidth, hi.y - nwidth, hi.z - nwidth};

    int ncells = box.numPts();
    const auto len = amrex::length(box);
    const auto lenxy = len.x * len.y;
    const auto lenx = len.x;
    const auto ec = amrex::Gpu::ExecutionConfig(ncells);

    AMREX_LAUNCH_KERNEL(
        ec.numBlocks, ec.numThreads, 0, amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE() noexcept {
          for (int icell = blockDim.x * blockIdx.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;
               icell < ncells; icell += stride) {
            int k = icell / lenxy;
            int j = (icell - k * lenxy) / lenx;
            int i = (icell - k * lenxy) - j * lenx;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            // if (i,j,k) are in the outer region, call the function
            if (((i < ilo.x) || (i > ihi.x)) || ((j < ilo.y) || (j > ihi.y)) ||
                ((k < ilo.z) || (k > ihi.z))) {
              detail::call_f(f, i, j, k, ncomp,
                             (ncells - icell + (int)threadIdx.x));
            }
          }
        });
    AMREX_GPU_ERROR_CHECK();
  } else {
    amrex::ParallelFor(box, ncomp, std::forward<L>(f));
  }
}

template <typename L>
void ParallelFor(const int nwidth, amrex::Box const &box, L &&f) noexcept {
  quokka::ParallelFor(amrex::Gpu::KernelInfo{}, nwidth, box,
                      std::forward<L>(f));
}

template <typename T, typename L,
          typename M = std::enable_if_t<std::is_integral<T>::value>>
void ParallelFor(const int nwidth, amrex::Box const &box, T ncomp,
                 L &&f) noexcept {
  quokka::ParallelFor(amrex::Gpu::KernelInfo{}, nwidth, box, ncomp,
                      std::forward<L>(f));
}

#else

template <typename L>
void ParallelFor(const int nwidth, amrex::Box const &box, L &&f) noexcept {
  if (nwidth > 0) {
    const amrex::Dim3 lo = amrex::lbound(box);
    const amrex::Dim3 hi = amrex::ubound(box);
    const amrex::Dim3 ilo{lo.x + nwidth, lo.y + nwidth, lo.z + nwidth};
    const amrex::Dim3 ihi{hi.x - nwidth, hi.y - nwidth, hi.z - nwidth};

    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          // if (i,j,k) are in the outer region, call the function
          if (((i < ilo.x) || (i > ihi.x)) || ((j < ilo.y) || (j > ihi.y)) ||
              ((k < ilo.z) || (k > ihi.z))) {
            amrex::detail::call_f(f, i, j, k);
          }
        }
      }
    }
  } else {
    amrex::ParallelFor(box, std::forward<L>(f));
  }
}

template <typename T, typename L,
          typename M = std::enable_if_t<std::is_integral<T>::value>>
void ParallelFor(const int nwidth, amrex::Box const &box, T ncomp,
                 L &&f) noexcept {
  if (nwidth > 0) {
    const amrex::Dim3 lo = amrex::lbound(box);
    const amrex::Dim3 hi = amrex::ubound(box);
    const amrex::Dim3 ilo{lo.x + nwidth, lo.y + nwidth, lo.z + nwidth};
    const amrex::Dim3 ihi{hi.x - nwidth, hi.y - nwidth, hi.z - nwidth};

    for (T n = 0; n < ncomp; ++n) {
      for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
          AMREX_PRAGMA_SIMD
          for (int i = lo.x; i <= hi.x; ++i) {
            // if (i,j,k) are in the outer region, call the function
            if (((i < ilo.x) || (i > ihi.x)) || ((j < ilo.y) || (j > ihi.y)) ||
                ((k < ilo.z) || (k > ihi.z))) {
              amrex::detail::call_f(f, i, j, k, n);
            }
          }
        }
      }
    }
  } else {
    amrex::ParallelFor(box, ncomp, std::forward<L>(f));
  }
}

#endif

} // namespace quokka

#endif // PARALLEL_FOR_HPP_