#ifndef PARALLEL_FOR_HPP_
#define PARALLEL_FOR_HPP_

#include <type_traits>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_Dim3.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Geometry.H"

namespace quokka {

template <typename L>
void ParallelFor(const int nwidth, amrex::Box const &box, L &&func) noexcept {
  AMREX_ALWAYS_ASSERT(nwidth >= 0);
  const amrex::Dim3 boxlo = amrex::lbound(box);
  const amrex::Dim3 boxhi = amrex::ubound(box);
  const amrex::Dim3 ilo{boxlo.x + nwidth, boxlo.y + nwidth, boxlo.z + nwidth};
  const amrex::Dim3 ihi{boxhi.x - nwidth, boxhi.y - nwidth, boxhi.z - nwidth};

  if (nwidth > 0) {
    amrex::ParallelFor(box, [=](int ix, int iy, int iz) noexcept {
      // if (i,j,k) are in the outer region, call the function
      if (((ix < ilo.x) || (ix > ihi.x)) || ((iy < ilo.y) || (iy > ihi.y)) ||
          ((iz < ilo.z) || (iz > ihi.z))) {
        func(ix, iy, iz);
      }
    });
  } else {
    amrex::ParallelFor(box, std::forward<L>(func));
  }
}

template <typename T, typename L>
void ParallelFor(const int nwidth, amrex::Box const &box, T ncomp,
                 L &&func) noexcept {
  AMREX_ALWAYS_ASSERT(nwidth >= 0);
  const amrex::Dim3 boxlo = amrex::lbound(box);
  const amrex::Dim3 boxhi = amrex::ubound(box);
  const amrex::Dim3 ilo{boxlo.x + nwidth, boxlo.y + nwidth, boxlo.z + nwidth};
  const amrex::Dim3 ihi{boxhi.x - nwidth, boxhi.y - nwidth, boxhi.z - nwidth};

  if (nwidth > 0) {
    amrex::ParallelFor(box, ncomp, [=](int ix, int iy, int iz, int n) noexcept {
      // if (i,j,k) are in the outer region, call the function
      if (((ix < ilo.x) || (ix > ihi.x)) || ((iy < ilo.y) || (iy > ihi.y)) ||
          ((iz < ilo.z) || (iz > ihi.z))) {
        func(ix, iy, iz, n);
      }
    });
  } else {
    amrex::ParallelFor(box, ncomp, std::forward<L>(func));
  }
}

} // namespace quokka

#endif // PARALLEL_FOR_HPP_