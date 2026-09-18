#ifndef ARRAY_UTILITIES_H
#define ARRAY_UTILITIES_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

namespace ArrayUtil
{
    using namespace amrex::literals;

    template <int XLO, int XHI>
    struct MathArray1D
    {
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        void zero()
        {
            for (int i = 0; i < (XHI-XLO+1); ++i) {
                arr[i] = 0.0_rt;
            }
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        const amrex::Real& operator() (int i) const noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI);
            return arr[i-XLO];
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real& operator() (int i) noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI);
            return arr[i-XLO];
        }

        amrex::Real arr[(XHI-XLO+1)];
    };

    template <typename T, int XLO, int XHI, int YLO, int YHI>
    struct MathArray2D
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void zero()
        {
            AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
            for (int i = 0; i < (YHI-YLO+1)*(XHI-XLO+1); ++i) {
                arr[i] = 0.0_rt;
            }
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void mul (const amrex::Real x) noexcept {
            for (int i = 0; i < (YHI-YLO+1)*(XHI-XLO+1); ++i) {
                arr[i] *= x;
            }
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void set (const int i, const int j, const amrex::Real x) noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI && j >= YLO && j <= YHI);
            arr[i+j*(XHI-XLO+1)-(YLO*(XHI-XLO+1)+XLO)] = x;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void add (const int i, const int j, const amrex::Real x) noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI && j >= YLO && j <= YHI);
            arr[i+j*(XHI-XLO+1)-(YLO*(XHI-XLO+1)+XLO)] += x;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void mul (const int i, const int j, const amrex::Real x) noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI && j >= YLO && j <= YHI);
            arr[i+j*(XHI-XLO+1)-(YLO*(XHI-XLO+1)+XLO)] *= x;
        }

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        T get (const int i, const int j) const noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI && j >= YLO && j <= YHI);
            return arr[i+j*(XHI-XLO+1)-(YLO*(XHI-XLO+1)+XLO)];
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void add_identity () noexcept {
            for (int i = XLO; i <= XHI; ++i) {
                arr[i+i*(XHI-XLO+1)-(YLO*(XHI-XLO+1)+XLO)] += 1.0_rt;
            }
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        const T& operator() (int i, int j) const noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI && j >= YLO && j <= YHI);
            return arr[i+j*(XHI-XLO+1)-(YLO*(XHI-XLO+1)+XLO)];
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        T& operator() (int i, int j) noexcept {
            AMREX_ASSERT(i >= XLO && i <= XHI && j >= YLO && j <= YHI);
            return arr[i+j*(XHI-XLO+1)-(YLO*(XHI-XLO+1)+XLO)];
        }

        T arr[(XHI-XLO+1)*(YHI-YLO+1)];
    };

    namespace Math
    {
        template <int XLO, int XHI>
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        amrex::Real sum(amrex::Array1D<amrex::Real, XLO, XHI> const& xarray,
                 const int xlo_s = XLO, const int xhi_s = XHI)
        {
            amrex::Real xsum = 0.0_rt;

            for (int i = xlo_s; i <= xhi_s; ++i) {
                xsum += xarray(i);
            }

            return xsum;
        }
    }
}

#endif
