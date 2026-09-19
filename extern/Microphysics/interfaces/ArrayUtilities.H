#ifndef ARRAY_UTILITIES_H
#define ARRAY_UTILITIES_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

namespace ArrayUtil
{
    using namespace amrex::literals;

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

}

#endif
