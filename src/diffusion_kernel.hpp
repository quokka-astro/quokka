#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>
#include "ArrayView.hpp"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_phi (int i, int j, int k,
               amrex::Array4<amrex::Real> const& phi,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = Real(0.);
    Real z = Real(0.);

	if constexpr (amrex::SpaceDim > 1) {
        y = prob_lo[1] + (j+Real(0.5)) * dx[1];
    }

    if constexpr (amrex::SpaceDim > 2) {
        z = prob_lo[2] + (k+Real(0.5)) * dx[2];
    }

    Real r2 = ((x-Real(0.25))*(x-Real(0.25))+(y-Real(0.25))*(y-Real(0.25))+(z-Real(0.25))*(z-Real(0.25)))/Real(0.01);
    phi(i,j,k) = Real(1.) + std::exp(-r2);
}

template <int N>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxx_in,
                     amrex::Array4<amrex::Real const> const& phi_in, amrex::Real dxinv)
{
    Array4View<amrex::Real const, N> phi(phi_in);
    Array4View<amrex::Real, N> fluxx(fluxx_in);

    fluxx(i,j,k) = (phi(i,j,k)-phi(i-1,j,k)) * dxinv;
}

#if 0
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxy,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dyinv)
{
    fluxy(i,j,k) = (phi(i,j,k)-phi(i,j-1,k)) * dyinv;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxz,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dzinv)
{
    fluxz(i,j,k) = (phi(i,j,k)-phi(i,j,k-1)) * dzinv;
}
#endif


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
                 std::vector<amrex::Array4<amrex::Real const>> const& fluxes,
                 amrex::Real dt,
                 std::vector<amrex::Real> dinv)
{
    phinew(i,j,k) = phiold(i,j,k)
        + dt * dinv[0] * (fluxes[0](i+1,j  ,k  ) - fluxes[0](i,j,k));

	if constexpr (amrex::SpaceDim > 1) {
        phinew(i,j,k) =+ dt * dinv[1] * (fluxes[1](i  ,j+1,k  ) - fluxes[1](i,j,k));
    }

	if constexpr (amrex::SpaceDim > 2) {
        phinew(i,j,k) =+ dt * dinv[2] * (fluxes[2](i  ,j  ,k+1) - fluxes[2](i,j,k));
    }
}

#endif
