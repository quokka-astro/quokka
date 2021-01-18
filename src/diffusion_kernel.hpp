#ifndef DIFFUSION_KERNEL_H_
#define DIFFUSION_KERNEL_H_

#include <AMReX_FArrayBox.H>
#include "ArrayView.hpp"


template <int DIR> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux(int i_in, int j_in, int k_in,
                  amrex::Array4<amrex::Real> const &fluxx_in,
	              amrex::Array4<amrex::Real const> const &phi_in,
                  amrex::Real dxinv)
{
    // these boilerplate defintions could be replaced via a (complicated) function definition macro
	auto [i, j, k] = reorderMultiIndex<DIR>(i_in, j_in, k_in);
	Array4View<amrex::Real const, DIR> phi(phi_in);
	Array4View<amrex::Real, DIR> fluxx(fluxx_in);

    // compute the flux along direction DIR
	fluxx(i, j, k) = (phi(i, j, k) - phi(i - 1, j, k)) * dxinv;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_phi (int i, int j, int k,
               amrex::Array4<amrex::Real> const& phi,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM > 2)
    Real z = prob_lo[2] + (k+Real(0.5)) * dx[2];
#else
    Real z = Real(0.);
#endif
    Real r2 = ((x-Real(0.25))*(x-Real(0.25))+(y-Real(0.25))*(y-Real(0.25))+(z-Real(0.25))*(z-Real(0.25)))/Real(0.01);
    phi(i,j,k) = Real(1.) + std::exp(-r2);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                              amrex::Array4<amrex::Real const> const& fluxy,
                              amrex::Array4<amrex::Real const> const& fluxz),
                 amrex::Real dt,
                 AMREX_D_DECL(amrex::Real dxinv,
                              amrex::Real dyinv,
                              amrex::Real dzinv))
{
    phinew(i,j,k) = phiold(i,j,k)
        + dt * dxinv * (fluxx(i+1,j  ,k  ) - fluxx(i,j,k))
        + dt * dyinv * (fluxy(i  ,j+1,k  ) - fluxy(i,j,k))
#if (AMREX_SPACEDIM > 2)
        + dt * dzinv * (fluxz(i  ,j  ,k+1) - fluxz(i,j,k));
#else
        ;
#endif
}

#endif
