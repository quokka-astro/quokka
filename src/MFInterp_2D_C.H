#ifndef MF_INTERP_2D_C_H_
#define MF_INTERP_2D_C_H_

#include <AMReX_Geometry.H>

namespace amrex {

// limits the slopes of all components in a given cell the same way
//  *and* prevents new minima or maxima from appearing in the fine grid

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mf_quokka_interp_llslope (int i, int j, int, Array4<Real> const& slope,
                                      Array4<Real const> const& u, int scomp, int ncomp,
                                      Box const& domain, IntVect const& ratio, BCRec const* bc) noexcept
{
    Real sfx = Real(1.0);
    Real sfy = Real(1.0);

    for (int ns = 0; ns < ncomp; ++ns) {
        int nu = ns + scomp;

        // x-direction
        Real dcx = mf_compute_slopes_x(i, j, 0, u, nu, domain, bc[ns]);
        Real df = Real(2.0) * (u(i+1,j,0,nu) - u(i  ,j,0,nu));
        Real db = Real(2.0) * (u(i  ,j,0,nu) - u(i-1,j,0,nu));
        Real sx = (df*db >= Real(0.0)) ? amrex::min(amrex::Math::abs(df),amrex::Math::abs(db)) : Real(0.);
        sx = amrex::Math::copysign(Real(1.),dcx)*amrex::min(sx,amrex::Math::abs(dcx));
        slope(i,j,0,ns        ) = dcx;

        // y-direction
        Real dcy = mf_compute_slopes_y(i, j, 0, u, nu, domain, bc[ns]);
        df = Real(2.0) * (u(i,j+1,0,nu) - u(i,j  ,0,nu));
        db = Real(2.0) * (u(i,j  ,0,nu) - u(i,j-1,0,nu));
        Real sy = (df*db >= Real(0.0)) ? amrex::min(amrex::Math::abs(df),amrex::Math::abs(db)) : Real(0.);
        sy = amrex::Math::copysign(Real(1.),dcy)*amrex::min(sy,amrex::Math::abs(dcy));
        slope(i,j,0,ns+  ncomp) = dcy;

        // adjust limited slopes to prevent new min/max for this component
        Real alpha = Real(1.0);
        if (sx != Real(0.0) || sy != Real(0.0)) {
            Real dumax = amrex::Math::abs(sx) * Real(ratio[0]-1)/Real(2*ratio[0])
                +        amrex::Math::abs(sy) * Real(ratio[1]-1)/Real(2*ratio[1]);
            Real umax = u(i,j,0,nu);
            Real umin = u(i,j,0,nu);
            for (int joff = -1; joff <= 1; ++joff) {
            for (int ioff = -1; ioff <= 1; ++ioff) {
                umin = amrex::min(umin, u(i+ioff,j+joff,0,nu));
                umax = amrex::max(umax, u(i+ioff,j+joff,0,nu));
            }}
            if (dumax * alpha > (umax - u(i,j,0,nu))) {
                alpha = (umax - u(i,j,0,nu)) / dumax;
            }
            if (dumax * alpha > (u(i,j,0,nu) - umin)) {
                alpha = (u(i,j,0,nu) - umin) / dumax;
            }
        }
        sx *= alpha;
        sy *= alpha;

        // for each direction, compute minimum of the ratio of limited to unlimited slopes
        if (dcx != Real(0.0)) {
            sfx = amrex::min(sfx, sx / dcx);
        }
        if (dcy != Real(0.0)) {
            sfy = amrex::min(sfy, sy / dcy);
        }
    }

    // multiply unlimited slopes by the minimum of the ratio of limited to unlimited slopes
    for (int ns = 0; ns < ncomp; ++ns) {
        slope(i,j,0,ns        ) *= sfx;
        slope(i,j,0,ns+  ncomp) *= sfy;
    }
}

} // namespace amrex

#endif
