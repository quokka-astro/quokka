#ifndef MF_INTERP_1D_C_H_
#define MF_INTERP_1D_C_H_

namespace amrex {

// limits the slopes of all components in a given cell the same way
//  *and* prevents new minima or maxima from appearing in the fine grid

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mf_quokka_interp_llslope (int i, int, int, Array4<Real> const& slope,
                                      Array4<Real const> const& u, int scomp, int ncomp,
                                      Box const& domain, IntVect const& ratio, BCRec const* bc) noexcept
{
    Real sfx = Real(1.0);

    for (int ns = 0; ns < ncomp; ++ns) {
        int nu = ns + scomp;

        // x-direction
        Real dc = mf_compute_slopes_x(i, 0, 0, u, nu, domain, bc[ns]);
        Real df = Real(2.0) * (u(i+1,0,0,nu) - u(i  ,0,0,nu));
        Real db = Real(2.0) * (u(i  ,0,0,nu) - u(i-1,0,0,nu));
        Real sx = (df*db >= Real(0.0)) ? amrex::min(amrex::Math::abs(df),amrex::Math::abs(db)) : Real(0.);
        sx = amrex::Math::copysign(Real(1.),dc)*amrex::min(sx,amrex::Math::abs(dc));
        slope(i,0,0,ns) = dc;

        // additional limiting is unnecessary in 1D

        if (dc != Real(0.0)) {
            sfx = amrex::min(sfx, sx / dc);
        }
    }

    for (int ns = 0; ns < ncomp; ++ns) {
        slope(i,0,0,ns) *= sfx;
    }
}

}

#endif
