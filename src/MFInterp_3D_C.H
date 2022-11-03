#ifndef MF_INTERP_3D_C_H_
#define MF_INTERP_3D_C_H_

namespace amrex {

// limits the slopes of all components in a given cell the same way
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mf_quokka_interp_llslope (int i, int j, int k, Array4<Real> const& slope,
                                      Array4<Real const> const& u, int scomp, int ncomp,
                                      Box const& domain, IntVect const& ratio, BCRec const* bc) noexcept
{
    Real sfx = Real(1.0);
    Real sfy = Real(1.0);
    Real sfz = Real(1.0);

    for (int ns = 0; ns < ncomp; ++ns) {
        int nu = ns + scomp;

        // x-direction
        Real dcx = mf_compute_slopes_x(i, j, k, u, nu, domain, bc[ns]);
        Real df = Real(2.0) * (u(i+1,j,k,nu) - u(i  ,j,k,nu));
        Real db = Real(2.0) * (u(i  ,j,k,nu) - u(i-1,j,k,nu));
        Real sx = (df*db >= Real(0.0)) ? amrex::min(amrex::Math::abs(df),amrex::Math::abs(db)) : Real(0.);
        sx = amrex::Math::copysign(Real(1.),dcx)*amrex::min(sx,amrex::Math::abs(dcx));
        slope(i,j,k,ns        ) = dcx;  // unlimited slope

        // y-direction
        Real dcy = mf_compute_slopes_y(i, j, k, u, nu, domain, bc[ns]);
        df = Real(2.0) * (u(i,j+1,k,nu) - u(i,j  ,k,nu));
        db = Real(2.0) * (u(i,j  ,k,nu) - u(i,j-1,k,nu));
        Real sy = (df*db >= Real(0.0)) ? amrex::min(amrex::Math::abs(df),amrex::Math::abs(db)) : Real(0.);
        sy = amrex::Math::copysign(Real(1.),dcy)*amrex::min(sy,amrex::Math::abs(dcy));
        slope(i,j,k,ns+  ncomp) = dcy;  // unlimited slope

        // z-direction
        Real dcz = mf_compute_slopes_z(i, j, k, u, nu, domain, bc[ns]);
        df = Real(2.0) * (u(i,j,k+1,nu) - u(i,j,k  ,nu));
        db = Real(2.0) * (u(i,j,k  ,nu) - u(i,j,k-1,nu));
        Real sz = (df*db >= Real(0.0)) ? amrex::min(amrex::Math::abs(df),amrex::Math::abs(db)) : Real(0.);
        sz = amrex::Math::copysign(Real(1.),dcz)*amrex::min(sz,amrex::Math::abs(dcz));
        slope(i,j,k,ns+2*ncomp) = dcz;  // unlimited slope

        // adjust limited slopes to prevent new min/max for this component
        Real alpha = 1.0;
        if (sx != Real(0.0) || sy != Real(0.0) || sz != Real(0.0)) {
            Real dumax = amrex::Math::abs(sx) * Real(ratio[0]-1)/Real(2*ratio[0])
                +        amrex::Math::abs(sy) * Real(ratio[1]-1)/Real(2*ratio[1])
                +        amrex::Math::abs(sz) * Real(ratio[2]-1)/Real(2*ratio[2]);
            Real umax = u(i,j,k,nu);
            Real umin = u(i,j,k,nu);
            for (int koff = -1; koff <= 1; ++koff) {
            for (int joff = -1; joff <= 1; ++joff) {
            for (int ioff = -1; ioff <= 1; ++ioff) {
                umin = amrex::min(umin, u(i+ioff,j+joff,k+koff,nu));
                umax = amrex::max(umax, u(i+ioff,j+joff,k+koff,nu));
            }}}
            if (dumax * alpha > (umax - u(i,j,k,nu))) {
                alpha = (umax - u(i,j,k,nu)) / dumax;
            }
            if (dumax * alpha > (u(i,j,k,nu) - umin)) {
                alpha = (u(i,j,k,nu) - umin) / dumax;
            }
        }
        sx *= alpha;
        sy *= alpha;
        sz *= alpha;

        // for each direction, compute minimum of the ratio of limited to unlimited slopes
        if (dcx != Real(0.0)) {
            sfx = amrex::min(sfx, sx / dcx);
        }
        if (dcy != Real(0.0)) {
            sfy = amrex::min(sfy, sy / dcy);
        }
        if (dcz != Real(0.0)) {
            sfz = amrex::min(sfz, sz / dcz);
        }
    }

    // multiply unlimited slopes by the minimum of the ratio of limited to unlimited slopes
    for (int ns = 0; ns < ncomp; ++ns) {
        slope(i,j,k,ns        ) *= sfx;
        slope(i,j,k,ns+  ncomp) *= sfy;
        slope(i,j,k,ns+2*ncomp) *= sfz;
    }
}

} // namespace amrex

#endif
