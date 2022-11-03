#include <AMReX_Interp_C.H>
#include <AMReX_MFInterp_C.H>
#include <AMReX_MFInterpolater.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "MFInterpolater.H"

namespace amrex {

// Cell centered
MFQuokkaInterp mf_linear_slope_minmax_interp;

Box
MFQuokkaInterp::CoarseBox (const Box& fine, const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
}

Box
MFQuokkaInterp::CoarseBox (const Box& fine, int ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
}

void
MFQuokkaInterp::interp (MultiFab const& crsemf, int ccomp, MultiFab& finemf, int fcomp, int nc,
                             IntVect const& ng, Geometry const& cgeom, Geometry const& fgeom,
                             Box const& dest_domain, IntVect const& ratio,
                             Vector<BCRec> const& bcs, int bcomp)
{
    AMREX_ASSERT(crsemf.nGrowVect() == 0);
    amrex::ignore_unused(fgeom);

    Box const& cdomain = cgeom.Domain();

#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion()) {
        MultiFab crse_tmp(crsemf.boxArray(), crsemf.DistributionMap(), AMREX_SPACEDIM*nc, 0);
        auto const& crse = crsemf.const_arrays();
        auto const& tmp = crse_tmp.arrays();
        auto const& ctmp = crse_tmp.const_arrays();
        auto const& fine = finemf.arrays();

        Gpu::DeviceVector<BCRec> d_bc(nc);
        BCRec const* pbc = d_bc.data();
        Gpu::copyAsync(Gpu::hostToDevice, bcs.begin()+bcomp, bcs.begin()+bcomp+nc, d_bc.begin());

        ParallelFor(crsemf, IntVect(-1),
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            mf_quokka_interp_llslope(i,j,k, tmp[box_no], crse[box_no], ccomp, nc,
                                     cdomain, ratio, pbc);
        });

        ParallelFor(finemf, ng, nc,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
        {
            if (dest_domain.contains(i,j,k)) {
                mf_cell_cons_lin_interp(i,j,k,n, fine[box_no], fcomp, ctmp[box_no],
                                        crse[box_no], ccomp, nc, ratio);
            }
        });

        Gpu::streamSynchronize();
    } else
#endif
    {
        BCRec const* pbc = bcs.data() + bcomp;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        {
            FArrayBox tmpfab;
            for (MFIter mfi(finemf); mfi.isValid(); ++mfi) {
                auto const& fine = finemf.array(mfi);
                auto const& crse = crsemf.const_array(mfi);

                Box const& cbox = amrex::grow(crsemf[mfi].box(), -1);
                tmpfab.resize(cbox, AMREX_SPACEDIM*nc);
                auto const& tmp = tmpfab.array();
                auto const& ctmp = tmpfab.const_array();

                Box const& fbox = amrex::grow(mfi.validbox(), ng) & dest_domain;

                amrex::LoopConcurrentOnCpu(cbox,
                [&] (int i, int j, int k) noexcept
                {
                    mf_quokka_interp_llslope(i,j,k, tmp, crse, ccomp, nc,
                                             cdomain, ratio, pbc);
                });

                amrex::LoopConcurrentOnCpu(fbox, nc,
                [&] (int i, int j, int k, int n) noexcept
                {
                    mf_cell_cons_lin_interp(i,j,k,n, fine, fcomp, ctmp,
                                            crse, ccomp, nc, ratio);
                });
            }
        }
    }
}

} // namespace amrex
