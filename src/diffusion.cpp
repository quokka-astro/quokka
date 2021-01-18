
#include "diffusion.hpp"
#include "diffusion_kernel.hpp"

using namespace amrex;

void advance (MultiFab& phi_old,
              MultiFab& phi_new,
	      Array<MultiFab, AMREX_SPACEDIM>& flux,
	      Real dt,
              Geometry const& geom)
{

    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries.
    // There are no physical domain boundaries to fill in this example.
    phi_old.FillBoundary(geom.periodicity());

    //
    // Note that this simple example is not optimized.
    // The following two MFIter loops could be merged
    // and we do not have to use flux MultiFab.
    // 
    // =======================================================

    // This example supports both 2D and 3D.  Otherwise,
    // we would not need to use AMREX_D_TERM.

    // Compute fluxes one grid at a time
    for (MFIter mfi(phi_old); mfi.isValid(); ++mfi) {
	    auto const &phi = phi_old.const_array(mfi);

	    const Box &xbx = mfi.nodaltilebox(0);
	    auto const &fluxx = flux[0].array(mfi);
        const Real dxinv = geom.InvCellSize(0);

	    amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE(LOOP_ORDER_X1(int i, int j, int k)) {
		    compute_flux<X1>(i, j, k, fluxx, phi, dxinv);
	    });

	    if constexpr (amrex::SpaceDim > 1) {
            const Box &ybx = mfi.nodaltilebox(1);
		    auto const &fluxy = flux[1].array(mfi);
            const Real dyinv = geom.InvCellSize(1);

		    amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE(LOOP_ORDER_X2(int i, int j, int k)) {
			    compute_flux<X2>(i, j, k, fluxy, phi, dyinv);
		    });
	    }

	    if constexpr (amrex::SpaceDim > 2) {
            const Box &zbx = mfi.nodaltilebox(2);
		    auto const &fluxz = flux[2].array(mfi);
            const Real dzinv = geom.InvCellSize(2);

		    amrex::ParallelFor(zbx, [=] AMREX_GPU_DEVICE(LOOP_ORDER_X3(int i, int j, int k)) {
			    compute_flux<X3>(i, j, k, fluxz, phi, dzinv);
		    });
	    }
    }

    // Advance the solution one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        std::vector<amrex::Array4<amrex::Real const>> fluxes(amrex::SpaceDim);
        std::vector<amrex::Real> dinv(amrex::SpaceDim);

        const Box& vbx = mfi.validbox();
        auto const& fluxx = flux[0].const_array(mfi);
        fluxes[0] = fluxx;
        dinv[0] = geom.InvCellSize(0);

	    if constexpr (amrex::SpaceDim > 1) {
            auto const& fluxy = flux[1].const_array(mfi);
            fluxes[1] = fluxy;
            dinv[1] = geom.InvCellSize(1);
        }
	    if constexpr (amrex::SpaceDim > 2) {
            auto const& fluxz = flux[2].const_array(mfi);
            fluxes[2] = fluxz;
            dinv[2] = geom.InvCellSize(2);
        }

        auto const& phiOld = phi_old.const_array(mfi);
        auto const& phiNew = phi_new.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            update_phi(i,j,k,phiOld,phiNew,
                       fluxes,
                       dt,
                       dinv);
        });
    }
}

void init_phi(amrex::MultiFab& phi_new, amrex::Geometry const& geom){

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    // =======================================
    // Initialize phi_new by calling a Fortran routine.
    // MFIter = MultiFab Iterator
    for (MFIter mfi(phi_new); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto const& phiNew = phi_new.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_phi(i,j,k,phiNew,dx,prob_lo);
        });
    }
}
