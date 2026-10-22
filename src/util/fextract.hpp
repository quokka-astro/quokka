#ifndef FEXTRACT_HPP_
#define FEXTRACT_HPP_

#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

// Extract the line along idir through the cells containing the transverse
// coordinates. The coordinate along idir is ignored. Out-of-domain coordinates
// are clamped to the nearest cell. center selects the upper midpoint cell in
// each transverse direction and ignores coordinates. Full sorted profiles are
// gathered on the I/O rank; other ranks retain their local data.
auto fextract(amrex::MultiFab &mf, amrex::Geometry &geom, int idir, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &slice_coords, bool center = false)
    -> std::tuple<amrex::Vector<amrex::Real>, amrex::Vector<amrex::Gpu::HostVector<amrex::Real>>>;

// Convenience overload: use the same physical coordinate on every transverse axis.
auto fextract(amrex::MultiFab &mf, amrex::Geometry &geom, int idir, amrex::Real slice_coord, bool center = false)
    -> std::tuple<amrex::Vector<amrex::Real>, amrex::Vector<amrex::Gpu::HostVector<amrex::Real>>>;

#endif // FEXTRACT_HPP_