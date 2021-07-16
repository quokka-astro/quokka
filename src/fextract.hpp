#ifndef FEXTRACT_HPP_
#define FEXTRACT_HPP_

#include "AMReX_Config.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

using namespace amrex;

auto fextract(MultiFab &mf, Geometry &geom, int idir,
              Real slice_coord, bool center = false)
    -> std::tuple<Vector<Real>, Vector<Vector<Real>>>;

#endif // FEXTRACT_HPP_