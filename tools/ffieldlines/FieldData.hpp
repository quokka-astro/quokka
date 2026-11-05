#ifndef FFIELDLINES_FIELDDATA_HPP_
#define FFIELDLINES_FIELDDATA_HPP_
/// \file FieldData.hpp
/// \brief Plotfile fields needed for field-line tracing, with filled ghost cells.

#include <string>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include "Options.hpp"

namespace ffieldlines
{

/// Component layout of FieldData::state.
enum Comp : int { Bx = 0, By, Bz, Rho, Mx, My, Mz, Temp, NComp };

/// Ghost cells on every level. A particle only integrates while its own cell is
/// in the valid region of its grid and RK4 stages stay within half a cell of
/// the step start, so a trilinear stencil needs at most one ghost cell; two
/// leave a margin.
constexpr int nGhost = 2;

struct FieldData {
	int finestLevel = 0;
	amrex::Vector<amrex::Geometry> geom;
	amrex::Vector<amrex::BoxArray> grids;
	amrex::Vector<amrex::DistributionMapping> dmap;
	amrex::Vector<int> refRatio; ///< refRatio[lev] = ratio between lev and lev+1
	amrex::Vector<amrex::MultiFab> state;
	amrex::Vector<amrex::iMultiFab> fineMask; ///< 1 where covered by level lev+1 (lev < finestLevel)
	bool hasTemperature = false;
	double time = 0.0;
};

/// Read the fields named in `opts` from `pf`, levels 0..finest, and fill all
/// ghost cells: same-level and periodic neighbors first, then (on fine levels)
/// trilinear interpolation from the next-coarser level, and first-order
/// extrapolation across non-periodic domain faces.
auto LoadFieldData(amrex::PlotFileData &pf, Options const &opts) -> FieldData;

/// Abort with the available names if `name` is not a plotfile variable.
void RequireVariable(amrex::PlotFileData const &pf, std::string const &name, std::string const &option);

} // namespace ffieldlines

#endif // FFIELDLINES_FIELDDATA_HPP_
