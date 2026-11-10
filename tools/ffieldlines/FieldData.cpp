/// \file FieldData.cpp
/// \brief Plotfile fields needed for field-line tracing, with filled ghost cells.

#include "FieldData.hpp"

#include <algorithm>
#include <limits>

#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_RealBox.H>

namespace ffieldlines
{

void RequireVariable(amrex::PlotFileData const &pf, std::string const &name, std::string const &option)
{
	auto const &names = pf.varNames();
	if (std::find(names.begin(), names.end(), name) != names.end()) {
		return;
	}
	std::string available;
	for (auto const &n : names) {
		available += "\n  " + n;
	}
	amrex::Abort("ffieldlines: variable '" + name + "' (" + option + ") is not in the plotfile. Available variables:" + available);
}

auto LoadFieldData(amrex::PlotFileData &pf, Options const &opts) -> FieldData
{
	if (pf.spaceDim() != 3) {
		amrex::Abort("ffieldlines: only 3D plotfiles are supported");
	}
	if (pf.coordSys() != 0) {
		amrex::Abort("ffieldlines: only Cartesian plotfiles are supported");
	}

	FieldData fd;
	fd.finestLevel = pf.finestLevel();
	if (opts.finestLevel >= 0) {
		fd.finestLevel = std::min(fd.finestLevel, opts.finestLevel);
	}
	fd.time = pf.time();

	// (component, plotfile variable) pairs to read
	amrex::Vector<std::pair<int, std::string>> reads = {{Bx, opts.bfield[0]},   {By, opts.bfield[1]},   {Bz, opts.bfield[2]},  {Rho, opts.density},
							    {Mx, opts.momentum[0]}, {My, opts.momentum[1]}, {Mz, opts.momentum[2]}};
	RequireVariable(pf, opts.bfield[0], "--bfield");
	RequireVariable(pf, opts.bfield[1], "--bfield");
	RequireVariable(pf, opts.bfield[2], "--bfield");
	RequireVariable(pf, opts.density, "--density");
	RequireVariable(pf, opts.momentum[0], "--momentum");
	RequireVariable(pf, opts.momentum[1], "--momentum");
	RequireVariable(pf, opts.momentum[2], "--momentum");
	fd.hasTemperature = !opts.temperature.empty();
	if (fd.hasTemperature) {
		RequireVariable(pf, opts.temperature, "--temperature");
		reads.emplace_back(Temp, opts.temperature);
	}

	const auto probLo = pf.probLo();
	const auto probHi = pf.probHi();
	const amrex::RealBox realBox(probLo.data(), probHi.data());
	const amrex::Array<int, AMREX_SPACEDIM> isPeriodic{opts.periodic[0], opts.periodic[1], opts.periodic[2]};

	const int nlev = fd.finestLevel + 1;
	fd.geom.resize(nlev);
	fd.grids.resize(nlev);
	fd.dmap.resize(nlev);
	fd.refRatio.resize(std::max(nlev - 1, 0));
	for (int lev = 0; lev < nlev; ++lev) {
		fd.geom[lev].define(pf.probDomain(lev), realBox, pf.coordSys(), isPeriodic);
		fd.grids[lev] = pf.boxArray(lev);
		fd.dmap[lev] = pf.DistributionMap(lev);
		if (lev < nlev - 1) {
			const amrex::IntVect ratio = pf.refRatioVect(lev);
			if (ratio[0] != ratio[1] || ratio[0] != ratio[2]) {
				amrex::Abort("ffieldlines: anisotropic refinement ratios are not supported");
			}
			fd.refRatio[lev] = ratio[0];
		}
	}

	// read valid data (no ghost cells)
	amrex::Vector<amrex::MultiFab> valid(nlev);
	for (int lev = 0; lev < nlev; ++lev) {
		valid[lev].define(fd.grids[lev], fd.dmap[lev], NComp, 0);
		valid[lev].setVal(0.0);
		for (auto const &[comp, name] : reads) {
			const amrex::MultiFab mf = pf.get(lev, name);
			amrex::MultiFab::Copy(valid[lev], mf, 0, comp, 1, 0);
		}
	}

	// boundary conditions: periodic where requested, first-order extrapolation elsewhere
	amrex::Vector<amrex::BCRec> bcs(NComp);
	for (auto &bc : bcs) {
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			const int type = (opts.periodic[d] != 0) ? amrex::BCType::int_dir : amrex::BCType::foextrap;
			bc.setLo(d, type);
			bc.setHi(d, type);
		}
	}
	using PhysBC = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<amrex::FabFillNoOp>>;

	fd.state.resize(nlev);
	for (int lev = 0; lev < nlev; ++lev) {
		fd.state[lev].define(fd.grids[lev], fd.dmap[lev], NComp, nGhost);
		PhysBC fineBC(fd.geom[lev], bcs, amrex::GpuBndryFuncFab<amrex::FabFillNoOp>{});
		if (lev == 0) {
			amrex::FillPatchSingleLevel(fd.state[lev], amrex::IntVect(nGhost), fd.time, {&valid[lev]}, {fd.time}, 0, 0, NComp, fd.geom[lev], fineBC,
						    0);
		} else {
			PhysBC coarseBC(fd.geom[lev - 1], bcs, amrex::GpuBndryFuncFab<amrex::FabFillNoOp>{});
			amrex::FillPatchTwoLevels(fd.state[lev], amrex::IntVect(nGhost), fd.time, {&valid[lev - 1]}, {fd.time}, {&valid[lev]}, {fd.time}, 0, 0,
						  NComp, fd.geom[lev - 1], fd.geom[lev], coarseBC, 0, fineBC, 0, amrex::IntVect(fd.refRatio[lev - 1]),
						  &amrex::cell_bilinear_interp, bcs, 0);
		}
		if (!fd.hasTemperature) {
			fd.state[lev].setVal(std::numeric_limits<amrex::Real>::quiet_NaN(), Temp, 1, nGhost);
		}
	}

	fd.fineMask.resize(nlev);
	for (int lev = 0; lev < nlev - 1; ++lev) {
		fd.fineMask[lev] = amrex::makeFineMask(fd.grids[lev], fd.dmap[lev], amrex::IntVect(0), fd.grids[lev + 1], amrex::IntVect(fd.refRatio[lev]),
						       fd.geom[lev].periodicity(), 0, 1);
	}
	return fd;
}

} // namespace ffieldlines
