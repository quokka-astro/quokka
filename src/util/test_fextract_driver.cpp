// Simple driver to exercise the src/util/fextract.hpp helper on a plotfile.
// Reads a plotfile, extracts a slice, and writes a deterministic ASCII dump for diffing.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_Geometry.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_REAL.H"
#include "AMReX_RealBox.H"
#include "AMReX_Vector.H"

#include "util/fextract.hpp"

using namespace amrex; // NOLINT

static auto buildGeometry(PlotFileData &pf) -> Geometry
{
	const int level = 0;
	const Box domain = pf.probDomain(level);
	const Array<Real, AMREX_SPACEDIM> problo = pf.probLo();
	const Array<Real, AMREX_SPACEDIM> probhi = pf.probHi();
	const RealBox rb(problo.data(), probhi.data());
	Vector<int> periodic(AMREX_SPACEDIM, 0);
	return Geometry(domain, &rb, pf.coordSys(), periodic.data());
}

static auto buildMultiFab(PlotFileData &pf, const Vector<std::string> &names) -> MultiFab
{
	const int level = 0;
	auto full = pf.get(level);
	if (names.empty()) {
		return full;
	}

	MultiFab subset(full.boxArray(), full.DistributionMap(), static_cast<int>(names.size()), full.nGrowVect());
	subset.setVal(0.0);

	for (int idx = 0; idx < static_cast<int>(names.size()); ++idx) {
		auto var = pf.get(level, names[idx]);
		MultiFab::Copy(subset, var, 0, idx, 1, full.nGrowVect());
	}

	return subset;
}

static void writeSlice(const std::string &outfile, const Vector<Real> &pos, const Vector<Gpu::HostVector<Real>> &data, const Vector<std::string> &names)
{
	if (!ParallelDescriptor::IOProcessor()) {
		return;
	}

	std::vector<int> indices(pos.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::ranges::sort(indices, [&pos](int a, int b) { return pos[a] < pos[b]; });

	std::ofstream ofs(outfile, std::ios::trunc);
	ofs.setf(std::ios::scientific);
	ofs << std::setprecision(17);

	if (!names.empty()) {
		ofs << "#";
		for (const auto &n : names) {
			ofs << " " << n;
		}
		ofs << "\n";
	}

	for (const int idx : indices) {
		ofs << pos[idx];
		for (const auto &comp : data) {
			ofs << " " << comp[idx];
		}
		ofs << "\n";
	}
}

auto problem_main() -> int
{
	try {
		ParmParse pp;

		std::string plotfile;
		const bool has_plotfile = pp.query("plotfile", plotfile) != 0;
		if (!has_plotfile || plotfile.empty()) {
			throw std::runtime_error("plotfile must be provided (plotfile=/path/to/pltXXXX).");
		}

		std::string outfile = "fextract.out";
		pp.query("outfile", outfile);

		int dir = 0;
		pp.query("dir", dir);
		Real coord = std::numeric_limits<Real>::lowest();
		pp.query("coord", coord);
		bool center = true;
		pp.query("center", center);

		Vector<std::string> names;
		pp.queryarr("vars", names);

		if (dir < 0 || dir >= AMREX_SPACEDIM) {
			throw std::runtime_error("dir must be within [0, AMREX_SPACEDIM).");
		}

		PlotFileData pf(plotfile);
		Geometry geom = buildGeometry(pf);
		MultiFab mf = buildMultiFab(pf, names);

		const bool has_coord = coord != std::numeric_limits<Real>::lowest();
		const Array<Real, AMREX_SPACEDIM> problo = pf.probLo();
		const Array<Real, AMREX_SPACEDIM> probhi = pf.probHi();
		const Real slice_coord = has_coord ? coord : 0.5 * (problo[dir] + probhi[dir]);
		const bool use_center = has_coord ? false : center;

		auto [pos, data] = fextract(mf, geom, dir, slice_coord, use_center);
		writeSlice(outfile, pos, data, names);
		
		return 0;
	} catch (const std::runtime_error &ex) {
		amrex::Print() << ex.what() << "\n";
		return 1;
	}
}
