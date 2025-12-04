// Simple driver to exercise the src/util/fextract.hpp helper on a plotfile.
// Reads a plotfile, extracts a slice, and writes a deterministic ASCII dump for diffing.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
<<<<<<< Updated upstream
=======
#include <ranges>
#include <stdexcept>
#include <string>
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
static auto buildGeometry(PlotFileData &pf) -> Geometry
=======
namespace {

auto buildGeometry(PlotFileData &pf) -> Geometry
>>>>>>> Stashed changes
{
	const int level = 0;
	const Box domain = pf.probDomain(level);
	const Array<Real, AMREX_SPACEDIM> problo = pf.probLo();
	const Array<Real, AMREX_SPACEDIM> probhi = pf.probHi();
	const RealBox rb(problo.data(), probhi.data());
	Vector<int> periodic(AMREX_SPACEDIM, 0);
	return Geometry(domain, &rb, pf.coordSys(), periodic.data());
}

<<<<<<< Updated upstream
static auto buildMultiFab(PlotFileData &pf, const Vector<std::string> &names) -> MultiFab
=======
auto buildMultiFab(PlotFileData &pf, const Vector<std::string> &names) -> MultiFab
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
static void writeSlice(const std::string &outfile, const Vector<Real> &pos, const Vector<Gpu::HostVector<Real>> &data, const Vector<std::string> &names)
=======
void writeSlice(const std::string &outfile, const Vector<Real> &pos, const Vector<Gpu::HostVector<Real>> &data,
		const Vector<std::string> &names)
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	try {
		const ParmParse pp;

		std::string plotfile;
		const bool has_plotfile = pp.query("plotfile", plotfile) != 0;
		if (!has_plotfile) {
			amrex::Abort("plotfile must be provided (plotfile=/path/to/pltXXXX).");
		}
=======
} // namespace

int main(int argc, char **argv)
{
	amrex::Initialize(argc, argv);
	int retval = 0;
	try {
		ParmParse pp;

		std::string plotfile;
		const bool has_plotfile = pp.query("plotfile", plotfile) != 0;
		if (!has_plotfile || plotfile.empty()) {
			throw std::runtime_error("plotfile must be provided (plotfile=/path/to/pltXXXX).");
		}

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
		if (plotfile.empty()) {
			amrex::Abort("plotfile must be provided (plotfile=/path/to/pltXXXX).");
=======
		if (dir < 0 || dir >= AMREX_SPACEDIM) {
			throw std::runtime_error("dir must be within [0, AMREX_SPACEDIM).");
>>>>>>> Stashed changes
		}

		PlotFileData pf(plotfile);

<<<<<<< Updated upstream
		if (dir < 0 || dir >= AMREX_SPACEDIM) {
			amrex::Abort("dir must be within [0, AMREX_SPACEDIM).");
		}

=======
>>>>>>> Stashed changes
		Geometry geom = buildGeometry(pf);
		MultiFab mf = buildMultiFab(pf, names);

		const bool has_coord = coord != std::numeric_limits<Real>::lowest();
		const Array<Real, AMREX_SPACEDIM> problo = pf.probLo();
		const Array<Real, AMREX_SPACEDIM> probhi = pf.probHi();
<<<<<<< Updated upstream
		const Real slice_coord = has_coord ? coord : static_cast<Real>(0.5) * (problo[dir] + probhi[dir]);
=======
		const Real slice_coord = has_coord ? coord : 0.5 * (problo[dir] + probhi[dir]);
>>>>>>> Stashed changes
		const bool use_center = has_coord ? false : center;

		auto [pos, data] = fextract(mf, geom, dir, slice_coord, use_center);
		writeSlice(outfile, pos, data, names);
<<<<<<< Updated upstream
	} catch (const std::exception &ex) {
		amrex::Abort(ex.what());
	} catch (...) {
		amrex::Abort("Unknown exception in test_fextract_driver");
	}
	amrex::Finalize();
	return 0;
=======
	} catch (const std::runtime_error &ex) {
		amrex::Print() << ex.what() << "\n";
		retval = 1;
	}
	amrex::Finalize();
	return retval;
>>>>>>> Stashed changes
}
