/// \file MakeTestPlotfile.cpp
/// \brief Write small synthetic Quokka-style plotfiles with analytic fields for ffieldlines tests.
///
/// Usage: ffieldlines_make_test_plotfile CASE OUTDIR NCELL MAX_GRID_SIZE NLEVELS
///
/// Domain [0,1]^3, NCELL^3 cells on level 0; with NLEVELS = 2 a refined patch
/// (ratio 2) covers [0.5,0.75] x [0.25,0.75] x [0.25,0.75]. All fields are
/// linear in position, so trilinear interpolation reproduces them exactly:
///
///  uniform:  B = (1, 0, 0),           rho = 2,     v = (1, 1, 0),  T = 100
///  diagonal: B = (1, 1, 0),           rho = 2,     v = (1, 0, 0),  T = 100
///  circle:   B = (-(y-1/2), x-1/2, 0), rho = 1 + x, v = (1, 0, 0),  T = 100 + 10 z

#include <cstdlib>
#include <string>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

namespace
{

struct Fields {
	double b[3];
	double rho;
	double v[3];
	double temp;
};

auto Evaluate(std::string const &name, const double x, const double y, const double z) -> Fields
{
	if (name == "uniform") {
		return {{1.0, 0.0, 0.0}, 2.0, {1.0, 1.0, 0.0}, 100.0};
	}
	if (name == "diagonal") {
		return {{1.0, 1.0, 0.0}, 2.0, {1.0, 0.0, 0.0}, 100.0};
	}
	if (name == "circle") {
		return {{-(y - 0.5), x - 0.5, 0.0}, 1.0 + x, {1.0, 0.0, 0.0}, 100.0 + 10.0 * z};
	}
	amrex::Abort("unknown test case '" + name + "'");
	return {};
}

} // namespace

auto main(int argc, char *argv[]) -> int
{
	amrex::Initialize(argc, argv, false);
	{
		if (amrex::command_argument_count() != 5) {
			amrex::Abort("usage: ffieldlines_make_test_plotfile CASE OUTDIR NCELL MAX_GRID_SIZE NLEVELS");
		}
		const std::string testCase = amrex::get_command_argument(1);
		const std::string outdir = amrex::get_command_argument(2);
		const int ncell = std::stoi(amrex::get_command_argument(3));
		const int maxGridSize = std::stoi(amrex::get_command_argument(4));
		const int nlevels = std::stoi(amrex::get_command_argument(5));
		if (nlevels < 1 || nlevels > 2) {
			amrex::Abort("NLEVELS must be 1 or 2");
		}

		const amrex::RealBox realBox({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
		const amrex::Array<int, 3> notPeriodic{0, 0, 0};
		const int ratio = 2;

		amrex::Vector<amrex::Geometry> geom(nlevels);
		amrex::Vector<amrex::MultiFab> state(nlevels);
		amrex::Box domain(amrex::IntVect(0), amrex::IntVect(ncell - 1));
		for (int lev = 0; lev < nlevels; ++lev) {
			geom[lev].define(domain, realBox, 0, notPeriodic);
			amrex::BoxArray ba;
			if (lev == 0) {
				ba.define(domain);
			} else {
				const int n = ncell * ratio;
				ba.define(amrex::Box(amrex::IntVect(n / 2, n / 4, n / 4), amrex::IntVect(3 * n / 4 - 1, 3 * n / 4 - 1, 3 * n / 4 - 1)));
			}
			ba.maxSize(maxGridSize);
			const amrex::DistributionMapping dm(ba);
			state[lev].define(ba, dm, 8, 0);

			const auto plo = geom[lev].ProbLoArray();
			const auto dx = geom[lev].CellSizeArray();
			for (amrex::MFIter mfi(state[lev]); mfi.isValid(); ++mfi) {
				const auto a = state[lev].array(mfi);
				const amrex::Box &bx = mfi.validbox();
				amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
					const double x = plo[0] + (i + 0.5) * dx[0];
					const double y = plo[1] + (j + 0.5) * dx[1];
					const double z = plo[2] + (k + 0.5) * dx[2];
					const Fields f = Evaluate(testCase, x, y, z);
					a(i, j, k, 0) = f.rho;
					for (int d = 0; d < 3; ++d) {
						a(i, j, k, 1 + d) = f.rho * f.v[d];
						a(i, j, k, 4 + d) = f.b[d];
					}
					a(i, j, k, 7) = f.temp;
				});
			}
			domain.refine(ratio);
		}

		const amrex::Vector<std::string> varnames{"gasDensity", "x-GasMomentum", "y-GasMomentum", "z-GasMomentum",
							  "x-BField",	"y-BField",	 "z-BField",	  "temperature"};
		amrex::WriteMultiLevelPlotfile(outdir, nlevels, amrex::GetVecOfConstPtrs(state), varnames, geom, 1.5, amrex::Vector<int>(nlevels, 7),
					       amrex::Vector<amrex::IntVect>(nlevels, amrex::IntVect(ratio)));
	}
	amrex::Finalize();
	return EXIT_SUCCESS;
}
