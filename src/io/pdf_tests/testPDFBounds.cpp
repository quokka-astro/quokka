#include "AMReX_ParmParse.H"
#include "DiagPDF.H"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

struct PDFTest {};

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int failures = 0;
	{
		const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(3));
		const amrex::RealBox physical(amrex::Array<amrex::Real, AMREX_SPACEDIM>{AMREX_D_DECL(0., 0., 0.)},
					      amrex::Array<amrex::Real, AMREX_SPACEDIM>{AMREX_D_DECL(2., 2., 2.)});
		const amrex::Array<int, AMREX_SPACEDIM> periodic{};
		const amrex::Geometry geometry(domain, physical, 0, periodic);
		amrex::BoxArray boxes(domain);
		boxes.maxSize(2);
		const amrex::DistributionMapping mapping(boxes);
		amrex::MultiFab state(boxes, mapping, 3, 0);
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const auto a = state.array(mfi);
			amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				a(i, j, k, 0) = 2.;
				a(i, j, k, 1) = 1. + i % 3;
				a(i, j, k, 2) = 4. - a(i, j, k, 1);
			});
		}
		amrex::Gpu::streamSynchronize();
		const amrex::Vector<const amrex::MultiFab *> states{&state};
		const amrex::Vector<std::string> names{"gasDensity", "q", "r"};
		const amrex::Vector<amrex::Geometry> geoms{geometry};
		const amrex::Vector<amrex::IntVect> ratios;
		int id = 0;
		for (const int rangeMode : {0, 1, 2}) {
			for (const int logarithmic : {0, 1}) {
				for (const int axes : {1, 2}) {
					for (const std::string weight : {"cell_counts", "volume", "mass"}) {
						const auto prefix = "pdf_test_" + std::to_string(id++);
						amrex::ParmParse pp(prefix);
						pp.add("int", 1);
						pp.add("weight_by", weight);
						pp.addarr("var_names", axes == 1 ? amrex::Vector<std::string>{"q"} : amrex::Vector<std::string>{"q", "r"});
						for (const auto &var : {"q", "r"}) {
							amrex::ParmParse vp(prefix + "." + var);
							vp.add("nBins", 2);
							vp.add("log_spaced_bins", logarithmic);
							if (rangeMode != 0 && std::string(var) == "q") {
								vp.addarr("range", rangeMode == 1 ? amrex::Vector<amrex::Real>{1., 2.}
												  : amrex::Vector<amrex::Real>{2., 3.});
							}
						}
						DiagPDF diagnostic;
						diagnostic.init(prefix, prefix);
						diagnostic.prepare(1, geoms, {boxes}, {mapping}, names);
						diagnostic.setDiagData<PDFTest>(nullptr, &states, &names, &geoms, &ratios, nullptr);
						diagnostic.processDiag<PDFTest>(1, 0.);
						if (amrex::ParallelDescriptor::IOProcessor()) {
							std::ifstream file(prefix + "0000001.dat");
							std::string line;
							amrex::Real total = 0.;
							int rows = 0;
							while (std::getline(file, line)) {
								if (line.empty() || line.front() == '#') {
									continue;
								}
								std::stringstream row(line);
								amrex::Real value = 0.;
								amrex::Real last = 0.;
								int columns = 0;
								while (row >> value) {
									last = value;
									++columns;
								}
								if (columns == 0) {
									continue;
								}
								if (columns != 3 * axes + 1) {
									++failures;
								}
								total += last;
								++rows;
							}
							amrex::Real expected =
							    weight == "cell_counts" ? std::pow(4., AMREX_SPACEDIM) : std::pow(2., AMREX_SPACEDIM);
							if (weight == "mass") {
								expected *= 2.;
							}
							if (rangeMode != 0) {
								expected *= rangeMode == 1 ? .75 : .5;
							}
							if (rows != static_cast<int>(1U << axes) || std::abs(total - expected) > 1.e-12) {
								std::cerr << prefix << " total=" << total << " expected=" << expected << '\n';
								++failures;
							}
						}
					}
				}
			}
		}
	}
	amrex::ParallelDescriptor::ReduceIntSum(failures);
	amrex::Finalize();
	return failures == 0 ? 0 : 1;
}
