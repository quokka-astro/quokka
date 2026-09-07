#include "AMReX_ParmParse.H"
#include "DiagPDF.H"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

struct PeriodicPDF {};

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int failures = 0;
	{
		int id = 0;
		for (const bool periodic : {false, true}) {
			for (const bool shifted : {false, true}) {
				if (shifted && !periodic) {
					continue;
				}
				const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(3));
				const amrex::RealBox physical(amrex::Array<amrex::Real, AMREX_SPACEDIM>{AMREX_D_DECL(0., 0., 0.)},
							      amrex::Array<amrex::Real, AMREX_SPACEDIM>{AMREX_D_DECL(2., 2., 2.)});
				amrex::Array<int, AMREX_SPACEDIM> periodicity{};
				periodicity[0] = periodic ? 1 : 0;
				const amrex::Geometry coarseGeom(domain, physical, 0, periodicity);
				const amrex::Geometry fineGeom(amrex::refine(domain, 2), physical, 0, periodicity);
				auto fineBox = amrex::refine(domain, 2);
				fineBox.setBig(0, 3);
				if (shifted) {
					fineBox.shift(0, 8);
				} // Represent the fine patch by its periodic image.
				amrex::BoxArray coarseBoxes(domain), fineBoxes(fineBox);
				coarseBoxes.maxSize(2);
				fineBoxes.maxSize(4);
				const amrex::DistributionMapping coarseMap(coarseBoxes), fineMap(fineBoxes);
				amrex::MultiFab coarse(coarseBoxes, coarseMap, 1, 0), fine(fineBoxes, fineMap, 1, 0);
				coarse.setVal(1.);
				fine.setVal(1.);
				const amrex::Vector<const amrex::MultiFab *> states{&coarse, &fine};
				const amrex::Vector<std::string> names{"gasDensity"};
				const amrex::Vector<amrex::Geometry> geoms{coarseGeom, fineGeom};
				const amrex::Vector<amrex::IntVect> ratios{amrex::IntVect(2)};
				for (const std::string weight : {"cell_counts", "volume", "mass"}) {
					const auto prefix = "periodic_pdf_" + std::to_string(id++);
					amrex::ParmParse pp(prefix);
					pp.add("int", 1);
					pp.add("weight_by", weight);
					pp.addarr("var_names", names);
					amrex::ParmParse vp(prefix + ".gasDensity");
					vp.add("nBins", 2);
					vp.addarr("range", amrex::Vector<amrex::Real>{.5, 1.5});
					DiagPDF diagnostic;
					diagnostic.init(prefix, prefix);
					diagnostic.prepare(2, geoms, {coarseBoxes, fineBoxes}, {coarseMap, fineMap}, names);
					diagnostic.setDiagData<PeriodicPDF>(nullptr, &states, &names, &geoms, &ratios, nullptr);
					diagnostic.processDiag<PeriodicPDF>(1, 0.);
					if (amrex::ParallelDescriptor::IOProcessor()) {
						std::ifstream file(prefix + "0000001.dat");
						std::string line;
						amrex::Real total = 0.;
						int rows = 0;
						while (std::getline(file, line)) {
							std::stringstream row(line);
							int bin;
							amrex::Real lo, hi, value;
							if (row >> bin >> lo >> hi >> value) {
								total += value;
								++rows;
							}
						}
						const amrex::Real expected = weight == "cell_counts"
										 ? .5 * std::pow(4., AMREX_SPACEDIM) + .5 * std::pow(8., AMREX_SPACEDIM)
										 : std::pow(2., AMREX_SPACEDIM);
						if (rows != 2 || std::abs(total - expected) > 1.e-12) {
							std::cerr << "periodic=" << periodic << " shifted=" << shifted << " weight=" << weight << " got "
								  << total << " expected " << expected << '\n';
							++failures;
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
