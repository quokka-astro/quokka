#include "AMReX_ParmParse.H"
#include "DiagFramePlane.H"
#include <iostream>

auto main(int argc, char **argv) -> int
{
	const bool filtered = argc > 1 && std::string(argv[1]) == "filtered";
	amrex::Initialize(argc, argv, false);
	int result = 0;
	{
		amrex::ParmParse pp("frame_test");
		pp.add("int", 1);
		pp.addarr("field_names", amrex::Vector<std::string>{"density"});
		pp.add("normal", 0);
		pp.addarr("center", amrex::Vector<amrex::Real>{.5});
		if (filtered) {
			pp.addarr("filters", amrex::Vector<std::string>{"unused"});
			amrex::ParmParse filter("frame_test.unused");
			filter.add("field_name", "unused_field");
			filter.add("value_greater", 1.);
		}
		DiagFramePlane diagnostic;
		diagnostic.init("frame_test", "frame_test");
		std::cout << "Initialization completed\n";
		amrex::Vector<std::string> fields;
		diagnostic.addVars(fields);
		if (fields != amrex::Vector<std::string>{"density"}) {
			std::cerr << "Unsupported filter still contributes variables\n";
			result = 1;
		}
#if AMREX_SPACEDIM == 3
		const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(3));
		const amrex::RealBox physical(amrex::Array<amrex::Real, 3>{0., 0., 0.}, amrex::Array<amrex::Real, 3>{1., 1., 1.});
		const amrex::Geometry geometry(domain, physical, 0, amrex::Array<int, 3>{0, 0, 0});
		const amrex::BoxArray boxes(domain);
		const amrex::DistributionMapping mapping(boxes);
		diagnostic.prepare(1, {geometry}, {boxes}, {mapping}, {"density"});
		std::cout << "Preparation completed\n";
#endif
	}
	amrex::Finalize();
	return result;
}
