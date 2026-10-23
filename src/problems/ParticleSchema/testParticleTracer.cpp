#include "AMReX.H"
#include "particle_tracer_test.hpp"
#include <array>

struct TracerProblem {};
auto main(int argc, char **argv) -> int
try {
	amrex::Initialize(argc, argv);
	{
		const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(3));
		amrex::RealBox bounds;
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			bounds.setLo(d, 0);
			bounds.setHi(d, 1);
		}
		const std::array<int, AMREX_SPACEDIM> periodic = {AMREX_D_DECL(0, 0, 0)};
		const amrex::Geometry geom(domain, &bounds, 0, periodic.data());
		amrex::BoxArray ba(domain);
		ba.maxSize(2);
		const amrex::DistributionMapping dm(ba);
		particle_test::CheckTracers<TracerProblem>(geom, dm, ba);
	}
	amrex::Finalize();
} catch (const std::exception &error) {
	amrex::Abort(error.what());
	return 1;
}
