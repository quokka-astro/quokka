#include "AMReX.H"
#include "particle_tracer_test.hpp"

struct TracerProblem {};
int main(int argc, char **argv)
{
	amrex::Initialize(argc, argv);
	{
		const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(3));
		amrex::RealBox bounds;
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			bounds.setLo(d, 0);
			bounds.setHi(d, 1);
		}
		const int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
		const amrex::Geometry geom(domain, &bounds, 0, periodic);
		amrex::BoxArray ba(domain);
		ba.maxSize(2);
		const amrex::DistributionMapping dm(ba);
		particle_test::CheckTracers<TracerProblem>(geom, dm, ba);
	}
	amrex::Finalize();
}
