#include "AMReX.H"
#include "AMReX_ParmParse.H"
#include "DiagBase.H"
#if AMREX_SPACEDIM == 3
#include "DiagProjectionPlot.H"
#endif
#include <iostream>

class ScheduledDiagnostic : public DiagBase
{
      public:
	explicit ScheduledDiagnostic(amrex::Real interval, int stepInterval = -1)
	{
		m_time_interval = interval;
		m_next_output_time = interval;
		m_interval = stepInterval;
	}
	void close() override {}
};

#if AMREX_SPACEDIM == 3
class ProjectionDiagnostic : public DiagProjectionPlot
{
      public:
	[[nodiscard]] auto prepared() const -> bool { return !first_time; }
};
#endif

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int failures = 0;
	{
		const auto check = [&failures](bool passed, const char *message) {
			if (!passed) {
				std::cerr << message << '\n';
				++failures;
			}
		};
		ScheduledDiagnostic regular(1.);
		check(regular.doDiag(0., 0), "Initial output must be enabled");
		check(!regular.doDiag(0.5, 1), "Small step must wait for deadline");
		check(regular.doDiag(1., 2) && regular.doDiag(1., 2), "Repeated calls on a due step must agree");
		check(!regular.doDiag(1.5, 3), "Step after output must wait");
		check(regular.doDiag(2., 4), "Next regular deadline must fire");

		ScheduledDiagnostic exact(1.);
		check(exact.doDiag(3., 1) && exact.doDiag(3., 1), "Exact jump must output consistently");
		check(!exact.doDiag(3.1, 2), "Consumed exact deadline must not fire again on next step");
		check(exact.doDiag(4., 3), "Output after exact jump must wait until next interval");

		ScheduledDiagnostic overshoot(1.);
		check(overshoot.doDiag(3.25, 1), "Overshooting jump must output");
		check(!overshoot.doDiag(3.5, 2), "Overshooting jump must catch up to future deadline");
		check(overshoot.doDiag(4., 3), "Next deadline after overshoot must fire");

		ScheduledDiagnostic combined(1., 2);
		check(combined.doDiag(3., 1), "Combined scheduler must handle time jump");
		check(combined.doDiag(3.1, 2), "Step interval must remain independent of time deadline");
		check(!combined.doDiag(3.2, 3), "Combined scheduler must not output between intervals");

#if AMREX_SPACEDIM == 3
		amrex::ParmParse pp("projection_test");
		pp.add("int", 1);
		pp.addarr("field_names", amrex::Vector<std::string>{"density"});
		ProjectionDiagnostic projection;
		projection.init("projection_test", "projection_test");
		projection.prepare(0, {}, {}, {}, {"density"});
		check(projection.prepared(), "Projection preparation must complete first-time setup");
		if (projection.prepared()) {
			// A completed setup must not repeat validation on subsequent output cycles.
			projection.prepare(0, {}, {}, {}, {});
			check(projection.prepared(), "Projection must remain prepared across output cycles");
		}
#endif
	}
	amrex::Finalize();
	return failures == 0 ? 0 : 1;
}
