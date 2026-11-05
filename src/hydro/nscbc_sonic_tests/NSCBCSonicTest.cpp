#include "AMReX_FArrayBox.H"
#include "AMReX_GpuContainers.H"
#include "hydro/NSCBC_outflow.hpp"

struct SonicProblem {};
template <> struct Physics_Traits<SonicProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = 1;
};
template <> struct quokka::EOS_Traits<SonicProblem> {
	// rho=1, P=1/2 has exactly representable c=1, including the conservative round trip.
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};
using Hydro = HydroSystem<SonicProblem>;
constexpr int nvar = Hydro::nvar_;
using State = quokka::valarray<amrex::Real, nvar>;

template <NSCBC::BoundarySide side> auto checkDerivatives(double mach) -> int
{
	constexpr bool lower = side == NSCBC::BoundarySide::Lower;
	amrex::Gpu::DeviceScalar<int> failure(0);
	auto *failed = failure.dataPtr();
	amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) {
		State q{1.0, lower ? -mach : mach, 0.0, 0.0, 0.5, 0.5, 0.25};
		State zero{};
		constexpr double mismatch = 1.e-6;
		const auto actual = NSCBC::detail::dQ_dx_outflow<SonicProblem, side>(q, zero, zero, zero, q[4] - mismatch, 1.0);
		// In 1D, K/(c-|u|) = (1+Mach)/(4L). No singular denominator is needed in this oracle.
		const double relaxation = mach < 1.0 ? 0.25 * (1.0 + mach) * (q[4] - (q[4] - mismatch)) : 0.0;
		State expected{};
		expected[0] = (lower ? 0.5 : -0.5) * relaxation;
		expected[1] = 0.5 * relaxation;
		expected[4] = expected[0];
		for (int n = 0; n < nvar; ++n) {
			if (!std::isfinite(actual[n]) || std::abs(actual[n] - expected[n]) > 1.e-12) {
				*failed = 1;
			}
		}
		if (mach >= 1.0) {
			State gradient{};
			for (int n = 0; n < nvar; ++n) {
				gradient[n] = 0.01 * (n + 1);
			}
			const auto outgoing = NSCBC::detail::dQ_dx_outflow<SonicProblem, side>(q, gradient, gradient, gradient, 0.1, 1.0);
			for (int n = 0; n < nvar; ++n) {
				if (!std::isfinite(outgoing[n]) || outgoing[n] != gradient[n]) {
					*failed = 1;
				}
			}
		}
		// Total Mach > 1 must not disable the incoming *normal* acoustic characteristic.
		q[1] = lower ? -0.6 : 0.6;
		q[2] = 0.9;
		State gradient{};
		gradient[4] = 0.02;
		const auto oblique = NSCBC::detail::dQ_dx_outflow<SonicProblem, side>(q, gradient, zero, zero, 0.1, 1.0);
		if (std::abs(oblique[0] + 0.01) > 1.e-12 || std::abs(oblique[1] - (lower ? -0.01 : 0.01)) > 1.e-12 || std::abs(oblique[4] - 0.01) > 1.e-12) {
			*failed = 1;
		}
	});
	return failure.dataValue();
}

template <FluxDir dir, NSCBC::BoundarySide side> auto checkGhosts(double mach, double mismatch) -> int
{
	constexpr int axis = static_cast<int>(dir);
	constexpr bool lower = side == NSCBC::BoundarySide::Lower;
	const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
	const amrex::RealBox physical({0., 0., 0.}, {1., 1., 1.});
	const amrex::Geometry geometry(domain, &physical, 0);
	const auto geom = geometry.data();
	amrex::FArrayBox fab(amrex::grow(domain, 6), nvar);
	fab.setVal(-12345.0);
	const auto a = fab.array();
	State q{1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.25};
	q[axis + 1] = lower ? -mach : mach;
	amrex::ParallelFor(domain, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto initial = Hydro::ComputeConsVars(q);
		for (int n = 0; n < nvar; ++n) {
			a(i, j, k, n) = initial[n];
		}
	});
	amrex::Gpu::DeviceVector<int> failures(6, 0);
	auto *failed = failures.data();
	amrex::ParallelFor(6, [=] AMREX_GPU_DEVICE(int depth) {
		const auto initial = Hydro::ComputeConsVars(q);
		amrex::IntVect ghost(3);
		ghost[axis] = lower ? -1 - depth : 8 + depth;
		NSCBC::setOutflowBoundary<SonicProblem, dir, side>(ghost, a, geom, 0.5 - mismatch);
		for (int n = 0; n < nvar; ++n) {
			const double actual = a(ghost, n);
			const double tolerance = mismatch == 0.0 ? 1.e-11 : 1.e-4;
			if (!std::isfinite(actual) || std::abs(actual - initial[n]) > tolerance) {
				failed[depth] = 1;
			}
		}
	});
	amrex::Gpu::HostVector<int> host(6);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, failures.begin(), failures.end(), host.begin());
	int count = 0;
	for (int depth = 0; depth < 6; ++depth) {
		count += host[depth];
	}
	if (count != 0) {
		amrex::Print() << "Ghost failures: axis " << axis << " lower " << lower << " Mach " << mach << " mismatch " << mismatch << '\n';
	}
	return count;
}

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int failures = 0;
	{
		eos_rp::eos_gamma = 2.0;
		eos_init();
		for (double mach :
		     {0.99, 1.0 - 1.e-8, 1.0 - 1.e-12, std::nextafter(1.0, 0.0), 1.0, std::nextafter(1.0, 2.0), 1.0 + 1.e-12, 1.0 + 1.e-8, 1.01}) {
			failures += checkDerivatives<NSCBC::BoundarySide::Lower>(mach);
			failures += checkDerivatives<NSCBC::BoundarySide::Upper>(mach);
			for (double mismatch : {0.0, 1.e-6}) {
				failures += checkGhosts<FluxDir::X1, NSCBC::BoundarySide::Lower>(mach, mismatch);
				failures += checkGhosts<FluxDir::X1, NSCBC::BoundarySide::Upper>(mach, mismatch);
				failures += checkGhosts<FluxDir::X2, NSCBC::BoundarySide::Lower>(mach, mismatch);
				failures += checkGhosts<FluxDir::X2, NSCBC::BoundarySide::Upper>(mach, mismatch);
				failures += checkGhosts<FluxDir::X3, NSCBC::BoundarySide::Lower>(mach, mismatch);
				failures += checkGhosts<FluxDir::X3, NSCBC::BoundarySide::Upper>(mach, mismatch);
			}
		}
		amrex::Print() << "NSCBC sonic failures: " << failures << '\n';
	}
	amrex::Finalize();
	return failures == 0 ? 0 : 1;
}
