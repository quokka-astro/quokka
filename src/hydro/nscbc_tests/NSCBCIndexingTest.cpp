#include "AMReX_FArrayBox.H"
#include "AMReX_GpuContainers.H"
#include "hydro/NSCBC_outflow.hpp"

struct BoundaryProblem {};
template <> struct Physics_Traits<BoundaryProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = 1;
};
template <> struct quokka::EOS_Traits<BoundaryProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};
using Hydro = HydroSystem<BoundaryProblem>;
constexpr int nvar = Hydro::nvar_;

// Affine primitive fields give distinct, analytically known transverse derivatives.
AMREX_GPU_DEVICE auto primitive(double y, double z) -> quokka::valarray<amrex::Real, nvar>
{
	quokka::valarray<amrex::Real, nvar> q{};
	for (int n = 0; n < nvar; ++n) {
		q[n] = 10.0 + n + (n + 1) * y + (2 * n + 3) * z;
	}
	return q;
}

template <NSCBC::BoundarySide side> auto checkTransverse() -> int
{
	const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
	const amrex::RealBox physical({0., 0., 0.}, {8., 16., 32.});
	const amrex::Geometry geometry(domain, &physical, 0);
	const auto geom = geometry.data();
	amrex::FArrayBox fab(domain, nvar);
	const auto a = fab.array();
	amrex::ParallelFor(domain, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto u = Hydro::ComputeConsVars(primitive((j + 0.5) * 2.0, (k + 0.5) * 4.0));
		for (int n = 0; n < nvar; ++n) {
			a(i, j, k, n) = u[n];
		}
	});
	amrex::Gpu::DeviceScalar<int> failure(0);
	auto *failed = failure.dataPtr();
	amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) {
		const auto [dy, dz] = NSCBC::detail::transverse_xdir_dQ_data<BoundaryProblem, side>(amrex::IntVect(0, 3, 3), a, geom);
		for (int n = 0; n < nvar; ++n) {
			if (!std::isfinite(dy[n]) || !std::isfinite(dz[n]) || std::abs(dy[n] - (n + 1)) > 1.e-9 || std::abs(dz[n] - (2 * n + 3)) > 1.e-9) {
				*failed = 1;
			}
		}
	});
	const int result = failure.dataValue();
	if (result != 0) {
		amrex::Print() << "Transverse derivatives failed: lower " << (side == NSCBC::BoundarySide::Lower) << '\n';
	}
	return result;
}

template <FluxDir dir, NSCBC::BoundarySide side> auto checkReflection() -> int
{
	constexpr int axis = static_cast<int>(dir);
	constexpr bool lower = side == NSCBC::BoundarySide::Lower;
	const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
	const amrex::RealBox physical({0., 0., 0.}, {8., 8., 8.});
	const amrex::Geometry geometry(domain, &physical, 0);
	const auto geom = geometry.data();
	amrex::FArrayBox fab(amrex::grow(domain, 6), nvar);
	fab.setVal(-12345.0);
	const auto a = fab.array();
	amrex::ParallelFor(domain, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::IntVect index(i, j, k);
		auto q = primitive(index[axis] * 0.01, 0.0);
		q[axis + 1] = lower ? 1.0 : -1.0;
		const auto u = Hydro::ComputeConsVars(q);
		for (int n = 0; n < nvar; ++n) {
			a(i, j, k, n) = u[n];
		}
	});
	amrex::Gpu::DeviceVector<int> failures(6, 0);
	auto *failed = failures.data();
	amrex::ParallelFor(6, [=] AMREX_GPU_DEVICE(int depth) {
		amrex::IntVect ghost(3), mirror(3);
		ghost[axis] = lower ? -1 - depth : 8 + depth;
		mirror[axis] = lower ? depth : 7 - depth;
		NSCBC::setOutflowBoundaryLowOrder<BoundaryProblem, dir, side>(ghost, a, geom, 1.0);
		for (int n = 0; n < nvar; ++n) {
			const double expected = (n == axis + 1 ? -1.0 : 1.0) * a(mirror, n);
			const double actual = a(ghost, n);
			if (!std::isfinite(actual) || std::abs(actual - expected) > 1.e-9) {
				failed[depth] = 1;
			}
		}
	});
	amrex::Gpu::HostVector<int> host(6);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, failures.begin(), failures.end(), host.begin());
	int count = 0;
	for (int depth = 0; depth < 6; ++depth) {
		count += host[depth];
		if (host[depth] != 0) {
			amrex::Print() << "Reflection failed: axis " << axis << " lower " << lower << " depth " << depth + 1 << '\n';
		}
	}
	return count;
}

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int failures = 0;
	{
		eos_rp::eos_gamma = 1.4;
		eos_init();
		failures += checkTransverse<NSCBC::BoundarySide::Lower>();
		failures += checkTransverse<NSCBC::BoundarySide::Upper>();
		failures += checkReflection<FluxDir::X1, NSCBC::BoundarySide::Lower>();
		failures += checkReflection<FluxDir::X1, NSCBC::BoundarySide::Upper>();
		failures += checkReflection<FluxDir::X2, NSCBC::BoundarySide::Lower>();
		failures += checkReflection<FluxDir::X2, NSCBC::BoundarySide::Upper>();
		failures += checkReflection<FluxDir::X3, NSCBC::BoundarySide::Lower>();
		failures += checkReflection<FluxDir::X3, NSCBC::BoundarySide::Upper>();
		amrex::Print() << "NSCBC indexing failures: " << failures << '\n';
	}
	amrex::Finalize();
	return failures == 0 ? 0 : 1;
}
