/// \file testRootFinding.cpp
/// \brief Unit test of the bracketing root finders (Brent, ModAB, and TOMS 748 for reference) on host and device.
///

#include <array>
#include <cmath>
#include <format>
#include <iostream>

#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_REAL.H"

#include "math/bracketing_root_finding.hpp"
#include "math/root_finding.hpp"

using amrex::Real;

namespace
{

constexpr int nfunc = 11;
constexpr int nsolver = 3;
constexpr int max_iter_budget = 200;
constexpr std::array<const char *, nsolver> solver_names = {"Brent", "ModAB", "TOMS748"};

struct TestCase {
	const char *name;
	Real a;
	Real b;
	Real root;
};

// clang-format off
const std::array<TestCase, nfunc> cases = {{
    {"x^2 - 2",               0.0, 2.0, 1.4142135623730951},
    {"x^3 - 2x - 5",          2.0, 3.0, 2.0945514815423265},
    {"cos(x) - x",            0.0, 1.0, 0.7390851332151607},
    {"exp(x) - 10",           0.0, 5.0, 2.302585092994046},
    {"x^10 - 1",              0.0, 1.3, 1.0},
    {"(x - 1)^3",             0.0, 3.0, 1.0},
    {"tanh(1000 (x - 0.3))",  0.0, 1.0, 0.3},
    {"step at 0.3",           0.0, 1.0, 0.3},
    {"log(x), reversed",      5.0, 0.5, 1.0},
    {"x, root at left end",   0.0, 1.0, 0.0},
    {"1e-10 (x - 1e5)",       0.0, 1.0e10, 1.0e5},
}};
// clang-format on

struct TestFunction {
	int id;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(Real x) const -> Real
	{
		switch (id) {
			case 0:
				return x * x - 2.0;
			case 1:
				return x * x * x - 2.0 * x - 5.0;
			case 2:
				return std::cos(x) - x;
			case 3:
				return std::exp(x) - 10.0;
			case 4:
				return std::pow(x, 10) - 1.0;
			case 5:
				return (x - 1.0) * (x - 1.0) * (x - 1.0);
			case 6:
				return std::tanh(1000.0 * (x - 0.3));
			case 7:
				return (x < 0.3) ? -1.0 : 1.0;
			case 8:
				return std::log(x);
			case 9:
				return x;
			default:
				return 1.0e-10 * (x - 1.0e5);
		}
	}
};

struct Result {
	Real root;
	int iter;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto solve(int solver, int func, Real a, Real b) -> Result
{
	const TestFunction f{func};
	const quokka::math::eps_tolerance<Real> tol{};
	int iter = max_iter_budget;
	std::pair<Real, Real> r{};
	if (solver == 0) {
		r = quokka::math::brent_solve(f, a, b, tol, iter);
	} else if (solver == 1) {
		r = quokka::math::modab_solve(f, a, b, tol, iter);
	} else {
		// toms748_solve requires a < b
		const Real lo = amrex::min(a, b);
		const Real hi = amrex::max(a, b);
		r = quokka::math::toms748_solve(f, lo, hi, tol, iter);
	}
	return Result{r.second / 2 + r.first / 2, iter};
}

} // namespace

auto problem_main() -> int
{
	constexpr int ntest = nfunc * nsolver;

	// host
	std::array<Result, ntest> host{};
	for (int n = 0; n < ntest; ++n) {
		host[n] = solve(n % nsolver, n / nsolver, cases[n / nsolver].a, cases[n / nsolver].b);
	}

	// device
	std::array<Real, nfunc> a_h{};
	std::array<Real, nfunc> b_h{};
	for (int i = 0; i < nfunc; ++i) {
		a_h[i] = cases[i].a;
		b_h[i] = cases[i].b;
	}
	amrex::Gpu::DeviceVector<Real> a_d(nfunc);
	amrex::Gpu::DeviceVector<Real> b_d(nfunc);
	amrex::Gpu::DeviceVector<Result> dev_d(ntest);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, a_h.begin(), a_h.end(), a_d.begin());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, b_h.begin(), b_h.end(), b_d.begin());
	const Real *a_ptr = a_d.data();
	const Real *b_ptr = b_d.data();
	Result *dev_ptr = dev_d.data();
	amrex::ParallelFor(ntest, [=] AMREX_GPU_DEVICE(int n) noexcept {
		const int func = n / nsolver;
		dev_ptr[n] = solve(n % nsolver, func, a_ptr[func], b_ptr[func]);
	});
	std::array<Result, ntest> dev{};
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, dev_d.begin(), dev_d.end(), dev.begin());

	// check: the bracket midpoint must match the exact root to a few ulp of the scale, within the iteration budget
	int status = 0;
	std::cout << std::format("{:<24}{:<9}{:>6}{:>6}{:>12}{:>12}\n", "function", "solver", "iter", "(dev)", "rel err", "(dev)");
	for (int n = 0; n < ntest; ++n) {
		const TestCase &tc = cases[n / nsolver];
		const Real scale = amrex::max(std::abs(tc.root), Real(1.0e-300));
		const Real reltol = 1.0e-13;
		const Real err_h = std::abs(host[n].root - tc.root) / (tc.root == 0 ? 1.0 : scale);
		const Real err_d = std::abs(dev[n].root - tc.root) / (tc.root == 0 ? 1.0 : scale);
		const bool ok = (err_h <= reltol) && (err_d <= reltol) && (host[n].iter < max_iter_budget) && (dev[n].iter < max_iter_budget);
		std::cout << std::format("{:<24}{:<9}{:>6}{:>6}{:>12.2e}{:>12.2e}{}\n", tc.name, solver_names[n % nsolver], host[n].iter, dev[n].iter, err_h,
					 err_d, ok ? "" : "  FAIL");
		if (!ok) {
			status = 1;
		}
	}

	std::cout << (status == 0 ? "RootFinding: all tests passed.\n" : "RootFinding: FAILED.\n");
	return status;
}
