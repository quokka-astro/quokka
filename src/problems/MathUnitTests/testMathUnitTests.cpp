//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMathUnitTests.cpp
/// \brief Unit tests of the math utilities: ODE integration and bracketing root finding (host and device).
///

#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "eos.H"
#include "extern_parameters.H"

#include "math/ODEIntegrate.hpp"
#include "math/bracketing_root_finding.hpp"
#include "math/root_finding.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/valarray.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>

struct ODETest {};

constexpr double seconds_in_year = 3.154e7;

// function definitions

using amrex::Real;

constexpr double Tgas0 = 6000.;			  // K
constexpr double rho0 = 0.01 * (C::m_p + C::m_e); // g cm^-3

template <> struct quokka::EOS_Traits<ODETest> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};

struct ODEUserData {
	amrex::Real rho;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cooling_function(Real const rho, Real const T) -> Real
{
	// use fitting function from Koyama & Inutsuka (2002)
	Real const gamma_heat = 2.0e-26;
	Real const lambda_cool = gamma_heat * (1.0e7 * std::exp(-114800. / (T + 1000.)) + 14. * std::sqrt(T) * std::exp(-92. / T));
	Real const rho_over_mh = rho / (C::m_p + C::m_e);
	Real const cooling_source_term = rho_over_mh * gamma_heat - (rho_over_mh * rho_over_mh) * lambda_cool;
	return cooling_source_term;
}

struct ODECoolingFunctor {
	Real rho;

	AMREX_GPU_HOST_DEVICE explicit ODECoolingFunctor(Real rho_in) : rho(rho_in) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs) const -> int
	{
		// compute temperature
		Real const Eint = y_data[0];
		Real const T = quokka::EOS<ODETest>::ComputeTgasFromEint(rho, Eint);

		// compute cooling function
		y_rhs[0] = cooling_function(rho, T);
		return 0;
	}
};

namespace
{

auto TestODEIntegration() -> int
{
	// initialize EOS
	init_extern_parameters();
	Real small_temp = 1e-10;
	Real small_dens = 1e-100;
	eos_init(small_temp, small_dens);

	// set up initial conditions
	const Real Eint0 = quokka::EOS<ODETest>::ComputeEintFromTgas(rho0, Tgas0);
	const Real Edot0 = cooling_function(rho0, Tgas0);
	const Real tcool = std::abs(Eint0 / Edot0);
	const Real max_time = 10.0 * tcool;

	std::cout << "Initial temperature: " << Tgas0 << '\n';
	std::cout << "Initial cooling time: " << tcool / seconds_in_year << '\n';
	std::cout << "Initial edot = " << Edot0 << '\n';

	// solve cooling
	ODECoolingFunctor const coolingFunctor(rho0);
	quokka::valarray<Real, 1> y = {Eint0};
	quokka::valarray<Real, 1> const abstol = 1.0e-20 * y;
	const Real rtol = 1.0e-4; // appropriate for RK12
	int steps_taken = 0;
	rk_adaptive_integrate(coolingFunctor, 0, y, max_time, rtol, abstol, steps_taken);

	const Real Tgas = quokka::EOS<ODETest>::ComputeTgasFromEint(rho0, y[0]);
	// for n_H = 0.01 cm^{-3} (for IK cooling function)
	const Real Teq = 160.52611612610758;
	const Real Terr_rel = std::abs(Tgas - Teq) / Teq;
	const Real reltol = 1.0e-4; // relative error tolerance

	std::cout << "Final temperature: " << Tgas << '\n';
	std::cout << "Relative error: " << Terr_rel << '\n';

	// Cleanup and exit
	int status = 0;
	if ((Terr_rel > reltol) || (std::isnan(Terr_rel))) {
		status = 1;
	}
	return status;
}

constexpr int nfunc = 12;
constexpr int nsolver = 3;
constexpr int max_iter_budget = 200;
constexpr int max_iter_smooth = 20; // bound for superlinearly convergent cases
constexpr std::array<const char *, nsolver> solver_names = {"Brent", "ModAB", "TOMS748"};

struct TestCase {
	const char *name;
	Real a;
	Real b;
	Real root;
	bool smooth; // simple root of a smooth function: superlinear convergence expected
};

// clang-format off
const std::array<TestCase, nfunc> cases = {{
    {.name = "x^2 - 2", .a = 0.0, .b = 2.0, .root = std::numbers::sqrt2, .smooth = true},
    {.name = "x^3 - 2x - 5", .a = 2.0, .b = 3.0, .root = 2.0945514815423265, .smooth = true},
    {.name = "cos(x) - x", .a = 0.0, .b = 1.0, .root = 0.7390851332151607, .smooth = true},
    {.name = "exp(x) - 10", .a = 0.0, .b = 5.0, .root = std::numbers::ln10, .smooth = true},
    {.name = "x^10 - 1", .a = 0.0, .b = 1.3, .root = 1.0, .smooth = true},
    {.name = "(x - 1)^3", .a = 0.0, .b = 3.0, .root = 1.0, .smooth = false},
    {.name = "tanh(1000 (x - 0.3))", .a = 0.0, .b = 1.0, .root = 0.3, .smooth = false},
    {.name = "step at 0.3", .a = 0.0, .b = 1.0, .root = 0.3, .smooth = false},
    {.name = "log(x), reversed", .a = 5.0, .b = 0.5, .root = 1.0, .smooth = true},
    {.name = "x, root at left end", .a = 0.0, .b = 1.0, .root = 0.0, .smooth = true},
    {.name = "1e-10 (x - 1e5)", .a = 0.0, .b = 1.0e10, .root = 1.0e5, .smooth = true},
    {.name = "1e300 (x^3 - 0.027)", .a = 0.0, .b = 1.0, .root = 0.3, .smooth = false}, // interpolation overflows
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
			case 10:
				return 1.0e-10 * (x - 1.0e5);
			default:
				return 1.0e300 * (x * x * x - 0.027);
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
	return Result{.root = r.second / 2 + r.first / 2, .iter = iter};
}

auto TestRootFinding() -> int
{
	constexpr int ntest = nfunc * nsolver;

	// The overflow test case raises FE_OVERFLOW/FE_INVALID on purpose, so suspend the FP-exception traps
	// that the test suite enables (amrex.fpe_trap_*) while the solvers run on the host.
	std::fenv_t fenv{};
	std::feholdexcept(&fenv);

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
	std::fesetenv(&fenv);

	// check: the bracket midpoint must match the exact root to a few ulp of the scale, and smooth simple roots must converge in few iterations
	int status = 0;
	std::cout << std::format("{:<24}{:<9}{:>6}{:>6}{:>12}{:>12}\n", "function", "solver", "iter", "(dev)", "rel err", "(dev)");
	for (int n = 0; n < ntest; ++n) {
		const TestCase &tc = cases[n / nsolver];
		const Real scale = (tc.root == 0) ? 1.0 : std::abs(tc.root);
		const Real reltol = 1.0e-13;
		const Real err_h = std::abs(host[n].root - tc.root) / scale;
		const Real err_d = std::abs(dev[n].root - tc.root) / scale;
		const int iter_limit = tc.smooth ? max_iter_smooth : max_iter_budget - 1;
		const bool ok = (err_h <= reltol) && (err_d <= reltol) && (host[n].iter <= iter_limit) && (dev[n].iter <= iter_limit);
		std::cout << std::format("{:<24}{:<9}{:>6}{:>6}{:>12.2e}{:>12.2e}{}\n", tc.name, solver_names[n % nsolver], host[n].iter, dev[n].iter, err_h,
					 err_d, ok ? "" : "  FAIL");
		if (!ok) {
			status = 1;
		}
	}

	std::cout << (status == 0 ? "Root finding: all tests passed.\n" : "Root finding: FAILED.\n");
	return status;
}

} // namespace

auto problem_main() -> int
{
	const int ode_status = TestODEIntegration();
	const int root_status = TestRootFinding();
	if (ode_status != 0) {
		std::cout << "MathUnitTests: ODE integration test FAILED.\n";
	}
	if (root_status != 0) {
		std::cout << "MathUnitTests: root finding test FAILED.\n";
	}
	return ((ode_status != 0) || (root_status != 0)) ? 1 : 0;
}
