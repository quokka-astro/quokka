// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testMathUnitTests.cpp
/// \brief Unit tests of the math utilities (ODE integration and bracketing root finding) and of the gamma-law EOS, on host and device.
///

#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"

#include "math/ODEIntegrate.hpp"
#include "math/bracketing_root_finding.hpp"
#include "math/linear_algebra.hpp"
#include "math/root_finding.hpp"
#include "math/rosenbrock/rosenbrock.hpp"
#include "math/rosenbrock/tableaus.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/valarray.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
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

// gas used by TestEOSIdeal
struct EOSTestGas {};
template <> struct quokka::EOS_Traits<EOSTestGas> {
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double gamma = 1.4; // not 5/3, so the test fails if gamma comes from anywhere but EOS_Traits
};

// isothermal gas used by TestEOSIdeal
struct EOSTestIso {};
template <> struct quokka::EOS_Traits<EOSTestIso> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 2.0e5;
	static constexpr double mean_molecular_weight = NAN;
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

auto TestEOSIdeal() -> int
{
	using EOS = quokka::EOS<EOSTestGas>;
	constexpr Real gamma = quokka::EOS_Traits<EOSTestGas>::gamma;
	constexpr Real mbar = quokka::EOS_Traits<EOSTestGas>::mean_molecular_weight;
	constexpr Real rho = 2.3e-24;
	constexpr Real T = 8000.;
	constexpr Real Eint = rho * C::k_B * T / ((gamma - 1.0) * mbar);
	constexpr Real P = (gamma - 1.0) * Eint;
	constexpr Real Tfloor = 1.0e-10; // Microphysics eos() resets e < 1e-200 to this temperature and raises any lower T to it
	constexpr Real rho_floor = 1.0e-100;
	constexpr Real reltol = 1.0e-14;

	int status = 0;
	const auto check = [&status](const char *name, Real value, Real expected) {
		const Real err = std::abs(value - expected) / std::abs(expected);
		const bool ok = err <= reltol;
		std::cout << std::format("{:<36}{:>24.16e}{:>24.16e}{:>12.2e}{}\n", name, value, expected, err, ok ? "" : "  FAIL");
		if (!ok) {
			status = 1;
		}
	};

	// regular state
	check("Tgas(rho, Eint)", EOS::ComputeTgasFromEint(rho, Eint), (gamma - 1.0) * mbar * Eint / (rho * C::k_B));
	check("Eint(rho, T)", EOS::ComputeEintFromTgas(rho, T), rho * C::k_B * T / ((gamma - 1.0) * mbar));
	check("Eint(rho, P)", EOS::ComputeEintFromPres(rho, P), P / (gamma - 1.0));
	check("dEint/dT(rho, T)", EOS::ComputeEintTempDerivative(rho, T), rho * C::k_B / ((gamma - 1.0) * mbar));
	check("P(rho, Eint)", EOS::ComputePressure(rho, Eint), (gamma - 1.0) * Eint);
	check("cs(rho, P)", EOS::ComputeSoundSpeed(rho, P), std::sqrt(gamma * P / rho));
	const auto [deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G] = EOS::ComputeOtherDerivatives(rho, P);
	if (deint_dRho != 0.0) {
		std::cout << "deint_dRho = " << deint_dRho << " (expected 0)  FAIL\n";
		status = 1;
	}
	check("deint_dP(rho, P)", deint_dP, 1.0 / ((gamma - 1.0) * rho));
	check("dRho_dP(rho, P)", dRho_dP, rho / P);
	check("dP_dRho_s(rho, P)", dP_dRho_s, gamma * P / rho);
	check("G(rho, P)", G, 0.5 * (1.0 + gamma));

	// floors
	check("Tgas(rho, 0)", EOS::ComputeTgasFromEint(rho, 0.0), Tfloor);
	check("Tgas(rho, -1e-12)", EOS::ComputeTgasFromEint(rho, -1.0e-12), Tfloor);
	check("Eint(rho, 0)", EOS::ComputeEintFromTgas(rho, 0.0), rho * C::k_B * Tfloor / ((gamma - 1.0) * mbar));
	check("P(rho, 0)", EOS::ComputePressure(rho, 0.0), rho * C::k_B * Tfloor / mbar);
	check("P(0, Eint)", EOS::ComputePressure(0.0, Eint), rho_floor * C::k_B * Tfloor / mbar);

	// NaN rho or T is raised to the floor, as Microphysics's eos() did (max(floor, NaN) = floor).
	// Comparisons with NaN may raise FE_INVALID, so suspend the FP-exception traps the test suite can enable.
	// The volatile reads and writes keep the NaN operations between feholdexcept and fesetenv.
	std::fenv_t fenv{};
	std::feholdexcept(&fenv);
	const volatile Real nan = std::numeric_limits<Real>::quiet_NaN();
	const volatile Real Eint_nanT = EOS::ComputeEintFromTgas(rho, nan);
	const volatile Real cs_nanrho = EOS::ComputeSoundSpeed(nan, P);
	std::fesetenv(&fenv);
	check("Eint(rho, NaN)", Eint_nanT, rho * C::k_B * Tfloor / ((gamma - 1.0) * mbar));
	check("cs(NaN, P)", cs_nanrho, std::sqrt(gamma * P / rho_floor));

	// isothermal
	check("P_iso(rho, Eint)", quokka::EOS<EOSTestIso>::ComputePressure(rho, Eint), rho * 2.0e5 * 2.0e5);

	// device: the floors must hold in device code too
	amrex::Gpu::DeviceVector<Real> T_d(2);
	Real *const T_ptr = T_d.data();
	amrex::ParallelFor(2, [=] AMREX_GPU_DEVICE(int n) noexcept { T_ptr[n] = EOS::ComputeTgasFromEint(rho, (n == 0) ? Eint : 0.0); });
	std::array<Real, 2> T_h{};
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, T_d.begin(), T_d.end(), T_h.begin());
	check("Tgas(rho, Eint) (device)", T_h[0], (gamma - 1.0) * mbar * Eint / (rho * C::k_B));
	check("Tgas(rho, 0) (device)", T_h[1], Tfloor);

	std::cout << (status == 0 ? "EOSIdeal: all tests passed.\n" : "EOSIdeal: FAILED.\n");
	return status;
}

/// y0' = -y0, y1' = L (y0 - y1), with L = 1e4: stiff, and with an exact solution. The integrator matrix
/// [[fac + 1, 0], [-L, fac + L]] needs a row swap whenever fac + 1 < L, i.e. on every step longer than about 1/L.
struct StiffLinearSystem {
	static constexpr int neqs = 2;
	static constexpr Real L = 1.0e4;

	AMREX_GPU_HOST_DEVICE static constexpr auto counts_toward_error(int /*n*/) -> bool { return true; }

	AMREX_GPU_HOST_DEVICE void rhs(Real /*t*/, amrex::GpuArray<Real, neqs> &y, amrex::GpuArray<Real, neqs> &ydot) const
	{
		ydot[0] = -y[0];
		ydot[1] = L * (y[0] - y[1]);
	}

	template <typename Matrix> AMREX_GPU_HOST_DEVICE void jac(Real /*t*/, amrex::GpuArray<Real, neqs> & /*y*/, Matrix &J) const
	{
		J(0, 0) = -1.0;
		J(0, 1) = 0.0;
		J(1, 0) = L;
		J(1, 1) = -L;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto rtol(int /*n*/) const -> Real { return 1.0e-8; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto atol(int /*n*/) const -> Real { return 1.0e-14; }

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto valid_state(const amrex::GpuArray<Real, neqs> &y) const -> bool
	{
		return std::isfinite(y[0]) && std::isfinite(y[1]);
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto valid_update(const amrex::GpuArray<Real, neqs> & /*y*/, const amrex::GpuArray<Real, neqs> & /*ynew*/) const
	    -> bool
	{
		return true;
	}
};

/// StiffLinearSystem with a NaN in its analytic Jacobian, in an entry that is zero.
struct NaNJacobianSystem : StiffLinearSystem {
	template <typename Matrix> AMREX_GPU_HOST_DEVICE void jac(Real t, amrex::GpuArray<Real, neqs> &y, Matrix &J) const
	{
		StiffLinearSystem::jac(t, y, J);
		J(0, 1) = std::numeric_limits<Real>::quiet_NaN();
	}
};

template <bool pivot> struct RosenbrockTestOptions : quokka::math::rosenbrock::DefaultOptions {
	static constexpr bool pivoting = pivot;
};

template <bool analytic, bool fallback> struct JacobianTestOptions : quokka::math::rosenbrock::DefaultOptions {
	static constexpr bool analytic_jacobian = analytic;
	static constexpr bool numerical_jacobian_fallback = fallback;
};

/// Integrates System from t = 0 to 1 on the device with y = (1, 0) at t = 0. Returns the solution, the status and the
/// number of steps.
template <typename Tableau, typename Options, typename System> auto RunRosenbrock() -> std::array<Real, 4>
{
	namespace rb = quokka::math::rosenbrock;
	constexpr int nout = 4;
	amrex::Gpu::DeviceVector<Real> out_d(nout);
	Real *out = out_d.data();
	amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*i*/) noexcept {
		System sys{};
		rb::State<System::neqs, Tableau::stages> rstate{};
		rstate.t = 0.0;
		rstate.tout = 1.0;
		rstate.y[0] = 1.0;
		rstate.y[1] = 0.0;
		const int status = rb::integrate<Tableau, Options>(sys, rstate, 100000);
		out[0] = rstate.y[0];
		out[1] = rstate.y[1];
		out[2] = status;
		out[3] = rstate.n_step;
	});
	std::array<Real, nout> out_h{};
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, out_d.begin(), out_d.end(), out_h.begin());
	return out_h;
}

/// The largest relative error of a solution of StiffLinearSystem at t = 1.
auto StiffLinearSystemError(const std::array<Real, 4> &out) -> Real
{
	constexpr Real L = StiffLinearSystem::L;
	const Real y0_exact = std::exp(-1.0);
	const Real y1_exact = L / (L - 1.0) * (std::exp(-1.0) - std::exp(-L));
	return std::max(std::abs(out[0] / y0_exact - 1.0), std::abs(out[1] / y1_exact - 1.0));
}

/// Integrates StiffLinearSystem from t = 0 to 1 on the device and compares with the exact solution.
template <typename Tableau, bool pivot> auto TestRosenbrockTableau() -> int
{
	namespace rb = quokka::math::rosenbrock;
	const auto out = RunRosenbrock<Tableau, RosenbrockTestOptions<pivot>, StiffLinearSystem>();
	const Real err = StiffLinearSystemError(out);
	constexpr Real reltol = 1.0e-6;
	const bool ok = static_cast<int>(out[2]) == rb::IERR_SUCCESS && err < reltol;
	std::cout << std::format("Rosenbrock {:8} pivoting {:3}: status {}, {} steps, relative error {:.3e} {}\n", Tableau::name, pivot ? "on" : "off",
				 static_cast<int>(out[2]), static_cast<int>(out[3]), err, ok ? "" : "FAILED");
	return ok ? 0 : 1;
}

/// A non-finite analytic Jacobian must fail the burn without numerical_jacobian_fallback, and be replaced by the
/// finite-difference Jacobian with it. The finite-difference Jacobian alone must also give an accurate solution.
auto TestJacobianOptions() -> int
{
	namespace rb = quokka::math::rosenbrock;
	constexpr Real reltol = 1.0e-6;
	int status = 0;

	const auto no_fallback = RunRosenbrock<rb::Rodas4P, JacobianTestOptions<true, false>, NaNJacobianSystem>();
	const bool ok_no_fallback = static_cast<int>(no_fallback[2]) == rb::IERR_JACOBIAN_NOT_FINITE;
	std::cout << std::format("Rosenbrock NaN Jacobian, no fallback: status {} {}\n", static_cast<int>(no_fallback[2]), ok_no_fallback ? "" : "FAILED");
	status += ok_no_fallback ? 0 : 1;

	const auto fallback = RunRosenbrock<rb::Rodas4P, JacobianTestOptions<true, true>, NaNJacobianSystem>();
	const Real err_fallback = StiffLinearSystemError(fallback);
	const bool ok_fallback = static_cast<int>(fallback[2]) == rb::IERR_SUCCESS && err_fallback < reltol;
	std::cout << std::format("Rosenbrock NaN Jacobian, fallback: status {}, {} steps, relative error {:.3e} {}\n", static_cast<int>(fallback[2]),
				 static_cast<int>(fallback[3]), err_fallback, ok_fallback ? "" : "FAILED");
	status += ok_fallback ? 0 : 1;

	const auto numerical = RunRosenbrock<rb::Rodas4P, JacobianTestOptions<false, false>, StiffLinearSystem>();
	const Real err_numerical = StiffLinearSystemError(numerical);
	const bool ok_numerical = static_cast<int>(numerical[2]) == rb::IERR_SUCCESS && err_numerical < reltol;
	std::cout << std::format("Rosenbrock numerical Jacobian: status {}, {} steps, relative error {:.3e} {}\n", static_cast<int>(numerical[2]),
				 static_cast<int>(numerical[3]), err_numerical, ok_numerical ? "" : "FAILED");
	status += ok_numerical ? 0 : 1;

	return status;
}

/// Solves A x = b on the device by dgefa and dgesl, for an n x n matrix with entries over twelve decades and a diagonal
/// too small to be the pivot, so that most steps interchange rows, and b = A (1, 2, ..., n). The factors are found with
/// and without pivoting. Returns the backward error of the pivoted solution, max |A x - b| / (max |A| max |x|) (near
/// machine epsilon for a stable factorization, however ill-conditioned A is), its info, and the info without pivoting.
template <int n> auto RunLinearAlgebra(const bool zero_first_pivot) -> std::array<Real, 3>
{
	constexpr int nout = 3;
	amrex::Gpu::DeviceVector<Real> out_d(nout);
	Real *out = out_d.data();
	amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*i*/) noexcept {
		using Matrix = amrex::Array2D<Real, 0, n - 1, 0, n - 1>;
		Matrix a{};
		amrex::GpuArray<Real, n> b{};
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				// a deterministic spread of signs and magnitudes, 10^-6 to 10^6
				const int h = (7 * i + 13 * j + 3 * i * j) % 25;
				a(i, j) = ((i + 2 * j) % 3 == 0 ? -1.0 : 1.0) * std::pow(10.0, 0.5 * (h - 12));
			}
		}
		if (zero_first_pivot) {
			a(0, 0) = 0.0;
		}
		for (int i = 0; i < n; ++i) {
			b[i] = 0.0;
			for (int j = 0; j < n; ++j) {
				b[i] += a(i, j) * static_cast<Real>(j + 1);
			}
		}

		Matrix lu = a;
		amrex::GpuArray<Real, n> x = b;
		amrex::GpuArray<std::int16_t, n> pivot{};
		int info = 0;
		quokka::math::dgefa<n, true>(lu, pivot, info);
		quokka::math::dgesl<n, true>(lu, pivot, x);

		Real amax = 0.0;
		Real xmax = 0.0;
		Real rmax = 0.0;
		for (int i = 0; i < n; ++i) {
			Real r = -b[i];
			for (int j = 0; j < n; ++j) {
				r += a(i, j) * x[j];
				amax = amrex::max(amax, std::abs(a(i, j)));
			}
			rmax = amrex::max(rmax, std::abs(r));
			xmax = amrex::max(xmax, std::abs(x[i]));
		}

		Matrix lu_nopivot = a;
		int info_nopivot = 0;
		quokka::math::dgefa<n, false>(lu_nopivot, pivot, info_nopivot);

		out[0] = rmax / (amax * xmax);
		out[1] = info;
		out[2] = info_nopivot;
	});
	std::array<Real, nout> out_h{};
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, out_d.begin(), out_d.end(), out_h.begin());
	return out_h;
}

/// dgefa and dgesl at the sizes of the networks (5-9 equations for the DType networks, 15 for primordial_chem, 23 for
/// GOW): with pivoting the backward error must be small, also with a zero first pivot; without pivoting, the zero
/// first pivot must be reported.
template <int n> auto TestLinearAlgebraSize() -> int
{
	constexpr Real tol = 1.0e-13;
	const auto out = RunLinearAlgebra<n>(false);
	const auto zero = RunLinearAlgebra<n>(true);
	const bool ok = out[0] < tol && static_cast<int>(out[1]) == 0 && zero[0] < tol && static_cast<int>(zero[1]) == 0 && static_cast<int>(zero[2]) == 1;
	std::cout << std::format("LU n = {:2}: backward error {:.3e}; zero first pivot: backward error {:.3e}, info without pivoting {} {}\n", n, out[0],
				 zero[0], static_cast<int>(zero[2]), ok ? "" : "FAILED");
	return ok ? 0 : 1;
}

/// dgefa on a matrix with a NaN and on one with an infinity, each in an entry that the elimination would not otherwise
/// read last: both must be reported with info = -1, with and without pivoting.
auto TestLinearAlgebraNonFinite() -> int
{
	constexpr int n = 4;
	constexpr int nout = 4;
	amrex::Gpu::DeviceVector<int> out_d(nout);
	int *out = out_d.data();
	amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*i*/) noexcept {
		using Matrix = amrex::Array2D<Real, 0, n - 1, 0, n - 1>;
		Matrix a{};
		for (int i = 0; i < n; ++i) {
			a(i, i) = 2.0;
		}
		amrex::GpuArray<std::int16_t, n> pivot{};
		Matrix a_nan = a;
		a_nan(0, n - 1) = std::numeric_limits<Real>::quiet_NaN();
		Matrix a_inf = a;
		a_inf(n - 1, 0) = std::numeric_limits<Real>::infinity();
		Matrix m = a_nan;
		quokka::math::dgefa<n, true>(m, pivot, out[0]);
		m = a_nan;
		quokka::math::dgefa<n, false>(m, pivot, out[1]);
		m = a_inf;
		quokka::math::dgefa<n, true>(m, pivot, out[2]);
		m = a_inf;
		quokka::math::dgefa<n, false>(m, pivot, out[3]);
	});
	std::array<int, nout> out_h{};
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, out_d.begin(), out_d.end(), out_h.begin());
	const bool ok = out_h[0] == -1 && out_h[1] == -1 && out_h[2] == -1 && out_h[3] == -1;
	std::cout << std::format("LU non-finite entry: info {} {} (NaN, pivoting on/off), {} {} (infinity, pivoting on/off) {}\n", out_h[0], out_h[1], out_h[2],
				 out_h[3], ok ? "" : "FAILED");
	return ok ? 0 : 1;
}

auto TestLinearAlgebra() -> int
{
	int status = 0;
	status += TestLinearAlgebraSize<3>();
	status += TestLinearAlgebraSize<8>();
	status += TestLinearAlgebraSize<15>();
	status += TestLinearAlgebraSize<23>();
	status += TestLinearAlgebraNonFinite();
	return status;
}

auto TestRosenbrock() -> int
{
	namespace rb = quokka::math::rosenbrock;
	int status = TestLinearAlgebra();
	status += TestRosenbrockTableau<rb::ROS2S, true>();
	status += TestRosenbrockTableau<rb::ROS2S, false>();
	status += TestRosenbrockTableau<rb::Rodas3P, true>();
	status += TestRosenbrockTableau<rb::Rodas3P, false>();
	status += TestRosenbrockTableau<rb::Rodas4P, true>();
	status += TestRosenbrockTableau<rb::Rodas4P, false>();
	status += TestRosenbrockTableau<rb::Rodas5P, true>();
	status += TestRosenbrockTableau<rb::Rodas5P, false>();
	status += TestJacobianOptions();
	std::cout << (status == 0 ? "Rosenbrock: all tests passed.\n" : "Rosenbrock: FAILED.\n");
	return status;
}

} // namespace

auto problem_main() -> int
{
	const int ode_status = TestODEIntegration();
	const int root_status = TestRootFinding();
	const int eos_status = TestEOSIdeal();
	const int rosenbrock_status = TestRosenbrock();
	if (ode_status != 0) {
		std::cout << "MathUnitTests: ODE integration test FAILED.\n";
	}
	if (root_status != 0) {
		std::cout << "MathUnitTests: root finding test FAILED.\n";
	}
	if (eos_status != 0) {
		std::cout << "MathUnitTests: EOS test FAILED.\n";
	}
	if (rosenbrock_status != 0) {
		std::cout << "MathUnitTests: Rosenbrock test FAILED.\n";
	}
	return ((ode_status != 0) || (root_status != 0) || (eos_status != 0) || (rosenbrock_status != 0)) ? 1 : 0;
}
