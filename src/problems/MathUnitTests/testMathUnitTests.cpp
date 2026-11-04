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

#include "math/ODEIntegrate.hpp"
#include "math/bracketing_root_finding.hpp"
#include "math/root_finding.hpp"
#include "radiation/radiation_system.hpp"
#include "rosenbrock/rosenbrock.hpp"
#include "util/BC.hpp"
#include "util/valarray.hpp"
#include <array>
#include <cfenv>
#include <cmath>
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
	constexpr Real Tfloor = 1.0e-10; // Microphysics eos() resets e <= 0 (and T <= 0) to this temperature
	constexpr Real rho_floor = 1.0e-100;
	constexpr Real reltol = 1.0e-14;

	int status = 0;
	auto check = [&status](const char *name, Real value, Real expected) {
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
	std::fenv_t fenv{};
	std::feholdexcept(&fenv);
	const Real nan = std::numeric_limits<Real>::quiet_NaN();
	const Real Eint_nanT = EOS::ComputeEintFromTgas(rho, nan);
	const Real cs_nanrho = EOS::ComputeSoundSpeed(nan, P);
	std::fesetenv(&fenv);
	check("Eint(rho, NaN)", Eint_nanT, rho * C::k_B * Tfloor / ((gamma - 1.0) * mbar));
	check("cs(NaN, P)", cs_nanrho, std::sqrt(gamma * P / rho_floor));

	// isothermal
	check("P_iso(rho, Eint)", quokka::EOS<EOSTestIso>::ComputePressure(rho, Eint), rho * 2.0e5 * 2.0e5);

	// device: the floors must hold in device code too
	amrex::Gpu::DeviceVector<Real> T_d(2);
	Real *T_ptr = T_d.data();
	amrex::ParallelFor(2, [=] AMREX_GPU_DEVICE(int n) noexcept { T_ptr[n] = EOS::ComputeTgasFromEint(rho, (n == 0) ? Eint : 0.0); });
	std::array<Real, 2> T_h{};
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, T_d.begin(), T_d.end(), T_h.begin());
	check("Tgas(rho, Eint) (device)", T_h[0], (gamma - 1.0) * mbar * Eint / (rho * C::k_B));
	check("Tgas(rho, 0) (device)", T_h[1], Tfloor);

	std::cout << (status == 0 ? "EOSIdeal: all tests passed.\n" : "EOSIdeal: FAILED.\n");
	return status;
}

// ---- Rosenbrock solver tests ----

// y' = -k y; optionally the Jacobian is non-finite on its first evaluation
struct DecaySystem {
	Real k = 1.0e3;
	bool nan_jacobian_once = false;
	int jac_calls = 0;
	AMREX_GPU_HOST_DEVICE void rhs(Real /*t*/, rosenbrock::Vector<1> &y, rosenbrock::Vector<1> &ydot) const { ydot[0] = -k * y[0]; }
	AMREX_GPU_HOST_DEVICE void jac(Real /*t*/, rosenbrock::Vector<1> const & /*y*/, rosenbrock::Matrix<1> &J)
	{
		J(0, 0) = (nan_jacobian_once && jac_calls == 0) ? std::numeric_limits<Real>::quiet_NaN() : -k;
		++jac_calls;
	}
};

// Robertson's stiff chemical kinetics problem
struct RobertsonSystem {
	static constexpr Real k1 = 0.04;
	static constexpr Real k2 = 3.0e7;
	static constexpr Real k3 = 1.0e4;
	AMREX_GPU_HOST_DEVICE void rhs(Real /*t*/, rosenbrock::Vector<3> &y, rosenbrock::Vector<3> &ydot) const
	{
		ydot[0] = -k1 * y[0] + k3 * y[1] * y[2];
		ydot[2] = k2 * y[1] * y[1];
		ydot[1] = -ydot[0] - ydot[2];
	}
	AMREX_GPU_HOST_DEVICE void jac(Real /*t*/, rosenbrock::Vector<3> const &y, rosenbrock::Matrix<3> &J) const
	{
		J(0, 0) = -k1;
		J(0, 1) = k3 * y[2];
		J(0, 2) = k3 * y[1];
		J(2, 0) = 0.0;
		J(2, 1) = 2.0 * k2 * y[1];
		J(2, 2) = 0.0;
		J(1, 0) = k1;
		J(1, 1) = -k3 * y[2] - 2.0 * k2 * y[1];
		J(1, 2) = -k3 * y[1];
	}
};

// rhs raises y[0] to at least 0 in place, as the photochemistry driver cleans the photon number densities
struct CleanSystem {
	AMREX_GPU_HOST_DEVICE void rhs(Real /*t*/, rosenbrock::Vector<1> &y, rosenbrock::Vector<1> &ydot) const
	{
		y[0] = amrex::max(y[0], Real(0.0));
		ydot[0] = 0.0;
	}
	AMREX_GPU_HOST_DEVICE void jac(Real /*t*/, rosenbrock::Vector<1> const & /*y*/, rosenbrock::Matrix<1> &J) const { J(0, 0) = 0.0; }
};

template <int N> AMREX_GPU_HOST_DEVICE auto UniformSettings(Real atol, Real rtol, int max_steps) -> rosenbrock::Settings<N>
{
	rosenbrock::Settings<N> s{};
	for (int n = 0; n < N; ++n) {
		s.atol[n] = atol;
		s.rtol[n] = rtol;
		s.in_error_norm[n] = true;
		s.min_value[n] = -std::numeric_limits<Real>::infinity();
		s.change_floor[n] = std::numeric_limits<Real>::infinity();
	}
	s.max_steps = max_steps;
	return s;
}

constexpr int n_rosenbrock_results = 12;

// Runs every Rosenbrock test case and writes the results to out[0..n_rosenbrock_results-1]; host and device run the same code.
AMREX_GPU_HOST_DEVICE void RunRosenbrockCases(Real *out)
{
	{ // decay
		DecaySystem sys{};
		rosenbrock::Vector<1> y{1.0};
		const auto r = rosenbrock::integrate<1>(sys, y, 1.0e-2, UniformSettings<1>(1.0e-14, 1.0e-8, 150000)); // atol << y(dt) = 4.5e-5, so rtol sets the accuracy
		out[0] = y[0];
		out[1] = static_cast<Real>(r.status);
	}
	{ // robertson
		RobertsonSystem sys{};
		rosenbrock::Vector<3> y{1.0, 0.0, 0.0};
		auto settings = UniformSettings<3>(1.0e-8, 1.0e-6, 150000);
		settings.atol[1] = 1.0e-14;
		const auto r = rosenbrock::integrate<3>(sys, y, 40.0, settings);
		out[2] = y[0];
		out[3] = y[1];
		out[4] = y[2];
		out[5] = static_cast<Real>(r.status);
	}
	{ // nan_jacobian_once
		DecaySystem sys{};
		sys.nan_jacobian_once = true;
		rosenbrock::Vector<1> y{1.0};
		const auto r = rosenbrock::integrate<1>(sys, y, 1.0e-2, UniformSettings<1>(1.0e-14, 1.0e-8, 150000)); // atol << y(dt) = 4.5e-5, so rtol sets the accuracy
		out[6] = y[0];
		out[7] = static_cast<Real>(r.status);
		out[8] = static_cast<Real>(r.n_jac);
	}
	{ // max_steps
		RobertsonSystem sys{};
		rosenbrock::Vector<3> y{1.0, 0.0, 0.0};
		const auto r = rosenbrock::integrate<3>(sys, y, 40.0, UniformSettings<3>(1.0e-8, 1.0e-6, 5));
		out[9] = static_cast<Real>(r.status);
	}
	{ // clean
		CleanSystem sys{};
		rosenbrock::Vector<1> y{-1.0e-12};
		const auto r = rosenbrock::integrate<1>(sys, y, 1.0, UniformSettings<1>(1.0e-8, 1.0e-8, 150000));
		out[10] = y[0];
		out[11] = static_cast<Real>(r.status);
	}
}

auto TestRosenbrock() -> int
{
	// the NaN-Jacobian case creates a NaN on purpose; suspend the FP-exception traps the test suite can enable
	std::fenv_t fenv{};
	std::feholdexcept(&fenv);
	std::array<Real, n_rosenbrock_results> host{};
	RunRosenbrockCases(host.data());
	amrex::Gpu::DeviceVector<Real> dev_d(n_rosenbrock_results);
	Real *dev_ptr = dev_d.data();
	amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) noexcept { RunRosenbrockCases(dev_ptr); });
	std::array<Real, n_rosenbrock_results> dev{};
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, dev_d.begin(), dev_d.end(), dev.begin());
	std::fesetenv(&fenv);

	const Real success = static_cast<Real>(rosenbrock::Status::Success);
	int status = 0;
	for (int pass = 0; pass < 2; ++pass) {
		const auto &v = (pass == 0) ? host : dev;
		const char *where = (pass == 0) ? "host" : "device";
		auto check = [&status, where](const char *name, bool ok) {
			std::cout << std::format("Rosenbrock {:<34}{:<8}{}\n", name, where, ok ? "PASS" : "FAIL");
			if (!ok) {
				status = 1;
			}
		};
		auto close = [](Real value, Real expected, Real reltol) { return std::abs(value - expected) <= reltol * std::abs(expected); };
		std::cout << std::format("Rosenbrock decay y = {:.10e} (exact {:.10e}), status {}; nan_jacobian_once y = {:.10e}, status {}, n_jac {}\n", v[0],
					 std::exp(-10.0), v[1], v[6], v[7], v[8]);
		check("decay", v[1] == success && close(v[0], std::exp(-10.0), 1.0e-5));
		check("robertson", v[5] == success && close(v[2], 0.7158270687, 1.0e-4) && close(v[3], 9.185534765e-6, 1.0e-3) &&
				       close(v[4], 0.2841637457, 1.0e-4));
		check("nan_jacobian_once", v[7] == success && v[8] >= 2.0 && close(v[6], std::exp(-10.0), 1.0e-5));
		check("max_steps", v[9] == static_cast<Real>(rosenbrock::Status::TooManySteps));
		check("clean", v[11] == success && v[10] >= 0.0);
	}
	return status;
}

} // namespace

auto problem_main() -> int
{
	const int ode_status = TestODEIntegration();
	const int root_status = TestRootFinding();
	const int eos_status = TestEOSIdeal();
	const int rosenbrock_status = TestRosenbrock();
	if (rosenbrock_status != 0) {
		std::cout << "MathUnitTests: Rosenbrock test FAILED.\n";
	}
	if (ode_status != 0) {
		std::cout << "MathUnitTests: ODE integration test FAILED.\n";
	}
	if (root_status != 0) {
		std::cout << "MathUnitTests: root finding test FAILED.\n";
	}
	if (eos_status != 0) {
		std::cout << "MathUnitTests: EOS test FAILED.\n";
	}
	return ((ode_status != 0) || (root_status != 0) || (eos_status != 0) || (rosenbrock_status != 0)) ? 1 : 0;
}
