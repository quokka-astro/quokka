//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMicrophysicsUnitTests.cpp
/// \brief Unit tests of the microphysics integrators' building blocks: the dense LU solver (math/linear_algebra.hpp)
/// on their matrix type, and the Rosenbrock tableaus (host and device). Transitional: it will be removed at the end
/// of the Microphysics port.
///

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <vector>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_REAL.H"

#include "math/linear_algebra.hpp"
#include "microphysics/array_utils.hpp"
#include "microphysics/rosenbrock/tableaus.hpp"

using amrex::Real;

namespace
{

using quokka::math::dgefa;
using quokka::math::dgesl;
using quokka::microphysics::MathArray2D;

// Evaluates Cases::Eval(n) for n = 0, ..., count - 1 on the host.
template <typename Cases> auto EvalOnHost(const int count) -> std::vector<Real>
{
	std::vector<Real> results(count);
	for (int n = 0; n < count; ++n) {
		results[n] = Cases::Eval(n);
	}
	return results;
}

// Evaluates Cases::Eval(n) for n = 0, ..., count - 1 on the device.
template <typename Cases> auto EvalOnDevice(const int count) -> std::vector<Real>
{
	amrex::Gpu::DeviceVector<Real> results_d(count);
	Real *results_ptr = results_d.data();
	amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(int n) noexcept { results_ptr[n] = Cases::Eval(n); });
	std::vector<Real> results(count);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, results_d.begin(), results_d.end(), results.begin());
	return results;
}

//------------------------------------------------------------------------------
// Dense LU factorization (dgefa) and solve (dgesl)
//------------------------------------------------------------------------------

// the smallest sizes, and the system sizes of OneZonePhotoionization (6), DTypeFront1D (8) and PrimordialChem (15)
constexpr int nsize = 5;
constexpr std::array<int, nsize> lu_sizes = {1, 2, 6, 8, 15};
constexpr int nsolve = 2 * nsize; // each size with and without pivoting

// Solves A x = b for x_i = i and returns the largest error in x relative to max_i x_i = N, or 1 if dgefa reports a
// zero pivot. M(i, j) = cos(1.3 i + 0.7 j^2) + 2 N delta_ij is strictly diagonally dominant, so it is well conditioned
// and needs no pivoting. With allow_pivot, A is M with its rows shifted cyclically, which moves the large entries off
// the diagonal.
template <int N, bool allow_pivot> AMREX_GPU_HOST_DEVICE auto SolveError() -> Real
{
	MathArray2D<N> a{};
	for (int j = 1; j <= N; ++j) {
		for (int i = 1; i <= N; ++i) {
			const int row = allow_pivot ? (i % N) + 1 : i;
			a(i, j) = std::cos(1.3 * row + 0.7 * j * j) + ((row == j) ? 2.0 * N : 0.0);
		}
	}
	amrex::Array1D<Real, 1, N> x{};
	for (int i = 1; i <= N; ++i) {
		for (int j = 1; j <= N; ++j) {
			x(i) += a(i, j) * j;
		}
	}

	amrex::Array1D<std::int16_t, 1, N> pivot{};
	int info = -1;
	dgefa<N, allow_pivot>(a, pivot, info);
	if (info != 0) {
		return 1.0;
	}
	dgesl<N, allow_pivot>(a, pivot, x);

	Real error = 0.0;
	for (int i = 1; i <= N; ++i) {
		error = amrex::max(error, std::abs(x(i) - i) / N);
	}
	return error;
}

struct SolveCases {
	AMREX_GPU_HOST_DEVICE static auto Eval(const int n) -> Real
	{
		const bool pivot = (n % 2 == 1);
		switch (n / 2) {
			case 0:
				return pivot ? SolveError<lu_sizes[0], true>() : SolveError<lu_sizes[0], false>();
			case 1:
				return pivot ? SolveError<lu_sizes[1], true>() : SolveError<lu_sizes[1], false>();
			case 2:
				return pivot ? SolveError<lu_sizes[2], true>() : SolveError<lu_sizes[2], false>();
			case 3:
				return pivot ? SolveError<lu_sizes[3], true>() : SolveError<lu_sizes[3], false>();
			default:
				return pivot ? SolveError<lu_sizes[4], true>() : SolveError<lu_sizes[4], false>();
		}
	}
};

// Small systems that are factored and solved exactly in floating point, to check LINPACK's conventions: pivoting
// takes the first largest entry of a column, and info is the index of the last zero pivot. Each case returns 0 if all
// of its checks pass and 1 otherwise.
constexpr int npivoting = 5;
constexpr std::array<const char *, npivoting> pivoting_names = {"3x3 tie, no pivoting", "3x3 tie, pivoting", "2x2 swap, no pivoting", "2x2 swap, pivoting",
								"2x2 singular, pivoting"};

struct PivotingCases {
	AMREX_GPU_HOST_DEVICE static auto Eval(const int n) -> Real
	{
		bool ok = false;
		if (n <= 1) {
			// [[1, 1, 0], [1, 1, 1], [0, 1, 1]] x = (3, 6, 5), so x = (1, 2, 3). Rows 1 and 2 tie in column 1, so row 1
			// stays the pivot row, and the second pivot is then zero unless rows 2 and 3 are swapped.
			MathArray2D<3> a{};
			a(1, 1) = 1.0;
			a(1, 2) = 1.0;
			a(2, 1) = 1.0;
			a(2, 2) = 1.0;
			a(2, 3) = 1.0;
			a(3, 2) = 1.0;
			a(3, 3) = 1.0;
			amrex::Array1D<std::int16_t, 1, 3> pivot{};
			int info = -1;
			if (n == 0) {
				dgefa<3, false>(a, pivot, info);
				ok = (info == 2);
			} else {
				dgefa<3, true>(a, pivot, info);
				amrex::Array1D<Real, 1, 3> x{};
				x(1) = 3.0;
				x(2) = 6.0;
				x(3) = 5.0;
				dgesl<3, true>(a, pivot, x);
				ok = (info == 0) && (pivot(1) == 1) && (pivot(2) == 3) && (pivot(3) == 3) && (x(1) == 1.0) && (x(2) == 2.0) && (x(3) == 3.0);
			}
		} else if (n <= 3) {
			// [[0, 1], [1, 0]] x = (3, 5), so x = (5, 3). Without pivoting, both pivots are zero.
			MathArray2D<2> a{};
			a(1, 2) = 1.0;
			a(2, 1) = 1.0;
			amrex::Array1D<std::int16_t, 1, 2> pivot{};
			int info = -1;
			if (n == 2) {
				dgefa<2, false>(a, pivot, info);
				ok = (info == 2);
			} else {
				dgefa<2, true>(a, pivot, info);
				amrex::Array1D<Real, 1, 2> x{};
				x(1) = 3.0;
				x(2) = 5.0;
				dgesl<2, true>(a, pivot, x);
				ok = (info == 0) && (pivot(1) == 2) && (pivot(2) == 2) && (x(1) == 5.0) && (x(2) == 3.0);
			}
		} else {
			// [[1, 2], [2, 4]] is singular: rows 1 and 2 are swapped, and the last pivot is zero
			MathArray2D<2> a{};
			a(1, 1) = 1.0;
			a(1, 2) = 2.0;
			a(2, 1) = 2.0;
			a(2, 2) = 4.0;
			amrex::Array1D<std::int16_t, 1, 2> pivot{};
			int info = -1;
			dgefa<2, true>(a, pivot, info);
			ok = (info == 2) && (pivot(1) == 2) && (pivot(2) == 2);
		}
		return ok ? 0.0 : 1.0;
	}
};

auto TestLinearAlgebra() -> int
{
	const std::vector<Real> solve_host = EvalOnHost<SolveCases>(nsolve);
	const std::vector<Real> solve_dev = EvalOnDevice<SolveCases>(nsolve);

	// The zero-pivot cases must not trap: dgefa skips the division by a zero pivot, but the compiler may still evaluate
	// it speculatively. So suspend the FP-exception traps that the test suite enables (amrex.fpe_trap_*) while they run.
	std::fenv_t fenv{};
	std::feholdexcept(&fenv);
	const std::vector<Real> pivoting_host = EvalOnHost<PivotingCases>(npivoting);
	const std::vector<Real> pivoting_dev = EvalOnDevice<PivotingCases>(npivoting);
	std::fesetenv(&fenv);

	int status = 0;
	const Real reltol = 1.0e-13;
	std::cout << std::format("{:<6}{:<12}{:>12}{:>12}\n", "N", "pivoting", "rel err", "(dev)");
	for (int n = 0; n < nsolve; ++n) {
		const bool ok = (solve_host[n] <= reltol) && (solve_dev[n] <= reltol);
		std::cout << std::format("{:<6}{:<12}{:>12.2e}{:>12.2e}{}\n", lu_sizes[n / 2], (n % 2 == 1) ? "yes" : "no", solve_host[n], solve_dev[n],
					 ok ? "" : "  FAIL");
		if (!ok) {
			status = 1;
		}
	}
	for (int n = 0; n < npivoting; ++n) {
		const bool ok = (pivoting_host[n] == 0.0) && (pivoting_dev[n] == 0.0);
		std::cout << std::format("{:<26}{}\n", pivoting_names[n], ok ? "ok" : "FAIL");
		if (!ok) {
			status = 1;
		}
	}

	std::cout << (status == 0 ? "Linear algebra: all tests passed.\n" : "Linear algebra: FAILED.\n");
	return status;
}

//------------------------------------------------------------------------------
// Rosenbrock tableaus
//------------------------------------------------------------------------------

using State = amrex::Array1D<Real, 1, 2>;

// Test problem: y' = |y|^2 (-y_2, y_1), which rotates y at the rate |y|^2, so from y(0) = (1, 0) the solution is
// y(t) = (cos t, sin t).
AMREX_GPU_HOST_DEVICE void Rhs(const State &y, State &ydot)
{
	const Real r2 = y(1) * y(1) + y(2) * y(2);
	ydot(1) = -r2 * y(2);
	ydot(2) = r2 * y(1);
}

AMREX_GPU_HOST_DEVICE void Jacobian(const State &y, MathArray2D<2> &jac)
{
	jac(1, 1) = -2.0 * y(1) * y(2);
	jac(1, 2) = -(y(1) * y(1) + 3.0 * y(2) * y(2));
	jac(2, 1) = 3.0 * y(1) * y(1) + y(2) * y(2);
	jac(2, 2) = 2.0 * y(1) * y(2);
}

// Takes one step of size h with the exact Jacobian, computed as Microphysics' Rosenbrock integrator computes a step,
// and returns the step's error estimate sum_i e(i) k_i in err.
template <typename Tableau> AMREX_GPU_HOST_DEVICE void Step(State &y, const Real h, State &err)
{
	MathArray2D<2> jac{};
	Jacobian(y, jac);
	const Real fac = 1.0 / (h * Tableau::gamma);
	for (int j = 1; j <= 2; ++j) {
		for (int i = 1; i <= 2; ++i) {
			jac(i, j) = -jac(i, j);
		}
		jac(j, j) += fac;
	}
	amrex::Array1D<std::int16_t, 1, 2> pivot{};
	int info = -1;
	dgefa<2, true>(jac, pivot, info);

	amrex::GpuArray<State, Tableau::stages> k{};
	Rhs(y, k[0]);
	dgesl<2, true>(jac, pivot, k[0]);
	for (int i = 2; i <= Tableau::stages; ++i) {
		State ystage = y;
		for (int n = 1; n <= 2; ++n) {
			for (int j = 1; j < i; ++j) {
				ystage(n) += Tableau::a(i, j) * k[j - 1](n);
				k[i - 1](n) += (Tableau::c(i, j) / h) * k[j - 1](n);
			}
		}
		State f{};
		Rhs(ystage, f);
		for (int n = 1; n <= 2; ++n) {
			k[i - 1](n) += f(n);
		}
		dgesl<2, true>(jac, pivot, k[i - 1]);
	}

	for (int n = 1; n <= 2; ++n) {
		Real update = 0.0;
		err(n) = 0.0;
		for (int i = 1; i <= Tableau::stages; ++i) {
			update += Tableau::b(i) * k[i - 1](n);
			err(n) += Tableau::e(i) * k[i - 1](n);
		}
		y(n) += update;
	}
}

// error at t = 1 after nsteps equal steps
template <typename Tableau> AMREX_GPU_HOST_DEVICE auto GlobalError(const int nsteps) -> Real
{
	State y{};
	y(1) = 1.0;
	State err{};
	const Real h = 1.0 / nsteps;
	for (int step = 0; step < nsteps; ++step) {
		Step<Tableau>(y, h, err);
	}
	return amrex::max(std::abs(y(1) - std::cos(1.0)), std::abs(y(2) - std::sin(1.0)));
}

// size of the error estimate of the first step, of size h
template <typename Tableau> AMREX_GPU_HOST_DEVICE auto ErrorEstimate(const Real h) -> Real
{
	State y{};
	y(1) = 1.0;
	State err{};
	Step<Tableau>(y, h, err);
	return amrex::max(std::abs(err(1)), std::abs(err(2)));
}

// Largest difference between ct(i) and the stage time sum_j alpha(i, j) that a and c imply, for i >= 2 (stage 1 is at
// time t, and ct(1) is not used). With Gamma the lower triangular matrix of the method's gamma(i, j), the tableaus
// hold a = alpha Gamma^-1 and c = diag(1 / gamma) - Gamma^-1 (Hairer and Wanner, Solving Ordinary Differential
// Equations II, Sect. IV.7), so alpha(i, j) = gamma (a(i, j) + sum_{j<k<i} alpha(i, k) c(k, j)).
template <typename Tableau> AMREX_GPU_HOST_DEVICE auto StageTimeError() -> Real
{
	Real error = 0.0;
	for (int i = 2; i <= Tableau::stages; ++i) {
		amrex::GpuArray<Real, Tableau::stages> alpha{}; // alpha[j - 1] = alpha(i, j)
		Real stage_time = 0.0;
		for (int j = i - 1; j >= 1; --j) {
			Real sum = Tableau::a(i, j);
			for (int k = j + 1; k < i; ++k) {
				sum += alpha[k - 1] * Tableau::c(k, j);
			}
			alpha[j - 1] = Tableau::gamma * sum;
			stage_time += alpha[j - 1];
		}
		error = amrex::max(error, std::abs(stage_time - Tableau::ct(i)));
	}
	return error;
}

struct TableauInfo {
	const char *name;
	int order;
};
constexpr int ntableau = 4;
constexpr std::array<TableauInfo, ntableau> tableaus = {
    {{.name = "ROS2S", .order = 2}, {.name = "Rodas3P", .order = 3}, {.name = "Rodas4P", .order = 4}, {.name = "Rodas5P", .order = 5}}};

// quantities computed for each tableau
constexpr int nquantity = 5;
constexpr int nsteps_coarse = 10;
constexpr Real h_coarse = 0.1;

template <typename Tableau> AMREX_GPU_HOST_DEVICE auto TableauQuantity(const int q) -> Real
{
	switch (q) {
		case 0:
			return StageTimeError<Tableau>();
		case 1:
			return GlobalError<Tableau>(nsteps_coarse);
		case 2:
			return GlobalError<Tableau>(2 * nsteps_coarse);
		case 3:
			return ErrorEstimate<Tableau>(h_coarse);
		default:
			return ErrorEstimate<Tableau>(h_coarse / 2);
	}
}

struct TableauCases {
	AMREX_GPU_HOST_DEVICE static auto Eval(const int n) -> Real
	{
		const int q = n % nquantity;
		switch (n / nquantity) {
			case 0:
				return TableauQuantity<quokka::microphysics::ROS2S>(q);
			case 1:
				return TableauQuantity<quokka::microphysics::Rodas3P>(q);
			case 2:
				return TableauQuantity<quokka::microphysics::Rodas4P>(q);
			default:
				return TableauQuantity<quokka::microphysics::Rodas5P>(q);
		}
	}
};

auto TestTableaus() -> int
{
	const int ncase = nquantity * ntableau;
	const std::vector<Real> host = EvalOnHost<TableauCases>(ncase);
	const std::vector<Real> dev = EvalOnDevice<TableauCases>(ncase);

	// The global error should fall as h^p when h is halved, and the error estimate of one step as h^p as well, since
	// each method's embedded solution has order p - 1.
	const Real stage_time_tol = 1.0e-13;
	const Real order_tol = 0.3;
	int status = 0;
	std::cout << std::format("{:<10}{:>7}{:>14}{:>8}{:>8}{:>12}{:>8}\n", "tableau", "order", "ct error", "p obs", "(dev)", "est. order", "(dev)");
	for (int t = 0; t < ntableau; ++t) {
		const int p = tableaus[t].order;
		const int i0 = t * nquantity; // results of tableau t: TableauQuantity(0), ..., TableauQuantity(nquantity - 1)
		const Real ct_error = amrex::max(host[i0], dev[i0]);
		const Real p_host = std::log2(host[i0 + 1] / host[i0 + 2]);
		const Real p_dev = std::log2(dev[i0 + 1] / dev[i0 + 2]);
		const Real q_host = std::log2(host[i0 + 3] / host[i0 + 4]);
		const Real q_dev = std::log2(dev[i0 + 3] / dev[i0 + 4]);
		const bool ok = (ct_error <= stage_time_tol) && (std::abs(p_host - p) <= order_tol) && (std::abs(p_dev - p) <= order_tol) &&
				(std::abs(q_host - p) <= order_tol) && (std::abs(q_dev - p) <= order_tol);
		std::cout << std::format("{:<10}{:>7}{:>14.2e}{:>8.2f}{:>8.2f}{:>12.2f}{:>8.2f}{}\n", tableaus[t].name, p, ct_error, p_host, p_dev, q_host,
					 q_dev, ok ? "" : "  FAIL");
		if (!ok) {
			status = 1;
		}
	}

	std::cout << (status == 0 ? "Rosenbrock tableaus: all tests passed.\n" : "Rosenbrock tableaus: FAILED.\n");
	return status;
}

} // namespace

auto problem_main() -> int
{
	const int lu_status = TestLinearAlgebra();
	const int tableau_status = TestTableaus();
	if (lu_status != 0) {
		std::cout << "MicrophysicsUnitTests: linear algebra test FAILED.\n";
	}
	if (tableau_status != 0) {
		std::cout << "MicrophysicsUnitTests: Rosenbrock tableau test FAILED.\n";
	}
	return ((lu_status != 0) || (tableau_status != 0)) ? 1 : 0;
}
