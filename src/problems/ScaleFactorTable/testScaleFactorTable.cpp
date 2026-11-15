/// \file testScaleFactorTable.cpp
/// \brief Accuracy test of ScaleFactorTable against the Einstein-de Sitter solution, compared with the adaptive RK integrator.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <string>

#include "AMReX.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "cosmology/Cosmology.hpp"
#include "cosmology/ScaleFactorTable.hpp"
#include "main.hpp"

namespace
{
using quokka::cosmology::CosmologyParams;
using quokka::cosmology::evolveScaleFactor;
using quokka::cosmology::ScaleFactorTable;

/// Number of ln a intervals of ScaleFactorTable::Build for spacing h
auto NumIntervals(amrex::Real const a_init, amrex::Real const a_max, amrex::Real const h) -> int
{
	return static_cast<int>(std::ceil((std::log(a_max) - std::log(a_init)) / h));
}

/// Observed order log2(e_coarse / e_fine) as text; marks '!' and sets status = 1 if outside [3.7, 4.3] while above the roundoff floor.
auto CheckOrder(amrex::Real const e_coarse, amrex::Real const e_fine, amrex::Real const floor, int &status) -> std::string
{
	if (e_coarse <= 0.0 || e_fine <= 0.0) { // avoid log2(0) under FPE traps
		return "-";
	}
	const amrex::Real order = std::log2(e_coarse / e_fine);
	std::string text = std::format("{:.2f}", order);
	if (e_fine > floor && (order < 3.7 || order > 4.3)) {
		status = 1;
		text += "!";
	}
	return text;
}

/// Self-convergence: max relative difference between tables built with h and h/2, at n_query + 1 uniform times in [0, t_query]
auto MaxRelDiff(CosmologyParams const &cosmo, amrex::Real const a_init, amrex::Real const a_max, amrex::Real const h, amrex::Real const t_query,
		int const n_query) -> amrex::Real
{
	ScaleFactorTable coarse; // h spacing
	ScaleFactorTable fine;	 // 0.5 * h spacing
	coarse.Build(cosmo, a_init, a_max, h);
	fine.Build(cosmo, a_init, a_max, 0.5 * h);

	amrex::Real max_diff = 0.0;
	for (int q = 0; q <= n_query; ++q) {
		const amrex::Real t = t_query * q / n_query;
		const amrex::Real a_fine = fine.ScaleFactorAt(t);
		const amrex::Real a_coarse = coarse.ScaleFactorAt(t);
		max_diff = std::max(max_diff, std::abs(a_coarse - a_fine) / a_fine);
	}
	return max_diff;
}

/// Einstein-de Sitter model: Omega_m = 1, flat, no radiation, no Lambda.
auto EdSParams() -> CosmologyParams
{
	CosmologyParams cosmo;
	cosmo.Omega_m = 1.0;
	cosmo.Omega_dm = 1.0;
	cosmo.Omega_b = 0.0;
	cosmo.Omega_r = 0.0;
	cosmo.Omega_L = 0.0;
	cosmo.validate();
	return cosmo;
}

/// Exact EdS time since a_init: t(a) = 2/(3 H0) (a^{3/2} - a_init^{3/2}).
auto TimeEdS(amrex::Real const a, amrex::Real const a_init, amrex::Real const H0) -> amrex::Real
{
	return 2.0 / (3.0 * H0) * (std::pow(a, 1.5) - std::pow(a_init, 1.5));
}

/// Exact EdS scale factor: (a_init^{3/2} + 3/2 H0 t)^{2/3}.
auto ScaleFactorEdS(amrex::Real const t, amrex::Real const a_init, amrex::Real const H0) -> amrex::Real
{
	return std::pow(std::pow(a_init, 1.5) + 1.5 * H0 * t, 2.0 / 3.0);
}

/// Exact EdS conformal time since a_init: tau(a) = 2/H0 (sqrt(a) - sqrt(a_init)).
auto TauEdS(amrex::Real const a, amrex::Real const a_init, amrex::Real const H0) -> amrex::Real { return 2.0 / H0 * (std::sqrt(a) - std::sqrt(a_init)); }

/// Max errors at fractional position s: relative error of a, and error of tau in units of 1/(a H).
struct TableErrors {
	amrex::Real a;
	amrex::Real tau;
};

/// Max errors of the table at fractional position s of each ln a interval (s = 0: nodes, s = 0.5: midpoints).
auto MaxRelError(ScaleFactorTable const &table, amrex::Real const a_init, amrex::Real const a_max, amrex::Real const dlna, amrex::Real const H0,
		 amrex::Real const s) -> TableErrors
{
	// same grid as ScaleFactorTable::Build
	const amrex::Real x_init = std::log(a_init);
	const int n_intervals = NumIntervals(a_init, a_max, dlna);

	TableErrors err{.a = 0.0, .tau = 0.0};
	for (int j = 0; j < n_intervals; ++j) {
		const amrex::Real a_lo = (j == 0) ? a_init : std::exp(x_init + j * dlna);
		const amrex::Real a_hi = std::exp(x_init + (j + 1) * dlna);
		const amrex::Real t_lo = TimeEdS(a_lo, a_init, H0);
		const amrex::Real t_hi = TimeEdS(a_hi, a_init, H0);
		const amrex::Real t = t_lo + s * (t_hi - t_lo); // trial point, s < 1: never query past the last node
		const amrex::Real a_exact = ScaleFactorEdS(t, a_init, H0);
		const amrex::Real aH = H0 / std::sqrt(a_exact); // a H = H0 a^{-1/2} in EdS
		err.a = std::max(err.a, std::abs(table.ScaleFactorAt(t) - a_exact) / a_exact);
		err.tau = std::max(err.tau, std::abs(table.ConformalTimeAt(t) - TauEdS(a_exact, a_init, H0)) * aH);
	}
	return err;
}

/// Max relative errors at the same times: old RK path and table for a; table and old midpoint rule for dtau.
struct PathErrors {
	amrex::Real rk;
	amrex::Real table;
	amrex::Real dtau_table;
	amrex::Real dtau_old;
};

/// Old approach: a advanced with evolveScaleFactor, two Strang half steps per step with H dt = H_dt; the table is queried at the same times.
auto MaxRelErrorRK(CosmologyParams const &cosmo, ScaleFactorTable const &table, amrex::Real const a_init, amrex::Real const a_max, amrex::Real const H_dt)
    -> PathErrors
{
	amrex::Real a = a_init;
	amrex::Real t = 0.0;
	PathErrors err{.rk = 0.0, .table = 0.0, .dtau_table = 0.0, .dtau_old = 0.0};

	while (a * (1.0 + H_dt) < a_max) { // margin for the ALWAYS_ASSERT of the table
		const amrex::Real dt = H_dt / (cosmo.H0 * quokka::cosmology::HubbleFactor(a, cosmo));
		const amrex::Real a_half = evolveScaleFactor(a, 0.5 * dt, cosmo); // old transport used dtau = dt / a(t + dt/2)
		const amrex::Real dtau_exact =
		    TauEdS(ScaleFactorEdS(t + dt, a_init, cosmo.H0), a_init, cosmo.H0) - TauEdS(ScaleFactorEdS(t, a_init, cosmo.H0), a_init, cosmo.H0);
		err.dtau_table = std::max(err.dtau_table, std::abs(table.Dtau(t, dt) - dtau_exact) / dtau_exact);
		err.dtau_old = std::max(err.dtau_old, std::abs(dt / a_half - dtau_exact) / dtau_exact);
		a = evolveScaleFactor(a_half, 0.5 * dt, cosmo);
		t += dt;
		const amrex::Real a_exact = ScaleFactorEdS(t, a_init, cosmo.H0);
		err.rk = std::max(err.rk, std::abs(a - a_exact) / a_exact);
		err.table = std::max(err.table, std::abs(table.ScaleFactorAt(t) - a_exact) / a_exact);
	}
	return err;
}
} // namespace

auto problem_main() -> int
{
	const CosmologyParams cosmo = EdSParams();
	const amrex::Real a_init = 0.01;
	const amrex::Real a_max = 1.0;
	const int n_conv = 9; // h = 0.2 / 2^k for k from 0 to 8
	const amrex::Real eps = std::numeric_limits<amrex::Real>::epsilon();
	amrex::Print() << std::format("machine epsilon (amrex::Real) = {:.16e}\n", eps);
	constexpr amrex::Real floor_factor = 10.0;
	int status = 0;
	ScaleFactorTable table;

	// 1. convergence: Hermite error dominates at s = 0.5, GL3 + summation at s = 0
	std::array<amrex::Real, n_conv> h = {};
	std::array<amrex::Real, n_conv> err_node{};
	std::array<amrex::Real, n_conv> err_mid{};
	std::array<amrex::Real, n_conv> err_tau{};
	std::array<amrex::Real, n_conv> err_tau_node{};

	amrex::Print() << "EdS (Omega_m = 1), a in [0.01, 1]: error vs exact a(t) and tau(t)\n";
	amrex::Print() << std::format("{:>10} {:>6} {:>11} {:>11} {:>9} {:>9} {:>13} {:>10} {:>11} {:>9} {:>13} {:>11} {:>9}\n", "h", "N", "err(s=0)",
				      "err(s=0.5)", "ord(s=0)", "ord(s=.5)", "err(s=.5)/h^4", "N eps", "tau(s=.5)", "ord(tau)", "tau/h^4", "tau(s=0)", "ord(t,s=0)");

	for (int k = 0; k < n_conv; ++k) {
		h[k] = std::ldexp(0.2, -k); // h = 0.2 / 2^k
		const int n_int = NumIntervals(a_init, a_max, h[k]);
		table.Build(cosmo, a_init, a_max, h[k]);
		const TableErrors e_node = MaxRelError(table, a_init, a_max, h[k], cosmo.H0, 0.0);
		const TableErrors e_mid = MaxRelError(table, a_init, a_max, h[k], cosmo.H0, 0.5);
		err_node[k] = e_node.a;
		err_mid[k] = e_mid.a;
		err_tau[k] = e_mid.tau;
		err_tau_node[k] = e_node.tau;

		const amrex::Real floor_k = floor_factor * n_int * eps;
		std::string ord_node = "-";
		std::string ord_mid = "-";
		std::string ord_tau = "-";
		std::string ord_tau_node = "-";
		if (k > 0) {
			ord_node = CheckOrder(err_node[k - 1], err_node[k], std::numeric_limits<amrex::Real>::max(), status); // printed only, never checked
			ord_mid = CheckOrder(err_mid[k - 1], err_mid[k], floor_k, status);
			ord_tau_node = CheckOrder(err_tau_node[k - 1], err_tau_node[k], std::numeric_limits<amrex::Real>::max(), status); // printed only
			ord_tau = CheckOrder(err_tau[k - 1], err_tau[k], floor_k, status);
		}
		amrex::Print() << std::format("{:>10.4e} {:>6} {:>11.3e} {:>11.3e} {:>9} {:>9} {:>13.4e} {:>10.3e} {:>11.3e} {:>9} {:>13.4e} {:>11.3e} {:>9}\n", h[k],
					      n_int, err_node[k], err_mid[k], ord_node, ord_mid, err_mid[k] / std::pow(h[k], 4), n_int * eps, err_tau[k], ord_tau,
					      err_tau[k] / std::pow(h[k], 4), err_tau_node[k], ord_tau_node);
	}

	// 2. production spacing: interpolation (9e-15) and roundoff
	const amrex::Real dlna_prod = 1.0e-3;
	const int n_prod = NumIntervals(a_init, a_max, dlna_prod);
	table.Build(cosmo, a_init, a_max, dlna_prod);
	const TableErrors e_node_prod = MaxRelError(table, a_init, a_max, dlna_prod, cosmo.H0, 0.0);
	const TableErrors e_mid_prod = MaxRelError(table, a_init, a_max, dlna_prod, cosmo.H0, 0.5);
	const amrex::Real err_node_prod = e_node_prod.a;
	const amrex::Real err_mid_prod = e_mid_prod.a;
	// Hermite constants for a as t^p, p = 2/3 (EdS): a: |(p-1)(p-2)(p-3)| / p^3 / 384 = 3.5/384; tau (in units of 1/(aH)): (p+1)(p+2)/p^2/384 = 5/192
	constexpr amrex::Real p = 2.0 / 3.0;
	const amrex::Real hermite_const = std::abs((p - 1.0) * (p - 2.0) * (p - 3.0)) / (p * p * p) / 384.0;
	const amrex::Real hermite_const_tau = (p + 1.0) * (p + 2.0) / (p * p) / 384.0;
	const amrex::Real tol = 2.0 * hermite_const * std::pow(dlna_prod, 4) + 100.0 * eps;
	const amrex::Real tol_tau = 2.0 * hermite_const_tau * std::pow(dlna_prod, 4) + 100.0 * eps;
	amrex::Print() << std::format("production h = {:.1e}: err(s=0) = {:.6e}, err(s=0.5) = {:.6e}, tol = 2 K h^4 + 100 eps = {:.6e}\n", dlna_prod,
				      err_node_prod, err_mid_prod, tol);
	amrex::Print() << std::format("production tau: err(s=0.5) = {:.6e}, tol = 2 (5/192) h^4 + 100 eps = {:.6e}\n", e_mid_prod.tau, tol_tau);
	if (err_node_prod > tol || err_mid_prod > tol || e_mid_prod.tau > tol_tau) {
		status = 1;
	}

	// 3. realistic LCDM (default parameters): self-convergence, no exact solution needed
	const CosmologyParams planck;		     // defaults of CosmologyParams
	const amrex::Real t_query = 0.9 / planck.H0; // flat LCDM: H0 t(a=1) is about 0.955 for Omega_m = 0.31, so stop at 0.9H0, before the table edge
	constexpr int n_query = 100000;
	amrex::Print() << "\nLCDM (default parameters): self-convergence D(h) = max |a_h - a_{h/2}| / a\n";
	amrex::Print() << std::format("{:>10} {:>11} {:>9} {:>10}\n", "h", "D(h)", "order", "N(h/2) eps");
	std::array<amrex::Real, n_conv> diff{};
	for (int k = 0; k < n_conv; ++k) {
		diff[k] = MaxRelDiff(planck, a_init, a_max, h[k], t_query, n_query);
		const amrex::Real floor_k = NumIntervals(a_init, a_max, 0.5 * h[k]) * eps;
		std::string ord = "-";
		if (k > 0) {
			ord = CheckOrder(diff[k - 1], diff[k], floor_factor * floor_k, status);
		}
		amrex::Print() << std::format("{:>10.4e} {:>11.3e} {:>9} {:>10.3e}\n", h[k], diff[k], ord, floor_k);
	}

	// 4. old adaptive RK12 vs table at the same times; two step histories show the path dependence of RK
	for (amrex::Real const H_dt : {0.01, 0.003}) {
		const PathErrors err = MaxRelErrorRK(cosmo, table, a_init, a_max, H_dt);
		amrex::Print() << std::format(
		    "H dt = {:.3f}: old RK max err = {:.3e}, table max err = {:.3e} (same times); dtau rel err: table {:.3e}, old midpoint {:.3e}\n", H_dt,
		    err.rk, err.table, err.dtau_table, err.dtau_old);
	}

	amrex::Print() << (status == 0 ? "ScaleFactorTable Test: PASSED\n" : "ScaleFactorTable Test: FAILED\n");
	return status;
}