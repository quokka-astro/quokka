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

/// Self-convergence: max relative difference between tables built with h and h/2, at n_query + 1 uniform times in [0, t_query]
auto MaxRelDiff(CosmologyParams const &cosmo, amrex::Real const a_init, amrex::Real const a_max, amrex::Real const h, amrex::Real const t_query,
		int const n_query) -> amrex::Real
{
	ScaleFactorTable coarse; // h spacing
	ScaleFactorTable fine; // 0.5 * h spacing
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

/// Max relative error of the table at fractional position s of each ln a interval (s = 0: nodes, s = 0.5: midpoints).
auto MaxRelError(ScaleFactorTable const &table, amrex::Real const a_init, amrex::Real const a_max, amrex::Real const dlna, amrex::Real const H0,
		 amrex::Real const s) -> amrex::Real
{
	// same grid as ScaleFactorTable::Build
	const amrex::Real x_init = std::log(a_init);
	const int n_intervals = NumIntervals(a_init, a_max, dlna);

	amrex::Real max_err = 0.0;
	for (int j = 0; j < n_intervals; ++j) {
		const amrex::Real a_lo = (j == 0) ? a_init : std::exp(x_init + j * dlna);
		const amrex::Real a_hi = std::exp(x_init + (j + 1) * dlna);
		const amrex::Real t_lo = TimeEdS(a_lo, a_init, H0);
		const amrex::Real t_hi = TimeEdS(a_hi, a_init, H0);
		const amrex::Real t = t_lo + s * (t_hi - t_lo); // trial point, s < 1: never query past the last node
		const amrex::Real a_table = table.ScaleFactorAt(t);
		const amrex::Real a_exact = ScaleFactorEdS(t, a_init, H0);
		max_err = std::max(max_err, std::abs(a_table - a_exact) / a_exact);
	}
	return max_err;
}

/// Max relative errors of the old RK path and of the table, at the same times.
struct PathErrors {
	amrex::Real rk;
	amrex::Real table;
};

/// Old approach: a advanced with evolveScaleFactor, two Strang half steps per step with H dt = H_dt; the table is queried at the same times.
auto MaxRelErrorRK(CosmologyParams const &cosmo, ScaleFactorTable const &table, amrex::Real const a_init, amrex::Real const a_max,
		   amrex::Real const H_dt) -> PathErrors
{
	amrex::Real a = a_init;
	amrex::Real t = 0.0;
	PathErrors err{.rk = 0.0, .table = 0.0};

	while (a * (1.0 + H_dt) < a_max) { // margin for the ALWAYS_ASSERT of the table
		const amrex::Real dt = H_dt / (cosmo.H0 * quokka::cosmology::HubbleFactor(a, cosmo));
		a = evolveScaleFactor(a, 0.5 * dt, cosmo);
		a = evolveScaleFactor(a, 0.5 * dt, cosmo);
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
	constexpr amrex::Real floor_factor = 10.0; 
	int status = 0;
	ScaleFactorTable table;

	// 1. convergence: Hermite error dominates at s = 0.5, GL3 + summation at s = 0
	std::array<amrex::Real, n_conv> h = {};
	std::array<amrex::Real, n_conv> err_node{};
	std::array<amrex::Real, n_conv> err_mid{};

	amrex::Print() << "EdS (Omega_m = 1), a in [0.01, 1]: error vs exact a(t)\n";
	amrex::Print() << std::format("{:>10} {:>6} {:>11} {:>11} {:>9} {:>9} {:>13} {:>10}\n", "h", "N", "err(s=0)", "err(s=0.5)", "ord(s=0)", "ord(s=.5)",
				      "err(s=.5)/h^4", "N eps");

	for (int k = 0; k < n_conv; ++k) {
		h[k] = std::ldexp(0.2, -k); // h = 0.2 / 2^k
		const int n_int =  NumIntervals(a_init, a_max, h[k]);
		table.Build(cosmo, a_init, a_max, h[k]);
		err_node[k] = MaxRelError(table, a_init, a_max, h[k], cosmo.H0, 0.0);
		err_mid[k] = MaxRelError(table, a_init, a_max, h[k], cosmo.H0, 0.5);
		
		std::string ord_node = "-";
		std::string ord_mid = "-";

		if (k > 0 && err_node[k] > 0.0 && err_node[k - 1] > 0.0) { // avoid log2(0) under FPE traps
			ord_node = std::format("{:.2f}", std::log2(err_node[k - 1] / err_node[k]));
		}

		if (k > 0 && err_mid[k] > 0.0 && err_mid[k - 1 ] > 0.0) {
			const amrex::Real order = std::log2(err_mid[k - 1] / err_mid[k]);
			ord_mid = std::format("{:.2f}", order);
			if (err_mid[k] > floor_factor * n_int * eps) { // order is meaningful in truncation-dominated regime.
				if (order < 3.7 || order > 4.3) {
					status = 1;
					ord_mid += "!";
				}

			}
		}	
		amrex::Print() << std::format("{:>10.4e} {:>6} {:>11.3e} {:>11.3e} {:>9} {:>9} {:>13.4e} {:>10.3e}\n", h[k], n_int, err_node[k], err_mid[k],
					      ord_node, ord_mid, err_mid[k] / std::pow(h[k], 4), n_int * eps);
	}

	// 2. production spacing: interpolation (9e-15) and roundoff
	const amrex::Real dlna_prod = 1.0e-3;
	const int n_prod = NumIntervals(a_init, a_max, dlna_prod);
	table.Build(cosmo, a_init, a_max, dlna_prod);
	const amrex::Real err_node_prod = MaxRelError(table, a_init, a_max, dlna_prod, cosmo.H0, 0.0);
	const amrex::Real err_mid_prod = MaxRelError(table, a_init, a_max, dlna_prod, cosmo.H0, 0.5);
	// Hermite constant for a proportional to t^p, p = 2/3 (EdS): |(p-1)(p-2)(p-3)| / p^3 / 384
	constexpr amrex::Real p = 2.0 / 3.0;
	const amrex::Real hermite_const = std::abs((p - 1.0) * (p - 2.0) * (p - 3.0)) / (p * p * p) / 384.0;
	const amrex::Real tol = 2.0 * hermite_const * std::pow(dlna_prod, 4) + 100.0 * eps;
	amrex::Print() << std::format("production h = {:.1e}: err(s=0) = {:.3e}, err(s=0.5) = {:.3e}, tol = 2 K h^4 + N eps = {:.3e}\n", dlna_prod,
				      err_node_prod, err_mid_prod, tol);
	if (err_node_prod > tol || err_mid_prod > tol) {
		status = 1;
	}

	// 3. realistic LCDM (default parameters): self-convergence, no exact solution needed
	const CosmologyParams planck; // defaults of CosmologyParams
	const amrex::Real t_query = 0.9 / planck.H0; // flat LCDM: H0 t(a=1) is about 0.955 for Omega_m = 0.31
	constexpr int n_query = 100000;
	amrex::Print() << "\nLCDM (default parameters): self-convergence D(h) = max |a_h - a_{h/2}| / a\n";
	amrex::Print() << std::format("{:>10} {:>11} {:>9} {:>10}\n", "h", "D(h)", "order", "N(h/2) eps");
	std::array<amrex::Real, n_conv> diff{};
	for (int k = 0; k < n_conv; ++k) {
		diff[k] = MaxRelDiff(planck, a_init, a_max, h[k], t_query, n_query);
		const amrex::Real floor_k = NumIntervals(a_init, a_max, 0.5 * h[k]) * eps;
		std::string ord = "-";
		if (k > 0 && diff[k] > 0.0 && diff[k - 1] > 0.0) {
			const amrex::Real order = std::log2(diff[k - 1] / diff[k]);
			ord = std::format("{:.2f}", order);
			if (diff[k] > floor_factor * floor_k && (order < 3.7 || order > 4.3)) {
				status = 1;
				ord += "!";
			}
		}
		amrex::Print() << std::format("{:>10.4e} {:>11.3e} {:>9} {:>10.3e}\n", h[k], diff[k], ord, floor_k);
	}
	
	// 4. old adaptive RK12 vs table at the same times; two step histories show the path dependence of RK
	for (amrex::Real const H_dt : {0.01, 0.003}) {
		const PathErrors err = MaxRelErrorRK(cosmo, table, a_init, a_max, H_dt);
		amrex::Print() << std::format("H dt = {:.3f}: old RK max err = {:.3e}, table max err = {:.3e} (same times)\n", H_dt, err.rk, err.table);
	}

	amrex::Print() << (status == 0 ? "ScaleFactorTable Test: PASSED\n" : "ScaleFactorTable Test: FAILED\n");
	return status;
}