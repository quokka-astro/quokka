// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef MICROPHYSICS_ROSENBROCK_HPP_
#define MICROPHYSICS_ROSENBROCK_HPP_
/// \file rosenbrock.hpp
/// \brief The Rosenbrock integrator. Ported from Microphysics' integration/Rosenbrock/rosenbrock_integrator.H.

#include <cmath>
#include <limits>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_ConstexprFor.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "gcem.hpp"

#include "math/linear_algebra.hpp"
#include "microphysics/integrator_common.hpp"
#include "microphysics/integrator_errors.hpp"
#include "microphysics/integrator_params.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "microphysics/rosenbrock/rosenbrock_state.hpp"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

namespace rosenbrock
{

// The tolerances of entry n. Only the entries that count toward the step error (counts_toward_error) have tolerances,
// so callers skip the others.
template <typename System, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rtol_for(const RosenbrockT &rstate, const int n) -> amrex::Real
{
	const VarRole role = System::role(n);
	AMREX_ASSERT(counts_toward_error(role));
	if (role == VarRole::Energy) {
		return rstate.rtol_enuc;
	}
	if (role == VarRole::ChemPhotonNumber) {
		return rstate.rtol_rad_num;
	}
	return rstate.rtol_spec;
}

template <typename System, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto atol_for(const RosenbrockT &rstate, const int n) -> amrex::Real
{
	const VarRole role = System::role(n);
	AMREX_ASSERT(counts_toward_error(role));
	if (role == VarRole::Energy) {
		return rstate.atol_enuc;
	}
	if (role == VarRole::ChemPhotonNumber) {
		return rstate.atol_rad_num;
	}
	return rstate.atol_spec;
}

template <typename System, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto error_norm(const RosenbrockT &rstate) -> amrex::Real
{
	constexpr int int_neqs = System::neqs;

	// unrolled, so that each entry's role and tolerances are compile-time constants
	amrex::Real err = 0.0_rt;
	amrex::constexpr_for<0, int_neqs>([&](auto nc) {
		constexpr int n = decltype(nc)::value;
		if constexpr (counts_toward_error(System::role(n))) {
			const amrex::Real sk =
			    atol_for<System>(rstate, n) + rtol_for<System>(rstate, n) * amrex::max(std::abs(rstate.y[n]), std::abs(rstate.ynew[n]));
			const amrex::Real term = rstate.work[n] / sk;
			err += term * term;
		}
	});
	// the root mean square over the entries that count toward the error, so that the entries outside error control do
	// not lower the norm (https://github.com/quokka-astro/quokka/pull/2295#discussion_r4051762482)
	constexpr int num_error_entries = System::NumErrorEntries();
	return std::sqrt(err / static_cast<amrex::Real>(num_error_entries));
}

template <typename System, typename RosenbrockT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto species_change_valid(const RosenbrockT &rstate, const IntegratorParams &params) -> bool
{
	constexpr amrex::Real increase_change_factor = 4.0_rt;
	constexpr amrex::Real decrease_change_factor = 0.25_rt;

	// accumulated without short-circuit evaluation, so that the test is a few compares and no branches on the GPU
	bool valid = true;
	for (int i = 0; i < System::num_spec; ++i) {
		const amrex::Real y_old = std::abs(rstate.y[i]);
		const amrex::Real y_new = std::abs(rstate.ynew[i]);
		const bool above_buffer = (y_old > params.X_reject_buffer * rstate.atol_spec) & (y_new > params.X_reject_buffer * rstate.atol_spec);
		const bool changed_too_much = (y_new > increase_change_factor * y_old) | (y_new < decrease_change_factor * y_old);
		valid &= !(above_buffer & changed_too_much);
	}

	return valid;
}

template <typename System, typename MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto matrix_is_finite(const MatrixType &matrix) -> bool
{
	constexpr int int_neqs = System::neqs;

	// A NaN or an infinity in any entry makes the sum of all entries non-finite, so one check of the sum settles the
	// usual case. Only a non-finite sum needs the entries checked one by one (the sum of finite entries can overflow).
	// The columns are summed separately first, so that the sum is not one long dependent chain.
	amrex::Real sum = 0.0_rt;
	for (int j = 0; j < int_neqs; ++j) {
		amrex::Real column_sum = 0.0_rt;
		for (int i = 0; i < int_neqs; ++i) {
			column_sum += matrix(i, j);
		}
		sum += column_sum;
	}
	if (std::isfinite(sum)) {
		return true;
	}

	for (int j = 0; j < int_neqs; ++j) {
		for (int i = 0; i < int_neqs; ++i) {
			const amrex::Real value = matrix(i, j);
			if (std::isnan(value) || std::isinf(value)) {
				return false;
			}
		}
	}

	return true;
}

template <typename System>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto valid_integrator_state(const amrex::GpuArray<amrex::Real, System::neqs> &y, const IntegratorParams &params)
    -> bool
{
	constexpr int int_neqs = System::neqs;

	// accumulated without short-circuit evaluation, so that the test is a few compares and no branches on the GPU
	bool valid = true;
	for (int n = 0; n < int_neqs; ++n) {
		valid &= std::isfinite(y[n]);
	}

	for (int n = 0; n < int_neqs; ++n) {
		switch (System::role(n)) {
			case VarRole::Species:
				valid &= !(y[n] < -params.atol_spec);
				break;
			case VarRole::Energy:
				valid &= !(y[n] <= 0.0_rt);
				break;
			case VarRole::ChemPhotonNumber:
				valid &= !(y[n] < -params.atol_rad_num);
				break;
			case VarRole::ChemPhotonFlux:
			case VarRole::ThermalBandEnergy:
			case VarRole::DustAbsorbedEnergy:
				break;
		}
	}

	return valid;
}

/// The finite-difference Jacobian, from int_neqs evaluations of the right-hand side around rstate.y. Expects the
/// right-hand side at rstate.y in rstate.k[0].
template <typename System, typename BurnT, typename RosenbrockT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void numerical_jacobian(BurnT &state, RosenbrockT &rstate, const amrex::Real time, const IntegratorParams &params)
{
	// TODO (james471): This is an unreviewed and untested function. It has never been used since we always
	// use the analytic Jacobian.
	constexpr int int_neqs = System::neqs;
	constexpr amrex::Real UROUND = std::numeric_limits<amrex::Real>::epsilon();
	amrex::GpuArray<amrex::Real, int_neqs> ewt{};
	amrex::GpuArray<amrex::Real, int_neqs> ybase{};

	// rhs() may clean or renormalize all components of rstate.y.
	for (int i = 0; i < int_neqs; ++i) {
		ybase[i] = rstate.y[i];
	}

	// error weights, only for the entries that count toward the step error (the others have no tolerances)
	amrex::Real fac = 0.0_rt;
	amrex::constexpr_for<0, int_neqs>([&](auto ic) {
		constexpr int i = decltype(ic)::value;
		if constexpr (counts_toward_error(System::role(i))) {
			ewt[i] = 1.0_rt / (rtol_for<System>(rstate, i) * std::abs(rstate.y[i]) + atol_for<System>(rstate, i));
			fac += (rstate.k[0][i] * ewt[i]) * (rstate.k[0][i] * ewt[i]);
		}
	});
	fac = std::sqrt(fac / static_cast<amrex::Real>(int_neqs));

	amrex::Real R0 = 1000.0_rt * std::abs(rstate.dt) * UROUND * static_cast<amrex::Real>(int_neqs) * fac;
	if (R0 == 0.0_rt) {
		R0 = 1.0_rt;
	}

	for (int j = 0; j < int_neqs; ++j) {
		const amrex::Real yj = ybase[j];
		// an entry without tolerances gets the relative step, with |y| floored at SMALL_X_SAFE so that a zero entry still moves
		const amrex::Real R = counts_toward_error(System::role(j)) ? amrex::max(std::sqrt(UROUND) * std::abs(yj), R0 / ewt[j])
									   : std::sqrt(UROUND) * amrex::max(std::abs(yj), params.SMALL_X_SAFE);
		for (int n = 0; n < int_neqs; ++n) {
			rstate.y[n] = ybase[n];
		}
		rstate.y[j] += R;

		rhs<System>(time, state, rstate, rstate.work, params);

		fac = 1.0_rt / R;
		for (int i = 0; i < int_neqs; ++i) {
			rstate.jac(i, j) = (rstate.work[i] - rstate.k[0][i]) * fac;
		}

		for (int n = 0; n < int_neqs; ++n) {
			rstate.y[n] = ybase[n];
		}
	}

	rstate.n_rhs += int_neqs;
	rstate.n_jac += 1;
}

/// Evaluates the right-hand side at rstate.y into rstate.k[0] and the Jacobian into rstate.jac. Returns false if the
/// Jacobian has a non-finite entry and numerical_jacobian_fallback is off.
template <typename System, typename BurnT, typename RosenbrockT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto evaluate_jacobian(BurnT &state, RosenbrockT &rstate, const amrex::Real time, const IntegratorParams &params)
    -> bool
{
	// a network that evaluates both at once reuses its rate coefficients between them
	constexpr bool combined =
	    System::traits::analytic_jacobian && System::traits::shared_rhs_and_jac && network_has_rhs_and_jac<System, BurnT, decltype(rstate.jac)>;
	if constexpr (combined) {
		rhs_and_jac<System>(time, state, rstate, rstate.k[0], rstate.jac, params);
		rstate.n_rhs += 1;
		rstate.n_jac += 1;
	} else {
		rhs<System>(time, state, rstate, rstate.k[0], params);
		rstate.n_rhs += 1;

		if constexpr (System::traits::analytic_jacobian) {
			jac<System>(state, rstate, rstate.jac);
			rstate.n_jac += 1;
		}
	}

	if constexpr (!System::traits::analytic_jacobian) {
		numerical_jacobian<System>(state, rstate, time, params);
		return true;
	} else if constexpr (System::traits::numerical_jacobian_fallback) {
		if (!matrix_is_finite<System>(rstate.jac)) {
			numerical_jacobian<System>(state, rstate, time, params);
		}
		return true;
	} else {
		// without the fallback the matrix is not checked here: decompose() sums its entries in the pass that forms the
		// integrator matrix (one pass over the matrix instead of two) and reports a non-finite sum
		return true;
	}
}

template <typename System, typename BurnT, typename RosenbrockT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs_at(const amrex::Real time, BurnT &state, RosenbrockT &rstate,
						     const amrex::GpuArray<amrex::Real, System::neqs> &y, amrex::GpuArray<amrex::Real, System::neqs> &ydot,
						     const IntegratorParams &params)
{
	constexpr int int_neqs = System::neqs;

	amrex::GpuArray<amrex::Real, int_neqs> ysave{};

	for (int n = 0; n < int_neqs; ++n) {
		ysave[n] = rstate.y[n];
		rstate.y[n] = y[n];
	}

	rhs<System>(time, state, rstate, ydot, params);

	for (int n = 0; n < int_neqs; ++n) {
		rstate.y[n] = ysave[n];
	}
}

/// The active block of the integrator matrix, in the form dgefa and dgesl take: entry (i, j) of the block is entry
/// (ActiveEntries()[i], ActiveEntries()[j]) of the full matrix. The indices are compile-time constants where the solver
/// is unrolled, so the block is addressed as directly as the full matrix.
template <typename System, typename MatrixType> struct ActiveBlock {
	static constexpr amrex::GpuArray<int, System::num_active> entries = System::ActiveEntries();
	MatrixType &matrix;
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(const int i, const int j) const -> amrex::Real &
	{
		return matrix(entries.arr[i], entries.arr[j]);
	}
};

/// Forms the integrator matrix M = fac I - J in rstate.jac and factors its active block (ODESystem::num_active): the
/// passive entries have zero columns in J off the diagonal, so M = [[A, 0], [C, D]] with D diagonal, and only A needs
/// an LU factorization. The passive diagonal is stored as its reciprocal, like the pivots of A. Returns 0, or a nonzero
/// code when A has a zero pivot or D a zero entry, in which case solve() must not be called.
template <typename System, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto decompose(RosenbrockT &rstate, const amrex::Real fac) -> int
{
	constexpr int int_neqs = System::neqs;

#ifdef AMREX_DEBUG
	for (int p = 0; p < System::num_passive; ++p) {
		const int col = System::PassiveEntries().arr[p];
		for (int i = 0; i < int_neqs; ++i) {
			AMREX_ASSERT_WITH_MESSAGE(i == col || rstate.jac(i, col) == 0.0_rt, "a passive entry of y has a nonzero off-diagonal Jacobian column");
		}
	}
#endif

	// The same pass sums the entries: a NaN or an infinity anywhere makes the sum non-finite (a sum of finite entries
	// can overflow, but then the factorization is hopeless too). The columns are summed separately, so that the sum is
	// not one long dependent chain.
	amrex::Real sum = 0.0_rt;
	for (int j = 0; j < int_neqs; ++j) {
		amrex::Real column_sum = 0.0_rt;
		for (int i = 0; i < int_neqs; ++i) {
			column_sum += rstate.jac(i, j);
			rstate.jac(i, j) = -rstate.jac(i, j);
		}
		sum += column_sum;
		rstate.jac(j, j) += fac;
	}
	if (!std::isfinite(sum)) {
		return -1;
	}

	int ierr_linpack = 0;
	ActiveBlock<System, decltype(rstate.jac)> block{rstate.jac};
	quokka::math::dgefa<System::num_active, System::traits::linalg_do_pivoting>(block, rstate.pivot, ierr_linpack);

	amrex::constexpr_for<0, System::num_passive>([&](auto pc) {
		constexpr int p = System::PassiveEntries().arr[decltype(pc)::value];
		if (rstate.jac(p, p) == 0.0_rt) {
			ierr_linpack = (ierr_linpack != 0) ? ierr_linpack : p + 1;
		} else {
			rstate.jac(p, p) = 1.0_rt / rstate.jac(p, p);
		}
	});

	return ierr_linpack;
}

/// Solves M x = b in place for the matrix decompose() factored: the active block by dgesl, then each passive entry p
/// from its row, x_p = (b_p - sum_a M(p, a) x_a) / M(p, p).
template <typename System, typename RosenbrockT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void solve(RosenbrockT &rstate, amrex::GpuArray<amrex::Real, System::neqs> &b)
{
	constexpr int num_active = System::num_active;

	// the active entries of b, gathered and scattered with compile-time indices so that b stays in registers on the GPU
	amrex::GpuArray<amrex::Real, num_active> b_active{};
	amrex::constexpr_for<0, num_active>([&](auto ac) {
		constexpr int a = decltype(ac)::value;
		b_active[a] = b[System::ActiveEntries().arr[a]];
	});
	const ActiveBlock<System, decltype(rstate.jac)> block{rstate.jac};
	quokka::math::dgesl<num_active, System::traits::linalg_do_pivoting>(block, rstate.pivot, b_active);
	amrex::constexpr_for<0, num_active>([&](auto ac) {
		constexpr int a = decltype(ac)::value;
		b[System::ActiveEntries().arr[a]] = b_active[a];
	});

	amrex::constexpr_for<0, System::num_passive>([&](auto pc) {
		constexpr int p = System::PassiveEntries().arr[decltype(pc)::value];
		amrex::Real x = b[p];
		amrex::constexpr_for<0, num_active>([&](auto ac) {
			constexpr int n = System::ActiveEntries().arr[decltype(ac)::value];
			x -= rstate.jac(p, n) * b[n];
		});
		b[p] = x * rstate.jac(p, p);
	});
}

template <typename problem_t, typename Tableau, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
rosenbrock_integrator(BurnT &state, RosenbrockState<ODESystem<problem_t, BurnT>::neqs, Tableau::stages, ODESystem<problem_t, BurnT>::num_active> &rstate,
		      const IntegratorParams &params) -> int
{
	using System = ODESystem<problem_t, BurnT>;
	constexpr int int_neqs = System::neqs;

	if (rstate.tout == rstate.t) {
		return IERR_SUCCESS;
	}

	for (int n = 0; n < int_neqs; ++n) {
		if (counts_toward_error(System::role(n)) &&
		    (atol_for<System>(rstate, n) <= 0.0_rt || rtol_for<System>(rstate, n) <= 10.0_rt * std::numeric_limits<amrex::Real>::epsilon())) {
			return IERR_TOO_MUCH_ACCURACY_REQUESTED;
		}
	}

	rstate.n_rhs = 0;
	rstate.n_jac = 0;
	rstate.n_step = 0;

	rstate.dt = rstate.tout - rstate.t;

	const amrex::Real posneg = rstate.tout >= rstate.t ? 1.0_rt : -1.0_rt;
	// make a copy of the trait to avoid compiler error "undefined in device code" (amrex::min takes references)
	constexpr amrex::Real ode_max_dt = System::traits::ode_max_dt;
	const amrex::Real hmax = amrex::min(ode_max_dt, std::abs(rstate.tout - rstate.t));
	amrex::Real h = amrex::min(std::abs(rstate.dt), hmax) * posneg;
	if (std::abs(h) <= 10.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
		h = 1.e-6_rt * posneg;
	}

	bool reject = false;
	bool last = false;
	int nsing = 0;
	int n_reject = 0;
	// The H211b controller below works with the logs of the previous factor and error (both start at 1), so it costs
	// one log and one exp per step instead of three pow calls.
	amrex::Real log_facold = 0.0_rt;
	amrex::Real log_errold = 0.0_rt;
	amrex::Real hopt = h;
	amrex::Real x = rstate.t;
	amrex::GpuArray<amrex::Real, int_neqs> rhs_tmp{};

	// main evolution loop

	// within this loop, x will be the current time and h will be the
	// step size we are attempting.  The solution at the current time
	// is rstate.y.

	while (true) {

		if (rstate.n_step > params.ode_max_steps) {
			rstate.t = x;
			rstate.dt = h;
			return IERR_TOO_MANY_STEPS;
		}

		if (0.1_rt * std::abs(h) <= std::abs(x) * std::numeric_limits<amrex::Real>::epsilon()) {
			rstate.t = x;
			rstate.dt = h;
			return IERR_DT_UNDERFLOW;
		}

		if (last) {
			rstate.t = x;
			rstate.dt = hopt;
			return IERR_SUCCESS;
		}

		hopt = h;
		if ((x + h * (1.0_rt + timestep_safety_factor) - rstate.tout) * posneg >= 0.0_rt) {
			h = rstate.tout - x;
			last = true;
		}

		// take a step -- this loop will keep trying until a step is
		// successful or a catastrophic error is encountered.

		while (true) {
			rstate.dt = h;

			if (0.1_rt * std::abs(h) <= std::abs(x) * std::numeric_limits<amrex::Real>::epsilon()) {
				rstate.t = x;
				rstate.dt = h;
				return IERR_DT_UNDERFLOW;
			}

			const amrex::Real inv_h = 1.0_rt / h;
			constexpr amrex::Real inv_gamma = 1.0_rt / Tableau::gamma;
			const amrex::Real fac = inv_h * inv_gamma;
			if (!evaluate_jacobian<System>(state, rstate, x, params)) {
				rstate.t = x;
				rstate.dt = h;
				return IERR_JACOBIAN_NOT_FINITE;
			}
			const int ierr_linpack = decompose<System>(rstate, fac);
			if (ierr_linpack < 0) {
				// a non-finite Jacobian entry (checked by evaluate_jacobian only with the finite-difference fallback)
				rstate.t = x;
				rstate.dt = h;
				return IERR_JACOBIAN_NOT_FINITE;
			}

			if (ierr_linpack != 0) {
				nsing += 1;
				if (nsing >= 5) {
					rstate.t = x;
					rstate.dt = h;
					return IERR_LU_DECOMPOSITION_ERROR;
				}
				h *= 0.5_rt;
				reject = true;
				last = false;
				continue;
			}

			solve<System>(rstate, rstate.k[0]);

			// The stage loops are unrolled at compile time so that every stage vector rstate.k[istage - 1] is indexed by a
			// constant: one run-time index into rstate would keep the whole RosenbrockState in scratch memory on the GPU.
			bool valid_stage_state = true;
			amrex::constexpr_for<2, Tableau::stages + 1>([&](auto is) {
				constexpr int istage = decltype(is)::value;
				if (!valid_stage_state) {
					return;
				}
				for (int n = 0; n < int_neqs; ++n) {
					rstate.ynew[n] = rstate.y[n];
					rstate.k[istage - 1][n] = 0.0_rt;

					amrex::constexpr_for<1, istage>([&](auto js) {
						constexpr int jstage = decltype(js)::value;
						const amrex::Real akj = rstate.k[jstage - 1][n];
						rstate.ynew[n] += Tableau::a(istage, jstage) * akj;
						rstate.k[istage - 1][n] += (Tableau::c(istage, jstage) * inv_h) * akj;
					});
				}

				if (!valid_integrator_state<System>(rstate.ynew, params)) {
					h *= 0.25_rt;
					reject = true;
					n_reject += 1;
					last = false;
					valid_stage_state = false;
					return;
				}

				rhs_at<System>(x + Tableau::ct(istage) * h, state, rstate, rstate.ynew, rhs_tmp, params);
				rstate.n_rhs += 1;

				for (int n = 0; n < int_neqs; ++n) {
					rstate.k[istage - 1][n] += rhs_tmp[n];
				}
				solve<System>(rstate, rstate.k[istage - 1]);
			});

			if (!valid_stage_state) {
				continue;
			}

			for (int n = 0; n < int_neqs; ++n) {
				amrex::Real y_update = 0.0_rt;
				amrex::Real err = 0.0_rt;

				amrex::constexpr_for<1, Tableau::stages + 1>([&](auto is) {
					constexpr int istage = decltype(is)::value;
					const amrex::Real aki = rstate.k[istage - 1][n];
					y_update += Tableau::b(istage) * aki;
					err += Tableau::e(istage) * aki;
				});

				rstate.ynew[n] = rstate.y[n] + y_update;
				rstate.work[n] = err;
			}

			rstate.n_step += 1;

			const bool valid_state = valid_integrator_state<System>(rstate.ynew, params);
			if (!valid_state) {
				reject = true;
				n_reject += 1;
				last = false;
				h *= 0.25_rt;
				continue;
			}

			const bool valid_update = species_change_valid<System>(rstate, params);
			constexpr amrex::Real err_min = 1.e-10_rt;
			// make copies of the traits to avoid compiler error "undefined in device code" (amrex::Clamp takes references)
			constexpr amrex::Real fac_min = System::traits::h211b_fac_min;
			constexpr amrex::Real fac_max = System::traits::h211b_fac_max;
			const amrex::Real err = error_norm<System>(rstate);
			// An accepted final step needs no new step size: hnew below then only enters the step-size underflow check at the
			// top of the loop, which no factor in [fac_min, fac_max] can make fire when this condition holds. The controller
			// is then skipped and the smallest factor used.
			const bool final_step_accepted = last && err <= 1.0_rt && valid_update &&
							 0.1_rt * (fac_min * std::abs(h)) > std::abs(x + h) * std::numeric_limits<amrex::Real>::epsilon();
			// H211b: fac = clamp((1 / err)^(1 / (b k)) (1 / errold)^(1 / (b k)) facold^(-1 / b), fac_min, fac_max), with err
			// and errold floored at err_min. In logs the clamp is a clamp of the exponent, so log(facold) needs no log call.
			constexpr amrex::Real inv_bk = 1.0_rt / (System::traits::h211b_b * System::traits::h211b_k);
			constexpr amrex::Real inv_b = 1.0_rt / System::traits::h211b_b;
			constexpr amrex::Real log_fac_min = gcem::log(fac_min);
			constexpr amrex::Real log_fac_max = gcem::log(fac_max);
			amrex::Real controller_fac = fac_min;
			if (!final_step_accepted) {
				const amrex::Real log_err = std::log(amrex::max(err, err_min));
				const amrex::Real log_fac = amrex::Clamp(-(log_err + log_errold) * inv_bk - log_facold * inv_b, log_fac_min, log_fac_max);
				controller_fac = std::exp(log_fac);
				log_facold = log_fac;
				log_errold = log_err;
			}

			amrex::Real hnew = h * controller_fac;

			// update the starting time and state for the next step
			// and predict the next timestep

			if (err <= 1.0_rt && valid_update) {
				for (int n = 0; n < int_neqs; ++n) {
					rstate.y[n] = rstate.ynew[n];
				}
				x += h;

				if (std::abs(hnew) > hmax) {
					hnew = posneg * hmax;
				}
				if (reject) {
					hnew = posneg * amrex::min(std::abs(hnew), std::abs(h));
				}
				reject = false;
				n_reject = 0;
				h = hnew;
				break;
			}

			// if we made it here, then we've failed
			// cut the timestep in prep for trying again

			reject = true;
			n_reject += 1;
			last = false;
			if (n_reject >= 2) {
				hnew *= System::traits::h211b_reduction_fac;
			}
			if (valid_update) {
				h = posneg * amrex::min(std::abs(hnew), System::traits::h211b_reduction_fac * std::abs(h));
			} else {
				h *= 0.25_rt;
			}
		}
	}
}

} // namespace rosenbrock

template <typename Tableau> struct Rosenbrock {
	using tableau = Tableau;

	template <typename problem_t, typename BurnT>
	AMREX_GPU_HOST_DEVICE AMREX_INLINE static void Integrate(BurnT &state, const amrex::Real dt, const IntegratorParams &params)
	{
		using System = ODESystem<problem_t, BurnT>;
		constexpr int int_neqs = System::neqs;

		auto rosenbrock_state = integrator_setup<System, RosenbrockState<int_neqs, Tableau::stages, System::num_active>>(state, dt, params);
		const auto state_save = integrator_backup<System>(state);

		const int istate = rosenbrock::rosenbrock_integrator<problem_t, Tableau>(state, rosenbrock_state, params);

		integrator_cleanup<System>(rosenbrock_state, state, istate, state_save, dt, params);
	}
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_ROSENBROCK_HPP_
