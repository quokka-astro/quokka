// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef MATH_ROSENBROCK_HPP_
#define MATH_ROSENBROCK_HPP_
/// \file rosenbrock.hpp
/// \brief An adaptive Rosenbrock integrator for small stiff ODE systems y' = f(t, y), one system per GPU thread.
/// Ported from Microphysics' integration/Rosenbrock/rosenbrock_integrator.H.

#include <cmath>
#include <cstdint>
#include <limits>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_ConstexprFor.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "math/linear_algebra.hpp"
#include "math/rosenbrock/tableaus.hpp"

namespace quokka::math::rosenbrock
{

using amrex::literals::operator""_rt;

/// Return codes of integrate(), with the values of Microphysics' integration/integrator_data.H (the gaps are its codes
/// for other integrators; IERR_JACOBIAN_NOT_FINITE is ours), so that a failed burn prints the same code as before the
/// port.
// TODO(james471): once Microphysics is removed, number these consecutively and make this an enum class (dropping the
// NOLINT).
enum IntegratorError : std::int8_t { // NOLINT(cppcoreguidelines-use-enum-class)
	IERR_SUCCESS = 1,
	IERR_DT_UNDERFLOW = -2,
	IERR_TOO_MANY_STEPS = -4,
	IERR_TOO_MUCH_ACCURACY_REQUESTED = -5,
	IERR_LU_DECOMPOSITION_ERROR = -7,
	IERR_JACOBIAN_NOT_FINITE = -9
};

/// When checking the integration time to see if we're done, be careful with roundoff issues.
constexpr amrex::Real timestep_safety_factor = 1.0e-4_rt;

/// The H211b step-size controller (Microphysics' defaults).
constexpr amrex::Real h211b_b = 4.0_rt;
constexpr amrex::Real h211b_k = 2.5_rt;
constexpr amrex::Real h211b_fac_min = 0.2_rt;
constexpr amrex::Real h211b_fac_max = 6.0_rt;
constexpr amrex::Real h211b_reduction_fac = 0.5_rt;

/// Compile-time settings of the integrator. A caller passes its own struct with the same members.
struct DefaultOptions {
	static constexpr bool analytic_jacobian = true;
	/// With analytic_jacobian, replace an analytic Jacobian that has a non-finite entry by finite differences. With
	/// false, such a step fails with IERR_JACOBIAN_NOT_FINITE.
	static constexpr bool numerical_jacobian_fallback = false;
	/// LU decomposition with partial pivoting.
	static constexpr bool pivoting = true;
	/// The largest internal timestep.
	static constexpr amrex::Real max_dt = 1.e30;
};

/// The state of the integrator for a system of neqs equations.
template <int neqs, int stages> struct State {
	amrex::Real t;	  // the starting time; on return, the time reached
	amrex::Real tout; // the stopping time
	amrex::Real dt;	  // the next internal timestep

	int n_step;
	int n_rhs;
	int n_jac;

	amrex::GpuArray<amrex::Real, neqs> y; // the solution at t
	amrex::GpuArray<amrex::Real, neqs> ynew;
	amrex::GpuArray<amrex::GpuArray<amrex::Real, neqs>, stages> k; // k[i - 1] is the stage vector of stage i
	amrex::GpuArray<amrex::Real, neqs> work;

	amrex::Array2D<amrex::Real, 0, neqs - 1, 0, neqs - 1> jac; // the Jacobian, then the factors of the integrator matrix
	amrex::GpuArray<std::int16_t, neqs> pivot;		   // the row swaps of the factorization
};

namespace detail
{

/// The divisor of the root mean square in error_norm. This accounts for passive variables that do not enter the norm.
template <typename System> constexpr auto ErrorNormDivisor() -> int
{
	if constexpr (requires { System::error_norm_divisor; }) {
		return System::error_norm_divisor;
	} else {
		int count = 0;
		for (int n = 0; n < System::neqs; ++n) {
			if (System::counts_toward_error(n)) {
				++count;
			}
		}
		return count;
	}
}

template <typename System, typename StateT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto error_norm(const System &sys, const StateT &rstate) -> amrex::Real
{
	constexpr int int_neqs = System::neqs;

	amrex::Real err = 0.0_rt;
	for (int n = 0; n < int_neqs; ++n) {
		const amrex::Real sk = sys.atol(n) + sys.rtol(n) * amrex::max(std::abs(rstate.y[n]), std::abs(rstate.ynew[n]));
		const amrex::Real term = System::counts_toward_error(n) ? rstate.work[n] / sk : 0.0_rt;
		err += term * term;
	}
	return std::sqrt(err / static_cast<amrex::Real>(ErrorNormDivisor<System>()));
}

/// Whether every entry of the Jacobian is finite: decides whether numerical_jacobian_fallback replaces it.
template <typename System, typename MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto matrix_is_finite(const MatrixType &matrix) -> bool
{
	constexpr int int_neqs = System::neqs;

	// A NaN or an infinity in any entry makes the sum of all entries non-finite, so one check of the sum settles the
	// usual case. Only a non-finite sum needs the entries checked one by one (the sum of finite entries can overflow).
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

/// The finite-difference Jacobian, from neqs evaluations of the right-hand side around rstate.y. Expects the
/// right-hand side at rstate.y in rstate.k[0].
template <typename System, typename StateT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void numerical_jacobian(System &sys, StateT &rstate, const amrex::Real time)
{
	// TODO (james471)): This code and its branch for passive variables is not tested.
	constexpr int int_neqs = System::neqs;
	constexpr amrex::Real UROUND = std::numeric_limits<amrex::Real>::epsilon();
	amrex::GpuArray<amrex::Real, int_neqs> ewt{};
	amrex::GpuArray<amrex::Real, int_neqs> ybase{};

	// rhs() may change rstate.y (e.g. clip or renormalize it).
	for (int i = 0; i < int_neqs; ++i) {
		ybase[i] = rstate.y[i];
	}

	amrex::Real fac = 0.0_rt;
	for (int i = 0; i < int_neqs; ++i) {
		ewt[i] = 1.0_rt / (sys.rtol(i) * std::abs(rstate.y[i]) + sys.atol(i));
		fac += (rstate.k[0][i] * ewt[i]) * (rstate.k[0][i] * ewt[i]);
	}
	fac = std::sqrt(fac / static_cast<amrex::Real>(int_neqs));

	amrex::Real R0 = 1000.0_rt * std::abs(rstate.dt) * UROUND * static_cast<amrex::Real>(int_neqs) * fac;
	if (R0 == 0.0_rt) {
		R0 = 1.0_rt;
	}

	for (int j = 0; j < int_neqs; ++j) {
		const amrex::Real yj = ybase[j];
		const amrex::Real R = amrex::max(std::sqrt(UROUND) * std::abs(yj), R0 / ewt[j]);
		for (int n = 0; n < int_neqs; ++n) {
			rstate.y[n] = ybase[n];
		}
		rstate.y[j] += R;

		sys.rhs(time, rstate.y, rstate.work);

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

/// Evaluates the right-hand side at rstate.y into rstate.k[0] and the Jacobian into rstate.jac. With
/// numerical_jacobian_fallback, an analytic Jacobian with a non-finite entry is replaced by the finite-difference one.
template <typename System, typename Options, typename StateT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void evaluate_jacobian(System &sys, StateT &rstate, const amrex::Real time)
{
	sys.rhs(time, rstate.y, rstate.k[0]);
	rstate.n_rhs += 1;

	if constexpr (!Options::analytic_jacobian) {
		numerical_jacobian<System>(sys, rstate, time);
	} else {
		sys.jac(time, rstate.y, rstate.jac);
		rstate.n_jac += 1;

		if constexpr (Options::numerical_jacobian_fallback) {
			if (!matrix_is_finite<System>(rstate.jac)) {
				numerical_jacobian<System>(sys, rstate, time);
			}
		}
	}
}

/// The right-hand side at y, evaluated with y in rstate.y so that the System sees the same state as in its other calls.
template <typename System, typename StateT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs_at(System &sys, const amrex::Real time, StateT &rstate, const amrex::GpuArray<amrex::Real, System::neqs> &y,
						     amrex::GpuArray<amrex::Real, System::neqs> &ydot)
{
	constexpr int int_neqs = System::neqs;

	amrex::GpuArray<amrex::Real, int_neqs> ysave{};

	for (int n = 0; n < int_neqs; ++n) {
		ysave[n] = rstate.y[n];
		rstate.y[n] = y[n];
	}

	sys.rhs(time, rstate.y, ydot);

	for (int n = 0; n < int_neqs; ++n) {
		rstate.y[n] = ysave[n];
	}
}

/// Forms the integrator matrix fac*I - J in rstate.jac and factors it with dgefa, whose info it returns:
///  - 0 on success;
///  - -1 if the integrator matrix has a non-finite entry. This is the only check of J when numerical_jacobian_fallback is off, and of a finite-difference J;
///  - k + 1 > 0 if the pivot of column k is zero (the matrix is singular for this step size).
template <typename System, typename Options, typename StateT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto decompose(StateT &rstate, const amrex::Real fac) -> int
{
	constexpr int int_neqs = System::neqs;

	for (int j = 0; j < int_neqs; ++j) {
		for (int i = 0; i < int_neqs; ++i) {
			rstate.jac(i, j) = -rstate.jac(i, j);
		}
		rstate.jac(j, j) += fac;
	}

	int ierr_linpack = 0;
	quokka::math::dgefa<int_neqs, Options::pivoting>(rstate.jac, rstate.pivot, ierr_linpack);

	return ierr_linpack;
}

template <typename System, typename Options, typename StateT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void solve(StateT &rstate, amrex::GpuArray<amrex::Real, System::neqs> &b)
{
	quokka::math::dgesl<System::neqs, Options::pivoting>(rstate.jac, rstate.pivot, b);
}

} // namespace detail

/// Integrates the System from rstate.t to rstate.tout, starting from rstate.y, with the Rosenbrock method Tableau
/// (tableaus.hpp). On return rstate.y holds the solution at rstate.t, which is rstate.tout on success, and rstate.n_step,
/// n_rhs and n_jac count the work. Returns IERR_SUCCESS or a negative IntegratorError.
/// Port only: a System may define `static constexpr int error_norm_divisor` to divide the error norm's sum of squares by
/// another count than the number of entries that count toward the error (Microphysics divides by neqs).
template <typename Tableau, typename Options = DefaultOptions, typename System>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto integrate(System &sys, State<System::neqs, Tableau::stages> &rstate, const int max_steps) -> int
{
	constexpr int int_neqs = System::neqs;

	if (rstate.tout == rstate.t) {
		return IERR_SUCCESS;
	}

	for (int n = 0; n < int_neqs; ++n) {
		if (sys.atol(n) <= 0.0_rt || sys.rtol(n) <= 10.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
			return IERR_TOO_MUCH_ACCURACY_REQUESTED;
		}
	}

	rstate.n_rhs = 0;
	rstate.n_jac = 0;
	rstate.n_step = 0;

	rstate.dt = rstate.tout - rstate.t;

	const amrex::Real posneg = rstate.tout >= rstate.t ? 1.0_rt : -1.0_rt;
	// make a copy of the option to avoid compiler error "undefined in device code" (amrex::min takes references)
	constexpr amrex::Real max_dt = Options::max_dt;
	const amrex::Real hmax = amrex::min(max_dt, std::abs(rstate.tout - rstate.t));
	amrex::Real h = amrex::min(std::abs(rstate.dt), hmax) * posneg;
	if (std::abs(h) <= 10.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
		h = 1.e-6_rt * posneg;
	}

	bool reject = false;
	bool last = false;
	int nsing = 0;
	int n_reject = 0;
	amrex::Real facold = 1.0_rt;
	amrex::Real errold = 1.0_rt;
	amrex::Real hopt = h;
	amrex::Real x = rstate.t;
	amrex::GpuArray<amrex::Real, int_neqs> rhs_tmp{};

	// main evolution loop

	// within this loop, x will be the current time and h will be the
	// step size we are attempting.  The solution at the current time
	// is rstate.y.

	while (true) {

		if (rstate.n_step > max_steps) {
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

			constexpr amrex::Real min_h_gamma = 1.0_rt / std::numeric_limits<amrex::Real>::max();
			if (0.1_rt * std::abs(h) <= std::abs(x) * std::numeric_limits<amrex::Real>::epsilon() || std::abs(h * Tableau::gamma) <= min_h_gamma) {
				rstate.t = x;
				rstate.dt = h;
				return IERR_DT_UNDERFLOW;
			}

			const amrex::Real fac = 1.0_rt / (h * Tableau::gamma);
			detail::evaluate_jacobian<System, Options>(sys, rstate, x);
			const int ierr_linpack = detail::decompose<System, Options>(rstate, fac);

			if (ierr_linpack < 0) {
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

			detail::solve<System, Options>(rstate, rstate.k[0]);

			// The stage loops are unrolled at compile time so that every stage vector rstate.k[istage - 1] is indexed by a constant.
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
						rstate.k[istage - 1][n] += (Tableau::c(istage, jstage) / h) * akj;
					});
				}

				if (!sys.valid_state(rstate.ynew)) {
					h *= 0.25_rt;
					reject = true;
					n_reject += 1;
					last = false;
					valid_stage_state = false;
					return;
				}

				detail::rhs_at<System>(sys, x + Tableau::ct(istage) * h, rstate, rstate.ynew, rhs_tmp);
				rstate.n_rhs += 1;

				for (int n = 0; n < int_neqs; ++n) {
					rstate.k[istage - 1][n] += rhs_tmp[n];
				}
				detail::solve<System, Options>(rstate, rstate.k[istage - 1]);
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

			const bool valid_state = sys.valid_state(rstate.ynew);
			if (!valid_state) {
				reject = true;
				n_reject += 1;
				last = false;
				h *= 0.25_rt;
				continue;
			}

			const bool valid_update = sys.valid_update(rstate.y, rstate.ynew);
			constexpr amrex::Real err_min = 1.e-10_rt;
			// local copies to avoid compiler error "undefined in device code" (amrex::Clamp takes references)
			constexpr amrex::Real fac_min = h211b_fac_min;
			constexpr amrex::Real fac_max = h211b_fac_max;
			const amrex::Real err = detail::error_norm(sys, rstate);
			// An accepted final step needs no new step size: hnew below then only enters the step-size underflow check at the
			// top of the loop, which no factor in [fac_min, fac_max] can make fire when this condition holds. The controller
			// (three pow calls) is then skipped and the smallest factor used.
			const bool final_step_accepted = last && err <= 1.0_rt && valid_update &&
							 0.1_rt * (fac_min * std::abs(h)) > std::abs(x + h) * std::numeric_limits<amrex::Real>::epsilon();
			const amrex::Real controller_fac = final_step_accepted
							       ? fac_min
							       : amrex::Clamp(std::pow(1.0_rt / amrex::max(err, err_min), 1.0_rt / (h211b_b * h211b_k)) *
										  std::pow(1.0_rt / amrex::max(errold, err_min), 1.0_rt / (h211b_b * h211b_k)) *
										  std::pow(facold, -1.0_rt / h211b_b),
									      fac_min, fac_max);
			facold = controller_fac;
			errold = err;

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
				hnew *= h211b_reduction_fac;
			}
			if (valid_update) {
				h = posneg * amrex::min(std::abs(hnew), h211b_reduction_fac * std::abs(h));
			} else {
				h *= 0.25_rt;
			}
		}
	}
}

} // namespace quokka::math::rosenbrock

#endif // MATH_ROSENBROCK_HPP_
