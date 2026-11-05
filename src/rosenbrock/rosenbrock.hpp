#ifndef ROSENBROCK_ROSENBROCK_HPP_
#define ROSENBROCK_ROSENBROCK_HPP_
/// \file rosenbrock.hpp
/// \brief A self-contained adaptive Rosenbrock (ros2s) integrator for one small stiff ODE system, usable in device code.
///
/// The control flow, error norm, H211b step-size controller and state checks are ported from Microphysics
/// (integration/Rosenbrock/rosenbrock_integrator.H). Differences: one tableau (ros2s); no numerical Jacobian (a
/// non-finite analytic Jacobian is treated like a singular matrix: the step is rejected and h is halved); and the
/// problem-specific checks are expressed through Settings. This directory includes only AMReX headers.

#include <cmath>
#include <limits>

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "rosenbrock/dense_lu.hpp"
#include "rosenbrock/ros2s_tableau.hpp"

namespace rosenbrock
{

enum class Status : int { Success = 0, TooManySteps, StepSizeUnderflow, SingularMatrix, ToleranceTooSmall };

template <int N> struct Settings {
	Vector<N> atol;				///< absolute tolerance; must be > 0 for every component in the error norm
	Vector<N> rtol;				///< relative tolerance; must be > 10 eps for every component in the error norm
	amrex::GpuArray<bool, N> in_error_norm; ///< false: left out of the sum of the RMS error norm (the sum is still divided by N)
	Vector<N> min_value;			///< a stage or step with y(n) < min_value(n) is rejected; -inf for no limit
	Vector<N> change_floor; ///< where |y_old| and |y_new| both exceed this, a step that changes |y| by more than 4x is rejected; +inf for none
	int max_steps;		///< maximum number of steps
};

struct Result {
	Status status;
	int n_step;
	int n_rhs;
	int n_jac;
};

namespace detail
{
// H211b step-size controller and safety constants (Microphysics defaults)
constexpr amrex::Real h211b_b = 4.0;
constexpr amrex::Real h211b_k = 2.5;
constexpr amrex::Real h211b_fac_min = 0.2;
constexpr amrex::Real h211b_fac_max = 6.0;
constexpr amrex::Real h211b_reduction_fac = 0.5;
constexpr amrex::Real err_min = 1.0e-10;
constexpr amrex::Real timestep_safety_factor = 1.0e-4;
constexpr amrex::Real increase_change_factor = 4.0;
constexpr amrex::Real decrease_change_factor = 0.25;
constexpr int max_singular = 5;

template <int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto error_norm(Settings<N> const &s, Vector<N> const &y, Vector<N> const &ynew, Vector<N> const &err) -> amrex::Real
{
	amrex::Real sum = 0.0;
	for (int n = 0; n < N; ++n) {
		if (s.in_error_norm[n]) {
			const amrex::Real sk = s.atol[n] + s.rtol[n] * amrex::max(std::abs(y[n]), std::abs(ynew[n]));
			const amrex::Real term = err[n] / sk;
			sum += term * term;
		}
	}
	return std::sqrt(sum / static_cast<amrex::Real>(N));
}

template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto valid_state(Settings<N> const &s, Vector<N> const &y) -> bool
{
	for (int n = 0; n < N; ++n) {
		if (std::isnan(y[n]) || std::isinf(y[n]) || y[n] < s.min_value[n]) {
			return false;
		}
	}
	return true;
}

template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto change_valid(Settings<N> const &s, Vector<N> const &y, Vector<N> const &ynew) -> bool
{
	for (int n = 0; n < N; ++n) {
		const amrex::Real yo = std::abs(y[n]);
		const amrex::Real yn = std::abs(ynew[n]);
		if (yo > s.change_floor[n] && yn > s.change_floor[n] && (yn > increase_change_factor * yo || yn < decrease_change_factor * yo)) {
			return false;
		}
	}
	return true;
}

template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto matrix_is_finite(Matrix<N> const &m) -> bool
{
	for (int j = 0; j < N; ++j) {
		for (int i = 0; i < N; ++i) {
			if (std::isnan(m(i, j)) || std::isinf(m(i, j))) {
				return false;
			}
		}
	}
	return true;
}
} // namespace detail

/// Integrates y' = f(t, y) from t = 0 to t = dt; y holds the initial state on entry and the final state on return.
///
/// System must provide, as host-device member functions,
///   void rhs(amrex::Real t, Vector<N> &y, Vector<N> &ydot);       // may modify ("clean") y in place
///   void jac(amrex::Real t, Vector<N> const &y, Matrix<N> &J);    // J(i, j) = d ydot_i / d y_j; called right after rhs at the same y
/// rhs is called on the accepted state itself at the start of every step attempt (so its cleaning persists) and on a
/// copy for the later stages.
template <int N, typename System> AMREX_GPU_HOST_DEVICE auto integrate(System &sys, Vector<N> &y, const amrex::Real dt, Settings<N> const &settings) -> Result
{
	using T = ros2s_tableau;
	constexpr amrex::Real eps = std::numeric_limits<amrex::Real>::epsilon();

	Result res{.status = Status::Success, .n_step = 0, .n_rhs = 0, .n_jac = 0};
	if (dt == 0.0) {
		return res;
	}
	for (int n = 0; n < N; ++n) {
		if (settings.in_error_norm[n] && (settings.atol[n] <= 0.0 || settings.rtol[n] <= 10.0 * eps)) {
			res.status = Status::ToleranceTooSmall;
			return res;
		}
	}

	const amrex::Real tout = dt;
	const amrex::Real posneg = (tout >= 0.0) ? 1.0 : -1.0;
	const amrex::Real hmax = std::abs(tout);
	amrex::Real h = hmax * posneg;
	if (std::abs(h) <= 10.0 * eps) {
		h = 1.0e-6 * posneg;
	}

	bool reject = false;
	bool last = false;
	int nsing = 0;
	int n_reject = 0;
	amrex::Real facold = 1.0;
	amrex::Real errold = 1.0;
	amrex::Real x = 0.0;

	Matrix<N> jac{};
	Pivots<N> pivot{};
	amrex::GpuArray<Vector<N>, T::stages> k{};
	Vector<N> ynew{};
	Vector<N> ystage{};
	Vector<N> rhs_tmp{};
	Vector<N> err{};

	while (true) {
		if (res.n_step > settings.max_steps) {
			res.status = Status::TooManySteps;
			return res;
		}
		if (0.1 * std::abs(h) <= std::abs(x) * eps) {
			res.status = Status::StepSizeUnderflow;
			return res;
		}
		if (last) {
			return res;
		}
		if ((x + h * (1.0 + detail::timestep_safety_factor) - tout) * posneg >= 0.0) {
			h = tout - x;
			last = true;
		}

		// try steps until one is accepted or the integration fails
		while (true) {
			if (0.1 * std::abs(h) <= std::abs(x) * eps) {
				res.status = Status::StepSizeUnderflow;
				return res;
			}

			// f and J at the accepted state; rhs may clean y in place
			sys.rhs(x, y, k[0]);
			res.n_rhs += 1;
			sys.jac(x, y, jac);
			res.n_jac += 1;

			// decompose (1 / (h gamma)) I - J
			int ierr = 1;
			if (detail::matrix_is_finite<N>(jac)) {
				const amrex::Real fac = 1.0 / (h * T::gamma);
				for (int j = 0; j < N; ++j) {
					for (int i = 0; i < N; ++i) {
						jac(i, j) = -jac(i, j);
					}
					jac(j, j) += fac;
				}
				ierr = lu_factor<N>(jac, pivot);
			}
			if (ierr != 0) {
				nsing += 1;
				if (nsing >= detail::max_singular) {
					res.status = Status::SingularMatrix;
					return res;
				}
				h *= 0.5;
				reject = true;
				last = false;
				continue;
			}

			lu_solve<N>(jac, pivot, k[0]);

			bool valid_stage_state = true;
			for (int istage = 2; istage <= T::stages; ++istage) {
				Vector<N> &ki = k[istage - 1];
				for (int n = 0; n < N; ++n) {
					ynew[n] = y[n];
					ki[n] = 0.0;
					for (int jstage = 1; jstage < istage; ++jstage) {
						const amrex::Real akj = k[jstage - 1][n];
						ynew[n] += T::a(istage, jstage) * akj;
						ki[n] += (T::c(istage, jstage) / h) * akj;
					}
				}
				if (!detail::valid_state<N>(settings, ynew)) {
					h *= 0.25;
					reject = true;
					n_reject += 1;
					last = false;
					valid_stage_state = false;
					break;
				}
				ystage = ynew; // rhs may clean its argument; the stage state itself is not changed
				sys.rhs(x + T::ct(istage) * h, ystage, rhs_tmp);
				res.n_rhs += 1;
				for (int n = 0; n < N; ++n) {
					ki[n] += rhs_tmp[n];
				}
				lu_solve<N>(jac, pivot, ki);
			}
			if (!valid_stage_state) {
				continue;
			}

			for (int n = 0; n < N; ++n) {
				amrex::Real y_update = 0.0;
				amrex::Real e = 0.0;
				for (int istage = 1; istage <= T::stages; ++istage) {
					y_update += T::b(istage) * k[istage - 1][n];
					e += T::e(istage) * k[istage - 1][n];
				}
				ynew[n] = y[n] + y_update;
				err[n] = e;
			}
			res.n_step += 1;

			if (!detail::valid_state<N>(settings, ynew)) {
				reject = true;
				n_reject += 1;
				last = false;
				h *= 0.25;
				continue;
			}

			const bool valid_update = detail::change_valid<N>(settings, y, ynew);
			const amrex::Real errnorm = detail::error_norm<N>(settings, y, ynew, err);
			// An accepted final step needs no new step size: hnew below then only enters the step-size underflow check at the
			// top of the loop, which no factor in [h211b_fac_min, h211b_fac_max] can make fire when this condition holds. The
			// controller (three pow calls) is then skipped and the smallest factor used.
			const bool final_step_accepted =
			    last && errnorm <= 1.0 && valid_update && 0.1 * (detail::h211b_fac_min * std::abs(h)) > std::abs(x + h) * eps;
			const amrex::Real exponent = 1.0 / (detail::h211b_b * detail::h211b_k);
			const amrex::Real controller_fac =
			    final_step_accepted ? detail::h211b_fac_min
						: std::fmin(std::fmax(std::pow(1.0 / std::fmax(errnorm, detail::err_min), exponent) *
									  std::pow(1.0 / std::fmax(errold, detail::err_min), exponent) *
									  std::pow(facold, -1.0 / detail::h211b_b),
								      detail::h211b_fac_min),
							    detail::h211b_fac_max);
			facold = controller_fac;
			errold = errnorm;
			amrex::Real hnew = h * controller_fac;

			if (errnorm <= 1.0 && valid_update) {
				y = ynew;
				x += h;
				if (std::abs(hnew) > hmax) {
					hnew = posneg * hmax;
				}
				if (reject) {
					hnew = posneg * std::fmin(std::abs(hnew), std::abs(h));
				}
				reject = false;
				n_reject = 0;
				h = hnew;
				break;
			}

			// rejected: cut the step and try again
			reject = true;
			n_reject += 1;
			last = false;
			if (n_reject >= 2) {
				hnew *= detail::h211b_reduction_fac;
			}
			if (valid_update) {
				h = posneg * std::fmin(std::abs(hnew), detail::h211b_reduction_fac * std::abs(h));
			} else {
				h *= 0.25;
			}
		}
	}
}

} // namespace rosenbrock

#endif // ROSENBROCK_ROSENBROCK_HPP_
