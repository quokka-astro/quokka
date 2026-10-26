#ifndef MICROPHYSICS_ROSENBROCK_HPP_
#define MICROPHYSICS_ROSENBROCK_HPP_
/// \file rosenbrock.hpp
/// \brief The Rosenbrock integrator. Ported from Microphysics' integration/Rosenbrock/rosenbrock_integrator.H.

#include <cmath>
#include <limits>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "math/linear_algebra.hpp"
#include "microphysics/integrator_common.hpp"
#include "microphysics/integrator_errors.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "microphysics/rosenbrock/rosenbrock_state.hpp"

// still from extern/Microphysics: burn_t, net_ienuc, the integrator parameters and integrator_neqs
#include "burn_type.H"
#include "extern_parameters.H"
#include "integrator_data.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

namespace rosenbrock
{

// rtol_for and atol_for keep the shape of Microphysics' versions instead of switching over the role, during the port
// only. See IntegratorLayout::role().
template <typename Layout, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rtol_for(const RosenbrockT &rstate, const int n) -> amrex::Real
{
	const VarRole role = Layout::role(n);
	if (role == VarRole::Energy) {
		return rstate.rtol_enuc;
	}
	if constexpr (Layout::photochemistry) {
		if (role != VarRole::Species) {
			return role == VarRole::ChemPhotonNumber ? rstate.rtol_rad_num : 1.0e30;
		}
	}
	return rstate.rtol_spec;
}

template <typename Layout, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto atol_for(const RosenbrockT &rstate, const int n) -> amrex::Real
{
	const VarRole role = Layout::role(n);
	if (role == VarRole::Energy) {
		return rstate.atol_enuc;
	}
	if constexpr (Layout::photochemistry) {
		if (role != VarRole::Species) {
			return role == VarRole::ChemPhotonNumber ? rstate.atol_rad_num : 1.0e30;
		}
	}
	return rstate.atol_spec;
}

template <typename Layout, int int_neqs, int stages>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto error_norm(const RosenbrockState<int_neqs, stages> &rstate) -> amrex::Real
{
	amrex::Real err = 0.0_rt;
	for (int n = 1; n <= int_neqs; ++n) {
		const amrex::Real sk = atol_for<Layout>(rstate, n) + rtol_for<Layout>(rstate, n) * amrex::max(std::abs(rstate.y(n)), std::abs(rstate.ynew(n)));
		const amrex::Real term = counts_toward_error(Layout::role(n)) ? rstate.work(n) / sk : 0.0_rt;
		err += term * term;
	}
	// Known bug, kept for bitwise identity with Microphysics and to be fixed later: the sum should be divided by the
	// number of equations that take part in error control, not by int_neqs. The excluded equations add zero terms, so
	// dividing by all of them lowers the norm and accepts steps whose monitored errors exceed the tolerances. See
	// https://github.com/quokka-astro/quokka/pull/2295#discussion_r4051762482 and
	// https://github.com/quokka-astro/quokka/pull/2295#issuecomment-5943124602.
	return std::sqrt(err / static_cast<amrex::Real>(int_neqs));
}

template <int int_neqs, int stages> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto species_change_valid(const RosenbrockState<int_neqs, stages> &rstate) -> bool
{
	constexpr amrex::Real increase_change_factor = 4.0_rt;
	constexpr amrex::Real decrease_change_factor = 0.25_rt;

	for (int i = 1; i <= NumSpec; ++i) {
		if (std::abs(rstate.y(i)) > integrator_rp::X_reject_buffer * rstate.atol_spec &&
		    std::abs(rstate.ynew(i)) > integrator_rp::X_reject_buffer * rstate.atol_spec &&
		    (std::abs(rstate.ynew(i)) > increase_change_factor * std::abs(rstate.y(i)) ||
		     std::abs(rstate.ynew(i)) < decrease_change_factor * std::abs(rstate.y(i)))) {
			return false;
		}
	}

	return true;
}

template <typename MatrixType, int int_neqs> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto matrix_is_finite(const MatrixType &matrix) -> bool
{
	for (int j = 1; j <= int_neqs; ++j) {
		for (int i = 1; i <= int_neqs; ++i) {
			const amrex::Real value = matrix(i, j);
			if (std::isnan(value) || std::isinf(value)) {
				return false;
			}
		}
	}

	return true;
}

template <typename Layout, int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto valid_integrator_state(const amrex::Array1D<amrex::Real, 1, int_neqs> &y) -> bool
{
	for (int n = 1; n <= int_neqs; ++n) {
		const amrex::Real yn = y(n);
		if (std::isnan(yn) || std::isinf(yn)) {
			return false;
		}
	}

	for (int n = 1; n <= int_neqs; ++n) {
		switch (Layout::role(n)) {
			case VarRole::Species:
				if (y(n) < -integrator_rp::atol_spec) {
					return false;
				}
				if (!integrator_rp::use_number_densities && y(n) > 1.0_rt) {
					return false;
				}
				break;
			case VarRole::Energy:
				if (y(n) <= 0.0_rt) {
					return false;
				}
				break;
			case VarRole::ChemPhotonNumber:
				if (y(n) < -integrator_rp::atol_rad_num) {
					return false;
				}
				break;
			case VarRole::ChemPhotonFlux:
			case VarRole::ThermalBandEnergy:
			case VarRole::DustAbsorbedEnergy:
				break;
		}
	}

	return true;
}

template <typename Layout, typename BurnT, int int_neqs, int stages>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void evaluate_jacobian(BurnT &state, RosenbrockState<int_neqs, stages> &rstate, const amrex::Real time)
{
	rhs<Layout>(time, state, rstate, rstate.k[0]);
	rstate.n_rhs += 1;

	if (rstate.jacobian_type == 1) {
		jac<Layout>(state, rstate, rstate.jac);
		rstate.n_jac += 1;

		if (matrix_is_finite<decltype(rstate.jac), int_neqs>(rstate.jac)) {
			return;
		}
	}

	constexpr amrex::Real UROUND = std::numeric_limits<amrex::Real>::epsilon();
	amrex::Array1D<amrex::Real, 1, int_neqs> ewt{};
	amrex::Array1D<amrex::Real, 1, int_neqs> ybase{};

	// rhs() may clean or renormalize all components of rstate.y.
	for (int i = 1; i <= int_neqs; ++i) {
		ybase(i) = rstate.y(i);
	}

	amrex::Real fac = 0.0_rt;
	for (int i = 1; i <= int_neqs; ++i) {
		ewt(i) = 1.0_rt / (rtol_for<Layout>(rstate, i) * std::abs(rstate.y(i)) + atol_for<Layout>(rstate, i));
		fac += (rstate.k[0](i) * ewt(i)) * (rstate.k[0](i) * ewt(i));
	}
	fac = std::sqrt(fac / static_cast<amrex::Real>(int_neqs));

	amrex::Real R0 = 1000.0_rt * std::abs(rstate.dt) * UROUND * static_cast<amrex::Real>(int_neqs) * fac;
	if (R0 == 0.0_rt) {
		R0 = 1.0_rt;
	}

	for (int j = 1; j <= int_neqs; ++j) {
		const amrex::Real yj = ybase(j);
		const amrex::Real R = amrex::max(std::sqrt(UROUND) * std::abs(yj), R0 / ewt(j));
		for (int n = 1; n <= int_neqs; ++n) {
			rstate.y(n) = ybase(n);
		}
		rstate.y(j) += R;

		rhs<Layout>(time, state, rstate, rstate.work);

		fac = 1.0_rt / R;
		for (int i = 1; i <= int_neqs; ++i) {
			rstate.jac.set(i, j, (rstate.work(i) - rstate.k[0](i)) * fac);
		}

		for (int n = 1; n <= int_neqs; ++n) {
			rstate.y(n) = ybase(n);
		}
	}

	rstate.n_rhs += int_neqs;
	rstate.n_jac += 1;
}

template <typename Layout, typename BurnT, int int_neqs, int stages>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs_at(const amrex::Real time, BurnT &state, RosenbrockState<int_neqs, stages> &rstate,
						     const amrex::Array1D<amrex::Real, 1, int_neqs> &y, amrex::Array1D<amrex::Real, 1, int_neqs> &ydot)
{
	amrex::Array1D<amrex::Real, 1, int_neqs> ysave{};

	for (int n = 1; n <= int_neqs; ++n) {
		ysave(n) = rstate.y(n);
		rstate.y(n) = y(n);
	}

	rhs<Layout>(time, state, rstate, ydot);

	for (int n = 1; n <= int_neqs; ++n) {
		rstate.y(n) = ysave(n);
	}
}

template <int int_neqs, int stages>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto decompose(RosenbrockState<int_neqs, stages> &rstate, const amrex::Real fac) -> int
{
	for (int j = 1; j <= int_neqs; ++j) {
		for (int i = 1; i <= int_neqs; ++i) {
			rstate.jac(i, j) = -rstate.jac(i, j);
		}
		rstate.jac(j, j) += fac;
	}

	int ierr_linpack = 0;
	if (integrator_rp::linalg_do_pivoting == 1) {
		constexpr bool allow_pivot{true};
		quokka::math::dgefa<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, ierr_linpack);
	} else {
		constexpr bool allow_pivot{false};
		quokka::math::dgefa<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, ierr_linpack);
	}

	return ierr_linpack;
}

template <int int_neqs, int stages>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void solve(RosenbrockState<int_neqs, stages> &rstate, amrex::Array1D<amrex::Real, 1, int_neqs> &b)
{
	if (integrator_rp::linalg_do_pivoting == 1) {
		constexpr bool allow_pivot{true};
		quokka::math::dgesl<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, b);
	} else {
		constexpr bool allow_pivot{false};
		quokka::math::dgesl<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, b);
	}
}

template <typename problem_t, typename Tableau, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rosenbrock_integrator(BurnT &state, RosenbrockState<integrator_neqs<BurnT>(), Tableau::stages> &rstate) -> int
{
	using Layout = IntegratorLayout<problem_t, BurnT>;
	constexpr int int_neqs = integrator_neqs<BurnT>();

	if (rstate.tout == rstate.t) {
		return IERR_SUCCESS;
	}

	for (int n = 1; n <= int_neqs; ++n) {
		if (atol_for<Layout>(rstate, n) <= 0.0_rt || rtol_for<Layout>(rstate, n) <= 10.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
			return IERR_TOO_MUCH_ACCURACY_REQUESTED;
		}
	}

	rstate.n_rhs = 0;
	rstate.n_jac = 0;
	rstate.n_step = 0;

	rstate.dt = rstate.tout - rstate.t;

	const amrex::Real posneg = rstate.tout >= rstate.t ? 1.0_rt : -1.0_rt;
	const amrex::Real hmax = amrex::min(integrator_rp::ode_max_dt, std::abs(rstate.tout - rstate.t));
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
	amrex::Array1D<amrex::Real, 1, int_neqs> rhs_tmp{};

	// main evolution loop

	// within this loop, x will be the current time and h will be the
	// step size we are attempting.  The solution at the current time
	// is rstate.y().

	while (true) {

		if (rstate.n_step > integrator_rp::ode_max_steps) {
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

			const amrex::Real fac = 1.0_rt / (h * Tableau::gamma);
			evaluate_jacobian<Layout>(state, rstate, x);
			const int ierr_linpack = decompose(rstate, fac);

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

			solve(rstate, rstate.k[0]);

			bool valid_stage_state = true;
			for (int istage = 2; istage <= Tableau::stages; ++istage) {
				for (int n = 1; n <= int_neqs; ++n) {
					rstate.ynew(n) = rstate.y(n);
					rstate.k[istage - 1](n) = 0.0_rt;

					for (int jstage = 1; jstage < istage; ++jstage) {
						const amrex::Real akj = rstate.k[jstage - 1](n);
						rstate.ynew(n) += Tableau::a(istage, jstage) * akj;
						rstate.k[istage - 1](n) += (Tableau::c(istage, jstage) / h) * akj;
					}
				}

				if (!valid_integrator_state<Layout>(rstate.ynew)) {
					h *= 0.25_rt;
					reject = true;
					n_reject += 1;
					last = false;
					valid_stage_state = false;
					break;
				}

				rhs_at<Layout>(x + Tableau::ct(istage) * h, state, rstate, rstate.ynew, rhs_tmp);
				rstate.n_rhs += 1;

				for (int n = 1; n <= int_neqs; ++n) {
					rstate.k[istage - 1](n) += rhs_tmp(n);
				}
				solve(rstate, rstate.k[istage - 1]);
			}

			if (!valid_stage_state) {
				continue;
			}

			for (int n = 1; n <= int_neqs; ++n) {
				amrex::Real y_update = 0.0_rt;
				amrex::Real err = 0.0_rt;

				for (int istage = 1; istage <= Tableau::stages; ++istage) {
					const amrex::Real aki = rstate.k[istage - 1](n);
					y_update += Tableau::b(istage) * aki;
					err += Tableau::e(istage) * aki;
				}

				rstate.ynew(n) = rstate.y(n) + y_update;
				rstate.work(n) = err;
			}

			rstate.n_step += 1;

			const bool valid_state = valid_integrator_state<Layout>(rstate.ynew);
			if (!valid_state) {
				reject = true;
				n_reject += 1;
				last = false;
				h *= 0.25_rt;
				continue;
			}

			const bool valid_update = species_change_valid(rstate);
			constexpr amrex::Real err_min = 1.e-10_rt;
			const amrex::Real err = error_norm<Layout>(rstate);
			const amrex::Real controller_fac =
			    amrex::Clamp(std::pow(1.0_rt / amrex::max(err, err_min), 1.0_rt / (integrator_rp::h211b_b * integrator_rp::h211b_k)) *
					     std::pow(1.0_rt / amrex::max(errold, err_min), 1.0_rt / (integrator_rp::h211b_b * integrator_rp::h211b_k)) *
					     std::pow(facold, -1.0_rt / integrator_rp::h211b_b),
					 integrator_rp::h211b_fac_min, integrator_rp::h211b_fac_max);
			facold = controller_fac;
			errold = err;

			amrex::Real hnew = h * controller_fac;

			// update the starting time and state for the next step
			// and predict the next timestep

			if (err <= 1.0_rt && valid_update) {
				for (int n = 1; n <= int_neqs; ++n) {
					rstate.y(n) = rstate.ynew(n);
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
				hnew *= integrator_rp::h211b_reduction_fac;
			}
			if (valid_update) {
				h = posneg * amrex::min(std::abs(hnew), integrator_rp::h211b_reduction_fac * std::abs(h));
			} else {
				h *= 0.25_rt;
			}
		}
	}
}

} // namespace rosenbrock

template <typename Tableau> struct Rosenbrock {
	using tableau = Tableau;

	template <typename problem_t, typename BurnT> AMREX_GPU_HOST_DEVICE AMREX_INLINE static void Integrate(BurnT &state, const amrex::Real dt)
	{
		using Layout = IntegratorLayout<problem_t, BurnT>;
		constexpr int int_neqs = integrator_neqs<BurnT>();

		auto rosenbrock_state = integrator_setup<Layout, RosenbrockState<int_neqs, Tableau::stages>>(state, dt);
		const auto state_save = integrator_backup<Layout>(state);

		const int istate = rosenbrock::rosenbrock_integrator<problem_t, Tableau>(state, rosenbrock_state);

		integrator_cleanup<Layout>(rosenbrock_state, state, istate, state_save, dt);
	}
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_ROSENBROCK_HPP_
