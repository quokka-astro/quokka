#ifndef MICROPHYSICS_ROSENBROCK_HPP_
#define MICROPHYSICS_ROSENBROCK_HPP_
/// \file rosenbrock.hpp
/// \brief The Rosenbrock integrator. Ported from Microphysics' integration/Rosenbrock/rosenbrock_integrator.H.

#include <cmath>
#include <limits>
#include <type_traits>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "math/linear_algebra.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "microphysics/rosenbrock/rosenbrock_state.hpp"

// still from extern/Microphysics: burn_t, net_ienuc, the integrator parameters, integrator_neqs, rhs and jac
#include "burn_type.H"
#include "extern_parameters.H"
#include "integrator_data.H"
#include "integrator_rhs_strang.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

template <typename problem_t, typename BurnT> struct IntegratorLayout {
	static constexpr bool photochemistry = requires { &BurnT::rn; };

	static constexpr auto NumChemRadEqs() -> int
	{
		if constexpr (photochemistry) {
			return static_cast<int>(std::extent_v<decltype(BurnT::rn)>);
		} else {
			return 0;
		}
	}

	static constexpr auto NumThermalBands() -> int
	{
		if constexpr (photochemistry) {
			return static_cast<int>(std::extent_v<decltype(BurnT::re_thermal)>);
		} else {
			return 0;
		}
	}

	static constexpr int num_chem_rad_eqs = NumChemRadEqs();
	static constexpr int num_thermal_bands = NumThermalBands();
	static constexpr int vars_per_group = Microphysics_Traits<problem_t>::integrate_rad_flux ? 2 : 1;
	static constexpr bool q_dust = requires { &BurnT::e_dust_absorbed; };

	static_assert(num_chem_rad_eqs % vars_per_group == 0, "every chemical band must have vars_per_group variables");
	static_assert(integrator_neqs<BurnT>() == NumSpec + 1 + num_chem_rad_eqs + num_thermal_bands + (q_dust ? 1 : 0),
		      "the layout must match Microphysics' integrator_neqs");
};

namespace rosenbrock
{

/// Only chemical band photon number densities participate in error control.
template <typename Layout> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rad_eq_in_error_control(const int n) -> bool
{
	if (n > net_ienuc + Layout::num_chem_rad_eqs) {
		return false;
	}
	const int group_offset = (n - net_ienuc - 1) % Layout::vars_per_group;
	return group_offset == 0;
}

template <typename Layout, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rtol_for(const RosenbrockT &rstate, const int n) -> amrex::Real
{
	if (n == net_ienuc) {
		return rstate.rtol_enuc;
	}
	if constexpr (Layout::photochemistry) {
		if (n > net_ienuc) {
			return rad_eq_in_error_control<Layout>(n) ? rstate.rtol_rad_num : 1.0e30;
		}
	}
	return rstate.rtol_spec;
}

template <typename Layout, typename RosenbrockT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto atol_for(const RosenbrockT &rstate, const int n) -> amrex::Real
{
	if (n == net_ienuc) {
		return rstate.atol_enuc;
	}
	if constexpr (Layout::photochemistry) {
		if (n > net_ienuc) {
			return rad_eq_in_error_control<Layout>(n) ? rstate.atol_rad_num : 1.0e30;
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
		amrex::Real term{};
		if constexpr (Layout::photochemistry) {
			if (n > net_ienuc && !rad_eq_in_error_control<Layout>(n)) {
				term = 0.0_rt;
			} else {
				term = rstate.work(n) / sk;
			}
		} else {
			term = rstate.work(n) / sk;
		}
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

	for (int n = 1; n <= NumSpec; ++n) {
		if (y(n) < -integrator_rp::atol_spec) {
			return false;
		}

		if (!integrator_rp::use_number_densities && y(n) > 1.0_rt) {
			return false;
		}
	}

	if constexpr (Layout::photochemistry) {
		for (int n = net_ienuc + 1; n <= net_ienuc + Layout::num_chem_rad_eqs; n += Layout::vars_per_group) {
			if (y(n) < -integrator_rp::atol_rad_num) {
				return false;
			}
		}
	}

	if (y(net_ienuc) <= 0.0_rt) {
		return false;
	}

	return true;
}

template <typename Layout, typename BurnT, int int_neqs, int stages>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void evaluate_jacobian(BurnT &state, RosenbrockState<int_neqs, stages> &rstate, const amrex::Real time)
{
	constexpr bool in_jacobian = true;

	::rhs(time, state, rstate, rstate.k[0], in_jacobian);
	rstate.n_rhs += 1;

	if (rstate.jacobian_type == 1) {
		::jac(time, state, rstate, rstate.jac);
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

		::rhs(time, state, rstate, rstate.work, in_jacobian);

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

template <typename BurnT, int int_neqs, int stages>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs_at(const amrex::Real time, BurnT &state, RosenbrockState<int_neqs, stages> &rstate,
						     const amrex::Array1D<amrex::Real, 1, int_neqs> &y, amrex::Array1D<amrex::Real, 1, int_neqs> &ydot)
{
	amrex::Array1D<amrex::Real, 1, int_neqs> ysave{};

	for (int n = 1; n <= int_neqs; ++n) {
		ysave(n) = rstate.y(n);
		rstate.y(n) = y(n);
	}

	::rhs(time, state, rstate, ydot);

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

} // namespace rosenbrock

} // namespace quokka::microphysics

#endif // MICROPHYSICS_ROSENBROCK_HPP_
