#ifndef MICROPHYSICS_JAFF_NETWORK_ADAPTER_HPP_
#define MICROPHYSICS_JAFF_NETWORK_ADAPTER_HPP_
/// \file jaff_network_adapter.hpp
/// \brief Runs the jaff-generated network (src/networks/photoionization_H_caseB), whose actual_rhs and actual_jac take
/// Microphysics' burn_t, on a BurnState<problem_t>: they copy the state into a burn_t and call the generated functions.
/// The generated files stay untouched, so jaff's regeneration check keeps passing. The adapter goes once jaff emits
/// networks for BurnState.

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "microphysics/burn_state.hpp"

// still from extern/Microphysics and the generated network: burn_t, neqs, ::actual_rhs and ::actual_jac
#include "actual_rhs.H"
#include "burn_type.H"

namespace quokka::microphysics
{

namespace detail
{
/// Copy every member a burn_t shares with a BurnState.
template <typename problem_t> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto to_burn_t(const BurnState<problem_t> &state) -> burn_t
{
	burn_t burn_state;
	burn_state.rho = state.rho;
	burn_state.T = state.T;
	burn_state.e = state.e;
	for (int n = 0; n < NumSpec; ++n) {
		burn_state.xn[n] = state.xn[n];
	}
	for (int n = 0; n < NumChemRadEqs; ++n) {
		burn_state.rn[n] = state.rn[n];
	}
	for (int n = 0; n < NumThermalBands; ++n) {
		burn_state.re_thermal[n] = state.re_thermal[n];
	}
	burn_state.c_hat = state.c_hat;
	burn_state.dedT = state.dedT;
	burn_state.dedr = state.dedr;
	burn_state.mu = state.mu;
	burn_state.mu_e = state.mu_e;
	burn_state.y_e = state.y_e;
	burn_state.abar = state.abar;
	burn_state.zbar = state.zbar;
	burn_state.time = state.time;
	return burn_state;
}
} // namespace detail

template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_rhs(const BurnState<problem_t> &state, amrex::Array1D<amrex::Real, 1, neqs> &ydot)
{
	const burn_t burn_state = detail::to_burn_t(state);
	::actual_rhs(burn_state, ydot);
}

template <typename problem_t, class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_jac(const BurnState<problem_t> &state, MatrixType &jac)
{
	const burn_t burn_state = detail::to_burn_t(state);
	::actual_jac(burn_state, jac);
}

} // namespace quokka::microphysics

#endif // MICROPHYSICS_JAFF_NETWORK_ADAPTER_HPP_
