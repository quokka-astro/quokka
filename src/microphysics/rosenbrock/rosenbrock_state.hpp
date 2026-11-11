// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef MICROPHYSICS_ROSENBROCK_STATE_HPP_
#define MICROPHYSICS_ROSENBROCK_STATE_HPP_
/// \file rosenbrock_state.hpp
/// \brief The state of the Rosenbrock integrator. Ported from Microphysics' integration/Rosenbrock/rosenbrock_type.H.

#include <cstdint>

#include "AMReX_Array.H"
#include "AMReX_REAL.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// When checking the integration time to see if we're done, be careful with roundoff issues.
constexpr amrex::Real timestep_safety_factor = 1.0e-4_rt;

/// num_active is the size of the block of the integrator matrix that is factored (ODESystem::num_active).
template <int int_neqs, int stages, int num_active = int_neqs> struct RosenbrockState {
	amrex::Real t;	  // the starting time
	amrex::Real tout; // the stopping time
	amrex::Real dt;	  // next internal timestep

	int n_step;
	int n_rhs;
	int n_jac;

	amrex::Real atol_spec;
	amrex::Real rtol_spec;

	amrex::Real atol_enuc;
	amrex::Real rtol_enuc;

	// Used only with photochemistry, where the setup sets them.
	amrex::Real atol_rad_num;
	amrex::Real rtol_rad_num;
	amrex::Real c_hat;

	amrex::GpuArray<amrex::Real, int_neqs> y;
	amrex::GpuArray<amrex::Real, int_neqs> ynew;
	amrex::GpuArray<amrex::GpuArray<amrex::Real, int_neqs>, stages> k; // k[i - 1] is the stage vector of stage i
	amrex::GpuArray<amrex::Real, int_neqs> work;

	amrex::Array2D<amrex::Real, 0, int_neqs - 1, 0, int_neqs - 1> jac; // stored column by column
	amrex::GpuArray<std::int16_t, num_active> pivot;		   // the row swaps of the factored active block
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_ROSENBROCK_STATE_HPP_
