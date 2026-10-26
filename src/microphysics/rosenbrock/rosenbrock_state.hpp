#ifndef MICROPHYSICS_ROSENBROCK_STATE_HPP_
#define MICROPHYSICS_ROSENBROCK_STATE_HPP_
/// \file rosenbrock_state.hpp
/// \brief The state of the Rosenbrock integrator. Ported from Microphysics' integration/Rosenbrock/rosenbrock_type.H.

#include <cstdint>

#include "AMReX_Array.H"
#include "AMReX_REAL.H"

#include "microphysics/array_utils.hpp"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// When checking the integration time to see if we're done, be careful with roundoff issues.
constexpr amrex::Real timestep_safety_factor = 1.0e-4_rt;

template <int int_neqs, int stages> struct RosenbrockState {
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

	amrex::Array1D<amrex::Real, 1, int_neqs> y;
	amrex::Array1D<amrex::Real, 1, int_neqs> ynew;
	amrex::GpuArray<amrex::Array1D<amrex::Real, 1, int_neqs>, stages> k; // k[i - 1] is the stage vector of stage i
	amrex::Array1D<amrex::Real, 1, int_neqs> work;

	MathArray2D<int_neqs> jac;
	amrex::Array1D<std::int16_t, 1, int_neqs> pivot;

	std::int16_t jacobian_type;
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_ROSENBROCK_STATE_HPP_
