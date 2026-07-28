#ifndef actual_rhs_H
#define actual_rhs_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <cmath>

#include <actual_network.H>
#include <burn_type.H>
#include <extern_parameters.H>
#include <jacobian_utilities.H>

using namespace ArrayUtil;
using namespace network_rp;

AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_rhs(const burn_t &state, amrex::Array1D<amrex::Real, 1, neqs> &ydot)
{

	using namespace Rates;

	// $JAFF REPEAT idx, rhs, cse IN rhses $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T RADIATION True]$
	const amrex::Real cse0 = 6.4745807555437135e-8 * state.xn[0];
	const amrex::Real cse1 = cse0 * state.rn[0];
	const amrex::Real cse2 = cse1 - 2.6299999999999999e-13 * state.xn[1] * state.xn[2];

	ydot(1) = -cse2;
	ydot(2) = cse2;
	ydot(3) = cse2;
	ydot(4) = 0;
	ydot(5) = -cse1;
	ydot(6) = -cse0 * state.rn[2 * 0 + 1];
	// $JAFF END
	// $JAFF SUB nspec
	ydot(4) /= state.rho;
	// $JAFF END
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{

	// $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE dedt state.dedT RADIATION True USE_DEDT
	// True]$
	const amrex::Real cse0 = 6.4745807555437135e-8 * state.rn[0];
	const amrex::Real cse1 = -cse0;
	const amrex::Real cse2 = 2.6299999999999999e-13 * state.xn[2];
	const amrex::Real cse3 = 2.6299999999999999e-13 * state.xn[1];
	const amrex::Real cse4 = 6.4745807555437135e-8 * state.xn[0];
	const amrex::Real cse5 = -cse4;
	const amrex::Real cse6 = -cse2;
	const amrex::Real cse7 = -cse3;

	jac(1, 1) = cse1;
	jac(1, 2) = cse2;
	jac(1, 3) = cse3;
	jac(1, 5) = cse5;
	jac(2, 1) = cse0;
	jac(2, 2) = cse6;
	jac(2, 3) = cse7;
	jac(2, 5) = cse4;
	jac(3, 1) = cse0;
	jac(3, 2) = cse6;
	jac(3, 3) = cse7;
	jac(3, 5) = cse4;
	jac(5, 1) = cse1;
	jac(5, 5) = cse5;
	jac(6, 1) = -6.4745807555437135e-8 * state.rn[2 * 0 + 1];
	jac(6, 6) = cse5;
	// $JAFF END
	// $JAFF SUB nspec, nbands
	int total_eqns = 3 + 2 + 1;
	for (int i = 1; i <= total_eqns; i++) {
		jac(4, i) /= state.rho;
	}
	// $JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
