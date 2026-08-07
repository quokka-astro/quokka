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
	const amrex::Real cse0 = 6.4745807555437137e-11 * state.xn[0] * state.rn[0];
	const amrex::Real cse1 = std::sqrt(state.T);
	const amrex::Real cse2 = 1.0 / state.T;
	const amrex::Real cse3 = cse0 + 5.84e-11 * cse1 * std::exp(-157896.75724967025 * cse2) * state.xn[0] * state.xn[2] -
				 1.6594178159829075e-10 * std::pow(state.T, -0.69999999999999996) * state.xn[1] * state.xn[2];
	const amrex::Real cse4 = state.xn[1] * state.xn[2];

	ydot(1) = -cse3;
	ydot(2) = cse3;
	ydot(3) = cse3;
	ydot(4) = -cse4 * (1.4e-27 * cse1 + 9.9999999999999998e-20 * std::exp(-118348.0 * cse2)) -
		  cse4 * ((state.T < 100.0) ? (0.0) : (8.4219589e-26 * std::pow(state.T, 0.10999999999999999))) -
		  (2.8000000000000002e-28 * cse1 * std::exp(-92.0 * cse2) + 2.0000000000000002e-19 * std::exp(-118400.0 / (state.T + 1000.0))) *
		      std::pow(state.xn[0], 2) +
		  4.1437316835479768e-22 * state.xn[0] * state.rn[0] + 2.0000000000000001e-26 * state.xn[0];
	ydot(5) = -cse0;
	ydot(6) = -6.4745807555437137e-11 * state.xn[0] * state.rn[2 * 0 + 1];
	// $JAFF END
	// $JAFF SUB nspec
	ydot(4) /= state.rho;
	// $JAFF END
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{

	// $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T RADIATION True USE_DEDT True]$
	const amrex::Real cse0 = 6.4745807555437137e-11 * state.rn[0];
	const amrex::Real cse1 = 1.0 / state.T;
	const amrex::Real cse2 = std::exp(-157896.75724967025 * cse1);
	const amrex::Real cse3 = cse2 * state.xn[2];
	const amrex::Real cse4 = std::sqrt(state.T);
	const amrex::Real cse5 = 5.84e-11 * cse4;
	const amrex::Real cse6 = cse0 + cse3 * cse5;
	const amrex::Real cse7 = 1.6594178159829075e-10 * std::pow(state.T, -0.69999999999999996);
	const amrex::Real cse8 = cse7 * state.xn[2];
	const amrex::Real cse9 = -cse2 * cse5 * state.xn[0] + cse7 * state.xn[1];
	const amrex::Real cse10 = state.xn[1] * state.xn[2];
	const amrex::Real cse11 = std::pow(state.T, -3.0 / 2.0);
	const amrex::Real cse12 = cse3 * state.xn[0];
	const amrex::Real cse13 = 1.0 / cse4;
	const amrex::Real cse14 = 1.1615924711880352e-10 * cse10 * std::pow(state.T, -1.7) + 9.2211706233807419e-6 * cse11 * cse12 + 2.92e-11 * cse12 * cse13;
	const amrex::Real cse15 = 0.66666666666670005 *
				  (1.6737729999999998e-24 * state.xn[0] + 1.6728620616289998e-24 * state.xn[1] + 9.1093837099999994e-28 * state.xn[2]) /
				  (1.380649e-16 * state.xn[0] + 1.380649e-16 * state.xn[1] + 1.380649e-16 * state.xn[2]);
	const amrex::Real cse16 = 6.4745807555437137e-11 * state.xn[0];
	const amrex::Real cse17 = -cse16;
	const amrex::Real cse18 = -cse8;
	const amrex::Real cse19 = -cse9;
	const amrex::Real cse20 = cse14 * cse15;
	const amrex::Real cse21 = state.T + 1000.0;
	const amrex::Real cse22 = std::exp(-118400.0 / cse21);
	const amrex::Real cse23 = std::exp(-92.0 * cse1);
	const amrex::Real cse24 = state.T < 100.0;
	const amrex::Real cse25 = ((cse24) ? (0.0) : (8.4219589e-26 * std::pow(state.T, 0.10999999999999999)));
	const amrex::Real cse26 = std::exp(-118348.0 * cse1);
	const amrex::Real cse27 = 9.9999999999999998e-20 * cse26 + 1.4e-27 * cse4;

	jac(1, 1) = -cse6;
	jac(1, 2) = cse8;
	jac(1, 3) = cse9;
	jac(1, 4) = -cse14 * cse15;
	jac(1, 5) = cse17;
	jac(2, 1) = cse6;
	jac(2, 2) = cse18;
	jac(2, 3) = cse19;
	jac(2, 4) = cse20;
	jac(2, 5) = cse16;
	jac(3, 1) = cse6;
	jac(3, 2) = cse18;
	jac(3, 3) = cse19;
	jac(3, 4) = cse20;
	jac(3, 5) = cse16;
	jac(4, 1) = 4.1437316835479768e-22 * state.rn[0] - 2 * state.xn[0] * (2.0000000000000002e-19 * cse22 + 2.8000000000000002e-28 * cse23 * cse4) +
		    2.0000000000000001e-26;
	jac(4, 2) = -cse25 * state.xn[2] - cse27 * state.xn[2];
	jac(4, 3) = -cse25 * state.xn[1] - cse27 * state.xn[1];
	jac(4, 4) = cse15 * (-cse10 * (6.9999999999999999e-28 * cse13 + 1.1834799999999999e-14 * cse26 / std::pow(state.T, 2)) -
			     cse10 * ((cse24) ? (0) : (9.2641547899999994e-27 * std::pow(state.T, -0.89000000000000001))) -
			     std::pow(state.xn[0], 2) * (2.5760000000000003e-26 * cse11 * cse23 + 1.4000000000000001e-28 * cse13 * cse23 +
							 2.3680000000000003e-14 * cse22 / std::pow(cse21, 2)));
	jac(4, 5) = 4.1437316835479768e-22 * state.xn[0];
	jac(5, 1) = -cse0;
	jac(5, 5) = cse17;
	jac(6, 1) = -6.4745807555437137e-11 * state.rn[2 * 0 + 1];
	jac(6, 6) = cse17;
	// $JAFF END
	// $JAFF SUB nspec, nbands
	for (int i = 1; i <= INT_NEQS; i++) {
		jac(4, i) /= state.rho;
	}
	// $JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
