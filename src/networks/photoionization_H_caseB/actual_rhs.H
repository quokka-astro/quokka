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
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat  RADIATION True]$
	const amrex::Real cse0 = 2.1596876715103067e-18 * state.c_hat * state.xn[0] * state.rn[0];
	const amrex::Real cse1 = std::sqrt(state.T);
	const amrex::Real cse2 = 1.0 / state.T;
	const amrex::Real cse3 = cse1 * std::exp(-157896.75724967025 * cse2) * state.xn[0] * state.xn[2];
	const amrex::Real cse4 = cse0 + 5.84e-11 * cse3 - 1.6594178159829075e-10 * std::pow(state.T, -0.69999999999999996) * state.xn[1] * state.xn[2];
	const amrex::Real cse5 = state.T < 100.0;
	const amrex::Real cse6 = state.xn[1] * state.xn[2];

	ydot(1) = -cse4;
	ydot(2) = cse4;
	ydot(3) = cse4;
	ydot(4) =
	    4.4448471445392093e-29 * state.c_hat * state.xn[0] * state.rn[0] - 1.27251276978816e-21 * cse3 -
	    cse6 * (1.8551e-27 * cse1 +
		    ((cse5) ? (3.47e-29 * std::pow(state.T, 1.915))
			    : ((state.T < 630.957344480193)
				   ? (2.34e-26 * std::sqrt(state.T))
				   : ((state.T < 3981.0717055349733)
					  ? (1.11e-24 * std::pow(state.T, -0.099000000000000005))
					  : ((state.T < 10000.0)
						 ? (1.08e-32 * std::pow(state.T, 2.1269999999999998))
						 : ((state.T < 31622.776601683792)
							? (2.6700000000000001e-30 * std::pow(state.T, 1.5289999999999999))
							: ((state.T < 100000.0) ? (1.7399999999999998e-24 * std::pow(state.T, 0.23699999999999999))
										: ((state.T < 1000000.0) ? (1.1e-21 * std::pow(state.T, -0.32300000000000001))
													 : (7.4900000000000001e-21 *
													    std::pow(state.T, -0.46200000000000002)))))))))) -
	    cse6 * ((cse5) ? (0.0) : (8.4219589e-26 * std::pow(state.T, 0.10999999999999999))) -
	    (2.8000000000000002e-28 * cse1 * std::exp(-92.0 * cse2) + 2.0000000000000002e-19 * std::exp(-118400.0 / (state.T + 1000.0))) *
		std::pow(state.xn[0], 2) +
	    2.0000000000000001e-26 * state.xn[0];
	ydot(5) = -cse0;
	ydot(6) = -2.1596876715103067e-18 * state.c_hat * state.xn[0] * state.rn[2 * 0 + 1];
	// $JAFF END
	// $JAFF SUB nspec
	ydot(4) /= state.rho;
	// $JAFF END
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{

	// $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT
	// True]$
	const amrex::Real cse0 = 2.1596876715103067e-18 * state.c_hat;
	const amrex::Real cse1 = cse0 * state.rn[0];
	const amrex::Real cse2 = std::sqrt(state.T);
	const amrex::Real cse3 = 1.0 / state.T;
	const amrex::Real cse4 = std::exp(-157896.75724967025 * cse3);
	const amrex::Real cse5 = cse2 * cse4;
	const amrex::Real cse6 = 5.84e-11 * cse5;
	const amrex::Real cse7 = cse1 + cse6 * state.xn[2];
	const amrex::Real cse8 = 1.6594178159829075e-10 * std::pow(state.T, -0.69999999999999996);
	const amrex::Real cse9 = cse8 * state.xn[2];
	const amrex::Real cse10 = -cse6 * state.xn[0] + cse8 * state.xn[1];
	const amrex::Real cse11 = state.xn[1] * state.xn[2];
	const amrex::Real cse12 = std::pow(state.T, -3.0 / 2.0);
	const amrex::Real cse13 = cse4 * state.xn[0] * state.xn[2];
	const amrex::Real cse14 = cse12 * cse13;
	const amrex::Real cse15 = 1.0 / cse2;
	const amrex::Real cse16 = cse13 * cse15;
	const amrex::Real cse17 = 1.1615924711880352e-10 * cse11 * std::pow(state.T, -1.7) + 9.2211706233807419e-6 * cse14 + 2.92e-11 * cse16;
	const amrex::Real cse18 = 0.66666666666670005 *
				  (1.6737729999999998e-24 * state.xn[0] + 1.6728620616289998e-24 * state.xn[1] + 9.1093837099999994e-28 * state.xn[2]) /
				  (1.380649e-16 * state.xn[0] + 1.380649e-16 * state.xn[1] + 1.380649e-16 * state.xn[2]);
	const amrex::Real cse19 = cse0 * state.xn[0];
	const amrex::Real cse20 = -cse19;
	const amrex::Real cse21 = -cse9;
	const amrex::Real cse22 = -cse10;
	const amrex::Real cse23 = cse17 * cse18;
	const amrex::Real cse24 = 4.4448471445392093e-29 * state.c_hat;
	const amrex::Real cse25 = 1.27251276978816e-21 * cse5;
	const amrex::Real cse26 = state.T + 1000.0;
	const amrex::Real cse27 = std::exp(-118400.0 / cse26);
	const amrex::Real cse28 = std::exp(-92.0 * cse3);
	const amrex::Real cse29 = state.T < 100.0;
	const amrex::Real cse30 = ((cse29) ? (0.0) : (8.4219589e-26 * std::pow(state.T, 0.10999999999999999)));
	const amrex::Real cse31 = std::sqrt(state.T);
	const amrex::Real cse32 = state.T < 630.957344480193;
	const amrex::Real cse33 = state.T < 3981.0717055349733;
	const amrex::Real cse34 = state.T < 10000.0;
	const amrex::Real cse35 = state.T < 31622.776601683792;
	const amrex::Real cse36 = state.T < 100000.0;
	const amrex::Real cse37 = state.T < 1000000.0;
	const amrex::Real cse38 =
	    1.8551e-27 * cse2 +
	    ((cse29)
		 ? (3.47e-29 * std::pow(state.T, 1.915))
		 : ((cse32) ? (2.34e-26 * cse31)
			    : ((cse33) ? (1.11e-24 * std::pow(state.T, -0.099000000000000005))
				       : ((cse34) ? (1.08e-32 * std::pow(state.T, 2.1269999999999998))
						  : ((cse35) ? (2.6700000000000001e-30 * std::pow(state.T, 1.5289999999999999))
							     : ((cse36) ? (1.7399999999999998e-24 * std::pow(state.T, 0.23699999999999999))
									: ((cse37) ? (1.1e-21 * std::pow(state.T, -0.32300000000000001))
										   : (7.4900000000000001e-21 * std::pow(state.T, -0.46200000000000002)))))))));

	jac(1, 1) = -cse7;
	jac(1, 2) = cse9;
	jac(1, 3) = cse10;
	jac(1, 4) = -cse17 * cse18;
	jac(1, 5) = cse20;
	jac(2, 1) = cse7;
	jac(2, 2) = cse21;
	jac(2, 3) = cse22;
	jac(2, 4) = cse23;
	jac(2, 5) = cse19;
	jac(3, 1) = cse7;
	jac(3, 2) = cse21;
	jac(3, 3) = cse22;
	jac(3, 4) = cse23;
	jac(3, 5) = cse19;
	jac(4, 1) = cse24 * state.rn[0] - cse25 * state.xn[2] - 2 * state.xn[0] * (2.8000000000000002e-28 * cse2 * cse28 + 2.0000000000000002e-19 * cse27) +
		    2.0000000000000001e-26;
	jac(4, 2) = -cse30 * state.xn[2] - cse38 * state.xn[2];
	jac(4, 3) = -cse25 * state.xn[0] - cse30 * state.xn[1] - cse38 * state.xn[1];
	jac(4, 4) =
	    cse18 *
	    (-cse11 *
		 (9.2755000000000002e-28 * cse15 +
		  ((cse29) ? (6.6450500000000003e-29 * std::pow(state.T, 0.91500000000000004))
			   : ((cse32) ? (1.17e-26 / cse31)
				      : ((cse33) ? (-1.0989000000000001e-25 * std::pow(state.T, -1.099))
						 : ((cse34) ? (2.2971599999999999e-32 * std::pow(state.T, 1.1269999999999998))
							    : ((cse35) ? (4.0824300000000001e-30 * std::pow(state.T, 0.52899999999999991))
								       : ((cse36) ? (4.1237999999999993e-25 * std::pow(state.T, -0.76300000000000001))
										  : ((cse37) ? (-3.5530000000000001e-22 * std::pow(state.T, -1.323))
											     : (-3.4603799999999999e-21 * std::pow(state.T, -1.462)))))))))) -
	     cse11 * ((cse29) ? (0) : (9.2641547899999994e-27 * std::pow(state.T, -0.89000000000000001))) - 2.0092563990834663e-16 * cse14 -
	     6.3625638489408001e-22 * cse16 -
	     std::pow(state.xn[0], 2) *
		 (2.5760000000000003e-26 * cse12 * cse28 + 1.4000000000000001e-28 * cse15 * cse28 + 2.3680000000000003e-14 * cse27 / std::pow(cse26, 2)));
	jac(4, 5) = cse24 * state.xn[0];
	jac(5, 1) = -cse1;
	jac(5, 5) = cse20;
	jac(6, 1) = -cse0 * state.rn[2 * 0 + 1];
	jac(6, 6) = cse20;
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
