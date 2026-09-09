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
	const amrex::Real cse0 = 1.0 / state.T;
	const amrex::Real cse1 = std::pow(cse0, 0.40699999999999997);
	const amrex::Real cse2 = std::pow(cse0, 1.5) * state.xn[2];
	const amrex::Real cse3 = state.xn[0] * state.xn[2];
	const amrex::Real cse4 = std::sqrt(state.T);
	const amrex::Real cse5 = 1.0 / (0.0031622776601683794 * cse4 + 1.0);
	const amrex::Real cse6 = cse4 * cse5;
	const amrex::Real cse7 = cse6 * std::exp(-157809.10000000001 * cse0);
	const amrex::Real cse8 = state.c_hat * state.rn[0];
	const amrex::Real cse9 = 3.387457892191344e-18 * cse8;
	const amrex::Real cse10 = state.c_hat * state.rn[1];
	const amrex::Real cse11 = 5.594274880278918e-19 * cse10;
	const amrex::Real cse12 = state.c_hat * state.rn[2];
	const amrex::Real cse13 = 1.0473472055970601e-19 * cse12;
	const amrex::Real cse14 = -4.8813573300130879e-6 * cse2 * std::pow(114.81327852071512 * cse1 + 1.0, -2.242) * state.xn[1] +
				  5.8500000000000005e-11 * cse3 * cse7 + (cse11 + cse13 + cse9) * state.xn[0];
	const amrex::Real cse15 = std::pow(cse0, 0.75);
	const amrex::Real cse16 = state.xn[2] * state.xn[4];
	const amrex::Real cse17 = 2.6161300353896405e-10 * cse15 * cse16;
	const amrex::Real cse18 = cse6 * std::exp(-285335.40000000002 * cse0) * state.xn[2] * state.xn[3];
	const amrex::Real cse19 = 2.3800000000000001e-11 * cse18;
	const amrex::Real cse20 = 1.6991057332915337e-19 * cse8;
	const amrex::Real cse21 = 4.2972652145734761e-18 * cse10;
	const amrex::Real cse22 = 1.649602553712884e-18 * cse12;
	const amrex::Real cse23 = (cse20 + cse21 + cse22) * state.xn[3];
	const amrex::Real cse24 = -cse17 + cse19 + cse23;
	const amrex::Real cse25 = 1.3616870968077311e-18 * cse12 * state.xn[4];
	const amrex::Real cse26 = cse6 * std::exp(-631515.0 * cse0);
	const amrex::Real cse27 =
	    5.68e-12 * cse16 * cse26 - 7.8154989015322843e-5 * cse2 * std::pow(201.88817541951113 * cse1 + 1.0, -2.242) * state.xn[5] + cse25;
	const amrex::Real cse28 = std::pow(cse0, 0.376);
	const amrex::Real cse29 = std::pow(cse0, 1.97) * state.T * state.xn[2];
	const amrex::Real cse30 = state.c_hat * state.rn[2 * 0 + 1];
	const amrex::Real cse31 = state.c_hat * state.rn[2 * 1 + 1];
	const amrex::Real cse32 = state.c_hat * state.rn[2 * 2 + 1];

	ydot(1) = -cse14;
	ydot(2) = cse14;
	ydot(3) = cse14 + cse24 + cse27;
	ydot(4) = -cse24;
	ydot(5) = -cse17 + cse19 + cse23 - cse27;
	ydot(6) = cse27;
	ydot(7) = 6.9047067617829618e-30 * state.c_hat * state.xn[4] * state.rn[2] -
		  cse16 * (3.611957317230672e-26 * cse15 * state.T + 4.9500000000000001e-22 * cse26 +
			   5.5399999999999998e-17 * cse5 * std::pow(state.T, -0.39700000000000002) * std::exp(-473638.0 * cse0) +
			   1.24e-13 * std::pow(state.T, -1.5) * (1.0 + 0.29999999999999999 * std::exp(-94000.0 * cse0)) * std::exp(-470000.0 * cse0)) -
		  9.3799999999999993e-22 * cse18 -
		  2.3402624934605136e-19 * cse29 * std::pow(86.15135782518999 * cse28 + 1.0, -3.7200000000000002) * state.xn[1] -
		  2.876084561370594e-17 * cse29 * std::pow(145.11453404889724 * cse28 + 1.0, -3.7200000000000002) * state.xn[5] -
		  cse3 * (7.4999999999999996e-19 * cse5 * std::exp(-118348.0 * cse0) + 1.2700000000000001e-21 * cse7) -
		  1.42e-27 * cse4 * (state.xn[1] + 1.0 * state.xn[4] + 4.0 * state.xn[5]) * state.xn[2] +
		  (3.9640472374168384e-30 * cse10 + 7.4213975641402087e-31 * cse12 + 2.4003187878278645e-29 * cse8) * state.xn[0] +
		  (7.6057297032735943e-29 * cse10 + 2.9196315598164334e-29 * cse12 + 3.0072472373526852e-30 * cse8) * state.xn[3];
	ydot(8) = -cse20 * state.xn[3] - cse9 * state.xn[0];
	ydot(9) = -3.387457892191344e-18 * cse30 * state.xn[0] - 1.6991057332915337e-19 * cse30 * state.xn[3];
	ydot(10) = -cse11 * state.xn[0] - cse21 * state.xn[3];
	ydot(11) = -5.594274880278918e-19 * cse31 * state.xn[0] - 4.2972652145734761e-18 * cse31 * state.xn[3];
	ydot(12) = -cse13 * state.xn[0] - cse22 * state.xn[3] - cse25;
	ydot(13) = -1.0473472055970601e-19 * cse32 * state.xn[0] - 1.649602553712884e-18 * cse32 * state.xn[3] - 1.3616870968077311e-18 * cse32 * state.xn[4];
	// $JAFF END
	// $JAFF SUB nspec
	ydot(7) /= state.rho;
	// $JAFF END
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{

	// $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT
	// True]$
	const amrex::Real cse0 = 3.387457892191344e-18 * state.c_hat;
	const amrex::Real cse1 = cse0 * state.rn[0];
	const amrex::Real cse2 = 5.594274880278918e-19 * state.c_hat;
	const amrex::Real cse3 = cse2 * state.rn[1];
	const amrex::Real cse4 = 1.0473472055970601e-19 * state.c_hat;
	const amrex::Real cse5 = cse4 * state.rn[2];
	const amrex::Real cse6 = std::sqrt(state.T);
	const amrex::Real cse7 = cse6 * state.xn[2];
	const amrex::Real cse8 = 0.0031622776601683794 * cse6 + 1.0;
	const amrex::Real cse9 = 1.0 / cse8;
	const amrex::Real cse10 = 1.0 / state.T;
	const amrex::Real cse11 = std::exp(-157809.10000000001 * cse10);
	const amrex::Real cse12 = cse11 * cse9;
	const amrex::Real cse13 = cse1 + 5.8500000000000005e-11 * cse12 * cse7 + cse3 + cse5;
	const amrex::Real cse14 = std::pow(cse10, 1.5);
	const amrex::Real cse15 = cse14 * state.xn[2];
	const amrex::Real cse16 = std::pow(cse10, 0.40699999999999997);
	const amrex::Real cse17 = 114.81327852071512 * cse16 + 1.0;
	const amrex::Real cse18 = std::pow(cse17, -2.242);
	const amrex::Real cse19 = 4.8813573300130879e-6 * cse18;
	const amrex::Real cse20 = cse15 * cse19;
	const amrex::Real cse21 = cse6 * cse9;
	const amrex::Real cse22 = cse11 * cse21;
	const amrex::Real cse23 = -cse14 * cse19 * state.xn[1] + 5.8500000000000005e-11 * cse22 * state.xn[0];
	const amrex::Real cse24 = std::pow(cse8, -2);
	const amrex::Real cse25 = cse11 * cse24;
	const amrex::Real cse26 = state.xn[0] * state.xn[2];
	const amrex::Real cse27 = std::pow(state.T, -3.0 / 2.0);
	const amrex::Real cse28 = cse12 * cse27;
	const amrex::Real cse29 = cse10 * state.xn[1];
	const amrex::Real cse30 = std::pow(cse10, 1.907) * state.xn[2];
	const amrex::Real cse31 = 1.0 / cse6;
	const amrex::Real cse32 = cse12 * cse31;
	const amrex::Real cse33 = 7.3220359950196314e-6 * cse15 * cse18 * cse29 - 0.00051140237013672606 * std::pow(cse17, -3.242) * cse29 * cse30 -
				  9.2496621559925108e-14 * cse25 * cse26 + 9.2318323500000018e-6 * cse26 * cse28 + 2.9250000000000002e-11 * cse26 * cse32;
	const amrex::Real cse34 = 0.66666666666670005 *
				  (1.6737729999999998e-24 * state.xn[0] + 1.6728620616289998e-24 * state.xn[1] + 9.1093837099999994e-28 * state.xn[2] +
				   6.6464729999999996e-24 * state.xn[3] + 6.6455620616289996e-24 * state.xn[4] + 6.6446511232579995e-24 * state.xn[5]) /
				  (1.380649e-16 * state.xn[0] + 1.380649e-16 * state.xn[1] + 1.380649e-16 * state.xn[2] + 1.380649e-16 * state.xn[3] +
				   1.380649e-16 * state.xn[4] + 1.380649e-16 * state.xn[5]);
	const amrex::Real cse35 = cse0 * state.xn[0];
	const amrex::Real cse36 = cse2 * state.xn[0];
	const amrex::Real cse37 = cse4 * state.xn[0];
	const amrex::Real cse38 = -cse20;
	const amrex::Real cse39 = std::pow(cse10, 0.75);
	const amrex::Real cse40 = 2.6161300353896405e-10 * cse39;
	const amrex::Real cse41 = std::exp(-285335.40000000002 * cse10);
	const amrex::Real cse42 = 2.3800000000000001e-11 * cse41;
	const amrex::Real cse43 = cse21 * state.xn[3];
	const amrex::Real cse44 = -cse40 * state.xn[4] + cse42 * cse43;
	const amrex::Real cse45 = 201.88817541951113 * cse16 + 1.0;
	const amrex::Real cse46 = std::pow(cse45, -2.242);
	const amrex::Real cse47 = 7.8154989015322843e-5 * cse46;
	const amrex::Real cse48 = cse14 * cse47 * state.xn[5];
	const amrex::Real cse49 = std::exp(-631515.0 * cse10);
	const amrex::Real cse50 = cse21 * cse49;
	const amrex::Real cse51 = 5.68e-12 * cse50 * state.xn[4];
	const amrex::Real cse52 = -cse48 + cse51;
	const amrex::Real cse53 = 1.6991057332915337e-19 * state.c_hat;
	const amrex::Real cse54 = cse53 * state.rn[0];
	const amrex::Real cse55 = 4.2972652145734761e-18 * state.c_hat;
	const amrex::Real cse56 = cse55 * state.rn[1];
	const amrex::Real cse57 = state.c_hat * state.rn[2];
	const amrex::Real cse58 = 1.649602553712884e-18 * cse57;
	const amrex::Real cse59 = cse7 * cse9;
	const amrex::Real cse60 = cse42 * cse59 + cse54 + cse56 + cse58;
	const amrex::Real cse61 = cse40 * state.xn[2];
	const amrex::Real cse62 = 1.3616870968077311e-18 * cse57;
	const amrex::Real cse63 = cse49 * cse9;
	const amrex::Real cse64 = cse62 + 5.68e-12 * cse63 * cse7;
	const amrex::Real cse65 = cse15 * cse47;
	const amrex::Real cse66 = -cse65;
	const amrex::Real cse67 = state.xn[2] * state.xn[4];
	const amrex::Real cse68 = cse41 * state.xn[3];
	const amrex::Real cse69 = cse68 * state.xn[2];
	const amrex::Real cse70 = cse27 * cse69 * cse9;
	const amrex::Real cse71 = cse31 * state.xn[2];
	const amrex::Real cse72 = cse68 * cse71 * cse9;
	const amrex::Real cse73 = 1.9620975265422304e-10 * cse10 * cse39 * cse67 - 3.7631104156003716e-14 * cse24 * cse69 + 6.7909825200000008e-6 * cse70 +
				  1.1900000000000001e-11 * cse72;
	const amrex::Real cse74 = cse24 * cse49;
	const amrex::Real cse75 = 8.9808685548781973e-15 * cse67 * cse74;
	const amrex::Real cse76 = cse27 * cse63;
	const amrex::Real cse77 = 3.5870051999999999e-6 * cse67 * cse76;
	const amrex::Real cse78 = cse10 * state.xn[5];
	const amrex::Real cse79 = 0.014397848749256066 * cse30 * std::pow(cse45, -3.242) * cse78;
	const amrex::Real cse80 = 0.00011723248352298426 * cse15 * cse46 * cse78;
	const amrex::Real cse81 = cse31 * cse63;
	const amrex::Real cse82 = 2.84e-12 * cse67 * cse81;
	const amrex::Real cse83 = -cse75 + cse77 - cse79 + cse80 + cse82;
	const amrex::Real cse84 = cse53 * state.xn[3];
	const amrex::Real cse85 = cse35 + cse84;
	const amrex::Real cse86 = cse55 * state.xn[3];
	const amrex::Real cse87 = cse36 + cse86;
	const amrex::Real cse88 = state.c_hat * state.xn[3];
	const amrex::Real cse89 = 1.649602553712884e-18 * cse88;
	const amrex::Real cse90 = state.c_hat * state.xn[4];
	const amrex::Real cse91 = 1.3616870968077311e-18 * cse90;
	const amrex::Real cse92 = cse37 + cse89 + cse91;
	const amrex::Real cse93 = state.c_hat * state.rn[0];
	const amrex::Real cse94 = state.c_hat * state.rn[1];
	const amrex::Real cse95 = std::exp(-118348.0 * cse10);
	const amrex::Real cse96 = cse9 * cse95;
	const amrex::Real cse97 = 1.2700000000000001e-21 * cse22 + 7.4999999999999996e-19 * cse96;
	const amrex::Real cse98 = 1.42e-27 * cse6;
	const amrex::Real cse99 = cse98 * state.xn[2];
	const amrex::Real cse100 = std::pow(cse10, 1.97);
	const amrex::Real cse101 = cse100 * state.T;
	const amrex::Real cse102 = cse101 * state.xn[2];
	const amrex::Real cse103 = std::pow(cse10, 0.376);
	const amrex::Real cse104 = 86.15135782518999 * cse103 + 1.0;
	const amrex::Real cse105 = std::pow(cse104, -3.7200000000000002);
	const amrex::Real cse106 = 2.3402624934605136e-19 * cse105;
	const amrex::Real cse107 = state.xn[1] + 1.0 * state.xn[4] + 4.0 * state.xn[5];
	const amrex::Real cse108 = 145.11453404889724 * cse103 + 1.0;
	const amrex::Real cse109 = std::pow(cse108, -3.7200000000000002);
	const amrex::Real cse110 = 2.876084561370594e-17 * cse109;
	const amrex::Real cse111 = 9.3799999999999993e-22 * cse41;
	const amrex::Real cse112 = std::exp(-473638.0 * cse10);
	const amrex::Real cse113 = cse112 * cse9;
	const amrex::Real cse114 = (1.0 + 0.29999999999999999 * std::exp(-94000.0 * cse10)) * std::exp(-470000.0 * cse10);
	const amrex::Real cse115 = 5.5399999999999998e-17 * cse113 * std::pow(state.T, -0.39700000000000002) + 1.24e-13 * cse114 * std::pow(state.T, -1.5) +
				   3.611957317230672e-26 * cse39 * state.T + 4.9500000000000001e-22 * cse50;
	const amrex::Real cse116 = std::pow(cse10, 2.3460000000000001) * state.xn[2];
	const amrex::Real cse117 = std::pow(state.T, -3.5);
	const amrex::Real cse118 = state.c_hat * state.xn[0];
	const amrex::Real cse119 = -cse85;
	const amrex::Real cse120 = -cse87;
	const amrex::Real cse121 = -cse92;
	const amrex::Real cse122 = state.c_hat * state.rn[2 * 2 + 1];

	jac(1, 1) = -cse13;
	jac(1, 2) = cse20;
	jac(1, 3) = -cse23;
	jac(1, 7) = -cse33 * cse34;
	jac(1, 8) = -cse35;
	jac(1, 10) = -cse36;
	jac(1, 12) = -cse37;
	jac(2, 1) = cse13;
	jac(2, 2) = cse38;
	jac(2, 3) = cse23;
	jac(2, 7) = cse33 * cse34;
	jac(2, 8) = cse35;
	jac(2, 10) = cse36;
	jac(2, 12) = cse37;
	jac(3, 1) = cse13;
	jac(3, 2) = cse38;
	jac(3, 3) = cse23 + cse44 + cse52;
	jac(3, 4) = cse60;
	jac(3, 5) = -cse61 + cse64;
	jac(3, 6) = cse66;
	jac(3, 7) = cse34 * (cse33 + cse73 + cse83);
	jac(3, 8) = cse85;
	jac(3, 10) = cse87;
	jac(3, 12) = cse92;
	jac(4, 3) = -cse44;
	jac(4, 4) = -cse60;
	jac(4, 5) = cse61;
	jac(4, 7) = -cse34 * cse73;
	jac(4, 8) = -cse84;
	jac(4, 10) = -cse86;
	jac(4, 12) = -cse89;
	jac(5, 3) = cse44 + cse48 - cse51;
	jac(5, 4) = cse60;
	jac(5, 5) = -cse61 - cse64;
	jac(5, 6) = cse65;
	jac(5, 7) = cse34 * (cse73 + cse75 - cse77 + cse79 - cse80 - cse82);
	jac(5, 8) = cse84;
	jac(5, 10) = cse86;
	jac(5, 12) = cse89 - cse91;
	jac(6, 3) = cse52;
	jac(6, 5) = cse64;
	jac(6, 6) = cse66;
	jac(6, 7) = cse34 * cse83;
	jac(6, 12) = cse91;
	jac(7, 1) = 7.4213975641402087e-31 * cse57 + 2.4003187878278645e-29 * cse93 + 3.9640472374168384e-30 * cse94 - cse97 * state.xn[2];
	jac(7, 2) = -cse102 * cse106 - cse99;
	jac(7, 3) =
	    -cse101 * cse106 * state.xn[1] - cse101 * cse110 * state.xn[5] - cse107 * cse98 - cse111 * cse43 - cse115 * state.xn[4] - cse97 * state.xn[0];
	jac(7, 4) = -cse111 * cse59 + 2.9196315598164334e-29 * cse57 + 3.0072472373526852e-30 * cse93 + 7.6057297032735943e-29 * cse94;
	jac(7, 5) = 6.9047067617829618e-30 * state.c_hat * state.rn[2] - cse115 * state.xn[2] - cse99;
	jac(7, 6) = -cse102 * cse110 - 5.6799999999999998e-27 * cse7;
	jac(7, 7) =
	    cse34 *
	    (2.2700546186566977e-19 * cse100 * cse105 * state.xn[1] * state.xn[2] + 2.7898020245294758e-17 * cse100 * cse109 * state.xn[2] * state.xn[5] -
	     2.8200543857749021e-17 * std::pow(cse104, -4.7200000000000006) * cse116 * state.xn[1] - 7.0999999999999998e-28 * cse107 * cse71 -
	     5.8377211647303819e-15 * std::pow(cse108, -4.7200000000000006) * cse116 * state.xn[5] +
	     1.4831082226189699e-24 * cse24 * cse41 * state.xn[2] * state.xn[3] -
	     cse26 * (-1.1858541225631422e-21 * cse24 * cse31 * cse95 - 2.0080463142069211e-24 * cse25 + 2.0041755700000002e-16 * cse28 +
		      6.3500000000000003e-22 * cse32 + 8.8760999999999989e-14 * cse96 / std::pow(state.T, 2)) -
	     cse67 * (-8.7595091186664108e-20 * cse112 * cse24 * std::pow(state.T, -0.89700000000000002) +
		      2.62395452e-11 * cse113 * std::pow(state.T, -2.3970000000000002) - 2.1993800000000001e-17 * cse113 * std::pow(state.T, -1.397) +
		      5.8280000000000003e-8 * cse114 * cse117 - 1.8600000000000002e-13 * cse114 * std::pow(state.T, -2.5) +
		      3.4968000000000002e-9 * cse117 * std::exp(-564000.0 * cse10) + 9.0298932930766814e-27 * cse39 - 7.8266372089167394e-25 * cse74 +
		      3.12599925e-16 * cse76 + 2.4750000000000001e-22 * cse81) -
	     2.6764460520000001e-16 * cse70 - 4.6899999999999997e-22 * cse72);
	jac(7, 8) = 2.4003187878278645e-29 * cse118 + 3.0072472373526852e-30 * cse88;
	jac(7, 10) = 3.9640472374168384e-30 * cse118 + 7.6057297032735943e-29 * cse88;
	jac(7, 12) = 7.4213975641402087e-31 * cse118 + 2.9196315598164334e-29 * cse88 + 6.9047067617829618e-30 * cse90;
	jac(8, 1) = -cse1;
	jac(8, 4) = -cse54;
	jac(8, 8) = cse119;
	jac(9, 1) = -cse0 * state.rn[2 * 0 + 1];
	jac(9, 4) = -cse53 * state.rn[2 * 0 + 1];
	jac(9, 9) = cse119;
	jac(10, 1) = -cse3;
	jac(10, 4) = -cse56;
	jac(10, 10) = cse120;
	jac(11, 1) = -cse2 * state.rn[2 * 1 + 1];
	jac(11, 4) = -cse55 * state.rn[2 * 1 + 1];
	jac(11, 11) = cse120;
	jac(12, 1) = -cse5;
	jac(12, 4) = -cse58;
	jac(12, 5) = -cse62;
	jac(12, 12) = cse121;
	jac(13, 1) = -cse4 * state.rn[2 * 2 + 1];
	jac(13, 4) = -1.649602553712884e-18 * cse122;
	jac(13, 5) = -1.3616870968077311e-18 * cse122;
	jac(13, 13) = cse121;
	// $JAFF END
	// $JAFF SUB nspec, nbands
	for (int i = 1; i <= INT_NEQS; i++) {
		jac(7, i) /= state.rho;
	}
	// $JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
