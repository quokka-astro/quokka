#ifndef actual_rhs_H
#define actual_rhs_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include <actual_network.H>
#include <burn_type.H>
#include <extern_parameters.H>
#include <jacobian_utilities.H>

using namespace ArrayUtil;
using namespace network_rp;

// Debug hook: abort with a full state/ydot dump on the Nth call to actual_rhs.
// Change this value to target a different call.
constexpr long debug_abort_call_count = 3;
// If true, print (but don't abort) on every call from 1 up to debug_abort_call_count,
// instead of only printing on the Nth call.
constexpr bool debug_print_up_to_n = true;

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

#if 0
  // Debug hook: skip printing/aborting until the call count reaches
  // debug_abort_call_count. The count is persisted to a temp file so it
  // survives across invocations (this RHS may be re-entered fresh each call).
  const char *counter_path = "/tmp/quokka_actual_rhs_call_count_dtype_photoionization_jaff.txt";
  long call_count = 0;
  {
    std::ifstream counter_in(counter_path);
    if (counter_in) {
      counter_in >> call_count;
    }
    counter_in.close();
    ++call_count;
    std::ofstream counter_out(counter_path, std::ios::trunc);
    counter_out << call_count;
    counter_out.close();

    if (call_count < debug_abort_call_count && !debug_print_up_to_n) {
      return;
    }
    if (call_count > debug_abort_call_count) {
      return;
    }
  }

  std::cout << std::setprecision(17);
  std::cout << "call_count = " << call_count << std::endl;
  std::cout << "---- burn_t state ----" << std::endl;
  std::cout << "state.time = " << state.time << std::endl;
  std::cout << "state.rho = " << state.rho << std::endl;
  std::cout << "state.T = " << state.T << std::endl;
  std::cout << "state.xn[0] (n_HI) = " << state.xn[0] << std::endl;
  std::cout << "state.xn[1] (n_HII) = " << state.xn[1] << std::endl;
  std::cout << "state.xn[2] (n_e) = " << state.xn[2] << std::endl;
  std::cout << "state.rn[0] = " << state.rn[0] << std::endl;
  std::cout << "state.rn[1] = " << state.rn[1] << std::endl;
  std::cout << "state.c_hat = " << state.c_hat << std::endl;
  std::cout << "state.n_rhs, state.n_jac, state.n_step = " << state.n_rhs << ", " << state.n_jac << ", " << state.n_step << std::endl;
  std::cout << "state.i, state.j, state.k = " << state.i << ", " << state.j << ", " << state.k << std::endl;
  std::cout << "---- ydot (outputs) ----" << std::endl;
  std::cout << "ydot(1) (n_HI) = " << ydot(1) << std::endl;
  std::cout << "ydot(2) (n_HII) = " << ydot(2) << std::endl;
  std::cout << "ydot(3) (n_e) = " << ydot(3) << std::endl;
  std::cout << "ydot(4) (net_ienuc) = " << ydot(4) << std::endl;
  std::cout << "ydot(5) (n_gamma) = " << ydot(5) << std::endl;
  std::cout << "ydot(6) (n_gamma_flux) = " << ydot(6) << std::endl;

  if (call_count == debug_abort_call_count) {
    std::remove(counter_path);
    amrex::Abort("Check ydot values in actual_rhs.H");
  }
#endif
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{

	// $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE dedt state.dedT RADIATION True USE_DEDT
	// True]$
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
	const amrex::Real cse10 = 1.0 / state.dedT;
	const amrex::Real cse11 = state.xn[1] * state.xn[2];
	const amrex::Real cse12 = std::pow(state.T, -3.0 / 2.0);
	const amrex::Real cse13 = cse3 * state.xn[0];
	const amrex::Real cse14 = 1.0 / cse4;
	const amrex::Real cse15 = 1.1615924711880352e-10 * cse11 * std::pow(state.T, -1.7) + 9.2211706233807419e-6 * cse12 * cse13 + 2.92e-11 * cse13 * cse14;
	const amrex::Real cse16 = 6.4745807555437137e-11 * state.xn[0];
	const amrex::Real cse17 = -cse16;
	const amrex::Real cse18 = -cse8;
	const amrex::Real cse19 = -cse9;
	const amrex::Real cse20 = cse10 * cse15;
	const amrex::Real cse21 = state.T + 1000.0;
	const amrex::Real cse22 = std::exp(-118400.0 / cse21);
	const amrex::Real cse23 = std::exp(-92.0 * cse1);
	const amrex::Real cse24 = state.T < 100.0;
	const amrex::Real cse25 = ((cse24) ? (0.0) : (8.4219589e-26 * std::pow(state.T, 0.10999999999999999)));
	const amrex::Real cse26 = std::exp(-118348.0 * cse1);
	const amrex::Real cse27 = 9.9999999999999998e-20 * cse26 + 1.4e-27 * cse4;

	jac.zero();

	jac(1, 1) = -cse6;
	jac(1, 2) = cse8;
	jac(1, 3) = cse9;
	jac(1, 4) = -cse10 * cse15;
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
	jac(4, 4) = cse10 * (-cse11 * (6.9999999999999999e-28 * cse14 + 1.1834799999999999e-14 * cse26 / std::pow(state.T, 2)) -
			     cse11 * ((cse24) ? (0) : (9.2641547899999994e-27 * std::pow(state.T, -0.89000000000000001))) -
			     std::pow(state.xn[0], 2) * (2.5760000000000003e-26 * cse12 * cse23 + 1.4000000000000001e-28 * cse14 * cse23 +
							 2.3680000000000003e-14 * cse22 / std::pow(cse21, 2)));
	jac(4, 5) = 4.1437316835479768e-22 * state.xn[0];
	jac(5, 1) = -cse0;
	jac(5, 5) = cse17;
	jac(6, 1) = -6.4745807555437137e-11 * state.rn[2 * 0 + 1];
	jac(6, 6) = cse17;
	// $JAFF END
	// $JAFF SUB nspec, nbands
	// LOCAL PATCH (not upstreamed): only the energy ROW (net_ienuc's own
	// ydot(nspec+1) was divided by state.rho in actual_rhs() above, converting
	// it from a volumetric rate (erg/cm^3/s) to a specific one (erg/g/s), so
	// every entry in that row must pick up the same 1/rho to stay consistent.
	// Non-energy rows' energy-COLUMN entries (species/radiation d(ydot_i)/de)
	// must NOT also be divided by rho: their numerator (ydot_i) was never
	// volumetric-to-specific converted, and the chain-rule division by dedt
	// (state.dedT, itself already specific, erg/g/K) already fully normalizes
	// that column. The previous version divided BOTH the row and the column by
	// rho, double-dividing the diagonal and spuriously dividing every other
	// row's energy-column entry by an extra, unjustified factor of rho.
	int total_eqns = 3 + 2 + 1;
	for (int i = 1; i <= total_eqns; i++) {
		jac(4, i) /= state.rho;
	}
	// $JAFF END

#if 0
  // Debug hook: print every jac(i,j) entry (i,j = 1..6) up to the Nth call to
  // actual_jac. Uses its own call counter/temp file, separate from
  // actual_rhs()'s, so the two hooks don't clobber each other's counts.
  const char *jac_counter_path = "/tmp/quokka_actual_jac_call_count_dtype_photoionization_jaff.txt";
  long jac_call_count = 0;
  {
    std::ifstream jac_counter_in(jac_counter_path);
    if (jac_counter_in) {
      jac_counter_in >> jac_call_count;
    }
    jac_counter_in.close();
    ++jac_call_count;
    std::ofstream jac_counter_out(jac_counter_path, std::ios::trunc);
    jac_counter_out << jac_call_count;
    jac_counter_out.close();

    if (jac_call_count < debug_abort_call_count && !debug_print_up_to_n) {
      return;
    }
    if (jac_call_count > debug_abort_call_count) {
      return;
    }
  }

  std::cout << std::setprecision(17);
  std::cout << "jac_call_count = " << jac_call_count << std::endl;
  for (int i = 1; i <= 6; ++i) {
    for (int j = 1; j <= 6; ++j) {
      std::cout << "jac(" << i << ", " << j << ") = " << jac(i, j) << std::endl;
    }
  }

  if (jac_call_count == debug_abort_call_count) {
    std::remove(jac_counter_path);
    amrex::Abort("Check jac values in actual_rhs.H");
  }
#endif
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
