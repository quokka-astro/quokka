#ifndef BURN_TYPE_H
#define BURN_TYPE_H

#include <iomanip>

#include <AMReX_REAL.H>

#include <network.H>
#include <eos_type.H>
#include <eos_composition.H>
#include <extern_parameters.H>

#include <ArrayUtilities.H>

using namespace amrex::literals;
using namespace network_rp;

// A generic structure holding data necessary to do a nuclear burn.

// Set the number of independent variables -- this should be
// enuc + the number of species which participate
// in the evolution equations.

// Number of equations evolved by the Strang-split network solve.
#if defined (PHOTOCHEMISTRY)
const int neqs = 1 + NumSpec + NumRadEqs;
#else
const int neqs = 1 + NumSpec;
#endif

// Index of the energy variable in the work arrays.

const int net_ienuc = NumSpec + 1;

// this is the data type that is used to get the ydots from the actual
// RHS of the network.
using YdotNetArray1D = amrex::Array1D<amrex::Real, 1, neqs>;

// this is the data type of the dense Jacobian that the network wants.

// the Jacobian can have a different type that our system
#ifdef SINGLE_PRECISION_JACOBIAN
using jac_t = float;
#else
using jac_t = amrex::Real;
#endif

using  JacNetArray2D = ArrayUtil::MathArray2D<jac_t, 1, neqs, 1, neqs>;

struct burn_t
{

  // this first group are the quantities the network RHS uses

  amrex::Real rho{};
  amrex::Real T{};
  amrex::Real e{};
  amrex::Real xn[NumSpec]{};
#if NAUX_NET > 0
  amrex::Real aux[NumAux]{};
#endif
#ifdef PHOTOCHEMISTRY
  amrex::Real rn[NumRadEqs]{};
  amrex::Real c_hat{};
#endif

  // scaling used to make e we integrate dimensionless
  amrex::Real e_scale{};

  // derivatives needed for calling the EOS
  amrex::Real dedr{};
  amrex::Real dedT{};
  amrex::Real dedA{};
  amrex::Real dedZ{};

  // for diagnostics / error reporting

  int i{};
  int j{};
  int k{};

  // for drive_initial_convection, we will fix T during the
  // integration to a passed in value.  We will interpret a positive
  // T_fixed as setting this feature.
  amrex::Real T_fixed{-1.0};

  // all coupling types need the specific heats to transform the
  // reaction Jacobian elements from T to e
  amrex::Real cv{};

  amrex::Real mu{};
  amrex::Real mu_e{};
  amrex::Real y_e{};
  amrex::Real eta{};
  amrex::Real abar{};
  amrex::Real zbar{};


  // diagnostics
  int n_rhs{}, n_jac{}, n_step{};

  // Was the burn successful?
  bool success{};

};

inline
std::ostream& operator<< (std::ostream& o, burn_t const& burn_state)
{
  o << std::setprecision(12);
  o << "rho = " << burn_state.rho << std::endl;
  o << "T =   " << burn_state.T << std::endl;
  o << "xn = ";
  for (double X : burn_state.xn) {
    o << X << " ";
  }
#ifdef PHOTOCHEMISTRY
  o << "rn = ";
  for (double Rn : burn_state.rn) {
    o << Rn << " ";
  }
#endif
  o << std::endl;
  o << "(i, j, k) = " << burn_state.i << " " << burn_state.j << " " << burn_state.k << std::endl;
#if NAUX_NET > 0
  o << "aux = ";
  for (double A : burn_state.aux) {
    o << A << " ";
  }
  o << std::endl;
#endif

  return o;
}


template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void normalize_abundances_burn (BurnT& state)
{

    amrex::Real sum = 0.0_rt;
    for (double &x : state.xn) {
        x = amrex::max(network_rp::small_x, amrex::min(1.0_rt, x));
        sum += x;
    }
    for (double &x : state.xn) {
        x /= sum;
    }

}

#endif
